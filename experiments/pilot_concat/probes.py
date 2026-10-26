"""Linear probes on frozen per-point features (pilot steps 1b and 4) and their application in step 5."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from appearance_pt.smoke import semseg_metrics


def split_probe(feat: torch.Tensor, gt: np.ndarray, x: np.ndarray, num_classes: int = 20, steps: int = 300,
                seed: int = 0) -> dict:
    """Per-point features (N, C): a linear head trained on the points on one side of the median x and scored on the
    other side, both ways (features standardised on the training half; AdamW, full batch). A health check of a
    feature set on one room, not a benchmark."""
    dev = feat.device
    y = torch.from_numpy(gt).to(dev)
    right = torch.from_numpy(x >= np.median(x)).to(dev)
    folds = []
    for train in (~right, right):
        tr = train & (y >= 0)
        xt = feat[tr].float()  # a normal tensor even if feat came out of inference mode
        mu, sd = xt.mean(0), xt.std(0) + 1e-6
        xt = (xt - mu) / sd
        torch.manual_seed(seed)
        head = torch.nn.Linear(xt.shape[1], num_classes).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad()
            F.cross_entropy(head(xt), y[tr]).backward()
            opt.step()
        with torch.no_grad():
            pred = head((feat[~train].float() - mu) / sd).argmax(1).cpu().numpy()
        folds.append(semseg_metrics(pred, gt[(~train).cpu().numpy()], num_classes=num_classes))
    return dict(miou=round(float(np.mean([f["miou"] for f in folds])), 4), acc=round(float(np.mean([f["acc"] for f in folds])), 4),
                miou_folds=[round(f["miou"], 4) for f in folds])


# ---- step 4: one linear probe per configuration, on the step-3 feature store ----------------------------------------

NUM_CLASSES = 100  # ScanNet++ top-100 semantic benchmark
EQ_WIDTH = 512  # "-eq" configs: every block PCA-reduced to this width before concatenation
CONFIGS = {  # name: (blocks, equal-width)
    "A": (("dino",), False),
    "B": (("utonia",), False),
    "C": (("utonia", "dino"), False),
    "C-eq": (("utonia", "dino"), True),
    "S": (("sonata",), False),
    "S+": (("sonata", "dino"), False),
    "K": (("concerto",), False),
    "K+": (("concerto", "dino"), False),
    "K+-eq": (("concerto", "dino"), True),
    "R": (("random",), False),
    "R+": (("random", "dino"), False),
    "X": (("raw",), False),
}


class FeatureWriter:
    """Step-3 output: <root>/<block>.npy (rows x width, memory-mapped), label.npy (Pointcept segment[:, 0]),
    scene.npy (scene index), views.npy (DSLR views that see the point), meta.json."""

    def __init__(self, root, n_max: int, widths: dict, dtypes: dict):
        from numpy.lib.format import open_memmap

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.widths, self.dtypes = widths, dtypes
        self.arrays = {b: open_memmap(self.root / f"{b}.npy", "w+", dtypes[b], (n_max, w)) for b, w in widths.items()}
        self.columns = {k: open_memmap(self.root / f"{k}.npy", "w+", np.int16, (n_max,)) for k in ("label", "scene", "views")}
        self.n, self.scenes = 0, []

    def add(self, scene: str, blocks: dict, label: np.ndarray, views: np.ndarray) -> None:
        rows = slice(self.n, self.n + len(label))
        for b, arr in blocks.items():
            self.arrays[b][rows] = arr
        self.columns["label"][rows] = label
        self.columns["scene"][rows] = len(self.scenes)
        self.columns["views"][rows] = np.minimum(views, 2**15 - 1)
        self.scenes.append(dict(id=scene, start=self.n, n=len(label)))
        self.n += len(label)

    def close(self, **meta) -> dict:
        for arr in (*self.arrays.values(), *self.columns.values()):
            arr.flush()
        meta = dict(n=self.n, widths=self.widths, dtypes={b: np.dtype(d).name for b, d in self.dtypes.items()},
                    scenes=self.scenes, **meta)
        (self.root / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        return meta


class FeatureStore:
    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.widths = self.meta["widths"]
        self.arrays = {b: np.load(self.root / f"{b}.npy", mmap_mode="r") for b in self.widths}
        self.label = np.load(self.root / "label.npy", mmap_mode="r")
        self.views = np.load(self.root / "views.npy", mmap_mode="r")

    @property
    def scenes(self) -> list[dict]:
        return self.meta["scenes"]

    def read(self, scene_ids: list[int], blocks) -> tuple[dict, np.ndarray]:
        """All rows of these scenes (in this order): {block: array}, labels."""
        ranges = [(self.scenes[i]["start"], self.scenes[i]["start"] + self.scenes[i]["n"]) for i in scene_ids]
        cat = lambda a: np.concatenate([a[s:e] for s, e in ranges])  # noqa: E731  (contiguous reads)
        return {b: cat(self.arrays[b]) for b in blocks}, cat(self.label).astype(np.int64)

    def groups(self, scene_ids, size: int, rng=None):
        """Scene ids in groups of `size` (shuffled with rng)."""
        ids = list(scene_ids) if rng is None else list(rng.permutation(list(scene_ids)))
        return [ids[i:i + size] for i in range(0, len(ids), size)]


PHASES = (("concerto", "dino"), ("utonia", "dino"), ("sonata", "dino"), ("random", "dino"), ("raw",))  # step 4 trains
# the configurations of one phase together, with only that phase's blocks in memory (<= ~11 GB for 2M rows)


def phase_of(config: str) -> tuple:
    return next(ph for ph in PHASES if set(CONFIGS[config][0]) <= set(ph))


def place(data: dict, label: np.ndarray, device, reserve_gb: float = 2.5) -> tuple[dict, torch.Tensor]:
    """Blocks and labels as torch tensors: on the GPU if they fit with `reserve_gb` to spare, else on the CPU
    (zero-copy) and moved batch by batch."""
    nbytes = sum(a.nbytes for a in data.values())
    fits = device != "cpu" and torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > nbytes + reserve_gb * 1e9
    where = device if fits else "cpu"
    return {b: torch.from_numpy(a).to(where) for b, a in data.items()}, torch.from_numpy(label).to(where)


def block_stats(data: dict, chunk: int = 100_000) -> dict:
    """{block: (mean, std)} float32 numpy, accumulated in float64 over chunks of rows."""
    out = {}
    for b, x in data.items():
        s1, s2 = 0.0, 0.0
        for start in range(0, len(x), chunk):
            c = x[start:start + chunk].double()
            s1, s2 = s1 + c.sum(0), s2 + (c**2).sum(0)
        mean = s1 / len(x)
        std = (s2 / len(x) - mean**2).clamp(min=0).sqrt()
        out[b] = (mean.float().cpu().numpy(), std.float().cpu().numpy() + 1e-6)
    return out


def fit_pca(x: torch.Tensor, stats_b: tuple, width: int = EQ_WIDTH, n_rows: int = 200_000, seed: int = 0,
            device="cuda") -> torch.Tensor:
    """(W, width) leading principal directions of a standardised block, from up to n_rows random rows."""
    g = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(x), generator=g)[:n_rows].sort().values
    mean, std = (torch.from_numpy(a).to(device) for a in stats_b)
    z = (x[pick.to(x.device)].to(device).float() - mean) / std
    _, vecs = torch.linalg.eigh(z.T @ z / len(z))
    return vecs[:, -width:].flip(1).contiguous()


class Probe(torch.nn.Module):
    """A linear head on standardised blocks, each optionally projected by a fixed matrix (the -eq configs)."""

    def __init__(self, blocks, widths: dict, proj: dict | None = None, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.blocks, self.proj = tuple(blocks), proj
        dims = sum(proj[b].shape[1] if proj else widths[b] for b in self.blocks)
        self.linear = torch.nn.Linear(dims, num_classes)

    def forward(self, x: dict) -> torch.Tensor:
        return self.linear(torch.cat([x[b] @ self.proj[b] if self.proj else x[b] for b in self.blocks], dim=1))

    def fold(self, stats: dict) -> dict:
        """Weights on the raw (unstandardised, unprojected) blocks: logits = sum_b x_b @ A_b.T + bias."""
        w, bias = self.linear.weight.detach().float(), self.linear.bias.detach().float().clone()
        out, offset = {}, 0
        for b in self.blocks:
            mean, std = (torch.from_numpy(a).to(w.device) for a in stats[b])
            d = self.proj[b].shape[1] if self.proj else len(mean)
            a = w[:, offset:offset + d]
            offset += d
            if self.proj:
                a = a @ self.proj[b].T
            a = a / std
            bias -= a @ mean
            out[b] = a.cpu()
        return dict(blocks=out, bias=bias.cpu())


def standardise(data: dict, idx, stats_t: dict, device) -> dict:
    return {b: (x[idx].to(device, non_blocking=True).float() - stats_t[b][0]) / stats_t[b][1] for b, x in data.items()}


def train_probes(data: dict, label: torch.Tensor, probes: dict, lrs: dict, stats: dict, epochs: int, batch: int = 4096,
                 seed: int = 0, weight_decay: float = 1e-4, device="cuda", log=print) -> list[float]:
    """Train every probe in `probes` ({key: Probe}) together on the rows of `data` ({block: tensor}, from place()):
    the same shuffled batches, AdamW with each probe's learning rate (lrs[key]), cosine decay to 0, cross-entropy.
    Returns the mean loss per probe for each epoch."""
    stats_t = {b: tuple(torch.from_numpy(a).to(device) for a in stats[b]) for b in data}
    for p in probes.values():
        p.to(device).train()
    opt = torch.optim.AdamW([dict(params=p.parameters(), lr=lrs[k]) for k, p in probes.items()], weight_decay=weight_decay)
    n = len(label)
    total = epochs * -(-n // batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.5 * (1 + np.cos(np.pi * min(step, total) / total)))
    g, curve = torch.Generator().manual_seed(seed), []
    torch.manual_seed(seed)
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g).to(label.device)
        loss_sum = torch.zeros((), device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            x = standardise(data, idx, stats_t, device)
            y = label[idx].to(device).long()
            loss = sum(F.cross_entropy(p(x), y, ignore_index=-1) for p in probes.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach()
        curve.append(round(float(loss_sum) / -(-n // batch) / len(probes), 4))  # one GPU sync per epoch
        log(f"    epoch {epoch + 1}/{epochs}: mean loss per probe {curve[-1]}")
    return curve


@torch.no_grad()
def evaluate_probes(data: dict, label: torch.Tensor, probes: dict, stats: dict, batch: int = 16384, device="cuda") -> dict:
    """Point-level mIoU (classes in GT or prediction) of every probe on the rows of `data`."""
    stats_t = {b: tuple(torch.from_numpy(a).to(device) for a in stats[b]) for b in data}
    preds = {k: [] for k in probes}
    for s in range(0, len(label), batch):
        x = standardise(data, slice(s, s + batch), stats_t, device)
        for k, p in probes.items():
            preds[k].append(p.eval()(x).argmax(1).cpu().numpy())
    gt = label.cpu().numpy().astype(np.int64)
    return {k: round(semseg_metrics(np.concatenate(v), gt, NUM_CLASSES)["miou"], 4) for k, v in preds.items()}


class FoldedSet:
    """Folded probes ({key: Probe.fold()}) applied together: one matrix product per block for all probes using it."""

    def __init__(self, folded: dict, device="cuda"):
        self.keys, self.device = list(folded), device
        self.bias = {k: folded[k]["bias"].to(device) for k in self.keys}
        self.mats, self.owners = {}, {}
        for b in sorted({b for f in folded.values() for b in f["blocks"]}):
            users = [k for k in self.keys if b in folded[k]["blocks"]]
            self.mats[b] = torch.cat([folded[k]["blocks"][b] for k in users]).to(device)
            self.owners[b] = users

    @property
    def blocks(self) -> list[str]:
        return list(self.mats)

    def logits(self, feats: dict) -> dict:
        """feats: {block: (n, W_b)} tensors on the device; returns {key: (n, classes)} float32."""
        out = {k: self.bias[k].expand(len(next(iter(feats.values()))), -1).clone() for k in self.keys}
        for b, mat in self.mats.items():
            z = feats[b].float() @ mat.T
            start = 0
            for k in self.owners[b]:
                c = len(self.bias[k])
                out[k] += z[:, start:start + c]
                start += c
        return out
