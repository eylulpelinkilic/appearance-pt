"""Experiment 2 (cross-attention teacher, point-only Sonata student): the pieces its steps share.

- Configuration (config.json next to this file) and the output layout under /kaggle/working/Experiment2_CrossAttention.
- The observation store written by step 1: per training scene, target points with their frozen Sonata feature (the
  teacher's query) and every DINOv2 observation of them (one per camera that sees the point; contiguous per point).
- Camera triples (two input cameras and a held-out third, all distinct) and the cross-attention teacher.
- The student's crops, trainable-layer choice, prediction head and loss.
- Point-only scene loading for evaluation, through an archive that refuses anything but the point cloud and the
  anonymisation list, so the evaluation cannot touch images, cameras, DINO features or teacher outputs.

Imports nothing from lifting.py or pipeline.py (and so no DINOv2 / transformers): step 4 checks that.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from appearance_pt import REPO_ROOT
from appearance_pt.scannetpp_store import SceneArchive, anonymised_vertices

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
OUT = Path("/kaggle/working/Experiment2_CrossAttention")
EXPERIMENT1 = REPO_ROOT / "experiments" / "Experiment1_Concatenation"


def load_config(path: Path = CONFIG, mode: str = "full") -> dict:
    """config.json with the mode's settings ("smoke" or "full") merged over the shared ones."""
    cfg = json.loads(Path(path).read_text())
    run = dict(cfg["shared"], mode=mode)
    for key, value in cfg["modes"][mode].items():  # a mode's dict (e.g. "probe") overrides key by key
        run[key] = {**run[key], **value} if isinstance(value, dict) and isinstance(run.get(key), dict) else value
    return dict(cfg, run=run)


def train_scenes(archives: dict, limit: int | None = None) -> list[str]:
    """The pilot's train scenes (configs/scannetpp_subset.json order) that are attached, the first `limit` of them."""
    subset = json.loads((REPO_ROOT / "configs" / "scannetpp_subset.json").read_text())
    return [s for s in subset["train"] if s in archives][:limit]


def val_scenes(archives: dict, limit: int | None = None) -> list[str]:
    subset = json.loads((REPO_ROOT / "configs" / "scannetpp_subset.json").read_text())
    return [s for s in subset["val"] if s in archives][:limit]


def dev_split(scenes: list[str], n_dev: int) -> list[str]:
    """Train scenes held out from teacher training for the diagnostic: the first n_dev of a seed-0 permutation (the
    pilot's step 4 draws its learning-rate hold-out the same way)."""
    order = np.random.default_rng(0).permutation(len(scenes))
    return [scenes[i] for i in sorted(order[:n_dev].tolist())]


def sample_points(segment: np.ndarray, anonymised: np.ndarray, n: int, seed: int) -> np.ndarray:
    """The pilot's labelled training sample (Experiment1 step3_features.sample_points; a test checks they agree)."""
    keep = segment[:, 0] >= 0
    keep[anonymised] = False
    cand = np.flatnonzero(keep)
    return np.sort(np.random.default_rng(seed).choice(cand, min(n, len(cand)), replace=False))


SKIP_DIRS = {"scenes", "wheelhouse"}  # the core datasets' scene folders hold ~100,000 files; the wheelhouse is not output


def walk_dirs(root: Path, max_depth: int = 7):
    """Directories below root, without descending into scene folders or the wheelhouse: a plain rglob over
    /kaggle/input with the ScanNet++ datasets attached takes minutes."""
    root = Path(root)
    base = len(root.parts)
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        depth = len(Path(dirpath).parts) - base
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if d not in SKIP_DIRS)
        yield Path(dirpath)


_PREVIOUS: dict = {}


def previous_runs(inputs: Path) -> list[Path]:
    """The Experiment2_CrossAttention folders of attached earlier runs (searched once per process)."""
    key = str(inputs)
    if key not in _PREVIOUS:
        _PREVIOUS[key] = [d for d in walk_dirs(inputs) if d.name == "Experiment2_CrossAttention"] if Path(inputs).exists() else []
    return _PREVIOUS[key]


def run_mode(root: Path) -> str | None:
    path = Path(root) / "run.json"
    return json.loads(path.read_text()).get("mode") if path.exists() else None


def mark_mode(out: Path, mode: str) -> None:
    """Record this run's mode in <out>/run.json; refuse to mix a smoke and a full run in one output folder."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    found = run_mode(out)
    if found not in (None, mode):
        raise SystemExit(f"!! {out} holds a {found} run; this is a {mode} run. Use another --out or start a new session.")
    (out / "run.json").write_text(json.dumps(dict(mode=mode)))


def find_previous(rel: str, mode: str, roots=(OUT,), inputs: Path = Path("/kaggle/input")) -> Path | None:
    """`rel` under this run's output, else under an attached earlier Experiment2_CrossAttention output of the same
    mode (resume). An earlier run of the other mode is ignored: a smoke run's small outputs must not end up in a full
    run."""
    for root in roots:
        if (Path(root) / rel).exists():
            return Path(root) / rel
    for prev in previous_runs(inputs):
        if (prev / rel).exists():
            if run_mode(prev) == mode:
                return prev / rel
            print(f"  ignoring {prev / rel}: it belongs to a {run_mode(prev)} run, this is a {mode} run")
    return None


def peak_gpu_gb() -> float | None:
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None


def dir_gb(path: Path) -> float:
    path = Path(path)
    return round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e9, 3) if path.exists() else 0.0


# ---- observation store (step 1 writes, steps 2 and 3 read) -------------------------------------------------------

def save_scene_observations(root: Path, scene: str, vertex: np.ndarray, query: np.ndarray, row: np.ndarray,
                            view: np.ndarray, uv: np.ndarray, feat: np.ndarray, stats: dict) -> Path:
    """One scene's target points (vertex ids, frozen Sonata features) and their observations (point row, camera,
    pixel, DINOv2 feature), regrouped so that each point's observations are contiguous, cameras in order."""
    order = np.lexsort((view, row))
    count = np.bincount(row, minlength=len(vertex)).astype(np.int16)
    path = Path(root) / "scenes" / f"{scene}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, vertex=vertex.astype(np.int64), query=query.astype(np.float16), count=count,
             offset=np.r_[0, np.cumsum(count)].astype(np.int64), obs_view=view[order].astype(np.int16),
             obs_uv=uv[order].astype(np.float32), obs_feat=feat[order].astype(np.float16),
             stats=np.array(json.dumps(stats)))
    tmp.replace(path)  # a scene is either complete or absent, so an interrupted step 1 resumes cleanly
    return path


class ObservationStore:
    """All scenes of a step-1 store in memory: points (query, count, offset into the observations) and observations."""

    def __init__(self, root: Path, scenes: list[str]):
        parts = [np.load(Path(root) / "scenes" / f"{s}.npz") for s in scenes]
        self.scenes = scenes
        self.scene = np.concatenate([np.full(len(p["vertex"]), i, np.int32) for i, p in enumerate(parts)])
        self.vertex = np.concatenate([p["vertex"] for p in parts])
        self.query = np.concatenate([p["query"] for p in parts])
        self.count = np.concatenate([p["count"] for p in parts]).astype(np.int64)
        self.offset = np.r_[0, np.cumsum(self.count)]
        self.obs_view = np.concatenate([p["obs_view"] for p in parts])
        self.obs_feat = np.concatenate([p["obs_feat"] for p in parts])

    def __len__(self) -> int:
        return len(self.vertex)

    def points_of(self, scene_ids) -> np.ndarray:
        return np.flatnonzero(np.isin(self.scene, list(scene_ids)))


# ---- camera triples and the teacher ------------------------------------------------------------------------------

def three_distinct(count: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(P, 3) distinct observation indices per point (each point has count >= 3), uniform over ordered triples:
    columns 0 and 1 are the input cameras, column 2 the held-out camera."""
    n = np.asarray(count, np.int64)
    assert (n >= 3).all(), "every point needs at least three cameras"
    a = (rng.random(len(n)) * n).astype(np.int64)
    b = (rng.random(len(n)) * (n - 1)).astype(np.int64)
    b += b >= a
    c = (rng.random(len(n)) * (n - 2)).astype(np.int64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    c += c >= lo
    c += c >= hi
    return np.stack([a, b, c], 1)


class CrossAttentionTeacher(torch.nn.Module):
    """query = W_q LN(Sonata feature), key_i = W_k LN(DINO feature of input camera i), value_i = that DINO feature as
    it is; output = softmax_i(query . key_i / sqrt(d)) weighted sum of the values, i.e. a point in DINO space. Only W_q
    and W_k (with biases) are trained; W_q starts at zero, so an untrained teacher is the uniform average."""

    def __init__(self, query_dim: int, key_dim: int, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.q = torch.nn.Linear(query_dim, dim)
        self.k = torch.nn.Linear(key_dim, dim)
        torch.nn.init.zeros_(self.q.weight)
        torch.nn.init.zeros_(self.q.bias)

    def weights(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q(F.layer_norm(query.float(), query.shape[-1:]))
        k = self.k(F.layer_norm(keys.float(), keys.shape[-1:]))
        logits = (q[:, None, :] * k).sum(-1) / self.dim ** 0.5
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits.softmax(-1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor | None = None):
        """query (B, Dq), keys (B, V, Dk) raw DINO features (also the values), mask (B, V) for padded cameras."""
        w = self.weights(query, keys, mask)
        return (w[..., None] * keys.float()).sum(1), w


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """mean over points of 1 - cos(pred, target); the teacher's and the student's loss."""
    return (1 - F.cosine_similarity(pred.float(), target.float(), dim=-1)).mean()


def gather_triples(store: ObservationStore, points: np.ndarray, triples: np.ndarray, device) -> tuple:
    """Query (B, Dq), input keys/values (B, 2, Dk) and held-out target (B, Dk) as float tensors on `device`."""
    obs = store.offset[points][:, None] + triples
    feats = torch.from_numpy(store.obs_feat[obs.reshape(-1)]).view(len(points), 3, -1).to(device).float()
    return torch.from_numpy(store.query[points]).to(device).float(), feats[:, :2], feats[:, 2]


def gather_all(store: ObservationStore, points: np.ndarray, device) -> tuple:
    """Query (B, Dq), every camera's DINO feature padded to (B, Vmax, Dk), and the padding mask (B, Vmax)."""
    count = store.count[points]
    vmax = int(count.max())
    local = np.arange(vmax)[None, :]
    mask = local < count[:, None]
    obs = np.where(mask, store.offset[points][:, None] + local, 0)
    feats = torch.from_numpy(store.obs_feat[obs.reshape(-1)]).view(len(points), vmax, -1).to(device).float()
    return torch.from_numpy(store.query[points]).to(device).float(), feats, torch.from_numpy(mask).to(device)


# ---- student --------------------------------------------------------------------------------------------------------

TRAINABLE = {  # parameter-name prefixes of Sonata's PTv3 (third_party/sonata/sonata/model.py: embedding, enc.enc0-4)
    "enc4": ("enc.enc4.",),
    "enc3-4": ("enc.enc3.", "enc.enc4."),
    "enc2-4": ("enc.enc2.", "enc.enc3.", "enc.enc4."),
    "all": ("",),
}


def set_trainable(model: torch.nn.Module, choice: str) -> list[str]:
    """requires_grad on the parameters of the chosen stages only; returns their names."""
    prefixes = TRAINABLE[choice]
    names = []
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(prefixes))
        if p.requires_grad:
            names.append(name)
    assert names, f"no parameter matches {prefixes}"
    return names


class PredictionHead(torch.nn.Module):
    """The student's small head from its per-point feature (the pilot's upcast layer) to DINOv2 space; discarded
    after training."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, hidden), torch.nn.GELU(),
                                       torch.nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def crop_indices(coord: np.ndarray, center: np.ndarray, n: int) -> np.ndarray:
    """The n vertices nearest to `center` (all of them if the scene is smaller), sorted."""
    if len(coord) <= n:
        return np.arange(len(coord))
    d = ((coord - center) ** 2).sum(1)
    return np.sort(np.argpartition(d, n - 1)[:n])


def lr_factor(step: int, steps: int, warmup: int) -> float:
    """Linear warm-up, then cosine decay to zero."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + np.cos(np.pi * min(step - warmup, steps - warmup) / max(steps - warmup, 1)))


class Clock:
    """Wall time since start, and whether `every` seconds have passed since the last tick (checkpoints)."""

    def __init__(self, every: float):
        self.start = self.last = time.time()
        self.every = every

    def due(self) -> bool:
        if time.time() - self.last >= self.every:
            self.last = time.time()
            return True
        return False

    @property
    def hours(self) -> float:
        return (time.time() - self.start) / 3600


# ---- point-only evaluation ------------------------------------------------------------------------------------------

POINT_ONLY = ("pointcept/", "scans/mesh_aligned_0.05_mask.txt")


class PointOnlyArchive(SceneArchive):
    """A scene archive that serves the point cloud and the anonymisation list and refuses everything else (DSLR and
    iPhone images, camera poses, masks); every name read is logged in READS."""

    READS: set = set()

    def read(self, name: str) -> bytes:
        if not name.startswith(POINT_ONLY):
            raise PermissionError(f"point-only evaluation tried to read {name}")
        PointOnlyArchive.READS.add(name.split("/")[0] if name.startswith("pointcept/") else name)
        return super().read(name)


def load_points_scene(path: Path, scene: str) -> dict:
    """coord, color, normal (the pilot's encoder inputs), segment and the anonymised vertex ids, nothing else."""
    with PointOnlyArchive(path) as arc:
        split = arc.names("pointcept/")[0].split("/")[1]
        pc = f"pointcept/{split}/{scene}/"
        s = {k: arc.npy(f"{pc}{k}.npy") for k in ("coord", "color", "normal", "segment")}
        s["anonymised"] = np.unique(anonymised_vertices(arc))
    return s


def prefetch_points(items: list[tuple[str, Path]], workers: int = 3, ahead: int = 4):
    """Yields (scene, point-only scene or the exception, seconds waited) in order, loading ahead in threads (as
    pipeline.prefetch, without importing it: pipeline pulls in the image code)."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(workers) as pool:
        queue, it = deque(), iter(items)

        def submit():
            nxt = next(it, None)
            if nxt is not None:
                queue.append((nxt[0], pool.submit(load_points_scene, nxt[1], nxt[0])))

        for _ in range(ahead):
            submit()
        while queue:
            scene, future = queue.popleft()
            submit()
            t0 = time.perf_counter()
            try:
                s = future.result()
            except Exception as e:  # noqa: BLE001  (reported per scene by the caller)
                s = e
            yield scene, s, round(time.perf_counter() - t0, 2)


def strip_to_points(path: Path, scene: str, dest: Path) -> Path:
    """A copy of a scene archive with only the files point-only evaluation may read (no dslr/, no iphone/): the
    image-free self-check of step 4 runs the student on it."""
    with SceneArchive(path) as arc:
        for name in arc.names("pointcept/") + ["scans/mesh_aligned_0.05_mask.txt"]:
            target = Path(dest) / scene / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(arc.read(name))
    return Path(dest) / scene

