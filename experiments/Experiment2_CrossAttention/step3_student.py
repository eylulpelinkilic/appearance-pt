"""Experiment 2, step 3: continued training of Sonata as a point-only student of the frozen teacher.

Targets: the trained teacher, frozen, is applied to every step-1 target point (>= 3 cameras; train scenes only) with
all of the point's cameras as keys and values (config student.target_keys = "all"); its weighted DINOv2 vector,
detached, is the point's target t(p).

Student: Sonata initialised from the same original checkpoint (sonata.pth at the pinned commit), with the pilot's
point inputs only (xyz, colour, normal; Sonata's own transform, 2 cm grid). Per step: one crop of a train scene (the
crop_points vertices nearest to a random target point), the student's per-point feature f(p) at the target points in
the crop (the pilot's upcast layer: all five stages concatenated, 1232 wide), a small prediction head h
(LayerNorm, 1232 -> 1024, GELU, 1024 -> 1024), and the loss

    L = mean over target points p in the crop of [ 1 - cos( h(f(p)), stop_gradient(t(p)) ) ].

Gradients reach the head and the chosen Sonata stages (xattn.TRAINABLE); everything else stays as in sonata.pth.
The model runs with norm statistics frozen and no stochastic depth (eval mode), fp16 autocast with loss scaling,
AdamW, linear warm-up then cosine decay, gradient clipping. This is continued training from Sonata, not
pretraining from scratch.

--mode smoke: measures, for each trainable choice x crop size in config, seconds per step and peak GPU memory over a
few steps (an out-of-memory is recorded, not fatal), checks that the first update changes the encoder, recommends a
choice and a step budget (student_budget_hours), and trains that choice for student_steps steps so that step 4 can
run end to end. The choice is then recorded in config.json (student_choice) before the full run; --mode full
refuses to train until it is. A checkpoint is written every checkpoint_minutes and at the end; a rerun (or a new
session with the earlier output attached) resumes from it.
Outputs: <out>/student/student.pth (the student's state_dict), train_log.json, checks.json, smoke.json (smoke).

Usage: python experiments/Experiment2_CrossAttention/step3_student.py --mode smoke|full
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from encoders import attention_backend, load_encoder, make_transform, seed_everything
from scenes import KEYS
from stages import Stages, is_oom
from xattn import (CONFIG, OUT, Clock, CrossAttentionTeacher, ObservationStore, PredictionHead, cosine_loss, crop_indices,
                   find_previous, gather_all, load_config, load_points_scene, lr_factor, mark_mode, peak_gpu_gb, set_trainable)

LOSS = "L = mean over target points p in the crop of [1 - cos(h(f(p)), stop_gradient(t(p)))]"
MAX_PEAK_GB = 13.5  # of a T4's 15 GB usable, leaving room for allocator fragmentation


def load_sonata(device: str):
    model, _ = load_encoder("sonata", seed=0)
    return model.to(device).eval(), make_transform("sonata")


def encode_train(model, transform, crop: dict, seed: int, device: str = "cuda") -> Stages:
    """One forward pass on a crop that keeps the autograd graph (encoders.encode_stages without inference mode)."""
    seed_everything(seed)
    point = transform({key: crop[key].copy() for key in KEYS})
    point = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in point.items()}
    with torch.autocast("cuda", dtype=torch.float16, enabled=device != "cpu"):
        point = model(point)
    return Stages.from_point(point, torch.float16 if device != "cpu" else torch.float32)


@torch.no_grad()
def make_targets(store: ObservationStore, teacher, device, batch: int = 2048) -> dict:
    """{scene: (vertex ids, targets (P, 1024) float16)}: the frozen teacher over all of each point's cameras."""
    out = []
    for s in range(0, len(store), batch):
        pts = np.arange(s, min(s + batch, len(store)))
        q, kv, mask = gather_all(store, pts, device)
        out.append(teacher(q, kv, mask)[0].half().cpu().numpy())
    target = np.concatenate(out) if out else np.zeros((0, store.obs_feat.shape[1]), np.float16)
    return {sc: (store.vertex[store.scene == i], target[store.scene == i]) for i, sc in enumerate(store.scenes)}


class Trainer:
    """Student, head, optimizer and the crop sampler; one call of step() is one update."""

    def __init__(self, model, transform, targets: dict, cache: Path, cfg: dict, trainable: str, crop_points: int,
                 steps: int, device: str, archives: dict | None = None):
        self.model, self.transform, self.targets, self.cache = model, transform, targets, Path(cache)
        self.cfg, self.crop_points, self.steps, self.device, self.archives = cfg, crop_points, steps, device, archives
        self.trainable = set_trainable(model, trainable)
        self.scenes = [s for s in targets if len(targets[s][0])]
        self.rng = np.random.default_rng(cfg["seed"])
        self.order, self.pos, self.current = [], 0, None
        self._load_next()
        with torch.no_grad():
            width = encode_train(model, transform, self._crop()[0], 0, device).width
        torch.manual_seed(cfg["seed"])
        self.head = PredictionHead(width, targets[self.scenes[0]][1].shape[1]).to(device)
        enc = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW([dict(params=enc, lr=cfg["lr"]), dict(params=self.head.parameters(), lr=cfg["head_lr"])],
                                     weight_decay=cfg["weight_decay"])
        self.base_lr = [cfg["lr"], cfg["head_lr"]]
        self.scaler = torch.amp.GradScaler("cuda", enabled=device != "cpu")
        self.step_no, self.curve, self.skipped = 0, [], 0

    def _load_next(self) -> None:
        if self.pos >= len(self.order):
            self.order, self.pos = list(self.rng.permutation(self.scenes)), 0
        scene = self.order[self.pos]
        self.pos += 1
        path = self.cache / f"{scene}.npz"
        if not path.exists():  # a new session: rebuild the point arrays from the core dataset
            s = load_points_scene(self.archives[scene], scene)
            self.cache.mkdir(parents=True, exist_ok=True)
            np.savez(path, coord=s["coord"].astype(np.float32), color=s["color"], normal=s["normal"].astype(np.float32))
        z = np.load(path)
        self.current = dict(scene=scene, coord=z["coord"], color=z["color"], normal=z["normal"], used=0)

    def _crop(self):
        """A crop around a random target point with at least min_targets targets inside (up to 5 tries)."""
        vertex, target = self.targets[self.current["scene"]]
        coord = self.current["coord"]
        for _ in range(5):
            centre = coord[vertex[self.rng.integers(len(vertex))]]
            idx = crop_indices(coord, centre, self.crop_points)
            pos = np.searchsorted(idx, vertex).clip(max=len(idx) - 1)
            inside = idx[pos] == vertex
            if inside.sum() >= self.cfg["min_targets"]:
                break
        crop = {k: self.current[k][idx] for k in KEYS}
        return crop, torch.from_numpy(pos[inside]), torch.from_numpy(target[inside].astype(np.float32))

    def step(self) -> dict:
        if self.current["used"] >= self.cfg["crops_per_scene"]:
            self._load_next()
        self.current["used"] += 1
        crop, local, target = self._crop()
        for _ in range(len(self.scenes)):  # too few targets around every centre tried: the next scene
            if len(local) >= self.cfg["min_targets"]:
                break
            self.skipped += 1
            self._load_next()
            crop, local, target = self._crop()
        assert len(local), "no scene has a crop with targets"
        for g, base in zip(self.opt.param_groups, self.base_lr):
            g["lr"] = base * lr_factor(self.step_no, self.steps, self.cfg["warmup"])
        t0 = time.perf_counter()
        stages = encode_train(self.model, self.transform, crop, int(self.rng.integers(2**31)), self.device)
        pred = self.head(stages.gather(local))
        loss = cosine_loss(pred, target.to(self.device))
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        enc_params = [p for p in self.model.parameters() if p.requires_grad]
        enc_grad = float(torch.norm(torch.stack([p.grad.norm() for p in enc_params if p.grad is not None]))) if enc_params else 0.0
        torch.nn.utils.clip_grad_norm_(enc_params + list(self.head.parameters()), self.cfg["grad_clip"])
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.device != "cpu":
            torch.cuda.synchronize()
        self.step_no += 1
        rec = dict(step=self.step_no, loss=float(loss.detach()), targets=int(len(local)), crop_vertices=len(crop["coord"]),
                   encoder_grad_norm=enc_grad, seconds=round(time.perf_counter() - t0, 3))
        self.curve.append(rec)
        return rec

    def state(self) -> dict:
        return dict(model=self.model.state_dict(), head=self.head.state_dict(), opt=self.opt.state_dict(),
                    scaler=self.scaler.state_dict(), step=self.step_no, curve=self.curve, rng=self.rng.bit_generator.state,
                    order=self.order, pos=self.pos)

    def load_state(self, st: dict) -> None:
        self.model.load_state_dict(st["model"])
        self.head.load_state_dict(st["head"])
        self.opt.load_state_dict(st["opt"])
        self.scaler.load_state_dict(st["scaler"])
        self.step_no, self.curve = st["step"], st["curve"]
        self.rng.bit_generator.state = st["rng"]
        self.order, self.pos = st["order"], st["pos"]


def encoder_changes(trainer: Trainer) -> dict:
    """One update: did a trainable encoder parameter change, and a frozen one stay?"""
    named = dict(trainer.model.named_parameters())
    train_name = trainer.trainable[-1]
    frozen = [n for n in named if n not in set(trainer.trainable)]
    before = {n: named[n].detach().clone() for n in [train_name] + frozen[:1]}
    rec = trainer.step()
    out = dict(loss=rec["loss"], encoder_grad_norm=rec["encoder_grad_norm"], trainable_parameter=train_name,
               trainable_changed=bool((named[train_name].detach() != before[train_name]).any()))
    if frozen:
        out.update(frozen_parameter=frozen[0], frozen_unchanged=bool((named[frozen[0]].detach() == before[frozen[0]]).all()))
    return out


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--points-cache", type=Path, default=Path("/tmp/exp2_points"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg_all = load_config(args.config, args.mode)
    run, cfg_s = cfg_all["run"], cfg_all["run"]["student"]
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    choice = cfg_all["student_choice"]
    if args.mode == "full" and not choice.get("recorded_from_smoke"):
        print("!! config.json student_choice is not recorded yet: run --mode smoke first and record its recommendation "
              "(trainable, crop_points, steps) in student_choice before the full run")
        return 3
    mark_mode(args.out, args.mode)
    out = args.out / "student"
    out.mkdir(parents=True, exist_ok=True)
    record = collect()
    print(f"git {json.dumps(record['git'])}\nmode {args.mode}: student {json.dumps(cfg_s)}")

    section("1. Targets from the frozen teacher")
    obs_summary = json.loads((args.out / "observations" / "summary.json").read_text())
    store = ObservationStore(args.out / "observations", obs_summary["scenes"])
    saved = torch.load(args.out / "teacher" / "teacher.pt", weights_only=False)
    teacher = CrossAttentionTeacher(saved["query_dim"], saved["key_dim"], saved["dim"]).to(dev)
    teacher.load_state_dict(saved["state_dict"])
    teacher.eval().requires_grad_(False)
    targets = make_targets(store, teacher, dev)
    del store
    print(f"  {sum(len(v[0]) for v in targets.values()):,} targets in {len(targets)} train scenes (keys: all cameras of a point)")
    archives = None
    if any(not (args.points_cache / f"{s}.npz").exists() for s in targets):
        from appearance_pt.scannetpp_store import find_scenes
        archives = {s: p[0] for s, p in find_scenes(args.input)["core"].items()}

    checks, smoke = {}, None
    with attention_backend("sdpa"):
        if args.mode == "smoke":
            section("2. Smoke: trainable layers x crop size (seconds per step, peak GPU memory)")
            opts = run["smoke_options"]
            table = []
            for trainable in opts["trainable"]:
                for crop_points in opts["crop_points"]:
                    rowr = dict(trainable=trainable, crop_points=crop_points)
                    try:
                        gc.collect()
                        if dev != "cpu":
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                        model, transform = load_sonata(dev)
                        tr = Trainer(model, transform, targets, args.points_cache, cfg_s, trainable, crop_points,
                                     opts["steps_each"], dev, archives)
                        first = encoder_changes(tr)
                        recs = [tr.step() for _ in range(opts["steps_each"] - 1)]
                        secs = [r["seconds"] for r in recs] or [first.get("seconds", 0)]
                        rowr.update(status="ok", seconds_per_step=round(float(np.median(secs)), 3), peak_gpu_gb=peak_gpu_gb(),
                                    trainable_parameters=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                                    first_update=first)
                    except RuntimeError as e:
                        if not is_oom(e):
                            raise
                        rowr.update(status="out of memory", peak_gpu_gb=peak_gpu_gb())
                    table.append(rowr)
                    print(f"  {json.dumps(rowr)}", flush=True)
                    tr = model = None
            rank = {name: i for i, name in enumerate(opts["trainable"])}
            fits = [r for r in table if r["status"] == "ok" and (r["peak_gpu_gb"] or 0) <= MAX_PEAK_GB
                    and r["first_update"]["trainable_changed"]]
            best = max(fits, key=lambda r: (rank[r["trainable"]], r["crop_points"]), default=None)
            budget_steps = None if best is None else int(run["student_budget_hours"] * 3600 / best["seconds_per_step"] // 100 * 100)
            smoke = dict(table=table, recommendation=None if best is None else dict(
                trainable=best["trainable"], crop_points=best["crop_points"], steps=budget_steps,
                seconds_per_step=best["seconds_per_step"], peak_gpu_gb=best["peak_gpu_gb"],
                rule=f"widest trainable set, then the largest crop, with peak GPU <= {MAX_PEAK_GB} GB and a first update that "
                     f"changes the encoder; steps = {run['student_budget_hours']} h / seconds per step"))
            (out / "smoke.json").write_text(json.dumps(smoke, indent=1))
            if best is None:
                print("!! no option fits; nothing to record")
                return 1
            trainable, crop_points, steps = best["trainable"], best["crop_points"], run["student_steps"]
        else:
            trainable, crop_points, steps = choice["trainable"], choice["crop_points"], choice["steps"]

        section(f"3. Training: {trainable}, crops of {crop_points:,} vertices, {steps} steps")
        tr = model = None
        gc.collect()
        if dev != "cpu":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model, transform = load_sonata(dev)
        tr = Trainer(model, transform, targets, args.points_cache, cfg_s, trainable, crop_points, steps, dev, archives)
        ckpt = out / "checkpoint.pt"
        prev = find_previous("student/checkpoint.pt", args.mode, roots=(args.out,), inputs=args.input)
        resumed_from = None
        if prev is not None:
            st = torch.load(prev, weights_only=False)
            if st.get("trainable") == trainable and st.get("crop_points") == crop_points and st.get("steps") == steps:
                tr.load_state(st)
                resumed_from = dict(path=str(prev), step=tr.step_no)
                if prev.resolve() != ckpt.resolve():
                    shutil.copy(prev, ckpt)
                print(f"  resumed from {prev} at step {tr.step_no}")
        if tr.step_no == 0:
            checks["first_update"] = encoder_changes(tr)
        clock = Clock(cfg_s["checkpoint_minutes"] * 60)
        t_start = time.time()
        while tr.step_no < steps:
            rec = tr.step()
            if rec["step"] % 50 == 0 or rec["step"] == steps:
                recent = tr.curve[-50:]
                print(f"  step {rec['step']}/{steps}: loss {np.mean([r['loss'] for r in recent]):.4f}, "
                      f"{np.median([r['seconds'] for r in recent]):.2f} s/step, peak GPU {peak_gpu_gb()} GB", flush=True)
            if clock.due():
                save_atomic(dict(tr.state(), trainable=trainable, crop_points=crop_points, steps=steps), ckpt)
        save_atomic(dict(tr.state(), trainable=trainable, crop_points=crop_points, steps=steps), ckpt)
        hours = (time.time() - t_start) / 3600

    torch.save(dict(state_dict=tr.model.state_dict(), encoder="sonata", init="sonata.pth (facebook/sonata @ df99897)",
                    trainable=trainable, crop_points=crop_points, steps=steps, git=record["git"]), out / "student.pth")
    curve = tr.curve
    log = dict(mode=args.mode, git=record["git"], description="continued training from Sonata (sonata.pth), point-only inputs",
               loss=LOSS, head="LayerNorm(1232) -> Linear(1232, 1024) -> GELU -> Linear(1024, 1024), discarded after training",
               target="frozen teacher, keys and values = all cameras of the point (>= 3), detached",
               trainable=trainable, trainable_parameters=int(sum(p.numel() for p in tr.model.parameters() if p.requires_grad)),
               crop_points=crop_points, steps=steps, config=cfg_s, resumed_from=resumed_from, skipped_crops=tr.skipped,
               seconds_per_step=float(np.median([r["seconds"] for r in curve])) if curve else None,
               session_hours=round(hours, 3), peak_gpu_gb=peak_gpu_gb(),
               loss_first_50=float(np.mean([r["loss"] for r in curve[:50]])) if curve else None,
               loss_last_50=float(np.mean([r["loss"] for r in curve[-50:]])) if curve else None,
               curve=[dict(step=r["step"], loss=r["loss"]) for r in curve])
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    (out / "checks.json").write_text(json.dumps(checks, indent=1))

    section("SUMMARY step 3 student (paste this block back)")
    if smoke is not None:
        for r in smoke["table"]:
            print(f"  {r['trainable']:7s} crop {r['crop_points']:>7,}: {r['status']}, "
                  f"{r.get('seconds_per_step', '-')} s/step, peak GPU {r.get('peak_gpu_gb')} GB, encoder changed after one update "
                  f"{r.get('first_update', {}).get('trainable_changed')}")
        print(f"recommendation (record in config.json student_choice before the full run): {json.dumps(smoke['recommendation'])}")
    if "first_update" in checks:
        print(f"first update: {json.dumps(checks['first_update'])}")
    print(f"{log['trainable']} ({log['trainable_parameters']:,} parameters), {steps} steps, {log['seconds_per_step']} s/step, "
          f"loss {log['loss_first_50']} -> {log['loss_last_50']}, {log['session_hours']} h, peak GPU {log['peak_gpu_gb']} GB; "
          f"saved {out / 'student.pth'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
