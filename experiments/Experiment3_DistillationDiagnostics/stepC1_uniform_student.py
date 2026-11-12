"""Experiment 3, Run C, part 1: a second student, identical to Experiment 2's step 3 except for the target.

Target: t_mean(p), the uniform average of all of the point's DINOv2 observations (the pilot's lifted feature), instead of
the frozen cross-attention teacher's weighted average. Everything else is Experiment 2's: the same observation store
(the same 409,600 target points), sonata.pth, the recorded student_choice (trainable layers, crop size, steps), the
student settings in Experiment 2's config.json (optimizer, schedule, head, seed 0), and Experiment 2's own Trainer
(step3_student.Trainer), whose random stream does not depend on the target values, so the crops and voxel samples come
in the same order.
Checks: the two students' configurations differ only in the target (printed diff); t_mean equals the pilot's lift() on
lift_check_scenes train scenes (recomputed from the images; t_mean is averaged from the stored fp16 observations, so
the difference is fp16 rounding); one update reaches the encoder. Also reported: cos(t_attn, t_mean) on the train
targets. Checkpoints every 30 min; a rerun, or a new session with this output attached, resumes.
Outputs: <out>/runC/student/ (student.pth, checkpoint.pt with the head, train_log.json, checks.json).

Usage: python experiments/Experiment3_DistillationDiagnostics/stepC1_uniform_student.py --mode smoke|full
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_store import find_scenes
from encoders import attention_backend
from lifting import OCCLUSION, lift, load_dino
from pipeline import add_dino, scene_views
from scenes import load_core_scene
from step3_student import LOSS, Trainer, encoder_changes, load_sonata, make_targets, save_atomic
from xattn import Clock, CrossAttentionTeacher, ObservationStore, gather_all, mark_mode, peak_gpu_gb, run_mode
from xdiag import CONFIG, EXP2_CONFIG, OUT, find_exp2, find_runs, load_config, write_json


@torch.no_grad()
def uniform_targets(store: ObservationStore, device, batch: int = 2048) -> dict:
    """{scene: (vertex ids, t_mean (P, 1024) float16)} in Experiment 2's make_targets layout."""
    out = []
    for s in range(0, len(store), batch):
        q, kv, mask = gather_all(store, np.arange(s, min(s + batch, len(store))), device)
        out.append(((kv * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)).half().cpu().numpy())
    target = np.concatenate(out)
    return {sc: (store.vertex[store.scene == i], target[store.scene == i]) for i, sc in enumerate(store.scenes)}


def student_config(cfg2: dict, target: str, steps: int) -> dict:
    choice = cfg2["student_choice"]
    return dict(target=target, init="sonata.pth (facebook/sonata @ df99897)", trainable=choice["trainable"],
                crop_points=choice["crop_points"], steps=steps, head="LayerNorm(1232) -> 1024 -> GELU -> 1024",
                loss=LOSS.replace("t(p)", "target(p)"), **cfg2["shared"]["student"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--exp2-config", type=Path, default=EXP2_CONFIG)
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--exp2", type=Path)
    ap.add_argument("--any-exp2", action="store_true", help="tests only")
    ap.add_argument("--points-cache", type=Path, default=Path("/tmp/exp3_points"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run = load_config(args.config, args.mode)["run"]
    cfg2 = json.loads(args.exp2_config.read_text())
    if not cfg2["student_choice"].get("recorded_from_smoke"):
        print("!! Experiment 2's config.json has no recorded student_choice")
        return 3
    steps = cfg2["student_choice"]["steps"] if args.mode == "full" else run["c"]["steps"]
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    mark_mode(args.out, args.mode)
    out = args.out / "runC" / "student"
    out.mkdir(parents=True, exist_ok=True)
    record = collect()
    print(f"git {json.dumps(record['git'])}\nmode {args.mode}: {steps} steps")

    section("1. Targets: uniform average and, for comparison, the attention teacher's")
    exp2 = find_exp2(args.input, args.exp2, args.any_exp2)
    obs_summary = json.loads((exp2 / "observations" / "summary.json").read_text())
    store = ObservationStore(exp2 / "observations", obs_summary["scenes"])
    t_mean = uniform_targets(store, dev)
    saved = torch.load(exp2 / "teacher" / "teacher.pt", weights_only=False, map_location="cpu")
    teacher = CrossAttentionTeacher(saved["query_dim"], saved["key_dim"], saved["dim"]).to(dev)
    teacher.load_state_dict(saved["state_dict"])
    t_attn = make_targets(store, teacher.eval().requires_grad_(False), dev)
    a = torch.from_numpy(np.concatenate([t_attn[s][1] for s in store.scenes])).float()
    m = torch.from_numpy(np.concatenate([t_mean[s][1] for s in store.scenes])).float()
    cos = F.cosine_similarity(a, m, dim=-1).numpy()
    count = store.count
    attn_vs_mean = dict(points=len(cos), mean=float(cos.mean()), p05=float(np.percentile(cos, 5)), p50=float(np.median(cos)),
                        by_cameras={k: float(cos[sel].mean()) for k, sel in (("3", count == 3), ("4-5", (count >= 4) & (count <= 5)),
                                                                              (">=6", count >= 6)) if sel.any()})
    print(f"  {len(cos):,} targets; cos(t_attn, t_mean) mean {attn_vs_mean['mean']:.4f}, 5th percentile {attn_vs_mean['p05']:.4f}")

    section("2. Checks before training: config diff, t_mean = the pilot's lifted feature")
    c_attn = student_config(cfg2, "frozen cross-attention teacher over all cameras (Experiment 2)", cfg2["student_choice"]["steps"])
    c_mean = student_config(cfg2, "uniform average of all cameras (the pilot's lifted feature)", cfg2["student_choice"]["steps"])
    diff = sorted(k for k in c_attn if c_attn[k] != c_mean[k])
    archives = {s: p[0] for s, p in find_scenes(args.input)["core"].items()}
    lift_check, dino = [], load_dino(dev)
    with torch.no_grad():
        for scene in store.scenes[:run["c"]["lift_check_scenes"]]:
            s = load_core_scene(archives[scene], scene, cfg2["shared"]["views"])
            pts = torch.from_numpy(s["coord"]).to(dev)
            nrm = torch.from_numpy(s["normal"]).float().to(dev)
            views = scene_views(s, pts, OCCLUSION[cfg2["shared"]["occlusion"]], device=dev)
            add_dino(dino, s, views, device=dev)
            vertex, target = t_mean[scene]
            vt = torch.from_numpy(vertex).to(dev)
            (_, lifted, _), = lift(pts[vt], views, nrm[vt], chunk=len(vertex) + 1)
            t = torch.from_numpy(target).float().to(dev)
            # stored observations and t_mean are fp16 (two roundings of ~2^-11 relative), lift() is fp32
            lift_check.append(dict(scene=scene, points=len(vertex), max_abs_diff=float((t - lifted).abs().max()),
                                   max_rel_diff=float((t - lifted).abs().max() / lifted.abs().max().clamp_min(1e-12)),
                                   min_cos=float(F.cosine_similarity(t, lifted, dim=-1).min())))
    del dino
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # DINOv2's memory back before the student trains
    checks = dict(config_diff=diff, config_attention=c_attn, config_uniform=c_mean, only_target_differs=diff == ["target"],
                  target_vs_lift=lift_check, target_vs_lift_ok=bool(lift_check) and all(c["max_rel_diff"] < 5e-3 and c["min_cos"] > 0.999 for c in lift_check))
    print(f"  config diff: {diff}; t_mean vs lift(): {json.dumps(lift_check)}")
    if not checks["only_target_differs"] or not checks["target_vs_lift_ok"]:
        write_json(out / "checks.json", checks)
        print("!! a check failed; not training")
        return 1

    section(f"3. Training the uniform-teacher student: {cfg2['student_choice']['trainable']}, crops of "
            f"{cfg2['student_choice']['crop_points']:,}, {steps} steps")
    del store
    trainable, crop_points, scfg = cfg2["student_choice"]["trainable"], cfg2["student_choice"]["crop_points"], cfg2["shared"]["student"]
    with attention_backend("sdpa"):
        model, transform = load_sonata(dev)
        tr = Trainer(model, transform, t_mean, args.points_cache, scfg, trainable, crop_points, steps, dev, archives)
        ckpt = out / "checkpoint.pt"
        prev = ckpt if ckpt.exists() else next((d / "runC" / "student" / "checkpoint.pt" for d in find_runs(args.input, "Experiment3_DistillationDiagnostics")
                                                if run_mode(d) == args.mode and (d / "runC" / "student" / "checkpoint.pt").exists()), None)
        resumed = None
        if prev is not None:
            st = torch.load(prev, weights_only=False)
            if st.get("trainable") == trainable and st.get("crop_points") == crop_points and st.get("steps") == steps:
                tr.load_state(st)
                resumed = dict(path=str(prev), step=tr.step_no)
                if prev.resolve() != ckpt.resolve():
                    shutil.copy(prev, ckpt)
                print(f"  resumed from {prev} at step {tr.step_no}")
        if tr.step_no == 0:
            checks["first_update"] = encoder_changes(tr)
        clock, t0 = Clock(scfg["checkpoint_minutes"] * 60), time.time()
        while tr.step_no < steps:
            rec = tr.step()
            if rec["step"] % 50 == 0 or rec["step"] == steps:
                recent = tr.curve[-50:]
                print(f"  step {rec['step']}/{steps}: loss {np.mean([r['loss'] for r in recent]):.4f}, "
                      f"{np.median([r['seconds'] for r in recent]):.2f} s/step, peak GPU {peak_gpu_gb()} GB", flush=True)
            if clock.due():
                save_atomic(dict(tr.state(), trainable=trainable, crop_points=crop_points, steps=steps, target="uniform"), ckpt)
        save_atomic(dict(tr.state(), trainable=trainable, crop_points=crop_points, steps=steps, target="uniform"), ckpt)
        hours = (time.time() - t0) / 3600
    torch.save(dict(state_dict=tr.model.state_dict(), encoder="sonata", init="sonata.pth (facebook/sonata @ df99897)", target="uniform",
                    trainable=trainable, crop_points=crop_points, steps=steps, git=record["git"]), out / "student.pth")
    curve = tr.curve
    log = dict(mode=args.mode, git=record["git"], target="uniform average of all cameras", loss=c_mean["loss"],
               trainable=trainable, trainable_parameters=int(sum(p.numel() for p in tr.model.parameters() if p.requires_grad)),
               crop_points=crop_points, steps=steps, resumed_from=resumed, seconds_per_step=float(np.median([r["seconds"] for r in curve])),
               session_hours=round(hours, 3), peak_gpu_gb=peak_gpu_gb(), loss_first_50=float(np.mean([r["loss"] for r in curve[:50]])),
               loss_last_50=float(np.mean([r["loss"] for r in curve[-50:]])), attn_vs_mean_train=attn_vs_mean,
               curve=[dict(step=r["step"], loss=r["loss"]) for r in curve])
    write_json(out / "train_log.json", log)
    write_json(out / "checks.json", checks)

    section("SUMMARY Run C training (paste this block back)")
    print(f"config diff (attention student vs uniform student): {diff}")
    print(f"t_mean vs the pilot's lift(): {json.dumps(lift_check)}")
    print(f"cos(t_attn, t_mean) on {attn_vs_mean['points']:,} train targets: mean {attn_vs_mean['mean']:.4f}, "
          f"5th percentile {attn_vs_mean['p05']:.4f}, by cameras {json.dumps(attn_vs_mean['by_cameras'])}")
    if "first_update" in checks:
        print(f"first update: {json.dumps(checks['first_update'])}")
    print(f"{trainable} ({log['trainable_parameters']:,} parameters), {steps} steps, {log['seconds_per_step']:.3f} s/step, loss "
          f"{log['loss_first_50']:.4f} -> {log['loss_last_50']:.4f}, {log['session_hours']} h, peak GPU {log['peak_gpu_gb']} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
