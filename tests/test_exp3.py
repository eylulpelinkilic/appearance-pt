"""Experiment 3 (experiments/Experiment3_DistillationDiagnostics) on CPU: Runs A, B, C and the report end to end on
synthetic rooms. Experiment 2's own end-to-end test first builds an Experiment 2 output (steps 1-4: observations,
teacher, student with its head, evaluation); Experiment 3 then runs on it with stand-ins for Sonata, DINOv2 and the
toolbox's ground truth."""

import importlib
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pytest

from conftest import ROOT, load_script

torch = pytest.importorskip("torch")
import test_exp2 as exp2_test  # noqa: E402  (its fakes, and its end-to-end run as the Experiment 2 output)

HERE = ROOT / "experiments" / "Experiment3_DistillationDiagnostics"
PROBE_MODULES = ("probes", "step4_probes", "xprobe")


def frozen_sonata(device):
    return exp2_test.FakeSonata().eval().requires_grad_(False), None


def build_student(state, device):
    model = exp2_test.FakeSonata()
    model.load_state_dict(state)
    return model.eval().requires_grad_(False)


def test_runs_a_b_c_and_the_report_on_synthetic_rooms(tmp_path, monkeypatch):
    exp2_test.test_steps_1_to_5_on_synthetic_rooms(tmp_path, monkeypatch)  # Experiment 2's (smoke) output: tmp_path/out
    exp2, inp, pilot = tmp_path / "out", tmp_path / "input", tmp_path / "pilot" / "pilot_concat" / "step5"
    e2e = exp2_test.e2e
    fake_dino = lambda device: exp2_test.scenes_test.FakeDino()  # noqa: E731
    for name in ("xdiag", "xprobe"):  # import afresh against the stubbed encoder packages
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(HERE))
    xdiag = importlib.import_module("xdiag")
    monkeypatch.setattr(xdiag, "build_student", build_student)
    monkeypatch.setattr(xdiag, "make_transform", lambda name: None)

    # Experiment 2's output is refused unless it is the full final run; a head checkpoint not at the end is refused
    with pytest.raises(SystemExit):
        xdiag.find_exp2(inp, exp2)
    assert xdiag.find_exp2(inp, exp2, any_status=True) == exp2
    early = tmp_path / "early_student"
    shutil.copytree(exp2 / "student", early)
    ck = torch.load(early / "checkpoint.pt", weights_only=False)
    torch.save(dict(ck, step=ck["step"] - 1), early / "checkpoint.pt")
    with pytest.raises(SystemExit):
        xdiag.load_student_head(early, "cpu")

    cfg = json.loads((HERE / "config.json").read_text())
    cfg["modes"]["smoke"] = dict(
        probe_train_scenes=4, val_scenes=2,
        probe=dict(lrs=[0.01, 0.05], holdout=1, sweep_epochs=1, epochs=2, seeds=2, batch=256, points_per_scene=2000),
        b=dict(train_scenes=2, val_scenes=2, points_per_scene=300, crops_per_scene=2, crop_points=3000, classes=5,
               head=dict(max_epochs=3, batch=64, warmup=1)),
        c=dict(steps=2, lift_check_scenes=1))
    config = tmp_path / "config3.json"
    config.write_text(json.dumps(cfg))
    cfg2 = json.loads((tmp_path / "config.json").read_text())  # Experiment 2's test config, with its choice recorded
    cfg2["student_choice"] = dict(recorded_from_smoke="test", trainable="all", crop_points=3000, steps=4)
    config2 = tmp_path / "config2.json"
    config2.write_text(json.dumps(cfg2))
    out = tmp_path / "out3"
    common = ["--mode", "smoke", "--config", str(config), "--input", str(inp), "--out", str(out), "--device", "cpu",
              "--exp2", str(exp2), "--any-exp2"]

    def run(module, extra=()):
        monkeypatch.setattr(sys, "argv", ["prog", *common, *[str(x) for x in extra]])
        return module.main()

    # ---- Run A ----
    stepA = load_script("experiments/Experiment3_DistillationDiagnostics/stepA_head_probes.py")
    xprobe = sys.modules["xprobe"]
    monkeypatch.setattr(xprobe, "encode_scene", exp2_test.fake_encode_scene)
    monkeypatch.setattr(stepA, "OfficialGT", e2e.FakeGT)
    assert run(stepA, ["--pilot", pilot, "--features", tmp_path / "featA", "--tmp", tmp_path / "tmpA", "--workers", 2]) == 0
    a = json.loads((out / "runA" / "summary.json").read_text())
    assert set(a["rows"]) == {"S", "S+", "A1", "A2", "A3"} and set(a["diffs"]) == {"A2 - A1", "A3 - A1"}
    rp = a["checks"]["reproduction"]
    assert rp["ok"] and rp["scenes"] == 2 and rp["seeds"] == 2 and rp["tiles_match"] and rp["identical_scenes"] == 2
    assert rp["diff_top1"] == 0 and rp["moved_share_top1"] == 0 and rp["max_abs_count_diff"] == 0
    w = a["checks"]["widths"]
    assert w["ok"] and w["f"] == exp2_test.WIDTH and w["f_plus_h"] == w["f"] + w["h"]
    assert a["probes"]["A3"]["width"] == w["f_plus_h"] and a["probes"]["A2"]["lr"] in (0.01, 0.05)
    assert a["checks"]["image_free"]["ok"] and a["checks"]["ended_training"]
    m2 = json.loads((exp2 / "reports" / "metrics.json").read_text())
    assert abs(a["rows"]["A1"]["top1"] - m2["rows"]["T"]["top1"]) < 1e-9  # A1 is Experiment 2's student row
    assert abs(a["rows"]["S"]["top1"] - m2["rows"]["S"]["top1"]) < 1e-9
    for c in ("A2", "A3"):
        r = a["rows"][c]
        assert abs(r["recovered_top1"] - (r["top1"] - a["rows"]["S"]["top1"]) / a["gain"]["top1"]) < 1e-12
        d = a["diffs"][f"{c} - A1"]["top1"]
        assert abs(d["diff"] - (r["top1"] - a["rows"]["A1"]["top1"])) < 1e-12 and d["lo"] <= d["hi"]
    assert a["decision"]["supported"] == any(a["diffs"][k]["top1"]["lo"] > 0 for k in a["diffs"])
    za = np.load(out / "runA" / "confmats.npz")
    assert {"A2#0|top1", "A2#1|top3", "A3#1|top1"} <= set(za.files)

    # probe features under runB/ are refused
    with pytest.raises(AssertionError):
        xprobe.fit_configs(out / "runB" / "features", {}, cfg["shared"]["probe"], "cpu", forbidden=out / "runB")

    # ---- Run B (a fresh process on Kaggle: no probe module loaded) ----
    for name in PROBE_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    stepB = load_script("experiments/Experiment3_DistillationDiagnostics/stepB_val_cosine.py")
    monkeypatch.setattr(stepB, "encode_scene", exp2_test.fake_encode_scene)
    monkeypatch.setattr(stepB, "load_frozen_sonata", frozen_sonata)
    monkeypatch.setattr(stepB, "load_dino", fake_dino)
    monkeypatch.setattr(stepB, "OfficialGT", e2e.FakeGT)
    assert run(stepB, ["--pilot", pilot, "--tmp", tmp_path / "tmpB", "--workers", 2]) == 0
    assert not any(m in sys.modules for m in PROBE_MODULES)
    b = json.loads((out / "runB" / "summary.json").read_text())
    assert b["checks"]["no_probe"] and b["checks"]["observations"]["mean_vs_lift_max_abs"] == 0.0
    assert b["checks"]["observations"]["counts_equal"] and b["checks"]["observations"]["cameras_distinct"]
    assert len(b["train_scenes"]) == 2 and set(b["train_scenes"]) <= set(json.loads((ROOT / "configs" / "scannetpp_subset.json").read_text())["train"])
    assert b["val_scenes"] == a["val_scenes"]
    for g in (b["b1"], b["b2"]):
        assert -1 <= g["val"] <= 1 and abs(g["gap"] - (g["train"] - g["val"])) < 1e-12 and g["lo"] <= g["hi"]
    assert abs(b["gap_difference"]["diff"] - (b["b1"]["gap"] - b["b2"]["gap"])) < 1e-12
    assert b["b2_head"]["epochs"] <= 3 and b["b2_head"]["steps"] > 0
    assert set(b["by_cameras"]) == {"1", "2", ">=3"} and sum(v["points"] for v in b["by_cameras"].values()) == 2 * 300
    assert 0 < len(b["classes"]) <= 5 and b["crops"]["points"] > 0
    assert sorted(p.name for p in (out / "runB" / "val_observations" / "scenes").glob("*.npz")) == sorted(f"{s}.npz" for s in b["val_scenes"])
    assert len(list((out / "runB" / "points").glob("*.npz"))) == 4 and not (tmp_path / "tmpB" / "train_observations").exists()
    one = np.load(out / "runB" / "points" / f"val_{b['val_scenes'][0]}.npz")
    single = one["cams"] == 1
    assert single.any() and np.allclose(one["b3"][single], 1, atol=1e-3)  # one camera: attention = average

    # ---- Run C ----
    stepC1 = load_script("experiments/Experiment3_DistillationDiagnostics/stepC1_uniform_student.py")
    monkeypatch.setattr(stepC1, "load_dino", fake_dino)
    monkeypatch.setattr(stepC1, "load_sonata", lambda device: (exp2_test.FakeSonata(), None))
    assert run(stepC1, ["--exp2-config", config2, "--points-cache", tmp_path / "pointsC"]) == 0
    checks = json.loads((out / "runC" / "student" / "checks.json").read_text())
    assert checks["config_diff"] == ["target"] and checks["only_target_differs"]
    assert checks["target_vs_lift_ok"] and checks["target_vs_lift"][0]["min_cos"] > 0.9999
    assert checks["first_update"]["trainable_changed"] and checks["first_update"]["encoder_grad_norm"] > 0
    log = json.loads((out / "runC" / "student" / "train_log.json").read_text())
    assert log["steps"] == 2 and len(log["curve"]) == 2 and log["attn_vs_mean_train"]["points"] > 0
    ckC = torch.load(out / "runC" / "student" / "checkpoint.pt", weights_only=False)
    assert ckC["target"] == "uniform" and ckC["step"] == ckC["steps"] == 2

    stepC2 = load_script("experiments/Experiment3_DistillationDiagnostics/stepC2_evaluate.py")
    monkeypatch.setattr(sys.modules["xprobe"], "encode_scene", exp2_test.fake_encode_scene)  # xprobe imported afresh
    monkeypatch.setattr(stepC2, "OfficialGT", e2e.FakeGT)
    assert run(stepC2, ["--pilot", pilot, "--features", tmp_path / "featC", "--tmp", tmp_path / "tmpC", "--workers", 2]) == 0
    c = json.loads((out / "runC" / "summary.json").read_text())
    assert set(c["rows"]) == {"S", "S+", "A1", "A2", "A3", "C1", "C2", "C3"}
    assert set(c["diffs"]) == {"C1 - A1", "C2 - A2", "C3 - A3"} and c["run_a"] == str(out / "runA")
    key = c["diffs"]["C1 - A1"]["top1"]
    assert c["decision"]["no_evidence"] == (abs(key["diff"]) < 1 or key["lo"] <= 0 <= key["hi"])
    assert abs(c["rows"]["A2"]["top1"] - a["rows"]["A2"]["top1"]) < 1e-12 and c["checks"]["image_free"]["ok"]
    assert c["second_seed_proposal"]["estimate_hours"] > 0

    # ---- report ----
    if all(importlib.util.find_spec(m) for m in ("reportlab", "matplotlib")):  # the dev group has no matplotlib
        report = load_script("experiments/Experiment3_DistillationDiagnostics/step_report.py")
        monkeypatch.setattr(sys, "argv", ["prog", "--mode", "smoke", "--config", str(config), "--input", str(inp), "--out", str(out)])
        assert report.main() == 0
        m = json.loads((out / "reports" / "metrics.json").read_text())
        assert m["status"] == "preliminary" and set(m["runs"]) == {"A", "B", "C"} and "smoke" in m["status_reason"]
        assert m["decision_rules"] == cfg["decision_rules"]
        pdf = out / "reports" / "distillation_diagnostics_report.pdf"
        assert pdf.exists() and pdf.stat().st_size > 30_000
        if os.environ.get("EXP3_KEEP_REPORT"):  # for a look at the rendered report
            shutil.copytree(out / "reports", os.environ["EXP3_KEEP_REPORT"], dirs_exist_ok=True)


def test_reproduction_check_tolerates_small_drift_only(tmp_path, monkeypatch):
    """config.json check_rules.A_reproduction: a drift of a few counts passes, a real change or other tiling fails."""
    exp2_test.e2e.stub_packages(monkeypatch)
    for name in ("xdiag", "xprobe"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(HERE))
    stepA = load_script("experiments/Experiment3_DistillationDiagnostics/stepA_head_probes.py")
    rng = np.random.default_rng(0)
    a1 = {kk: rng.integers(0, 50, (3, 4, 100, 100)).astype(np.int32) for kk in (1, 3)}
    for kk in (1, 3):
        a1[kk][:, :, np.arange(20), np.arange(20)] += 5000 + rng.integers(0, 400, (3, 4, 20))  # seed spread
    scenes = [f"s{i}" for i in range(4)]
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "results.json").write_text(json.dumps(dict(per_scene=[dict(scene=s, tiles=dict(student=1)) for s in scenes])))
    rows = [dict(scene=s, tiles=dict(student=1)) for s in scenes]
    ref = {f"T#{i}": {kk: a1[kk][i].copy() for kk in (1, 3)} for i in range(3)}
    for i in range(3):  # a few counts move from a diagonal cell to an off-diagonal one
        ref[f"T#{i}"][1][0, 3, 3] -= 58
        ref[f"T#{i}"][1][0, 3, 4] += 58
    ok = stepA.reproduction(ref, a1, tmp_path, scenes, rows)
    assert ok["ok"] and ok["max_abs_count_diff"] == 58 and ok["identical_scenes"] == 3 and 0 < ok["moved_share_top1"] < 1e-3
    big = {k: {kk: v.copy() for kk, v in d.items()} for k, d in ref.items()}
    for i in range(3):
        big[f"T#{i}"][1][:, np.arange(20), np.arange(20)] //= 2
    assert not stepA.reproduction(big, a1, tmp_path, scenes, rows)["ok"]
    rows[1] = dict(scene="s1", tiles=dict(student=2))
    tiled = stepA.reproduction(ref, a1, tmp_path, scenes, rows)
    assert not tiled["ok"] and tiled["tiles_mismatch"] == {"s1": {"now": 2, "experiment2": 1}}
