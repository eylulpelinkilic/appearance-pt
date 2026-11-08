"""Experiment 2 (experiments/Experiment2_CrossAttention) on CPU: the per-view observations against the pilot's lifting,
camera triples, the teacher's equal-weight and update behaviour, the student's trainable layers, and steps 1-5 end to
end on synthetic rooms with stand-ins for Sonata, DINOv2, the toolbox's ground truth and the pilot's saved outputs."""

import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pytest

from conftest import ROOT, load_script

torch = pytest.importorskip("torch")
sys.path.insert(0, str(ROOT / "experiments" / "Experiment1_Concatenation"))
sys.path.insert(0, str(ROOT / "experiments" / "Experiment2_CrossAttention"))
e2e = load_script("tests/test_pilot_steps345.py")
scenes_test = e2e.scenes_test
xattn = load_script("experiments/Experiment2_CrossAttention/xattn.py")
lifting = load_script("experiments/Experiment1_Concatenation/lifting.py")
WIDTH = 12  # the fake Sonata's feature width (the real one is 1232)


def synthetic_views(n_views=16, feat_channels=5):
    """The synthetic room seen from n_views cameras at its centre, with random patch features."""
    coord, _ = scenes_test.room()
    points = torch.from_numpy(coord)
    normals = torch.from_numpy(scenes_test.room_normals(coord)).float()
    K = torch.tensor([[scenes_test.F, 0, scenes_test.HW[1] / 2], [0, scenes_test.F, scenes_test.HW[0] / 2], [0, 0, 1]],
                     dtype=torch.float32)
    g = torch.Generator().manual_seed(0)
    views = []
    for i, yaw in enumerate(np.linspace(0, 2 * np.pi, n_views, endpoint=False)):
        v = lifting.View(f"v{i}", torch.tensor(scenes_test.w2c_looking(yaw), dtype=torch.float32), K, scenes_test.HW,
                         torch.ones(scenes_test.HW, dtype=torch.bool), occ=lifting.OCCLUSION["v2"])
        v.zbuf = lifting.zbuffer(points, v)
        v.feat = torch.randn(feat_channels, 9, 13, generator=g).half()
        views.append(v)
    return points, normals, views


def test_observations_average_to_the_pilots_lifted_feature():
    points, normals, views = synthetic_views()
    idx = torch.from_numpy(np.random.default_rng(0).choice(len(points), 3000, replace=False))
    obs = lifting.observe(points[idx], views, normals[idx])
    mean, count = lifting.observed_mean(obs, len(idx))
    (_, lifted, lifted_count), = lifting.lift(points[idx], views, normals[idx], chunk=len(idx) + 1)
    assert torch.equal(count, lifted_count) and (count >= 3).float().mean() > 0.3
    torch.testing.assert_close(mean, lifted, rtol=0, atol=0)  # the same sums in the same order
    pairs = obs["row"] * len(views) + obs["view"]
    assert len(pairs.unique()) == len(pairs)  # one observation per (point, camera)
    for vi in obs["view"].unique().tolist():  # every pixel is where that point projects in that camera
        m = obs["view"] == vi
        u, v, _ = lifting.project(points[idx][obs["row"][m]], views[vi].w2c, views[vi].K)
        torch.testing.assert_close(torch.stack([u, v], 1), obs["uv"][m], rtol=0, atol=1e-3)  # float noise only


def test_triples_are_distinct_and_cover_every_ordering():
    count = np.array([3, 4, 7, 30] * 2000)
    t = xattn.three_distinct(count, np.random.default_rng(0))
    assert ((t >= 0) & (t < count[:, None])).all()
    assert (t[:, 0] != t[:, 1]).all() and (t[:, 2] != t[:, 0]).all() and (t[:, 2] != t[:, 1]).all()
    three = t[count == 3]
    orderings = {tuple(r) for r in three.tolist()}
    assert len(orderings) == 6  # all 3! orderings of three cameras occur
    held = np.bincount(t[count == 4, 2], minlength=4) / (count == 4).sum()
    assert np.abs(held - 0.25).max() < 0.03  # the held-out camera is uniform


def test_teacher_starts_as_the_mean_and_learns_from_one_update():
    torch.manual_seed(0)
    teacher = xattn.CrossAttentionTeacher(WIDTH, 6, dim=8)
    q, kv = torch.randn(64, WIDTH), torch.randn(64, 2, 6)
    pred, w = teacher(q, kv)
    torch.testing.assert_close(pred, kv.mean(1))  # W_q = 0: equal weights
    assert torch.allclose(w, torch.full_like(w, 0.5))
    # the target is always the first input camera: the teacher should learn to weight it
    opt = torch.optim.Adam(teacher.parameters(), lr=0.05)
    first = None
    for _ in range(60):
        loss = xattn.cosine_loss(teacher(q, kv)[0], kv[:, 0])
        first = float(loss.detach()) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < first and teacher(q, kv)[1][:, 0].mean() > 0.6
    # padded cameras get no weight
    mask = torch.tensor([[True, True, False]] * 64)
    _, w3 = teacher(q, torch.cat([kv, torch.randn(64, 1, 6) * 100], 1), mask)
    assert float(w3[:, 2].detach().abs().max()) == 0.0


def test_sample_points_is_the_pilots_sample(monkeypatch):
    e2e.stub_packages(monkeypatch)
    step3 = load_script("experiments/Experiment1_Concatenation/step3_features.py")
    rng = np.random.default_rng(1)
    seg = np.c_[rng.integers(-1, 5, 5000), np.full((5000, 2), -1)].astype(np.int16)
    anon = rng.choice(5000, 300)
    s = dict(segment=seg, anonymised=anon)
    np.testing.assert_array_equal(step3.sample_points(s, 1000, 7), xattn.sample_points(seg, anon, 1000, 7))


class FakeSonata(torch.nn.Module):
    """Parameter names as Sonata's PTv3 (embedding, enc.enc0-4); maps the 9 point inputs to WIDTH features."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Linear(9, 16)
        self.enc = torch.nn.ModuleDict({f"enc{i}": torch.nn.Linear(16, 16 if i < 4 else WIDTH) for i in range(5)})

    def forward(self, x):
        h = torch.tanh(self.embedding(x))
        for i in range(5):
            h = torch.tanh(self.enc[f"enc{i}"](h))
        return h


def inputs9(s):
    c = s["coord"].astype(np.float32)
    return torch.from_numpy(np.c_[c - c.mean(0), s["color"] / 255.0, s["normal"]].astype(np.float32))


def fake_encode_train(model, transform, crop, seed, device="cpu"):
    stages = sys.modules["stages"]
    return stages.Stages([model(inputs9(crop))], [], torch.arange(len(crop["coord"])))


def fake_encode_scene(model, transform, s, seed, store="cpu", margin=1.0):
    """Frozen features of a whole scene: the fake model's output (the frozen-Sonata stand-in carries the label)."""
    stages = sys.modules["stages"]
    with torch.no_grad():
        f = model(inputs9(s)).half()
    if getattr(model, "label_channel", False):
        f = f.clone()
        f[torch.arange(len(f)), torch.from_numpy(s["segment"][:, 0].clip(0).astype(np.int64) % WIDTH)] += 3
    return stages.encode_tiled(lambda inputs: stages.Stages([f[torch.from_numpy(inputs)]], [], torch.arange(len(inputs))), s["coord"])


def test_trainable_layers_and_one_student_update(tmp_path):
    model = FakeSonata()
    names = xattn.set_trainable(model, "enc4")
    assert names == ["enc.enc4.weight", "enc.enc4.bias"]
    assert not model.embedding.weight.requires_grad and model.enc["enc4"].weight.requires_grad
    assert xattn.set_trainable(model, "all") == [n for n, _ in model.named_parameters()]
    coord = np.random.default_rng(0).random((5000, 3)).astype(np.float32)
    idx = xattn.crop_indices(coord, coord[0], 400)
    d = ((coord - coord[0]) ** 2).sum(1)
    assert len(idx) == 400 and d[idx].max() <= np.sort(d)[399] + 1e-9


def write_pilot(root, val, train, rng):
    """The pilot's saved step-5 outputs for these val scenes (fake but consistent): confmats.npz, results.json."""
    cms = {}
    for c in ("S", "S+"):
        for seed in range(2):
            for k in (1, 3):
                m = rng.integers(0, 30, (len(val), 100, 100)).astype(np.int32)
                m[:, np.arange(5), np.arange(5)] += 4000 if c == "S" else 5000
                cms[f"{c}#{seed}|top{k}"] = m
    root.mkdir(parents=True)
    np.savez_compressed(root / "confmats.npz", scenes=np.array(val), **cms)
    metrics = load_script("experiments/Experiment1_Concatenation/metrics.py")
    miou = {c: {"top1_all": [round(float(np.mean([100 * metrics.miou(cms[f"{c}#{s}|top1"].sum(0)) for s in range(2)])), 2), 0.0]}
            for c in ("S", "S+")}
    (root / "results.json").write_text(json.dumps(dict(
        git={"commit": "03228a2"}, miou=miou, features=dict(points_per_scene=2000, views=16, occlusion="v2",
                                                            encoder_names={"sonata": "sonata"}, git={"commit": "b949425"},
                                                            scenes=list(train)))))  # ids, as the pilot's step 4 saves them


def test_steps_1_to_5_on_synthetic_rooms(tmp_path, monkeypatch):
    e2e.stub_packages(monkeypatch)
    subset = json.loads((ROOT / "configs" / "scannetpp_subset.json").read_text())
    inp = tmp_path / "input"
    for scene in subset["train"][:4]:
        scenes_test.write_scene(inp / "core_train", scene, split="train", n_views=16)
    val = subset["val"][:2]
    scenes_test.write_scene(inp / "core_val", val[0], anonymised=range(0, 40_000, 7), n_views=16)
    scenes_test.write_scene(inp / "core_val", val[1], n_views=16)
    write_pilot(tmp_path / "pilot" / "pilot_concat" / "step5", val, subset["train"][:4], np.random.default_rng(0))

    cfg = json.loads((ROOT / "experiments" / "Experiment2_CrossAttention" / "config.json").read_text())
    cfg["modes"]["smoke"].update(train_scenes=4, dev_scenes=1, targets_per_scene=64, teacher_epochs=2, student_steps=4,
                                 smoke_options=dict(trainable=["enc4", "all"], crop_points=[3000], steps_each=2),
                                 probe_train_scenes=4, val_scenes=2,
                                 probe=dict(lrs=[0.05], holdout=1, sweep_epochs=1, epochs=2, seeds=2, batch=256, points_per_scene=2000))
    cfg["shared"]["teacher"].update(batch=64, dim=8)
    cfg["student_choice"] = dict(recorded_from_smoke=None, trainable=None, crop_points=None, steps=None)  # as before a smoke run
    cfg["shared"]["student"].update(min_targets=4, warmup=1)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    out, cache = tmp_path / "out", tmp_path / "points"
    common = ["--mode", "smoke", "--config", str(config), "--input", str(inp), "--out", str(out), "--device", "cpu"]

    def run(module, extra=()):
        monkeypatch.setattr(sys, "argv", ["prog", *common, *extra])
        return module.main()

    frozen = FakeSonata()
    frozen.label_channel = True
    step1 = load_script("experiments/Experiment2_CrossAttention/step1_observations.py")
    monkeypatch.setattr(step1, "load_models", lambda device: (frozen, None, scenes_test.FakeDino()))
    monkeypatch.setattr(step1, "encode_scene", fake_encode_scene)
    assert run(step1, ["--points-cache", str(cache), "--workers", "2"]) == 0
    obs = json.loads((out / "observations" / "summary.json").read_text())
    assert len(obs["scenes"]) == 4 and len(obs["dev_scenes"]) == 1 and obs["targets"] > 100
    assert 0 < obs["fraction_training_points_ge3"] <= 1 and obs["training_points"] == 4 * 2000
    assert obs["checks"]["mean_vs_lift_max_abs"] == 0.0 and obs["checks"]["reprojection_max_px"] < 1e-3
    assert obs["checks"]["counts_equal"] and obs["checks"]["cameras_distinct"] and obs["checks"]["vertex_ids_unique"]
    assert run(step1, ["--points-cache", str(cache)]) == 0  # resumes: nothing left to do

    step2 = load_script("experiments/Experiment2_CrossAttention/step2_teacher.py")
    assert run(step2) == 0
    checks = json.loads((out / "teacher" / "checks.json").read_text())
    assert checks["ok"] and checks["equal_weights"]["max_abs_diff_to_mean"] < 1e-5
    diag = json.loads((out / "teacher" / "diagnostic.json").read_text())
    assert diag["dev_points"] > 0 and len(diag["curve"]) == 2 and -1 <= diag["uniform_cos"] <= 1

    step3 = load_script("experiments/Experiment2_CrossAttention/step3_student.py")
    monkeypatch.setattr(step3, "load_sonata", lambda device: (FakeSonata(), None))
    monkeypatch.setattr(step3, "encode_train", fake_encode_train)
    assert run(step3, ["--points-cache", str(cache)]) == 0
    smoke = json.loads((out / "student" / "smoke.json").read_text())
    assert smoke["recommendation"]["trainable"] == "all" and all(r["status"] == "ok" for r in smoke["table"])
    assert all(r["first_update"]["trainable_changed"] for r in smoke["table"])
    enc4 = next(r for r in smoke["table"] if r["trainable"] == "enc4")["first_update"]
    assert enc4["frozen_unchanged"] and enc4["encoder_grad_norm"] > 0
    log = json.loads((out / "student" / "train_log.json").read_text())
    assert log["steps"] == 4 and len(log["curve"]) == 4 and "stop_gradient" in log["loss"]
    # the full mode refuses to train before the smoke run's choice is recorded
    monkeypatch.setattr(sys, "argv", ["prog", "--mode", "full", "--config", str(config), "--out", str(out), "--device", "cpu"])
    assert step3.main() == 3

    step4 = load_script("experiments/Experiment2_CrossAttention/step4_evaluate.py")

    def load_student(path, device):
        model = FakeSonata()
        saved = torch.load(path, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        return model.eval(), None, {k: v for k, v in saved.items() if k != "state_dict"}

    monkeypatch.setattr(step4, "load_student", load_student)
    monkeypatch.setattr(step4, "load_frozen_sonata", lambda device: (FakeSonata().eval(), None))
    monkeypatch.setattr(step4, "encode_scene", fake_encode_scene)
    monkeypatch.setattr(step4, "OfficialGT", e2e.FakeGT)
    assert run(step4, ["--pilot", str(tmp_path / "pilot" / "pilot_concat" / "step5"), "--features", str(tmp_path / "feat"),
                       "--tmp", str(tmp_path / "tmp4"), "--workers", "2"]) == 0
    m = json.loads((out / "reports" / "metrics.json").read_text())
    assert m["status"] == "preliminary" and set(m["rows"]) == {"S", "S+", "T"}
    assert m["rows"]["T"]["provenance"] == "new run (this experiment)" and m["rows"]["S"]["provenance"].startswith("pilot")
    assert m["protocol"]["recorded_matches_recomputed"] and m["protocol"]["val_scenes_same_order"]
    assert m["protocol"]["train_scenes_match"] and m["protocol"]["all_pilot_train_scenes"]
    assert m["image_free"]["ok"] and m["image_free"]["max_abs_feature_diff"] == 0.0
    assert set(m["image_free"]["archive_reads"]) <= {"pointcept", "scans/mesh_aligned_0.05_mask.txt"}
    ctrl = m["control"]
    assert ctrl["row"]["seeds"] == 2 and set(ctrl["diffs"]) == {"Sr - S", "T - Sr"}
    assert abs(ctrl["diffs"]["T - Sr"]["diff"] - (m["rows"]["T"]["top1"] - ctrl["row"]["top1"])) < 1e-12
    probes = torch.load(out / "evaluation" / "probes.pt", weights_only=False)
    assert set(probes["probes"]) == {"T#0", "T#1", "Sr#0", "Sr#1"} and set(probes["lr"]) == {"T", "Sr"}
    d = m["diffs"]["S+ - S"]
    assert d["diff"] > 0 and d["lo"] <= d["diff"] <= d["hi"]
    assert abs(m["recovered_fraction"] - (m["rows"]["T"]["top1"] - m["rows"]["S"]["top1"]) / m["concatenation_gain"]) < 1e-12

    if all(importlib.util.find_spec(m) for m in ("reportlab", "matplotlib")):  # the dev group has no matplotlib
        step5 = load_script("experiments/Experiment2_CrossAttention/step5_report.py")
        monkeypatch.setattr(sys, "argv", ["prog", "--out", str(out)])
        assert step5.main() == 0
        pdf = out / "reports" / "sonata_cross_attention_report.pdf"
        assert pdf.exists() and pdf.stat().st_size > 50_000
        if os.environ.get("EXP2_KEEP_REPORT"):  # for a look at the rendered report
            shutil.copytree(out / "reports", os.environ["EXP2_KEEP_REPORT"], dirs_exist_ok=True)


def test_a_full_run_never_reuses_a_smoke_run(tmp_path):
    smoke = tmp_path / "input" / "notebooks" / "me" / "exp2" / "Experiment2_CrossAttention"
    (smoke / "teacher").mkdir(parents=True)
    (smoke / "teacher" / "teacher.pt").write_bytes(b"x")
    xattn.mark_mode(smoke, "smoke")
    (tmp_path / "input" / "datasets" / "core" / "scenes" / "deep").mkdir(parents=True)  # not walked into
    out = tmp_path / "out"
    assert xattn.find_previous("teacher/teacher.pt", "smoke", roots=(out,), inputs=tmp_path / "input") == smoke / "teacher" / "teacher.pt"
    assert xattn.find_previous("teacher/teacher.pt", "full", roots=(out,), inputs=tmp_path / "input") is None
    xattn.mark_mode(out, "full")
    xattn.mark_mode(out, "full")  # the same mode again is fine
    with pytest.raises(SystemExit):
        xattn.mark_mode(out, "smoke")
    assert not any(d.name == "deep" for d in xattn.walk_dirs(tmp_path / "input"))
