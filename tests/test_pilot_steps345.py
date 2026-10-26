"""Pilot steps 3, 4 and 5 end to end on CPU, to catch glue bugs before a GPU run.

Synthetic train and val rooms (tests/test_pilot_step2.py) packed like Kaggle-unpacked core scenes; stub modules for
the encoder packages (the real ones need spconv and a GPU) and Hugging Face; fake encoders whose features carry the
label (the untrained "random" one does not); the DINOv2 stand-in; and a stand-in for the toolbox's GT that reads the
Pointcept labels. Checks sampling, gathering, lifting, the feature store, probe training and folding, per-chunk
prediction, the metric (against the toolbox's ConfMat on the written files) and the anonymised-vertex exclusion."""

import json
import sys
import types

import numpy as np
import pytest

from conftest import ROOT, load_script

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
sys.path.insert(0, str(ROOT / "experiments" / "pilot_concat"))
scenes_test = load_script("tests/test_pilot_step2.py")
toolbox = load_script("third_party/scannetpp/semantic/utils/confmat.py")
WIDTHS = {"utonia": 10, "sonata": 9, "concerto": 12, "random": 10}


def stub_packages(monkeypatch):
    for name in ("utonia", "sonata", "concerto"):
        mod = types.ModuleType(name)
        mod.model = types.SimpleNamespace(SerializedAttention=type("SerializedAttention", (), {"forward": None}),
                                          SDPA_AVAILABLE=True)
        monkeypatch.setitem(sys.modules, name, mod)
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    for name in ("encoders", "pipeline", "step2_lift"):  # import afresh against the stubs
        monkeypatch.delitem(sys.modules, name, raising=False)


class FakeModels:
    def __init__(self, concerto="concerto", seed=0, device="cpu"):
        self.encoders = {b: b for b in WIDTHS}
        self.transforms = {b: None for b in WIDTHS}
        self.widths = dict(WIDTHS)
        self.names = {b: concerto if b == "concerto" else b for b in WIDTHS}
        self.dino, self.dino_width = scenes_test.FakeDino(), 9


def fake_encode_scene(stages):
    def encode_scene(model, transform, s, seed, store="cpu", margin=1.0):
        rng = np.random.default_rng(seed)
        n, width = len(s["coord"]), WIDTHS[model]
        f = rng.normal(size=(n, width)).astype(np.float32)
        if model != "random":
            f[np.arange(n), s["segment"][:, 0].clip(0) % width] += 3.0
        return stages.encode_tiled(lambda inputs: stages.Stages([torch.from_numpy(f[inputs]).half()], [], torch.arange(len(inputs))),
                                   s["coord"])
    return encode_scene


class FakeGT:
    """The Pointcept labels in the toolbox's convention, and the toolbox's ConfMat on the written files (excluding
    the anonymised vertices, as eval_semantic does)."""

    def __init__(self, archives, meta_dir, root):
        self.archives, self.class_names = archives, [f"class{i}" for i in range(100)]

    def labels(self, scene):
        seg = np.load(self.archives[scene] / "pointcept" / "val" / scene / "segment.npy")
        return np.where(seg < 0, -100, seg).astype(np.int16)

    def evaluate_files(self, scenes, pred_dir, gt_dir):
        out = {}
        for k in (1, 3):
            cm = toolbox.ConfMat(100, top_k_pred=k, ignore_label=-100)
            for sc in scenes:
                gt = np.loadtxt(gt_dir / f"{sc}.txt", dtype=np.int32, delimiter=",")
                pred = np.loadtxt(pred_dir / f"{sc}.txt", dtype=np.int32, delimiter=",")
                text = (self.archives[sc] / "scans" / "mesh_aligned_0.05_mask.txt").read_text().split()
                keep = ~np.isin(np.arange(len(gt)), np.array(text, dtype=np.int64))
                cm.update(torch.from_numpy(pred[keep]), torch.from_numpy(gt[keep]))
            out[k] = float(cm.miou)
        return out


def run(module, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    return module.main()


def test_steps_3_4_5_on_synthetic_rooms(tmp_path, monkeypatch):
    stub_packages(monkeypatch)
    subset = json.loads((ROOT / "configs" / "scannetpp_subset.json").read_text())
    for scene in subset["train"][:4]:
        scenes_test.write_scene(tmp_path / "input" / "core_train", scene, split="train")
    val = subset["val"][:2]
    scenes_test.write_scene(tmp_path / "input" / "core_val", val[0], anonymised=range(0, 40_000, 7))
    scenes_test.write_scene(tmp_path / "input" / "core_val", val[1])

    step3 = load_script("experiments/pilot_concat/step3_features.py")
    stages = sys.modules["stages"]
    monkeypatch.setattr(step3, "Models", FakeModels)
    monkeypatch.setattr(sys.modules["pipeline"], "encode_scene", fake_encode_scene(stages))
    feats = tmp_path / "features"
    assert run(step3, monkeypatch, ["--input", str(tmp_path / "input"), "--features", str(feats), "--out", str(tmp_path / "s3"),
                                    "--points", "2000", "--limit", "4", "--device", "cpu", "--workers", "2"]) == 0
    meta = json.loads((feats / "meta.json").read_text())
    assert meta["n"] == 8000 and len(meta["scenes"]) == 4 and meta["encoder_names"]["concerto"] == "concerto"
    assert np.load(feats / "dino.npy", mmap_mode="r").shape[1] == 9 and (np.load(feats / "views.npy")[:8000] > 0).mean() > 0.3

    step4 = load_script("experiments/pilot_concat/step4_probes.py")
    assert run(step4, monkeypatch, ["--features", str(feats), "--out", str(tmp_path / "s4"), "--lrs", "0.05", "--holdout", "1",
                                    "--sweep-epochs", "2", "--epochs", "8", "--seeds", "2", "--batch", "256"]) == 0

    step5 = load_script("experiments/pilot_concat/step5_eval.py")
    monkeypatch.setattr(step5, "Models", FakeModels)
    monkeypatch.setattr(step5, "OfficialGT", FakeGT)
    assert run(step5, monkeypatch, ["--input", str(tmp_path / "input"), "--probes", str(tmp_path / "s4" / "probes.pt"),
                                    "--out", str(tmp_path / "s5"), "--tmp", str(tmp_path / "tmp5"), "--limit", "2",
                                    "--device", "cpu", "--official-scenes", "2", "--chunk", "50000"]) == 0
    res = json.loads((tmp_path / "s5" / "results.json").read_text())
    assert res["official_check"]["equal"], res["official_check"]
    assert res["scenes"] == val and not res["failed"]
    k, r = res["miou"]["K"]["top1_all"][0], res["miou"]["R"]["top1_all"][0]
    assert k > 80 and r < k - 30, (k, r)  # label-carrying features against noise
    assert set(res["bootstrap_top1"]) >= {"K+ - K", "C - B"} and "seen" in res["gains_top1"]["K+ - K"]
    assert res["per_scene"][0]["kept"] == res["per_scene"][0]["n_vertices"] - len(range(0, 40_000, 7))
