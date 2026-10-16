import numpy as np

from appearance_pt.runlog import emit_result, parse_result
from appearance_pt.smoke import semseg_metrics, voxel_downsample


def test_semseg_metrics_hand_computed():
    gt = np.array([0, 0, 1, 1, -1, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    m = semseg_metrics(pred, gt, num_classes=3)
    # valid points: 5; correct: 3. IoU: c0 = 1/(2+2-1)=1/3, c1 = 2/(3+2-2)=2/3, c2 = 0/(0+1-0)=0
    assert m["acc"] == 3 / 5
    np.testing.assert_allclose(m["iou"], [1 / 3, 2 / 3, 0.0])
    assert m["miou"] == np.mean([1 / 3, 2 / 3, 0.0])
    assert m["n_classes_present"] == 3


def test_semseg_metrics_ignores_out_of_range_labels():
    m = semseg_metrics(np.array([0, 0]), np.array([0, 99]), num_classes=3)
    assert m["acc"] == 1.0


def test_voxel_downsample_keeps_one_point_per_voxel():
    coord = np.array([[0.0, 0, 0], [0.005, 0, 0], [0.03, 0, 0], [0.031, 0.001, 0]])
    keep, grid = voxel_downsample(coord, 0.02)
    assert keep.tolist() == [0, 2]
    assert grid.tolist() == [[0, 0, 0], [1, 0, 0]]


def test_result_line_roundtrip(capsys):
    emit_result(acc=0.9, name="x")
    assert parse_result("noise\n" + capsys.readouterr().out) == {"acc": 0.9, "name": "x"}
