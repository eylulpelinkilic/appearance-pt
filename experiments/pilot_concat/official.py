"""The ScanNet++ toolbox's own semantic ground truth and evaluation, run in-process on the packed val scenes.

Ground truth: the toolbox's ScannetPP_Release_Dataset with the transforms of semantic/prep/prepare_training_data.py
(add_mesh_vertices, map_label_to_index, get_labels_on_vertices) and multilabel max_gt 3: top-100 classes through
map_benchmark.csv, ignore -100, up to 3 labels per vertex in annotation order. (Pointcept's segment.npy holds the
same labels but moves the smallest instance's label to the first column, so it is not the official GT for the
top-k metric; decisions.md P7.) The toolbox reads <data_root>/<scene>/scans/, which links to the unpacked archives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from appearance_pt import REPO_ROOT
from appearance_pt.scannetpp_store import SceneArchive

TOOLBOX = REPO_ROOT / "third_party" / "scannetpp"
VAL_FILES = ("mesh_aligned_0.05.ply", "mesh_aligned_0.05_mask.txt", "segments.json", "segments_anno.json")


def use_toolbox() -> None:
    if str(TOOLBOX) not in sys.path:
        sys.path.insert(0, str(TOOLBOX))


class OfficialGT:
    def __init__(self, archives: dict, meta_dir: Path, root: Path):
        use_toolbox()
        from munch import Munch
        from semantic.datasets.scannetpp_release import ScannetPP_Release_Dataset
        from semantic.transforms.factory import get_transform

        self.data_root = Path(root) / "data"
        for scene, archive in archives.items():
            scans = self.data_root / scene / "scans"
            if scans.exists():
                continue
            scans.parent.mkdir(parents=True, exist_ok=True)
            if Path(archive).is_dir():
                scans.symlink_to(Path(archive) / "scans", target_is_directory=True)
            else:
                scans.mkdir()
                with SceneArchive(archive) as arc:
                    for name in VAL_FILES:
                        (scans / name).write_bytes(arc.read(f"scans/{name}"))
        list_file = Path(root) / "scenes.txt"
        list_file.write_text("\n".join(archives) + "\n")
        bench = Path(meta_dir) / "semantic_benchmark"
        cfg = Munch(transforms=["add_mesh_vertices", "map_label_to_index", "get_labels_on_vertices"],
                    labels_path=str(bench / "top100.txt"), mapping_file=str(bench / "map_benchmark.csv"),
                    ignore_label=-100, use_instances=False, instance_labels_path=None,
                    multilabel=Munch(max_gt=3, multilabel_only=False))
        self.dataset = ScannetPP_Release_Dataset(str(self.data_root), list_file=str(list_file), transform=get_transform(cfg))
        self.index = {sample["scene_id"]: i for i, sample in enumerate(self.dataset.samples)}
        self.class_names = (bench / "top100.txt").read_text().splitlines()

    def labels(self, scene: str) -> np.ndarray:
        """(N, 3) int16 vertex labels, -100 = none."""
        return np.asarray(self.dataset[self.index[scene]]["vtx_labels"], dtype=np.int16)

    def evaluate_files(self, scenes: list[str], pred_dir: Path, gt_dir: Path) -> dict:
        """The toolbox's eval_semantic on <dir>/<scene>.txt files: {k: mIoU} for top-1 and top-3."""
        use_toolbox()
        from semantic.eval.eval_semantic import eval_semantic

        confmats = eval_semantic(scenes, str(pred_dir), str(gt_dir), str(self.data_root), len(self.class_names), -100, [1, 3])
        return {k: float(cm.miou) for k, cm in confmats.items()}


def compare_with_pointcept(official: np.ndarray, segment: np.ndarray) -> dict:
    """How Pointcept's segment.npy (ignore -1) relates to the official labels, per vertex."""
    seg = np.where(segment < 0, -100, segment).astype(np.int16)
    same_rows = (seg == official).all(1)
    same_sets = (np.sort(seg, 1) == np.sort(official, 1)).all(1)
    return dict(same_rows=round(float(same_rows.mean()), 6), same_label_sets=round(float(same_sets.mean()), 6),
                multilabel=round(float(((official >= 0).sum(1) > 1).mean()), 4))
