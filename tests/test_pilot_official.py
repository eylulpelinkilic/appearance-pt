"""official.py's fix for the toolbox's class mapping: the released map_benchmark.csv has no `count` column."""

import pytest

from conftest import load_script

pd = pytest.importorskip("pandas")
official = load_script("experiments/pilot_concat/official.py")


def table(with_count):
    rows = dict(**{"class": ["chair", "chairs", "wall", "junk", "lamp"]},
                semantic_map_to=["chair", "chair", None, "None", None], instance_map_to=[None] * 5)
    if with_count:
        rows["count"] = [50, 3, 900, 7, 1]
    return pd.DataFrame(rows)


def test_missing_count_column_keeps_every_class_and_the_mapping():
    official.use_toolbox()
    import semantic.prep.map_semantic as ms

    classes, mapping = ms.filter_map_classes(table(False), 0, count_type="count", mapping_type="semantic")
    assert mapping == {"chair": "chair", "chairs": "chair", "wall": "wall", "lamp": "lamp"}  # "None" target dropped
    assert classes == ["chair", "wall", "lamp"]
    official.use_toolbox()  # patching twice is harmless
    _, filtered = ms.filter_map_classes(table(True), 5, count_type="count", mapping_type="semantic")
    assert "lamp" not in filtered and "chairs" not in filtered  # with counts, the toolbox's filter still applies
