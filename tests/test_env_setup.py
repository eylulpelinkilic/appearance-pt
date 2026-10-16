"""Variant selection in build_wheels.py and the image check in install_env.py."""

import json

import pytest

from conftest import ROOT, load_script

bw = load_script("scripts/setup/build_wheels.py")
ie = load_script("scripts/setup/install_env.py")


@pytest.mark.parametrize(
    "cuda, expected",
    [("11.8", "cu118"), ("12.1", "cu121"), ("12.3", "cu121"), ("12.4", "cu124"), ("12.6", "cu126"),
     ("12.8", "cu126"), ("13.0", "cu126")],
)
def test_spconv_variant(cuda, expected):
    assert bw.spconv_variant(cuda) == expected


def test_spconv_variant_too_old():
    with pytest.raises(ValueError):
        bw.spconv_variant("11.3")


def test_pyg_index_candidates():
    assert bw.pyg_index_candidates("2.9.1+cu128") == [
        "https://data.pyg.org/whl/torch-2.9.1+cu128.html",
        "https://data.pyg.org/whl/torch-2.9.0+cu128.html",
    ]
    assert bw.pyg_index_candidates("2.8.0+cu126") == ["https://data.pyg.org/whl/torch-2.8.0+cu126.html"]
    with pytest.raises(ValueError):
        bw.pyg_index_candidates("2.8.0")  # CPU build


def test_requirements_template_renders():
    text = bw.render_requirements(bw.REQUIREMENTS.read_text(), "cu126")
    assert "spconv-cu126==2.3.8" in text and "{SPCONV}" not in text
    assert "yapf==0.40.1" in text  # needed by DITR's FormatCode(verify=True)


def test_parse_install_report_normalizes_names():
    report = {"install": [
        {"metadata": {"name": "torch_scatter", "version": "2.1.2+pt28cu126"}},
        {"metadata": {"name": "SharedArray", "version": "3.2.4"}},
    ]}
    assert bw.parse_install_report(report) == [("sharedarray", "3.2.4"), ("torch-scatter", "2.1.2+pt28cu126")]


def test_find_wheelhouses_and_compatibility(tmp_path):
    wh = tmp_path / "appearance-pt-wheels" / "wheelhouse"
    wh.mkdir(parents=True)
    (wh / "MANIFEST.json").write_text(json.dumps({"torch": "2.8.0+cu126", "python_tag": "cp313"}))
    assert ie.find_wheelhouses(tmp_path) == [wh]
    manifest = json.loads((wh / "MANIFEST.json").read_text())
    assert ie.compatibility_problems(manifest, "2.8.0+cu126", "cp313") == []
    problems = ie.compatibility_problems(manifest, "2.9.0+cu128", "cp312")
    assert len(problems) == 2 and "torch" in problems[0]


def test_envinfo_does_not_leak_kaggle_credentials(monkeypatch):
    from appearance_pt import envinfo

    monkeypatch.setenv("KAGGLE_USER_SECRETS_TOKEN", "super-secret-value")
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    record = envinfo.collect(ROOT)
    assert "super-secret-value" not in json.dumps(record, default=str)
    assert record["env"]["KAGGLE_KERNEL_RUN_TYPE"] == "Batch"
    assert record["git"]["commit"] is None or len(record["git"]["commit"]) == 40
