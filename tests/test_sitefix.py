"""The spconv/cumm .pth fix: right switches, right header dir, runs in a fresh interpreter."""

import os
import subprocess
import sys

import pytest

from appearance_pt import sitefix


@pytest.fixture
def site(tmp_path):
    (tmp_path / "cumm" / "include" / "tensorview").mkdir(parents=True)
    (tmp_path / "cumm" / "__init__.py").write_text("")
    (tmp_path / "setup.py").write_text("")  # the stray file that triggers pccm's dev-install check
    (tmp_path / "include").mkdir()
    return tmp_path


def test_pth_sets_switches_in_new_interpreters(site, monkeypatch):
    for k in (*sitefix.SWITCHES, "CUMM_INCLUDE_PATH"):
        monkeypatch.delenv(k, raising=False)
    report = sitefix.install_spconv_fix(site / "cumm")

    assert report["markers_in_site_dir"] == ["setup.py", "include"]
    assert report["env"]["CUMM_INCLUDE_PATH"] == str(site / "cumm" / "include")
    assert os.environ["CUMM_DISABLE_JIT"] == "1" and os.environ["SPCONV_DISABLE_JIT"] == "1"
    # a fresh interpreter that has the dir as a site dir processes the .pth (as Kaggle's dist-packages does)
    code = (f"import site; site.addsitedir({str(site)!r}); import os; "
            "print(os.environ['CUMM_DISABLE_JIT'], os.environ['SPCONV_DISABLE_JIT'], os.environ['CUMM_INCLUDE_PATH'])")
    env = {k: v for k, v in os.environ.items() if k not in (*sitefix.SWITCHES, "CUMM_INCLUDE_PATH")}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert out.split() == ["1", "1", str(site / "cumm" / "include")]


def test_missing_wheel_headers_fail_loudly(tmp_path):
    (tmp_path / "cumm").mkdir()
    with pytest.raises(FileNotFoundError, match="tensorview"):
        sitefix.install_spconv_fix(tmp_path / "cumm")
