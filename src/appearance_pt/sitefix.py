"""Make the prebuilt spconv/cumm wheels usable on Kaggle (docs/decisions.md D7).

cumm and spconv treat themselves as editable dev installs, and compile their C++ bindings on
import, when a `setup.py` or `.gitignore` sits next to the package directory
(`pccm.utils.project_is_editable`). Kaggle's site-packages contains one, so `import
spconv.pytorch` started a JIT build that failed. cumm also prefers `<site-packages>/include`
over the headers shipped in its wheel (`cumm/include`) when that directory exists, which it does
on Kaggle; those headers are what spconv's NVRTC fallback kernels compile against.

The fix is a `.pth` file next to cumm: Python runs it at startup of every interpreter (training
runs, DDP workers, import checks), setting the switches both packages read at import time.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

PTH_NAME = "appearance_pt_spconv.pth"
SWITCHES = {"CUMM_DISABLE_JIT": "1", "SPCONV_DISABLE_JIT": "1"}
MARKERS = ("setup.py", ".gitignore", "include")  # what makes pccm/cumm misdetect a dev install


def spconv_env(cumm_dir: Path) -> dict[str, str]:
    include = cumm_dir / "include"
    if not (include / "tensorview").is_dir():
        raise FileNotFoundError(f"cumm wheel headers missing: {include}/tensorview")
    return {**SWITCHES, "CUMM_INCLUDE_PATH": str(include)}


def install_spconv_fix(cumm_dir: Path | None = None) -> dict:
    """Write the .pth next to the installed cumm and apply it to this process. Returns a report."""
    if cumm_dir is None:
        spec = importlib.util.find_spec("cumm")  # locates the package without importing it
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError("cumm is not installed (it comes with spconv)")
        cumm_dir = Path(spec.origin).parent
    env = spconv_env(cumm_dir)
    site_dir = cumm_dir.parent
    pth = site_dir / PTH_NAME
    pth.write_text("import os; " + "; ".join(f"os.environ.setdefault({k!r}, {v!r})" for k, v in env.items()) + "\n")
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return {"pth": str(pth), "env": env, "markers_in_site_dir": [m for m in MARKERS if (site_dir / m).exists()]}
