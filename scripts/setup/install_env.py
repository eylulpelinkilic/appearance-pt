"""Install the prebuilt wheelhouse into this Kaggle session, offline.

Refuses to install when the session's image differs from the one the wheels were built for
(Kaggle images change over time; pin the notebook environment or rebuild the wheels).
Writes the session's environment record to <record> for the run log.

Usage: python scripts/setup/install_env.py [--wheelhouse DIR] [--record /kaggle/working/env_record.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from appearance_pt.envinfo import check_imports, collect
from appearance_pt.runlog import section
from appearance_pt.sitefix import install_spconv_fix


def find_wheelhouses(root: Path = Path("/kaggle/input"), max_depth: int = 4) -> list[Path]:
    found = []
    for depth in range(1, max_depth + 1):
        found += [p.parent for p in root.glob("/".join(["*"] * depth) + "/MANIFEST.json")]
    return sorted(found)


def compatibility_problems(manifest: dict, torch_version: str, python_tag: str) -> list[str]:
    problems = []
    if manifest.get("torch") != torch_version:
        problems.append(f"torch: wheels built for {manifest.get('torch')}, session has {torch_version}")
    if manifest.get("python_tag") != python_tag:
        problems.append(f"python: wheels built for {manifest.get('python_tag')}, session has {python_tag}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheelhouse", type=Path, help="default: search /kaggle/input for MANIFEST.json")
    ap.add_argument("--record", type=Path, default=Path("/kaggle/working/env_record.json"))
    args = ap.parse_args()

    import torch

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    section("1. Locate wheelhouse")
    candidates = [args.wheelhouse] if args.wheelhouse else find_wheelhouses()
    if not candidates:
        print("!! no wheelhouse found: attach the private dataset 'appearance-pt-wheels' (Add Input)")
        return 2
    chosen, problems = None, {}
    for wh in candidates:
        manifest = json.loads((wh / "MANIFEST.json").read_text())
        problems[wh] = compatibility_problems(manifest, torch.__version__, py_tag)
        print(f"  {wh}: built {manifest.get('created_utc')} for torch {manifest.get('torch')} -> "
              f"{'compatible' if not problems[wh] else 'INCOMPATIBLE'}")
        if not problems[wh] and chosen is None:
            chosen = wh
    if chosen is None:
        for wh, p in problems.items():
            print(f"!! {wh}: " + "; ".join(p))
        print("!! The Kaggle image changed. Pin this notebook's environment to the image the wheels were\n"
              "   built on (Settings -> Environment), or rerun m1_01_build_wheels and update the dataset.")
        return 2

    section(f"2. Offline install from {chosen}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--quiet",
         "-f", str(chosen / "wheels"), "-r", str(chosen / "requirements.lock.txt")],
        check=True,
    )
    print((chosen / "requirements.lock.txt").read_text())
    print(f"spconv/cumm fix (decisions.md D7): {json.dumps(install_spconv_fix())}")

    section("3. Import check")
    ok = check_imports()

    manifest = json.loads((chosen / "MANIFEST.json").read_text())
    record = collect()
    record["wheelhouse"] = {"path": str(chosen), **{k: manifest[k] for k in ("created_utc", "torch", "spconv_variant")}}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2))
    print(f"environment record: {args.record}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
