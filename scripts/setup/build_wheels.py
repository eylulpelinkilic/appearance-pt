"""Build the offline wheelhouse for Kaggle sessions (once per Kaggle image).

Everything is built against the torch preinstalled in the current image, which is never
replaced. Output, ready to become a private Kaggle Dataset:
  <out>/wheels/*.whl            everything added on top of the image (incl. compiled pointops)
  <out>/requirements.lock.txt   exact name==version list; sessions install it --no-index --no-deps
  <out>/MANIFEST.json           image torch/CUDA/Python, chosen variants, sha256 per wheel, git commits
  <out>/base_freeze.txt         pip freeze of the untouched image

Usage (Kaggle, GPU session, Internet on):
  python scripts/setup/build_wheels.py --out /kaggle/working/wheelhouse
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from appearance_pt import REPO_ROOT, THIRD_PARTY
from appearance_pt.envinfo import check_imports, collect
from appearance_pt.runlog import section
from appearance_pt.sitefix import install_spconv_fix

REQUIREMENTS = REPO_ROOT / "configs/env/kaggle-requirements.in"
POINTOPS_SRC = THIRD_PARTY / "Pointcept/libs/pointops"
SPCONV_CUDA = ((12, 6), (12, 4), (12, 1), (11, 8))  # spconv 2.3.8 wheels on PyPI (checked 2026-10-16)
PROTECTED = ("torch", "torchvision", "torchaudio", "triton", "numpy")  # image versions are pinned
TORCH_EXTENSIONS = {"torch-scatter", "torch-cluster"}  # built against installed torch if no PyG wheel


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def spconv_variant(cuda: str) -> str:
    """Newest spconv CUDA build not newer than torch's CUDA (also used for CUDA 13.x)."""
    major, minor = (int(x) for x in cuda.split(".")[:2])
    for v in SPCONV_CUDA:
        if (major, minor) >= v:
            return f"cu{v[0]}{v[1]}"
    raise ValueError(f"no spconv 2.3.8 wheel for CUDA {cuda}")


def pyg_index_candidates(torch_version: str) -> list[str]:
    """data.pyg.org pages to try: the exact torch version, then its X.Y.0 page."""
    base, _, local = torch_version.partition("+")
    if not local.startswith("cu"):
        raise ValueError(f"expected a CUDA torch build, got {torch_version}")
    major, minor = base.split(".")[:2]
    pages = dict.fromkeys([f"{base}+{local}", f"{major}.{minor}.0+{local}"])
    return [f"https://data.pyg.org/whl/torch-{p}.html" for p in pages]


def pick_pyg_index(torch_version: str, py_tag: str) -> str | None:
    for url in pyg_index_candidates(torch_version):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                page = r.read().decode()
        except Exception as e:  # noqa: BLE001
            print(f"  {url}: {type(e).__name__}")
            continue
        if re.search(rf"torch_scatter-[^\"<]*-{py_tag}-{py_tag}-linux_x86_64\.whl", page):
            print(f"  {url}: has torch_scatter wheels for {py_tag}")
            return url
        print(f"  {url}: no {py_tag} torch_scatter wheel")
    return None


def render_requirements(template: str, spconv: str) -> str:
    return template.replace("{SPCONV}", spconv)


def parse_install_report(report: dict) -> list[tuple[str, str]]:
    """(normalized name, version) of every distribution pip would add or change."""
    return sorted((normalize(i["metadata"]["name"]), i["metadata"]["version"]) for i in report.get("install", []))


def pip(*args, env=None, capture=False) -> str:
    cmd = [sys.executable, "-m", "pip", *map(str, args)]
    print("$ pip " + " ".join(map(str, args)), flush=True)
    r = subprocess.run(cmd, env=env, check=True, text=True, capture_output=capture)
    return r.stdout if capture else ""


def nvcc_release() -> str | None:
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    except OSError:
        return None
    m = re.search(r"release (\d+\.\d+)", out)
    return m.group(1) if m else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", default="7.5+PTX", help="TORCH_CUDA_ARCH_LIST for compiled extensions (T4 = 7.5)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    t_start = time.time()

    import torch

    section("1. Image facts")
    env_record = collect()
    print(json.dumps({k: env_record[k] for k in ("python", "torch", "git")}, indent=1))
    torch_version, torch_cuda = torch.__version__, torch.version.cuda
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if not torch_cuda:
        print("!! preinstalled torch has no CUDA support; use the GPU image")
        return 2
    if torch.cuda.is_available() and "sm_75" not in torch.cuda.get_arch_list():
        print(f"!! preinstalled torch lacks sm_75 kernels ({torch.cuda.get_arch_list()}); see docs/decisions.md fallback")
        return 2
    nvcc = nvcc_release()
    print(f"torch {torch_version} (CUDA {torch_cuda}), nvcc {nvcc}, python tag {py_tag}")
    if nvcc is None:
        print("!! nvcc not found; pointops cannot be compiled")
        return 2
    if nvcc.split(".")[0] != torch_cuda.split(".")[0]:
        print(f"!! nvcc {nvcc} and torch CUDA {torch_cuda} differ in major version; torch refuses to build extensions")
        return 2
    spconv = spconv_variant(torch_cuda)
    print(f"spconv variant: spconv-{spconv}")
    print("PyG wheel index:")
    pyg = pick_pyg_index(torch_version, py_tag)
    if pyg is None:
        print("  no matching PyG page; torch-scatter/cluster will be compiled from source (slow)")

    out, wheels = args.out, args.out / "wheels"
    if out.exists():
        shutil.rmtree(out)
    wheels.mkdir(parents=True)
    (out / "base_freeze.txt").write_text(pip("freeze", capture=True))
    find_links = ["-f", pyg] if pyg else []
    build_env = {**os.environ, "TORCH_CUDA_ARCH_LIST": args.arch, "MAX_JOBS": str(args.jobs)}

    section("2. Resolve what the image is missing (pip dry run; protected packages pinned)")
    reqs = out / "requirements.in"
    reqs.write_text(render_requirements(REQUIREMENTS.read_text(), spconv))
    constraints = out / "protected-constraints.txt"
    constraints.write_text("".join(f"{n}=={md.version(n)}\n" for n in PROTECTED if _installed(n)))
    print(constraints.read_text())
    report_path = out / "install-report.json"
    pip("install", "--dry-run", "--quiet", "--report", report_path, "-r", reqs, "-c", constraints, *find_links)
    plan = parse_install_report(json.loads(report_path.read_text()))
    for name, version in plan:
        before = _installed(name)
        print(f"  {name:28s} {version:22s} {'(new)' if not before else f'(image has {before})'}")

    section("3. Download / build wheels")
    regular = [f"{n}=={v}" for n, v in plan if n not in TORCH_EXTENSIONS]
    torch_ext = [f"{n}=={v}" for n, v in plan if n in TORCH_EXTENSIONS]
    if regular:
        pip("wheel", "--no-deps", "-w", wheels, *find_links, *regular)
    if torch_ext:
        pip("wheel", "--no-deps", "--no-build-isolation", "-w", wheels, *find_links, *torch_ext, env=build_env)

    section(f"4. Compile pointops (TORCH_CUDA_ARCH_LIST={args.arch}, MAX_JOBS={args.jobs})")
    t = time.time()
    with tempfile.TemporaryDirectory() as tmp:  # build outside the submodule to keep it clean
        src = shutil.copytree(POINTOPS_SRC, Path(tmp) / "pointops")
        pip("wheel", "--no-deps", "--no-build-isolation", "-w", wheels, src, env=build_env)
    pointops_whl = next(wheels.glob("pointops-*.whl"))
    print(f"built {pointops_whl.name} in {(time.time() - t) / 60:.1f} min")

    section("5. Lock file and manifest")
    lock = plan + [("pointops", pointops_whl.name.split("-")[1])]
    (out / "requirements.lock.txt").write_text("".join(f"{n}=={v}\n" for n, v in sorted(lock)))
    files = sorted(wheels.glob("*.whl"))
    manifest = {
        "created_utc": env_record["utc"],
        "python": sys.version.split()[0],
        "python_tag": py_tag,
        "torch": torch_version,
        "torch_cuda": torch_cuda,
        "nvcc": nvcc,
        "arch_list": args.arch,
        "spconv_variant": spconv,
        "pyg_index": pyg,
        "kaggle_env": env_record["env"],
        "git": env_record["git"],
        "wheels": [{"file": f.name, "sha256": sha256(f), "mb": round(f.stat().st_size / 2**20, 1)} for f in files],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(files)} wheels, {sum(w['mb'] for w in manifest['wheels']):.0f} MB total")

    section("6. Offline install from the wheelhouse (exactly what later sessions do)")
    pip("install", "--no-index", "--no-deps", "-f", wheels, "-r", out / "requirements.lock.txt")
    print(f"spconv/cumm fix (decisions.md D7): {json.dumps(install_spconv_fix())}")
    try:
        pip("check")
    except subprocess.CalledProcessError:
        print("!! pip check reports conflicts above (informational; the smoke test decides)")

    section("7. Import check")
    ok = check_imports()
    print(f"\nwheelhouse: {out}  ({(time.time() - t_start) / 60:.1f} min total)")
    print(
        "NEXT: when this committed run finishes, create a PRIVATE Kaggle Dataset from this\n"
        "notebook version's output (slug: appearance-pt-wheels), then run m1_02_smoke_test."
    )
    return 0 if ok else 1


def _installed(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


if __name__ == "__main__":
    sys.exit(main())
