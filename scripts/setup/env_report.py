"""M1 environment report for a Kaggle GPU session.

Prints GPU, driver/CUDA, compilers, the preinstalled PyTorch stack, disk, network and
secret availability, so the environment plan rests on measured numbers. Installs nothing
and never prints secret values (only whether a secret is attached).

Standalone on purpose (stdlib + preinstalled torch): it is embedded verbatim in
notebooks/m1_00_env_report.ipynb and must work before the repo is cloned.
Regenerate the notebook with `python scripts/make_notebooks.py` after editing.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import date, datetime, timezone

OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
REPORT = {}
SAFE_KAGGLE_VARS = ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_DOCKER_IMAGE", "KAGGLE_CONTAINER_NAME")
SCANNETPP_TOKEN_EXPIRY = date(2026, 10, 25)
PACKAGES = (
    "torch", "torchvision", "torchaudio", "triton", "numpy", "scipy", "pandas", "scikit-learn",
    "timm", "transformers", "peft", "accelerate", "huggingface-hub", "safetensors", "einops",
    "addict", "yapf", "termcolor", "tensorboard", "tensorboardX", "wandb", "h5py", "plyfile",
    "imageio", "opencv-python", "opencv-python-headless", "pillow", "matplotlib", "open3d",
    "trimesh", "albumentations", "munch", "regex", "SharedArray", "spconv-cu124", "spconv-cu126",
    "torch-scatter", "torch-cluster", "torch-sparse", "torch-geometric", "flash-attn", "xformers",
    "ninja", "cmake", "setuptools", "wheel", "pip", "packaging", "psutil",
)
NETWORK_URLS = {
    "github": "https://github.com",
    "github_api": "https://api.github.com",
    "pypi": "https://pypi.org/simple/pip/",
    "pytorch_wheels": "https://download.pytorch.org/whl/",
    "pyg_wheels": "https://data.pyg.org/whl/",
    "huggingface": "https://huggingface.co",
    "scannetpp_server": "https://kaldir.vc.in.tum.de/scannetpp/",
}
# public 434 MB checkpoint, used only to measure download bandwidth (stream is cut after BW_BYTES)
BW_URL = "https://huggingface.co/facebook/sonata/resolve/main/sonata.pth"
BW_BYTES = 200 * 2**20


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n== {title}\n{'=' * 78}", flush=True)


def sh(cmd: str, timeout: int = 60, echo: bool = True) -> str:
    """Run a shell command and print its output; never raises."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).rstrip()
    except subprocess.TimeoutExpired:
        out = f"<timeout after {timeout}s>"
    if echo:
        print(f"$ {cmd}\n{out if out else '<no output>'}", flush=True)
    return out


def system() -> None:
    section("1. System")
    info = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "kaggle_env": {k: os.environ.get(k) for k in SAFE_KAGGLE_VARS},
        # names only: several KAGGLE_* variables hold credentials
        "kaggle_env_names": sorted(k for k in os.environ if k.startswith("KAGGLE_")),
    }
    for k, v in info.items():
        print(f"{k:18s} {v}")
    sh("head -4 /etc/os-release")
    sh("lscpu | grep -E '^Model name|^CPU\\(s\\)|^Thread|^Core|^Socket'")
    sh("free -g")
    try:
        mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split(":")[0] in ("MemTotal", "MemAvailable")}
        info["ram_total_gb"] = round(mem["MemTotal"] / 2**20, 1)
        info["ram_available_gb"] = round(mem["MemAvailable"] / 2**20, 1)
    except OSError:
        pass
    REPORT["system"] = info


def gpu() -> None:
    section("2. GPU and driver (nvidia-smi)")
    sh("nvidia-smi")
    q = sh(
        "nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap,clocks.max.sm,"
        "power.limit,pcie.link.gen.max,pcie.link.width.max --format=csv"
    )
    sh("nvidia-smi topo -m")
    header = sh("nvidia-smi | grep -i 'CUDA Version'", echo=False)
    REPORT["gpu"] = {"query_csv": q, "driver_header": header.strip(" |")}


def toolchain() -> None:
    section("3. CUDA toolkit and compilers (needed to build pointops)")
    print(f"CUDA_HOME={os.environ.get('CUDA_HOME')}")
    sh("ls -d /usr/local/cuda*")
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
    print(f"nvcc: {nvcc}")
    REPORT["toolchain"] = {
        "nvcc_path": nvcc,
        "nvcc": sh(f"{nvcc} --version | grep release") if nvcc else None,
        "gcc": sh("gcc --version | head -1"),
        "gxx": sh("g++ --version | head -1"),
        "ninja": sh("ninja --version"),
        "cmake": sh("cmake --version | head -1"),
        "git": sh("git --version"),
    }


def python_packages() -> None:
    section("4. Python and preinstalled packages")
    print(f"python     {sys.version}\nexecutable {sys.executable}")
    sh("pip --version")
    sh("uv --version")
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    for name, v in versions.items():
        print(f"  {name:24s} {v or '-- not installed --'}")
    REPORT["python"] = {"version": sys.version.split()[0], "executable": sys.executable, "packages": versions}


def pytorch() -> None:
    section("5. PyTorch (preinstalled) on this GPU")
    info = {}
    REPORT["torch"] = info
    try:
        import torch
    except ImportError as e:
        print(f"torch not importable: {e}")
        return
    info.update(
        version=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        cuda_available=torch.cuda.is_available(), device_count=torch.cuda.device_count(),
    )
    if not torch.cuda.is_available():
        print(json.dumps(info, indent=1))
        print("!! CUDA not available: enable the GPU accelerator (Settings -> Accelerator -> GPU T4 x2).")
        return
    info["arch_list"] = torch.cuda.get_arch_list()
    info["supports_sm75"] = "sm_75" in info["arch_list"]
    info["devices"] = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        info["devices"].append({"name": p.name, "capability": f"{p.major}.{p.minor}",
                                "memory_gb": round(p.total_memory / 2**30, 2), "sms": p.multi_processor_count})
    try:
        info["bf16_native"] = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        info["bf16_native"] = torch.cuda.is_bf16_supported()
    try:
        info["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception as e:  # noqa: BLE001
        info["nccl"] = f"unavailable ({type(e).__name__})"
    info["nccl_backend"] = torch.distributed.is_nccl_available()
    print(json.dumps(info, indent=1))

    print("\n-- matmul throughput per GPU (4096^2, TFLOP/s) --")
    info["matmul_tflops"] = {}
    for i in range(torch.cuda.device_count()):
        dev = torch.device(f"cuda:{i}")
        res = {}
        for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
            a = torch.randn(4096, 4096, device=dev, dtype=dtype)
            b = torch.randn_like(a)
            for _ in range(3):
                a @ b
            torch.cuda.synchronize(dev)
            t = time.perf_counter()
            for _ in range(20):
                a @ b
            torch.cuda.synchronize(dev)
            res[name] = round(2 * 4096**3 * 20 / (time.perf_counter() - t) / 1e12, 1)
            del a, b
        info["matmul_tflops"][f"cuda:{i}"] = res
        print(f"cuda:{i} {res}")

    print("\n-- scaled_dot_product_attention backends (fp16, 4x8x1024x64) --")
    info["sdpa"] = {}
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = torch.randn(4, 8, 1024, 64, device="cuda", dtype=torch.float16)
        for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
            try:
                with sdpa_kernel(be):
                    torch.nn.functional.scaled_dot_product_attention(q, q, q)
                torch.cuda.synchronize()
                info["sdpa"][be.name] = "ok"
            except RuntimeError as e:
                info["sdpa"][be.name] = f"unavailable: {str(e).splitlines()[0][:100]}"
    except ImportError as e:
        info["sdpa"]["error"] = str(e)
    for k, v in info["sdpa"].items():
        print(f"  {k:22s} {v}")

    print("\n-- torch.compile (inductor/triton) on this GPU; DITR compiles DINOv2 --")
    try:
        def f(x, w):
            return torch.nn.functional.layer_norm(torch.nn.functional.gelu(x @ w), (x.shape[-1],)) * x.sin()

        x = torch.randn(4096, 256, device="cuda", dtype=torch.float16)
        w = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        t = time.perf_counter()
        y = torch.compile(f, fullgraph=True)(x, w)
        torch.cuda.synchronize()
        err = (y.float() - f(x, w).float()).abs().max().item()
        info["torch_compile"] = f"ok (compile+run {time.perf_counter() - t:.1f}s, max abs err {err:.2e})"
    except Exception as e:  # noqa: BLE001
        info["torch_compile"] = f"FAILED: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
    print(f"  {info['torch_compile']}")


def disk() -> None:
    section("6. Disk space")
    sh("df -h")
    rows = {}
    for path in ("/", "/kaggle/working", "/kaggle/input", "/kaggle/temp", "/kaggle/tmp", "/tmp", "/dev/shm", "/root"):
        if os.path.exists(path):
            u = shutil.disk_usage(path)
            rows[path] = {"total_gb": round(u.total / 1e9, 1), "free_gb": round(u.free / 1e9, 1), "writable": os.access(path, os.W_OK)}
    print(f"\n{'path':18s} {'total GB':>9s} {'free GB':>9s}  writable")
    for path, r in rows.items():
        print(f"{path:18s} {r['total_gb']:9.1f} {r['free_gb']:9.1f}  {r['writable']}")
    REPORT["disk"] = rows


def network() -> None:
    section("7. Network (Internet must be ON for setup notebooks)")
    res = {}
    for name, url in NETWORK_URLS.items():
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "appearance-pt-env-report"}), timeout=15) as r:
                res[name] = f"HTTP {r.status} in {1000 * (time.perf_counter() - t):.0f} ms"
        except Exception as e:  # noqa: BLE001
            res[name] = f"FAILED {type(e).__name__}: {str(e)[:80]}"
        print(f"  {name:18s} {url:48s} {res[name]}")
    t = time.perf_counter()
    got = 0
    try:
        with urllib.request.urlopen(BW_URL, timeout=30) as r:
            while got < BW_BYTES:
                chunk = r.read(8 * 2**20)
                if not chunk:
                    break
                got += len(chunk)
        dt = time.perf_counter() - t
        res["download_MBps"] = round(got / 2**20 / dt, 1)
        print(f"  download bandwidth: {got / 2**20:.0f} MiB in {dt:.1f}s = {res['download_MBps']} MiB/s (from huggingface.co)")
    except Exception as e:  # noqa: BLE001
        res["download_MBps"] = f"FAILED {type(e).__name__}"
        print(f"  download bandwidth test failed: {e}")
    REPORT["network"] = res


def secrets() -> None:
    section("8. Kaggle Secrets (values are never printed)")
    res = {}
    tokens = {}
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        for name in ("GITHUB_TOKEN", "SCANNETPP_TOKEN"):
            try:
                tokens[name] = client.get_secret(name)
                res[name] = "attached" if tokens[name] else "empty"
            except Exception as e:  # noqa: BLE001
                res[name] = f"not attached ({type(e).__name__})"
    except ImportError:
        res["error"] = "kaggle_secrets not importable (not a Kaggle session?)"
    for k, v in res.items():
        print(f"  {k:16s} {v}")
    days = (SCANNETPP_TOKEN_EXPIRY - date.today()).days
    print(f"  ScanNet++ token expires {SCANNETPP_TOKEN_EXPIRY} ({days} days from today); extension possible from 2026-10-18.")
    res["scannetpp_token_days_left"] = days

    repo = os.environ.get("APPT_GITHUB_REPO", "")
    token = tokens.get("GITHUB_TOKEN")
    if token:
        def gh(path, token=token):
            req = urllib.request.Request(f"https://api.github.com{path}", headers={"Authorization": f"Bearer {token}", "User-Agent": "appearance-pt-env-report"})
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.load(r)
        try:
            res["github_login"] = gh("/user")["login"]
            print(f"  GitHub token valid for user: {res['github_login']}")
        except Exception as e:  # noqa: BLE001
            res["github_login"] = f"FAILED {type(e).__name__}"
            print(f"  GitHub token check failed: {type(e).__name__}")
        if repo and "/" in repo and "OWNER" not in repo:
            try:
                r = gh(f"/repos/{repo}")
                res["github_repo"] = {"name": r["full_name"], "private": r["private"], "default_branch": r["default_branch"]}
                print(f"  repo {r['full_name']}: private={r['private']} default_branch={r['default_branch']} -> clone will work")
            except Exception as e:  # noqa: BLE001
                res["github_repo"] = f"FAILED {type(e).__name__}"
                print(f"  cannot access repo {repo}: {type(e).__name__} (check token scope: contents:read on this repo)")
        else:
            print("  APPT_GITHUB_REPO not set in the config cell; skipping repo access check.")
    REPORT["secrets"] = res


def summary() -> None:
    section("SUMMARY (paste this block back)")
    s, g, tc = REPORT.get("system", {}), REPORT.get("gpu", {}), REPORT.get("toolchain", {})
    t, d = REPORT.get("torch", {}), REPORT.get("disk", {})
    gpus = [f"{x['name']} sm{x['capability']} {x['memory_gb']}GB" for x in t.get("devices", [])]
    lines = [
        f"kaggle env       : {s.get('kaggle_env')}",
        f"python           : {REPORT.get('python', {}).get('version')}",
        f"gpus             : {gpus}",
        f"driver           : {g.get('driver_header')}",
        f"torch            : {t.get('version')} (cuda {t.get('cuda')}, cudnn {t.get('cudnn')}, sm_75 in arch list: {t.get('supports_sm75')})",
        f"nvcc             : {tc.get('nvcc') or 'NOT FOUND'}",
        f"gcc              : {tc.get('gcc')}",
        f"bf16 native      : {t.get('bf16_native')}   nccl: {t.get('nccl')}",
        f"matmul TFLOP/s   : {t.get('matmul_tflops')}",
        f"sdpa             : {t.get('sdpa')}",
        f"torch.compile    : {t.get('torch_compile')}",
        f"ram / cpus       : {s.get('ram_total_gb')} GB / {s.get('cpu_count')}",
        "disk free (GB)   : " + ", ".join(f"{p}={r['free_gb']}/{r['total_gb']}" for p, r in d.items()),
        f"network          : {REPORT.get('network')}",
        f"secrets          : {REPORT.get('secrets')}",
        f"section errors   : {REPORT.get('errors', 'none')}",
    ]
    print("\n".join(lines))
    path = os.path.join(OUT_DIR, "m1_env_report.json")
    with open(path, "w") as fh:
        json.dump(REPORT, fh, indent=2, default=str)
    print(f"\nfull report: {path}")


def main() -> None:
    for step in (system, gpu, toolchain, python_packages, pytorch, disk, network, secrets):
        try:
            step()
        except Exception as e:  # noqa: BLE001 - a broken section must not hide the others
            print(f"!! section {step.__name__} crashed: {type(e).__name__}: {e}")
            REPORT.setdefault("errors", []).append(f"{step.__name__}: {type(e).__name__}: {e}")
    summary()


if __name__ == "__main__":
    main()
