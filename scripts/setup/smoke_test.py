"""M1 smoke test: does the GPU stack work for Pointcept, DITR, Sonata and DDP on Kaggle T4s?

Each check runs in its own process (Pointcept and DITR both ship a top-level `pointcept`
package; a CUDA fault in one check must not take down the others) with a timeout, and
prints a SMOKE_RESULT line that ends up in the summary table and summary.json.
Two automatic fallbacks make failures informative: DDP is retried with NCCL_P2P_DISABLE=1,
and DITR is retried with torch.compile disabled.

Usage: python scripts/setup/smoke_test.py [--out /kaggle/working/smoke] [--only NAME ...]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from appearance_pt import REPO_ROOT, THIRD_PARTY
from appearance_pt.envinfo import collect
from appearance_pt.runlog import Check, Summary, parse_result, section

SMOKE = REPO_ROOT / "scripts/setup/smoke"
SRC = str(REPO_ROOT / "src")


def check_specs(n_gpus: int) -> list[dict]:
    py = sys.executable
    return [
        dict(name="extensions", cmd=[py, SMOKE / "check_extensions.py"], cwd=REPO_ROOT, path=[SRC], timeout=600),
        dict(name="scannetpp_eval_import", cmd=[py, "-c", "import semantic.eval.eval_semantic; print('ok')"],
             cwd=THIRD_PARTY / "scannetpp", path=["."], timeout=120),
        dict(name="sonata_demo", cmd=[py, SMOKE / "check_sonata.py"], cwd=REPO_ROOT,
             path=[str(THIRD_PARTY / "sonata"), SRC], timeout=1200),
        dict(name="pointcept_ptv3_train", cmd=[py, SMOKE / "check_pointcept.py"], cwd=THIRD_PARTY / "Pointcept",
             path=[".", SRC], timeout=1200),
        dict(name="ditr_image_train", cmd=[py, SMOKE / "check_ditr.py"], cwd=THIRD_PARTY / "ditr", path=[".", SRC],
             timeout=1500, fallback=dict(name="ditr_image_train[no torch.compile]", env={"TORCHDYNAMO_DISABLE": "1"})),
        dict(name=f"ddp_nccl_{n_gpus}gpu",
             cmd=[py, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n_gpus}", SMOKE / "check_ddp.py"],
             cwd=REPO_ROOT, path=[SRC], timeout=300, needs_gpus=2,
             fallback=dict(name=f"ddp_nccl_{n_gpus}gpu[NCCL_P2P_DISABLE=1]", env={"NCCL_P2P_DISABLE": "1"})),
    ]


def _stream(p: subprocess.Popen, lines: list[str]) -> None:
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)


def run_check(spec: dict, out_dir: Path, extra_env: dict | None = None, name: str | None = None) -> Check:
    name = name or spec["name"]
    section(f"CHECK {name}")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "SMOKE_OUT": str(out_dir),
           "PYTHONPATH": os.pathsep.join(spec["path"]), **(extra_env or {})}
    t = time.time()
    lines: list[str] = []
    # own process group, so a timeout also kills torchrun's worker processes
    p = subprocess.Popen([str(c) for c in spec["cmd"]], cwd=spec["cwd"], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    reader = threading.Thread(target=_stream, args=(p, lines), daemon=True)
    reader.start()
    try:
        rc = p.wait(timeout=spec["timeout"])  # also catches silent hangs (e.g. NCCL deadlock)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        rc = f"timeout after {spec['timeout']}s"
    reader.join(timeout=10)
    seconds = time.time() - t
    metrics = parse_result("".join(lines))
    if rc == 0:
        detail = ", ".join(f"{k}={v}" for k, v in metrics.items())
        return Check(name, "PASS", seconds, detail, metrics)
    last = [l.strip() for l in lines if l.strip()][-1:] or ["<no output>"]
    return Check(name, "FAIL", seconds, f"exit {rc}: {last[0][:160]}", metrics)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/smoke"))
    ap.add_argument("--only", nargs="*", help="run only checks whose name starts with one of these")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    n_gpus = torch.cuda.device_count()
    record = collect()
    section("environment")
    print(json.dumps({k: record[k] for k in ("python", "git", "torch", "packages")}, indent=1))
    if n_gpus == 0:
        print("!! no GPU visible: set Accelerator to GPU T4 x2")
        return 2

    summary = Summary("M1 smoke test")
    for spec in check_specs(n_gpus):
        if args.only and not any(spec["name"].startswith(o) for o in args.only):
            continue
        if n_gpus < spec.get("needs_gpus", 1):
            summary.add(Check(spec["name"], "SKIP", 0.0, f"needs {spec['needs_gpus']} GPUs, have {n_gpus}"))
            continue
        result = run_check(spec, args.out)
        summary.add(result)
        if result.status == "FAIL" and "fallback" in spec:
            fb = spec["fallback"]
            retry = run_check(spec, args.out, extra_env=fb["env"], name=fb["name"])
            if retry.status == "PASS":  # informative, but the default configuration still failed
                retry.status = "WARN"
                retry.detail = "passes only with this setting -> " + retry.detail
            summary.add(retry)

    summary.print()
    summary.save(args.out / "summary.json", environment=record)
    return 0 if summary.ok else 1


if __name__ == "__main__":
    sys.exit(main())
