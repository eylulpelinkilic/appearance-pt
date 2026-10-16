"""Readable logs for Kaggle entry points: timestamped sections and a PASS/FAIL summary."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

RESULT_PREFIX = "SMOKE_RESULT "


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n== {title}    [{now()}]\n{'=' * 78}", flush=True)


def emit_result(**metrics) -> None:
    """Last line of a check script: machine-readable metrics for the orchestrator."""
    print(RESULT_PREFIX + json.dumps(metrics, default=str), flush=True)


def parse_result(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    return {}


@dataclass
class Check:
    name: str
    status: str  # PASS | FAIL | WARN | SKIP
    seconds: float
    detail: str = ""
    metrics: dict = field(default_factory=dict)


class Summary:
    def __init__(self, title: str):
        self.title = title
        self.checks: list[Check] = []
        self.t0 = time.time()

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def ok(self) -> bool:
        return all(c.status != "FAIL" for c in self.checks)

    def print(self) -> None:
        section(f"SUMMARY: {self.title}")
        width = max([len(c.name) for c in self.checks] + [5])
        for c in self.checks:
            print(f"{c.status:4s}  {c.name:{width}s}  {c.seconds:7.1f}s  {c.detail}")
        print(f"\nOverall: {'PASS' if self.ok else 'FAIL'}  ({time.time() - self.t0:.0f}s total)", flush=True)

    def save(self, path: Path, **extra) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"title": self.title, "ok": self.ok, "checks": [asdict(c) for c in self.checks], **extra}
        path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"summary written to {path}")
