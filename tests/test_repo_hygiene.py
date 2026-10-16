"""Nothing secret or data-like may be committed (ScanNet++ terms forbid redistribution)."""

import re
import subprocess

from conftest import ROOT

SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"""(?i)\btoken\s*[:=]\s*["'][A-Za-z0-9_\-]{20,}["']"""),
]
DATA_SUFFIXES = (".ply", ".pth", ".pt", ".npy", ".npz", ".h5", ".mkv", ".mp4", ".lz4", ".whl", ".jpg", ".png")


def repo_files():
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split()
    return [ROOT / f for f in out if not f.startswith("third_party/")]


def test_no_secrets_in_repo():
    for path in repo_files():
        if not path.is_file() or path.suffix in DATA_SUFFIXES:
            continue
        # a line may opt out explicitly with the marker "not-a-secret" (test fixtures only)
        text = "\n".join(l for l in path.read_text(errors="ignore").splitlines() if "not-a-secret" not in l)
        for pattern in SECRET_PATTERNS:
            match = pattern.search(text)
            assert not match, f"possible secret in {path.relative_to(ROOT)}: {match.group()[:12]}..."


def test_no_data_files_tracked():
    tracked = [p for p in repo_files() if p.suffix in DATA_SUFFIXES and not str(p).startswith(str(ROOT / "docs/figures"))]
    assert not tracked, f"data-like files would be committed: {tracked}"


def test_gitignore_excludes_data_and_credentials():
    ignored = (ROOT / ".gitignore").read_text()
    for pattern in ("data/", "checkpoints/", "features/", "*.ply", "*.pth", "*.npz", "download_scannetpp*.yml"):
        assert pattern in ignored
