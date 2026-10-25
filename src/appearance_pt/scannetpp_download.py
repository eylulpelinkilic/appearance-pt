"""ScanNet++ downloads that never expose the token, driven by the official download config.

The official download_scannetpp.py (from the dataset website; kept out of git, no license) prints
the token (its line 291, and the full URL of every file when verbose or on errors) and waits for
a "y" on stdin, so it is not run. This module reads its config (download_scannetpp.yml: URL
template, zipped and excluded assets, meta files), resolves file paths with the toolbox's
ScannetppScene_Release exactly as that script does, and takes the token as an argument (from
Kaggle Secrets). Anything printed or raised has the token replaced by ***.
"""

from __future__ import annotations

import importlib.util
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from . import THIRD_PARTY

PLACEHOLDER_TOKEN = "<YOUR_TOKEN_HERE>"
RETRIES = 5


class RemoteError(RuntimeError):
    def __init__(self, code: int | None, message: str):
        super().__init__(message)
        self.code = code


def load_config(path: Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def _scene_release():
    path = THIRD_PARTY / "scannetpp" / "common" / "scene_release.py"
    spec = importlib.util.spec_from_file_location("scannetpp_scene_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ScannetppScene_Release


def scene_files(cfg: dict, scene_id: str, split: str, assets: list[str]) -> list[dict]:
    """Remote path and local path (relative to the data root) of each asset of one scene.

    Mirrors download_scannetpp.py main(): assets excluded for the split are skipped, zipped
    assets are fetched as <path>.zip and unpacked next to it.
    """
    scene = _scene_release()(scene_id, data_root="data")
    excluded = (cfg.get("exclude_assets") or {}).get(split, [])
    out = []
    for asset in assets:
        if asset in excluded:
            continue
        path = getattr(scene, asset)
        zipped = asset in cfg["zipped_assets"]
        out.append(dict(asset=asset, zipped=zipped, local=path.as_posix(),
                        remote=(path.with_suffix(".zip") if zipped else path).as_posix()))
    return out


def choose_subset(train_ids: list[str], val_ids: list[str], n_train: int, seed: int) -> dict:
    """All val scenes and a fixed random sample of train scenes (sorted, so the list diffs cleanly)."""
    rng = np.random.default_rng(seed)
    train = rng.choice(sorted(train_ids), size=n_train, replace=False)
    return {"seed": seed, "n_train": n_train, "train": sorted(train.tolist()), "val": sorted(val_ids)}


class ScannetppClient:
    def __init__(self, cfg: dict, token: str, pause: float = 0.2):
        if not token or token == PLACEHOLDER_TOKEN:
            raise ValueError("no ScanNet++ token: attach the Kaggle secret SCANNETPP_TOKEN")
        self.cfg, self._token, self.pause = cfg, token, pause

    def redact(self, text: str) -> str:
        return str(text).replace(self._token, "***")

    def _url(self, remote: str) -> str:
        return self.cfg["root_url"].replace("TOKEN", self._token).replace("FILEPATH", str(remote).replace("\\", "/"))

    def _open(self, remote: str, method: str = "GET", headers: dict | None = None):
        """urlopen with the official script's retry policy (back off on 429 and 5xx); errors carry no token."""
        request = urllib.request.Request(self._url(remote), method=method, headers=headers or {})
        for attempt in range(RETRIES):
            try:
                return urllib.request.urlopen(request, timeout=120)
            except urllib.error.HTTPError as e:
                if (e.code == 429 or 500 <= e.code < 600) and attempt < RETRIES - 1:
                    time.sleep(0.5 * 2**attempt)
                    continue
                try:
                    body = e.read().decode("utf-8", errors="replace").strip()[:300]
                except Exception:  # noqa: BLE001
                    body = ""
                hint = {401: " (token invalid or expired)", 406: " (download script version outdated: new config needed)"}
                raise RemoteError(e.code, self.redact(f"HTTP {e.code}{hint.get(e.code, '')} for {remote}: {body}")) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < RETRIES - 1:
                    time.sleep(0.5 * 2**attempt)
                    continue
                raise RemoteError(None, self.redact(f"{type(e).__name__} for {remote}: {e}")) from None

    def size(self, remote: str) -> int | None:
        """Bytes of a remote file without downloading it; None if it does not exist, -1 if unknown."""
        try:
            with self._open(remote, "HEAD") as r:
                length = r.headers.get("Content-Length")
            if length is None:  # 1-byte ranged GET; the body is never read
                with self._open(remote, "GET", {"Range": "bytes=0-0"}) as r:
                    total = (r.headers.get("Content-Range") or "").rpartition("/")[2]
                    length = total if r.status == 206 else r.headers.get("Content-Length")
        except RemoteError as e:
            if e.code == 404:
                return None
            raise
        finally:
            time.sleep(self.pause)
        return int(length) if length and str(length).isdigit() else -1

    def download(self, remote: str, local: Path) -> int:
        """Stream a remote file to `local` (via a .part file, so an interrupted download never looks done)."""
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        part = local.with_name(local.name + ".part")
        with self._open(remote) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        part.rename(local)
        time.sleep(self.pause)
        return local.stat().st_size
