"""Kaggle bootstrap: fetch this (private) repo at a fixed ref, then run entry points with logs.

Embedded verbatim in every notebook by scripts/make_notebooks.py; edit here, then regenerate.
The GitHub token comes from Kaggle Secrets and reaches git only through environment variables
(never argv, files or output), so it cannot leak into logs or CalledProcessError tracebacks.
"""

import base64
import os
import pathlib
import shutil
import subprocess
import sys
import time

_GIT_DEBUG_VARS = ("GIT_TRACE", "GIT_TRACE_CURL", "GIT_CURL_VERBOSE", "GIT_TRACE_PACKET")


def _git(args, cwd=None, env=None, capture=False):
    r = subprocess.run(["git", *args], cwd=cwd, env=env, check=True, text=True, capture_output=capture)
    return r.stdout.strip() if capture else None


def clone_repo(repo, ref, dest="/tmp/appearance-pt", get_secret=None):
    """Fetch `repo` ("owner/name") at `ref` (branch, tag or commit) into `dest`.

    Submodules (public) are fetched at their pinned commits. Returns (dest, commit).
    """
    if get_secret is None:
        from kaggle_secrets import UserSecretsClient

        get_secret = UserSecretsClient().get_secret
    auth = base64.b64encode(f"x-access-token:{get_secret('GITHUB_TOKEN')}".encode()).decode()
    env = {k: v for k, v in os.environ.items() if k not in _GIT_DEBUG_VARS}
    env.update(
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
        GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {auth}",
    )
    del auth
    dest = pathlib.Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    _git(["init", "-q", str(dest)])
    _git(["remote", "add", "origin", f"https://github.com/{repo}.git"], cwd=dest)
    _git(["fetch", "-q", "--depth", "1", "origin", ref], cwd=dest, env=env)
    _git(["checkout", "-q", "FETCH_HEAD"], cwd=dest)
    del env
    _git(["submodule", "update", "--init", "--depth", "1", "-q"], cwd=dest)
    commit = _git(["rev-parse", "HEAD"], cwd=dest, capture=True)
    print(f"repo {repo} @ {ref} -> {commit}\n{_git(['submodule', 'status'], cwd=dest, capture=True)}")
    return dest, commit


def run_logged(cmd, log_path, cwd, extra_env=None):
    """Run `cmd`, streaming output to the notebook and appending it to `log_path`.

    Raises if the command fails, so a failed step stops "Save & Run All" visibly.
    """
    log_path = pathlib.Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **(extra_env or {})}
    t0 = time.time()
    with open(log_path, "a") as log:
        log.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(map(str, cmd))}\n")
        with subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
            for line in p.stdout:
                sys.stdout.write(line)
                log.write(line)
            rc = p.wait()
    print(f"\n[exit code {rc} after {(time.time() - t0) / 60:.1f} min; log: {log_path}]")
    if rc != 0:
        raise RuntimeError(f"step failed with exit code {rc}; see {log_path}")
