"""Kaggle bootstrap: fetch this (private) repo at a fixed ref, then run entry points with logs.

Embedded verbatim in every notebook by scripts/make_notebooks.py; edit here, then regenerate.
The GitHub token comes from Kaggle Secrets and reaches git only through environment variables
(never argv, files or output), so it cannot leak into logs or tracebacks; git's error messages
are shown (a notebook would otherwise hide them) with the token redacted.
"""

import base64
import os
import pathlib
import shutil
import subprocess
import sys
import time

_GIT_DEBUG_VARS = ("GIT_TRACE", "GIT_TRACE_CURL", "GIT_CURL_VERBOSE", "GIT_TRACE_PACKET")


def _git(args, cwd=None, env=None, capture=False, redact=()):
    # stderr is captured: a notebook cell never shows a subprocess's stderr, only the exit code
    r = subprocess.run(["git", *args], cwd=cwd, env=env, text=True, capture_output=True)
    if r.returncode != 0:
        err = (r.stderr or "").strip()
        for s in redact:
            err = err.replace(s, "***")
        raise RuntimeError(f"git {args[0]} failed (exit {r.returncode}):\n{err}")
    return r.stdout.strip() if capture else None


def clone_repo(repo, ref, dest="/tmp/appearance-pt", get_secret=None):
    """Fetch `repo` ("owner/name") at `ref` (branch, tag or commit) into `dest`.

    Submodules (public) are fetched at their pinned commits. Returns (dest, commit).
    """
    if repo.startswith("OWNER/"):
        raise ValueError('set REPO in the config cell to your GitHub repo ("owner/name")')
    if get_secret is None:
        from kaggle_secrets import UserSecretsClient

        get_secret = UserSecretsClient().get_secret
    token = get_secret("GITHUB_TOKEN")
    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = {k: v for k, v in os.environ.items() if k not in _GIT_DEBUG_VARS}
    env.update(
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
        GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {auth}",
    )
    hide = (token, auth)
    del token, auth
    dest = pathlib.Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    _git(["init", "-q", str(dest)])
    _git(["remote", "add", "origin", f"https://github.com/{repo}.git"], cwd=dest)
    try:
        _git(["fetch", "-q", "--depth", "1", "origin", ref], cwd=dest, env=env, redact=hide)
    except RuntimeError as e:
        raise RuntimeError(
            f"{e}\n\nCould not fetch {repo} @ {ref}. Check: REPO/REF in the config cell; Internet on; "
            "GITHUB_TOKEN attached to this notebook and allowed to read this repo "
            "(fine-grained token: repo selected under 'Repository access', 'Contents: Read-only', not expired)."
        ) from None
    finally:
        del env, hide
    _git(["checkout", "-q", "FETCH_HEAD"], cwd=dest)
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
