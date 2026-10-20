"""The Kaggle bootstrap must never expose the GitHub token (argv, output, submodule fetches)."""

import base64
import subprocess
import sys

import pytest

from conftest import load_script

TOKEN = "fake-token-value-for-tests-0123456789"  # not-a-secret (deliberately not GitHub's token format)


@pytest.fixture
def boot():
    return load_script("scripts/setup/kaggle_bootstrap.py")


def test_clone_keeps_token_out_of_argv_and_output(boot, monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run(args, cwd=None, env=None, check=False, text=False, capture_output=False):
        calls.append((list(args), dict(env) if env else None))
        return subprocess.CompletedProcess(args, 0, stdout="abc123\n" if capture_output else None, stderr="")

    monkeypatch.setattr(boot.subprocess, "run", fake_run)
    dest, commit = boot.clone_repo("me/appearance-pt", "main", dest=tmp_path / "repo", get_secret=lambda name: TOKEN)

    encoded = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
    assert commit == "abc123"
    for args, env in calls:
        joined = " ".join(map(str, args))
        assert TOKEN not in joined and encoded not in joined
    fetch_envs = [env for args, env in calls if "fetch" in args]
    assert fetch_envs and all(encoded in env["GIT_CONFIG_VALUE_0"] for env in fetch_envs)
    # submodules are public: no credentials passed to those calls
    for args, env in calls:
        if "submodule" in args:
            assert env is None or all(encoded not in v for v in env.values())
    out = capsys.readouterr().out
    assert TOKEN not in out and encoded not in out


def test_failed_fetch_shows_git_error_without_token(boot, monkeypatch, tmp_path):
    encoded = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()

    def fake_run(args, cwd=None, env=None, check=False, text=False, capture_output=False):
        if "fetch" in args:  # worst case: git echoes the credentials
            return subprocess.CompletedProcess(args, 128, stdout="", stderr=f"remote: Repository not found. {TOKEN} {encoded}\n")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(boot.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as e:
        boot.clone_repo("me/appearance-pt", "main", dest=tmp_path / "repo", get_secret=lambda name: TOKEN)
    msg = str(e.value)
    assert "Repository not found" in msg and "exit 128" in msg
    assert TOKEN not in msg and encoded not in msg


def test_placeholder_repo_is_rejected(boot):
    with pytest.raises(ValueError, match="REPO"):
        boot.clone_repo("OWNER/appearance-pt", "main", get_secret=lambda name: TOKEN)


def test_run_logged_streams_logs_and_raises(boot, tmp_path):
    log = tmp_path / "logs" / "step.log"
    boot.run_logged([sys.executable, "-c", "print('hello from step')"], log, cwd=tmp_path)
    assert "hello from step" in log.read_text()
    with pytest.raises(RuntimeError, match="exit code 3"):
        boot.run_logged([sys.executable, "-c", "import sys; sys.exit(3)"], log, cwd=tmp_path)
