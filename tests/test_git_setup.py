import subprocess
from unittest.mock import patch

from bridge import git_setup


def test_bootstrap_git_skips_when_no_gh_token(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with patch("subprocess.run") as mock_run:
        git_setup.bootstrap_git()
    mock_run.assert_not_called()


def test_bootstrap_git_configures_identity_and_credential_helper(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "ghp_faketoken")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    with patch("subprocess.run", side_effect=fake_run):
        git_setup.bootstrap_git()

    assert ["git", "config", "--global", "user.name", git_setup.GIT_USER_NAME] in calls
    assert ["git", "config", "--global", "user.email", git_setup.GIT_USER_EMAIL] in calls
    helper_calls = [c for c in calls if c[:3] == ["git", "config", "--global"] and "credential" in c[3]]
    assert len(helper_calls) == 1
    assert "GH_TOKEN" in helper_calls[0][4]


def test_bootstrap_git_uses_claude_home_as_git_config_home(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "ghp_faketoken")
    captured_envs = []

    def fake_run(cmd, **kwargs):
        captured_envs.append(kwargs.get("env"))
        return subprocess.CompletedProcess(cmd, 0)

    with patch("subprocess.run", side_effect=fake_run), \
         patch.object(git_setup, "CLAUDE_HOME", "/data/claude-home"):
        git_setup.bootstrap_git()

    assert all(env["HOME"] == "/data/claude-home" for env in captured_envs)
