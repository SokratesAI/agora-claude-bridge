"""One-time git/gh bootstrap -- gives a claude-cli session the same
real git+gh workflow this session itself uses (git clone/commit/push,
gh pr create/diff/checks/merge), instead of Agora's purpose-built
github_read/create_pr/merge_pr tools those don't apply here at all.

Idempotent -- safe to call on every boot. Configures HOME (CLAUDE_HOME,
a PVC mount) so the resulting ~/.gitconfig survives pod restarts, same
reasoning as credentials.py's own bootstrap.
"""
import os
import subprocess

from bridge.config import CLAUDE_HOME
from bridge.log import log

GIT_USER_NAME = "SokratesAI Bot"
GIT_USER_EMAIL = "sokrates-ai-user@users.noreply.github.com"


def bootstrap_git():
    token = os.environ.get("GH_TOKEN", "")
    if not token:
        log("git_setup: no GH_TOKEN set, skipping git/gh bootstrap")
        return

    env = {**os.environ, "HOME": CLAUDE_HOME}
    subprocess.run(["git", "config", "--global", "user.name", GIT_USER_NAME], env=env, check=True)
    subprocess.run(["git", "config", "--global", "user.email", GIT_USER_EMAIL], env=env, check=True)

    # Token-based HTTPS credential helper -- no `gh auth login` needed
    # (that's an interactive/device-code flow, unsuitable for a pod).
    # `gh` itself already honors GH_TOKEN for every `gh` subcommand with
    # zero extra setup; this just makes plain `git push`/`git clone` over
    # HTTPS authenticate the same way, mirroring `gh auth setup-git`'s
    # actual effect without requiring a prior interactive login.
    helper = f'!f() {{ echo "username=x-access-token"; echo "password=${{GH_TOKEN}}"; }}; f'
    subprocess.run(
        ["git", "config", "--global", "credential.https://github.com.helper", helper],
        env=env, check=True,
    )
    log("git_setup: configured git identity + HTTPS credential helper from GH_TOKEN")
