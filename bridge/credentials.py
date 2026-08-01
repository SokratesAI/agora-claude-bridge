"""One-time credentials.json bootstrap from the claude-auth k8s secret's
three separate keys (access_token/refresh_token/expires_at) into the
single-file format the `claude` CLI actually reads for OAuth
(subscription) auth.

Deliberately first-boot-only -- skips entirely if the file already exists.
This is NOT an "always overwrite on start" step: once the CLI does its own
real token refresh, the fresh token lives ONLY on the CLAUDE_HOME PVC; the
k8s secret's refresh_token becomes single-use-stale at that point (the
exact same reason the old claude-auth-refresher CronJob died -- see
cli.py's own module docstring). Overwriting on every pod restart would
clobber a live-refreshed good token with the original stale one from the
Secret, every single time.
"""
import json
import os
import stat
import time

from bridge.config import CLAUDE_HOME
from bridge.log import log


def bootstrap_credentials():
    claude_dir = os.path.join(CLAUDE_HOME, ".claude")
    os.makedirs(claude_dir, exist_ok=True)
    dest = os.path.join(claude_dir, ".credentials.json")

    if os.path.exists(dest):
        log(f"credentials: {dest} already exists, leaving it alone (the CLI owns it now)")
        return

    access_token = os.environ.get("CLAUDE_ACCESS_TOKEN", "")
    refresh_token = os.environ.get("CLAUDE_REFRESH_TOKEN", "")
    expires_at = os.environ.get("CLAUDE_EXPIRES_AT", "")
    if not (access_token and refresh_token and expires_at):
        log("credentials: no CLAUDE_ACCESS_TOKEN/CLAUDE_REFRESH_TOKEN/CLAUDE_EXPIRES_AT set, "
            "skipping bootstrap (an ANTHROPIC_API_KEY env var, if set, still works independently)")
        return

    creds = {
        "claudeAiOauth": {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "expiresAt": int(expires_at),
        }
    }
    with open(dest, "w") as f:
        json.dump(creds, f)
    os.chmod(dest, stat.S_IRUSR | stat.S_IWUSR)  # 600 -- this is a real credential

    remaining_h = (int(expires_at) - int(time.time() * 1000)) / 3_600_000
    log(f"credentials: bootstrapped {dest}, expiresAt ~{remaining_h:.1f}h from now")
