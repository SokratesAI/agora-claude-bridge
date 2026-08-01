"""One-time credentials.json bootstrap -- pipes the claude-auth k8s secret's
raw JSON content straight to disk, unmodified. No field-by-field
reconstruction: an earlier version assembled {"claudeAiOauth": {...}} from
three separate secret keys (access_token/refresh_token/expires_at), which
the `claude` CLI's own client-side validation rejected instantly
("Not logged in") -- the real local credentials.json Edvard generated this
from carries additional fields (scopes, etc.) that the 3-key split had
silently dropped. Piping the whole original file through sidesteps needing
to know its exact schema at all.

Deliberately first-boot-only -- skips entirely if the file already exists.
This is NOT an "always overwrite on start" step: once the CLI does its own
real token refresh, the fresh token lives ONLY on the CLAUDE_HOME PVC; the
k8s secret's refresh token becomes single-use-stale at that point (the
exact same reason the old claude-auth-refresher CronJob died -- see
cli.py's own module docstring). Overwriting on every pod restart would
clobber a live-refreshed good token with the original stale one from the
Secret, every single time.
"""
import json
import os
import stat

from bridge.config import CLAUDE_HOME
from bridge.log import log


def bootstrap_credentials():
    claude_dir = os.path.join(CLAUDE_HOME, ".claude")
    os.makedirs(claude_dir, exist_ok=True)
    dest = os.path.join(claude_dir, ".credentials.json")

    if os.path.exists(dest):
        log(f"credentials: {dest} already exists, leaving it alone (the CLI owns it now)")
        return

    raw = os.environ.get("CLAUDE_CREDENTIALS_JSON", "")
    if not raw:
        log("credentials: no CLAUDE_CREDENTIALS_JSON set, skipping bootstrap "
            "(an ANTHROPIC_API_KEY env var, if set, still works independently)")
        return

    try:
        json.loads(raw)  # fail loudly on garbage rather than writing an unusable file
    except json.JSONDecodeError as e:
        log(f"credentials: CLAUDE_CREDENTIALS_JSON is not valid JSON, skipping bootstrap: {e}")
        return

    with open(dest, "w") as f:
        f.write(raw)
    os.chmod(dest, stat.S_IRUSR | stat.S_IWUSR)  # 600 -- this is a real credential
    log(f"credentials: bootstrapped {dest} from CLAUDE_CREDENTIALS_JSON")
