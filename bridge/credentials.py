"""One-time credentials.json bootstrap -- pipes the claude-auth k8s secret's
raw JSON content straight to disk, unmodified. No field-by-field
reconstruction: an earlier version assembled {"claudeAiOauth": {...}} from
three separate secret keys (access_token/refresh_token/expires_at), which
the `claude` CLI's own client-side validation rejected instantly
("Not logged in") -- the real local credentials.json Edvard generated this
from carries additional fields (scopes, etc.) that the 3-key split had
silently dropped. Piping the whole original file through sidesteps needing
to know its exact schema at all.

Deliberately not an "always overwrite on start" step: once the CLI does its
own real token refresh, the fresh token lives ONLY on the CLAUDE_HOME PVC;
the k8s secret's refresh token becomes single-use-stale at that point (the
exact same reason the old claude-auth-refresher CronJob died -- see cli.py's
own module docstring). Overwriting on every pod restart would clobber a
live-refreshed good token with the original stale one from the Secret, every
single time.

That invariant is preserved below and the rule is now stated as what it
always meant: **the newer credential wins.** A live-refreshed on-disk token
always carries a later `expiresAt` than the Secret's frozen snapshot, so
"newer wins" never clobbers it -- while a Secret a human has just refreshed
by hand *is* newer, and that case was silently ignored.

Both halves of that cost a 30-hour outage on 2026-08-17/19, measured from
this pod's own log on 2026-08-19 (Cycle 266):

- The PVC was replaced (`agora-claude-bridge-data` created 16:42:21Z, pod
  started 16:42:27Z), so the live credential -- the only copy -- was gone
  and this module bootstrapped onto an empty volume. The Secret it copied
  had `expiresAt` **2026-08-01T18:22:21Z**, sixteen days expired, and
  `subscriptionType: pro` against the live account's `max`. It wrote that
  and logged `bootstrapped ... from CLAUDE_CREDENTIALS_JSON`, which reads
  as success. The next CLI invocation, 20 hours later, exited 1; every
  heartbeat after it did too, until Edvard noticed and re-authed by hand.
- The documented recovery -- put a fresh credential in the Secret, restart
  the pod -- could not have worked either, because the dead file now
  existed and `os.path.exists` sent the bootstrap home.

So an expired Secret is still written when there is nothing else (a doomed
credential beats no credential, and the CLI's error message is clearer than
"Not logged in"), but it is logged as the alarm it is rather than as a
success.
"""
import json
import os
import stat
import time

from bridge.config import CLAUDE_HOME
from bridge.log import log


def _expires_at(raw):
    """`claudeAiOauth.expiresAt` as epoch seconds, or None if unreadable.

    None means "no opinion", and every caller below treats it as a reason to
    keep the existing behaviour rather than to act -- an unparseable
    credential is the absence of evidence, not evidence that replacing it is
    safe. Same reasoning as cli.refresh_window_clear.
    """
    try:
        return float(json.loads(raw)["claudeAiOauth"]["expiresAt"]) / 1000.0
    except Exception:
        return None


def _write(dest, raw):
    with open(dest, "w") as f:
        f.write(raw)
    os.chmod(dest, stat.S_IRUSR | stat.S_IWUSR)  # 600 -- this is a real credential


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""


def bootstrap_credentials(now=None):
    now = time.time() if now is None else now
    claude_dir = os.path.join(CLAUDE_HOME, ".claude")
    os.makedirs(claude_dir, exist_ok=True)
    dest = os.path.join(claude_dir, ".credentials.json")

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

    secret_expiry = _expires_at(raw)

    if os.path.exists(dest):
        disk_expiry = _expires_at(_read(dest))
        if secret_expiry is None or disk_expiry is None or secret_expiry <= disk_expiry:
            log(f"credentials: {dest} already exists, leaving it alone (the CLI owns it now)")
            return
        # Strictly newer. The CLI cannot have produced this -- its refreshes
        # only ever land on disk -- so a human updated the Secret, which is
        # the recovery path, and skipping it is how that path silently did
        # nothing on 2026-08-17.
        _write(dest, raw)
        log(f"credentials: replaced {dest} from CLAUDE_CREDENTIALS_JSON -- the Secret is "
            f"newer than what was on disk (secret expiresAt {secret_expiry:.0f} > "
            f"on-disk {disk_expiry:.0f}), so a human refreshed it")
        return

    _write(dest, raw)
    if secret_expiry is not None and secret_expiry <= now:
        hours = (now - secret_expiry) / 3600.0
        log(f"credentials: WROTE AN EXPIRED CREDENTIAL to {dest} -- the claude-auth Secret's "
            f"token expired {hours:.1f} hours ago and this volume had nothing else. Every CLI "
            f"invocation will fail until the Secret carries a fresh credentials.json and this "
            f"pod restarts. This is the whole outage, at second zero, not 20 hours later.")
        return
    log(f"credentials: bootstrapped {dest} from CLAUDE_CREDENTIALS_JSON")
