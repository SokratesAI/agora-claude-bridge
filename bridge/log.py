"""Plain stdout logging -- matches agora-persona-runner's log.py (kubectl logs is the sink)."""
import sys
from datetime import datetime, timezone


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", file=sys.stderr, flush=True)
