"""A durable record of how this process started and how it ended.

Everything else this bridge knows about its own shutdown goes to stdout,
and stdout belongs to a Pod that Kubernetes deletes. So when a cycle is
lost mid-turn, the one question that decides what to fix -- did the
process get SIGTERM and refuse to drain, or did it never get SIGTERM at
all -- has no evidence behind it and has to be inferred from ReplicaSet
timestamps.

That inference is unreliable, and cycle 1246 measured it going wrong.
The bridge Deployment's strategy is `Recreate`, and the deployment
controller creates the new ReplicaSet only *after* every old Pod is
gone, so a new ReplicaSet's creationTimestamp is the moment termination
ENDED. Cycle 1245 read that same timestamp as the moment the rollout
began and concluded from the five-second gap to the replacement
container that the outgoing process had ignored a 2880-second grace.
Both readings fit the same two numbers and they say opposite things.

This file settles it, on the PVC, where it outlives the Pod:

  started  -- this process began serving
  signal   -- SIGTERM/SIGINT arrived, with the turns in flight at that instant
  drained  -- the drain finished and the process is exiting cleanly

`signal` followed by `drained` is a clean drain. `signal` followed by
`started` and no `drained` in between is a process killed before it
finished draining -- the grace expiring, or a SIGKILL. A `started` with
no `signal` before it at all is a process that was never asked to stop:
a crash, an OOM kill, or a delete with no grace, and no drain of any
kind could have saved the turn. Those are three different fixes and
until now they left identical traces.

Never raises. A bridge that cannot append a diagnostic line still has to
serve, and `signal` is written from inside a signal handler where an
exception would escape into whatever the main thread was doing."""
import json
import os
import sys
import time
from datetime import datetime, timezone

from bridge.config import CLAUDE_HOME

# Under CLAUDE_HOME, which is the PVC, for the same reason quota-history
# lives there: the whole point is surviving the Pod.
LIFECYCLE_FILE = os.environ.get(
    "BRIDGE_LIFECYCLE_FILE", os.path.join(CLAUDE_HOME, "bridge-lifecycle.jsonl"))


def record(event, **fields):
    """Append one JSON line. Best effort, by design -- see the module docstring."""
    try:
        row = {
            "event": event,
            "at": datetime.now(timezone.utc).isoformat(),
            "monotonic": round(time.monotonic(), 3),
            "pid": os.getpid(),
        }
        row.update(fields)
        line = json.dumps(row, default=str) + "\n"
        # One open-append-close per row, and one write() call for the whole
        # line: O_APPEND makes a single small write atomic against other
        # writers, so a second process cannot interleave half a row into
        # this one. Buffering it would risk losing the `signal` row to the
        # very kill this file exists to record.
        with open(LIFECYCLE_FILE, "a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception as e:  # noqa: BLE001 -- see the module docstring
        print(f"[lifecycle_log] could not record {event!r}: {e}", file=sys.stderr, flush=True)


def read(path=None):
    """Every row, oldest first. A malformed line is skipped rather than
    raising: this file is appended to from a signal handler and a torn
    final line must not make the whole history unreadable."""
    rows = []
    try:
        with open(path or LIFECYCLE_FILE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except FileNotFoundError:
        return []
    return rows


def lives(rows):
    """Group the rows into one entry per process life, each with a verdict.

    The verdicts are the three causes the raw rows exist to separate, plus
    the one still in progress:

      clean            -- SIGTERM arrived and the drain finished. Any turn
                          in flight got to return its reply.
      killed_mid_drain -- SIGTERM arrived and no `drained` row followed, so
                          the process died inside the drain: the grace
                          period expiring, or a SIGKILL. `in_flight` on the
                          signal row is how many turns that cost.
      no_signal        -- the process ended and was never asked to stop. A
                          crash, an OOM kill, or a delete with no grace. No
                          drain could have helped and fixing the drain
                          would be fixing the wrong thing.
      running          -- the newest life; still serving, or draining now.
                          It cannot be judged until a later life starts.

    Only rows between two `started` rows belong to a life, so a file that
    was rotated or truncated mid-life yields fewer lives rather than a
    wrong verdict on a life it cannot see the start of."""
    out = []
    current = None
    for row in rows:
        if row.get("event") == "started":
            if current is not None:
                out.append(current)
            current = {"started": row, "signal": None, "drained": None}
            continue
        if current is None:
            continue
        if row.get("event") == "signal" and current["signal"] is None:
            current["signal"] = row
        elif row.get("event") == "drained":
            current["drained"] = row
    if current is not None:
        out.append(current)
    for i, life in enumerate(out):
        still_running = i == len(out) - 1
        if life["drained"] is not None:
            life["verdict"] = "clean"
        elif life["signal"] is not None:
            life["verdict"] = "running" if still_running else "killed_mid_drain"
        else:
            life["verdict"] = "running" if still_running else "no_signal"
        life["turns_lost"] = (
            life["signal"].get("in_flight", 0)
            if life["verdict"] == "killed_mid_drain" else 0)
    return out
