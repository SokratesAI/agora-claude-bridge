"""Tells the running session how much wall-clock it has left.

A turn is killed at CLI_TIMEOUT_SECONDS (2700s, config.py) and until now
nothing told the session that. It could read its own quota -- quota.py
built that whole channel -- but not its own clock, so it paced against a
budget it could see and ran into a deadline it could not.

That is not hypothetical. Measured 2026-08-10 across the last 30 Nova
cycles in Agora: 27 posted a reply, and Cycle 81's turn was killed at the
45-minute mark with `failed: timed out` and no reply at all. It had
already merged its PR and written its journal entry; it died three tool
calls into rewriting the digest. The owner was told only that it failed,
which is the opposite of what happened. Cycle durations are also
climbing -- the five completed cycles before it ran 33, 36, 36, 36 and 29
minutes against a 45-minute ceiling -- so this stops being a rare edge
the more the loop does per cycle.

Same shape as quota.py's snapshot for the same reason: the hook runs in
the CLI's own process tree, not this one, so the two communicate through
a file on CLAUDE_HOME rather than through memory.

The thresholds in hooks/deadline_notice.py are measured off the same 30
cycles, not guessed -- see that module.
"""
import json
import os
import time

from bridge.config import CLAUDE_HOME
from bridge.log import log

DEADLINE_FILE = os.path.join(CLAUDE_HOME, "turn-deadline.json")


def write(timeout_seconds, path=None):
    """Record that a turn just started and when it will be killed.

    Best-effort, like every other side-channel here: a turn that cannot
    write this file still runs, it just runs without a clock.
    """
    path = path or DEADLINE_FILE
    started = time.time()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump({
                "started_at": started,
                "deadline_at": started + timeout_seconds,
                "timeout_seconds": timeout_seconds,
            }, handle)
        return True
    except Exception as exc:
        log(f"deadline write failed: {type(exc).__name__}: {exc}")
        return False


def clear(path=None):
    """Drop the record when the turn ends.

    Matters more than it looks: the file has no session id in it, so a
    stale one left behind by a finished turn would be read by the *next*
    turn's hook and reported as that turn's clock -- already expired,
    every time. Removing it means a missing file is the normal resting
    state and the hook simply says nothing.
    """
    path = path or DEADLINE_FILE
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        log(f"deadline clear failed: {type(exc).__name__}: {exc}")


def read(path=None):
    """The live record, or None if there isn't a usable one."""
    path = path or DEADLINE_FILE
    try:
        with open(path) as handle:
            data = json.load(handle)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("deadline_at"), (int, float)):
        return None
    return data


def seconds_left(record=None, now=None):
    """Seconds until the turn is killed, or None if unknown.

    Can go negative: the kill is enforced by proc.wait() in cli.py, which
    only fires once the CLI's own work returns, so a session really can
    still be running past its deadline. Reporting that honestly is more
    use than clamping it to zero.
    """
    record = record if record is not None else read()
    if not record:
        return None
    now = now if now is not None else time.time()
    return record["deadline_at"] - now
