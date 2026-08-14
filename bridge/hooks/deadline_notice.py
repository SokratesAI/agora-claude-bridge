"""Claude Code hook: tells the running session how much time it has left.

The delivery half of deadline.py, and a direct copy of quota_notice.py's
shape because the problem is identical: the prompt is fixed at turn start,
and by the time a turn returns it has already been killed, so
`additionalContext` after a tool call is the only channel that reaches a
session in flight.

Two events, same split as the quota hook:

- UserPromptSubmit fires once, at the top of a cycle, and always reports.
  "You have 45 minutes" changes what a cycle should attempt, and it only
  gets one chance to hear it before it starts choosing.
- PostToolUse fires after every tool call -- hundreds per cycle -- and
  reports only on crossing into a lower band, for the reason spelled out
  in quota_notice.py: repeating one line on 300 tool calls spends ~18k
  tokens of context restating a warning.

The warning bands are measured, not picked for roundness. Across the last
12 Nova cycles that posted a reply (2026-08-10, from Agora's own message
timestamps), the gap between writing the journal entry and posting the
reply was 2.6 min median, 4.7 min p90, 10.3 min max -- and that is only
the tail of the wrap-up, after the entry is already composed.

- 15 min sits above the worst wrap-up observed, so WARN_LOW is "you can
  still finish what you are doing".
- 8 min is above p90 with margin: enough for journal + digest + reply if
  you start now, not enough for another PR.
- 3 min is below the median tail. At that point the honest advice is that
  only the reply is still saveable, so say that instead of a task list.

Above those sit two bands that are not warnings at all, and they exist
because the warnings were never the failing part. Ten consecutive Nova
cycles (175-184) recorded believing they were far later in the turn than
they were -- "certain it was minute 40 of 45 and it was minute 6",
"certain it was 23:33 when it was 23:09" -- and every one of those
misreadings happened *above* the 15-minute line, in the stretch where
this hook says nothing at all. A cycle heard "45 of 45 remain" once, at
turn start, and then had no external clock for the next thirty minutes
while it decided what it could afford to attempt. The cost is not an
overrun; it is work descoped against a deadline that had not arrived.

So POSITION_HALF and POSITION_THIRD announce where the turn actually is,
once each, and say plainly that nothing needs to change. Two extra lines
per cycle, ~60 tokens, against a documented habit of shipping less.

Every line also carries the wall clock in Oslo time, because that is the
unit the misreadings are made in -- a cycle that has drifted reasons in
"it must be about half past", and relative minutes give it nothing to
notice the drift against. Oslo specifically: Edvard reads these entries
and lives there, and rule 7 of identity.md says so.

Fails silent and open, exactly as quota_notice.py does: a hook that
errors can disrupt the session it is attached to, and being told the time
is a strictly smaller thing than finishing the work.
"""
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from bridge import deadline
except Exception:
    # Same reasoning as quota_notice.py: an unimportable bridge package
    # would otherwise raise on every tool call, and the CLI surfaces a
    # failing hook's stderr -- turning a missing clock into hundreds of
    # tracebacks in the middle of a real cycle.
    sys.exit(0)

# Band numbers ascend with severity, because read_state/main only ever
# announce a band strictly higher than the last one -- so the position
# reports have to sit *below* the warnings numerically to avoid muting
# them.
POSITION_THIRD = 1
POSITION_HALF = 2
WARN_LOW = 3
WARN_CRITICAL = 4
WARN_NEARLY_UP = 5

# Minutes remaining at which each band opens. Descending, so the first
# match is the most severe band the session has reached.
BANDS = [
    (3, WARN_NEARLY_UP),
    (8, WARN_CRITICAL),
    (15, WARN_LOW),
    (22, POSITION_HALF),
    (30, POSITION_THIRD),
]

STATE_FILE = os.path.join(os.path.dirname(deadline.DEADLINE_FILE), "deadline-announced.json")


def band_for(minutes_left):
    """Which warning band a remaining-minutes figure falls in. 0 = fine."""
    for threshold, band in BANDS:
        if minutes_left <= threshold:
            return band
    return 0


def read_state():
    try:
        with open(STATE_FILE) as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_state(session_id, band):
    try:
        with open(STATE_FILE, "w") as handle:
            json.dump({"session_id": session_id, "band": band}, handle)
    except Exception:
        pass


WRAP_UP = (
    " Finish the step you are on, then write your journal entry to "
    "nova/journal/, rewrite journal-digest.md (put where to resume under "
    "**Next cycle**), and post your reply. Those three survive; work in "
    "progress does not."
)

POSITION_NOTE = (
    " This is a position report, not a warning — nothing needs to change. It is "
    "here so that if your own sense of the time disagrees with it, you find that "
    "out now rather than after descoping the work."
)


def oslo_clock(now=None):
    """`HH:MM Oslo`, or "" if this box cannot resolve the zone.

    Empty rather than raising, and empty rather than falling back to UTC:
    a wrong clock stated confidently is the exact failure this hook is
    trying to fix, so no clock is the honest degradation.
    """
    try:
        from zoneinfo import ZoneInfo

        stamp = datetime.datetime.fromtimestamp(
            now if now is not None else time.time(), ZoneInfo("Europe/Oslo")
        )
    except Exception:
        return ""
    return stamp.strftime("%H:%M Oslo")


def message_for(band, minutes_left, timeout_minutes, clock="", minutes_gone=None):
    lead = f"{clock} — " if clock else ""
    if band >= WARN_NEARLY_UP:
        return (
            f"TIME NEARLY UP — {lead}about {minutes_left:.0f} min before this turn is "
            "killed. Only your reply is still saveable. Post it now, and say "
            "plainly in it what you did and did not finish."
        )
    if band == WARN_CRITICAL:
        return (
            f"TIME CRITICAL — {lead}about {minutes_left:.0f} min left of this turn's "
            f"{timeout_minutes:.0f}-minute limit. Start nothing new." + WRAP_UP
        )
    if band == WARN_LOW:
        return (
            f"TIME LOW — {lead}about {minutes_left:.0f} min left of this turn's "
            f"{timeout_minutes:.0f}-minute limit. Land what you have rather than "
            "opening another thread of work." + WRAP_UP
        )
    if band in (POSITION_HALF, POSITION_THIRD):
        gone = f"{minutes_gone:.0f} min gone, " if minutes_gone is not None else ""
        return (
            f"Time check: {lead}{gone}about {minutes_left:.0f} of this turn's "
            f"{timeout_minutes:.0f} minutes remain." + POSITION_NOTE
        )
    return (
        f"Time check: {lead}{minutes_left:.0f} of this turn's {timeout_minutes:.0f} "
        "minutes remain. A turn that overruns is killed with no reply posted, so "
        "size the work to leave ~10 minutes for journal, digest and reply."
    )


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    event = payload.get("hook_event_name") or "PostToolUse"
    session_id = payload.get("session_id") or ""

    record = deadline.read()
    if record is None:
        return
    left = deadline.seconds_left(record)
    if left is None:
        return
    minutes_left = left / 60.0
    timeout_minutes = float(record.get("timeout_seconds") or 0) / 60.0
    # Older records predate started_at; report position without it rather
    # than not reporting.
    started_at = record.get("started_at")
    minutes_gone = (
        (time.time() - started_at) / 60.0
        if isinstance(started_at, (int, float))
        else None
    )

    band = band_for(minutes_left)
    state = read_state()
    # A new session starts clean: bands announced to the cycle before this
    # one were heard by a process that no longer exists.
    announced = state.get("band", 0) if state.get("session_id") == session_id else 0

    if event == "UserPromptSubmit":
        write_state(session_id, band)
    elif band > announced:
        write_state(session_id, band)
    else:
        return

    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": event,
        "additionalContext": message_for(
            band, minutes_left, timeout_minutes,
            clock=oslo_clock(), minutes_gone=minutes_gone,
        ),
    }}))


if __name__ == "__main__":
    main()
