"""Claude Code hook: tells the running session how much quota it has left.

This is the delivery half of quota.py. The watcher writes a snapshot from
a background thread in the bridge process; this script runs inside the
CLI's own process tree, after each tool call, and turns that snapshot into
`additionalContext` -- the one channel that reaches a session already in
flight. Nothing else does: the prompt is fixed at turn start, and by the
time a turn returns, the cap has already been hit.

Two events, deliberately different:

- UserPromptSubmit fires once, at the top of a cycle. It always reports,
  even at 100% remaining, because "you are starting with 43% of the
  seven-day window left" changes what a cycle should attempt, and a cycle
  only gets one chance to hear it.
- PostToolUse fires after every tool call -- hundreds per cycle. It
  reports only when remaining has crossed into a lower band, because the
  alternative is measurable harm: ~60 tokens of identical text on every
  one of ~300 tool calls is ~18k tokens of context spent restating a
  warning, in a session whose whole problem is that it is running out of
  budget. Bands escalate (10% -> 5% -> spent) so the message still gets
  louder as it gets worse.

Fails silent and open. A hook that errors or writes garbage can disrupt
the session it is attached to, and being told the quota is a strictly
smaller thing than finishing the work -- so every failure path here exits
0 with no output, leaving the session exactly as it would have been.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from bridge import quota
except Exception:
    # "Fails silent and open" has to include this line too. An unimportable
    # bridge package would otherwise raise on every single tool call, and
    # the CLI surfaces a failing hook's stderr -- turning a missing quota
    # number into hundreds of tracebacks in the middle of a real cycle.
    sys.exit(0)

# The owner's number, 2026-08-04: "alerted if you have 10% or less on your
# token quota". Overridable, but the default is the one he asked for.
WARN_PCT = float(os.environ.get("QUOTA_WARN_PCT", "10"))

STATE_FILE = os.path.join(os.path.dirname(quota.SNAPSHOT_FILE), "quota-announced.json")

# A snapshot this old is reported with its age attached rather than
# suppressed. Silence is the failure this feature exists to remove, and a
# number that is five minutes stale still tells a cycle whether it is at
# 80% or at 4%.
STALE_AFTER_SECONDS = 300

WINDOW_LABELS = {"five_hour": "5-hour", "seven_day": "7-day"}


# Break-even, exactly. `pace` is used-share divided by elapsed-share, so
# `pace <= 1.0` is algebraically the same statement as "what is left of the
# budget covers what is left of the window at the rate spent so far": with
# r = 100*pace the spend per unit of elapsed share, the rest of the window
# costs r*(1-elapsed) and the budget left is 100 - r*elapsed, and those two
# meet exactly at pace == 1.
#
# Without this the band is a percentage and nothing else, and 10% remaining
# reads identically at hour 1 of a window and at hour 160 of 168. On
# 2026-09-16 the seven-day window sat at 10% remaining with pace 0.962 and
# the reset 10.8 hours away -- healthy by every instrument this loop owns,
# and `tools.quota_runway` said so in as many words -- while every cycle for
# the rest of that day was being told to start nothing and ship only a
# journal entry. The threshold was right about the number and wrong about
# the clock.
#
# Only band 1 softens. `pace` is window-to-date, so it dilutes a burst with
# whatever was quiet before it; at 5% left that dilution is the difference
# between finishing and not, and there is no margin there to spend on being
# clever.
def on_budget(tightest):
    """True when the window is spending at or under its own clock."""
    pace = tightest.get("pace")
    if not isinstance(pace, (int, float)):
        return False  # no pace, no argument -- warn the way it always did
    return pace <= 1.0


def severity(band, soft):
    """What has been announced, as one monotone number.

    The band alone cannot carry this. The dedupe only speaks again when the
    number goes up, and soft-to-hard happens *inside* band 1 -- so a window
    that starts on budget at 9% and then burns hard would never be allowed
    to say so.
    """
    if band == 0:
        return 0
    if band == 1 and soft:
        return 1
    return band * 2


def band_for(remaining_pct):
    """Which warning band a remaining-percentage falls in. 0 = fine."""
    if remaining_pct <= 0.5:
        return 3
    if remaining_pct <= WARN_PCT / 2:
        return 2
    if remaining_pct <= WARN_PCT:
        return 1
    return 0


def oslo_time(iso_utc):
    """Rule 7: anything a human reads is Oslo time, not UTC."""
    if not iso_utc:
        return ""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        parsed = datetime.fromisoformat(iso_utc)
        return parsed.astimezone(ZoneInfo("Europe/Oslo")).strftime("%H:%M")
    except Exception:
        return ""


def read_state():
    try:
        with open(STATE_FILE) as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_state(session_id, level):
    try:
        with open(STATE_FILE, "w") as handle:
            json.dump({"session_id": session_id, "level": level}, handle)
    except Exception:
        pass


def describe(snapshot):
    """The one-line budget report, or "" if the snapshot is unusable."""
    tightest = snapshot.get("tightest") or {}
    remaining = tightest.get("remaining_pct")
    if not isinstance(remaining, (int, float)):
        return ""
    label = WINDOW_LABELS.get(tightest.get("window"), tightest.get("window") or "quota")
    resets = oslo_time(tightest.get("resets_at"))
    notes = []
    if resets:
        notes.append(f"resets {resets} Oslo")
    try:
        import time
        age = time.time() - float(snapshot.get("fetched_at") or 0)
    except Exception:
        age = 0
    if age > STALE_AFTER_SECONDS:
        notes.append(f"reading is {int(age // 60)} min old")

    report = f"{remaining:.0f}% of your {label} Claude quota remains"
    if notes:
        report += " (" + ", ".join(notes) + ")"
    return report + "."


# journal.md was frozen on 2026-08-09 when the journal became one document
# per entry, and an entry appended to it is invisible to the site and to
# every cycle after -- so this line was telling a cycle to destroy its own
# handoff, at exactly the moment it was least likely to double-check.
# Found by the reviewer on Cycle 82 while checking the wording of the new
# deadline hook against this one.
WRAP_UP = (
    " Wrap up now rather than starting anything new: finish the step you are on, then "
    "write your journal entry to nova/journal/, rewrite journal-digest.md (put where to "
    "resume under **Next cycle**), and post your reply to the owner. Those three are what "
    "survive; work in progress is not."
)


def message_for(band, report, soft=False, pace=None):
    if band >= 3:
        return f"QUOTA SPENT — {report} Further calls will be refused." + WRAP_UP
    if band == 2:
        return f"QUOTA CRITICAL — {report}" + WRAP_UP
    if band == 1 and soft:
        return (
            f"QUOTA LOW, ON BUDGET — {report} The window is spending at or under its "
            f"own clock (pace {pace:.2f}), so what is left of the budget covers what "
            "is left of the window at the rate spent so far. Work at normal size; you "
            "will hear from me again if the pace goes above 1.0."
        )
    if band == 1:
        return f"QUOTA LOW — {report}" + WRAP_UP
    return f"Quota check: {report}"


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    event = payload.get("hook_event_name") or "PostToolUse"
    session_id = payload.get("session_id") or ""

    snapshot = quota.read_snapshot()
    if snapshot is None:
        return
    report = describe(snapshot)
    if not report:
        return

    tightest = snapshot["tightest"]
    band = band_for(tightest["remaining_pct"])
    soft = band == 1 and on_budget(tightest)
    level = severity(band, soft)
    state = read_state()
    # A new session starts with a clean slate: levels announced to the
    # cycle before this one were heard by a process that no longer exists.
    announced = state.get("level", 0) if state.get("session_id") == session_id else 0

    if event == "UserPromptSubmit":
        write_state(session_id, level)
    elif level > announced:
        write_state(session_id, level)
    else:
        return

    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": event,
        "additionalContext": message_for(band, report, soft, tightest.get("pace")),
    }}))


if __name__ == "__main__":
    main()
