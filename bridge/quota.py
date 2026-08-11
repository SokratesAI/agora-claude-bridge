"""Tracks how much of the Claude subscription quota is left, and makes that
number reachable from inside the running CLI session.

Edvard, 2026-08-04, after a cycle burned 78% of the quota in one run and
hit the cap mid-sentence: "Maybe an idea for you is to have you get
alerted if you have 10% or less on your token quota so that you can wrap
up and prepare to continue the next cycle instead of just hitting a
wall."

That is the whole feature. A cycle that knows it is nearly out can still
write its journal, its digest and its reply -- the three things that carry
work across to the next cycle. A cycle that finds out by being cut off
loses all three, which is what happened.

Two independent signals, because they fail differently:

- `rate_limit_event` on the CLI's own stream-json output (cli.py feeds it
  to note_rate_limit_event). Free, no extra request, and authoritative --
  it is what the API told the CLI. But it only carries a status
  (allowed / allowed_warning / rejected), never a percentage, so it can
  say "close" but not "10%".
- GET /api/oauth/usage with the same OAuth token the CLI uses. This is
  where the real per-window percentages live -- it is what `/usage` in an
  interactive session renders. Undocumented and internal, so it is
  treated as an enrichment that may vanish: every failure path here
  degrades to "no snapshot", never to a broken turn.

The percentages are *utilization* -- how much is USED. Everything this
module hands out is converted to remaining, because that is the number
Edvard asked to be warned about and the direction a threshold reads
naturally in.

Delivery is a file, not a return value, because the consumer is a hook
script running in a separate process (see hooks/quota_notice.py). The
watcher writes a snapshot; the hook reads it after each tool call and
tells the session when it is time to wrap up.
"""
import datetime
import json
import os
import sys
import threading
import time
import urllib.request

from bridge.config import CLAUDE_HOME
from bridge.log import log

USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
FETCH_TIMEOUT_SECONDS = 10

# Where the watcher writes and the hook reads. Under CLAUDE_HOME (the PVC)
# so a snapshot outlives a pod restart -- a cycle that starts right after
# one still learns it is resuming into a nearly-spent window.
SNAPSHOT_FILE = os.path.join(CLAUDE_HOME, "quota-snapshot.json")

# The snapshot is overwritten on every poll, so it answers "how much is
# left right now" and nothing else -- no burn rate, no per-cycle cost, no
# way to tell a spending week from a quiet one. Edvard asked (2026-08-08)
# for weekly spend logged so the cycle rate can be tuned on data rather
# than argument, and that needs a series, not a sample. This is the
# append-only companion: one JSON object per changed reading, alongside
# the snapshot on the same PVC so it outlives the pod.
HISTORY_FILE = os.path.join(CLAUDE_HOME, "quota-history.jsonl")

# The windows a Claude subscription actually enforces. Anything else the
# endpoint returns (the several null experiment keys) is ignored rather
# than guessed at -- an unknown window with a null utilization must not be
# read as "0% used, plenty left".
TRACKED_WINDOWS = ("five_hour", "seven_day")

# The same response also carries per-model caps, in a `limits[]` array this
# module never opened: a weekly window scoped to one model, enforced ON TOP
# of the shared weekly one rather than instead of it. Measured live
# 2026-08-09 by running four Claude Fable 5 calls and reading the endpoint
# either side -- `weekly_all` moved 11 -> 13 and the Fable-scoped window
# moved 0 -> 5, so the scoped cap fills ~2.7x faster than the shared one
# and is the one that would stop that model first. Every snapshot, warning
# and history row ever written here has been blind to it.
SCOPED_LIMIT_KIND = "weekly_scoped"

# Once a minute while a turn is running. A cycle runs up to 45 minutes
# (CLI_TIMEOUT_SECONDS), so this is ~45 requests per cycle against an
# endpoint whose response is a few hundred bytes -- far below anything
# worth rate-limiting, and fine-grained enough that the warning lands
# while there is still quota left to act on it.
POLL_SECONDS = 60

# The endpoint does throttle: six calls in a few seconds while building
# this earned a real 429. One a minute is well clear of that, but a
# failing poll must not keep knocking at the same rate -- on failure the
# wait doubles up to this cap, then resets on the next success. Quota
# moves slowly enough that a five-minute reading is still actionable, and
# the hook says how old its number is rather than pretending it is fresh.
POLL_BACKOFF_MAX_SECONDS = 300

# Backoff starts here rather than at POLL_SECONDS, because the failure
# this watcher actually meets is transient and self-healing. The first
# poll of a session races the CLI's own OAuth refresh and loses: measured
# 2026-08-08, the poll fired at 22:24:06.20 and the CLI rewrote
# .credentials.json at 22:24:09.34, so the watcher read a token that was
# three seconds from being replaced and got a 401. Doubling from 60s meant
# the first usable reading landed 120s into the session, and every cycle
# spent its opening minutes sizing its work against a snapshot taken
# before the *previous* cycle ran. Retrying in five seconds costs one
# request and clears the race; an endpoint that has genuinely moved still
# reaches the five-minute cap, on the seventh failure.
POLL_BACKOFF_START_SECONDS = 5

# How long close() waits for the final reading, matching activity.py's
# CLOSE_WAIT_SECONDS. Bounded rather than unbounded because the caller is
# holding a finished reply and a hung endpoint must not delay it -- but
# not zero, which is what this used to be. An unjoined daemon thread runs
# its last act *after* close() returns, i.e. outside whatever scope the
# caller thought it was in. In the test suite that scope is conftest's
# patch of fetch_usage, so the final reading escaped the guard and hit
# the live endpoint with this box's real credentials: measured 2026-08-09,
# one `pytest tests/` run appended three rows carrying the real live
# utilization to the production quota-history.jsonl, 19ms apart. The
# conftest docstring already records that this earned a genuine 429.
#
# Deliberately below FETCH_TIMEOUT_SECONDS (10), so a hung endpoint does
# outlast this join and write its row late. That is the right way round:
# a late history row harms nothing, and Edvard is the one who would feel
# the extra five seconds on his reply. The suite's safety does not rest on
# this number -- conftest's guard is session-scoped precisely so that it
# holds for the thread that wins this race rather than losing it.
CLOSE_WAIT_SECONDS = 5


def _read_access_token():
    """The CLI rotates this file on its own refresh schedule, so it is read
    fresh every fetch rather than cached at startup -- a cached token goes
    401 partway through a long cycle, which is exactly when the warning
    matters most."""
    path = os.path.join(CLAUDE_HOME, ".claude", ".credentials.json")
    try:
        with open(path) as handle:
            return json.load(handle).get("claudeAiOauth", {}).get("accessToken") or ""
    except Exception:
        return ""


def fetch_usage(token=None):
    """The raw endpoint payload, or None if it could not be read.

    Never raises. Every caller is either a background thread or a hook
    running inside someone else's turn, and neither has any business
    failing a real piece of work over a usage number.
    """
    token = token or _read_access_token()
    if not token:
        return None
    req = urllib.request.Request(
        USAGE_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-beta": "oauth-2025-04-20",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_SECONDS) as resp:
            return json.load(resp)
    except Exception as exc:
        log(f"quota fetch failed: {type(exc).__name__}: {exc}")
        return None


# How long each tracked window is, so "how far through it are we" can be
# computed from `resets_at` alone. The endpoint reports when a window ends
# and never when it began.
WINDOW_SECONDS = {
    "five_hour": 5 * 3600,
    "seven_day": 7 * 86400,
}


def _pace(name, resets_at, used_pct):
    """Used share divided by elapsed share -- 1.0 is exactly on the line.

    Edvard, 2026-08-09, cutting the heartbeat from 72 to 60 minutes: "i
    think an aggressive approach is better to start with". That is the
    right instinct and it needs an instrument, because the cadence change
    moves this loop across the line rather than towards it. Measured the
    same evening: `seven_day` went 2% -> 13% in 17.88 hours at the old
    cadence -- 14.76%/day against a window that only affords 14.3%/day, so
    24 cycles a day is roughly 118-124% of a week's quota and the window
    empties about a day early.

    The number a cycle actually needs is not "how much is left" but "is
    what is left ahead of or behind schedule", and every cycle so far has
    re-derived that by hand from `remaining_pct / days_left`. That formula
    was written when the heartbeat was 6-hourly and each cycle had a
    five-hour window to itself; five cycles now share one. Computing it
    here means every cycle reads the same number the same way, and it
    lands in `quota-history.jsonl` for free, which is the trial-and-error
    record he asked for.

    Deliberately a measurement and not a threshold. Nothing in this module
    acts on it -- no warning fires off pace, no cycle is told to stop --
    because what to do about a hot week is a judgement this file cannot
    make. It reports; `prompt.md` decides.

    Read it as "will *this* window hold out", not as "is the cadence
    sustainable". It averages over the whole window to date, so a burst is
    diluted by anything quiet before it: the live reading while this was
    written was 0.23, which looks like a very cold week and was actually a
    near-idle stretch 08-05 to 08-08 followed by 14 cycles in one day at
    14.76%/day. The cadence question needs the slope between two recent
    readings, which is what `quota-history.jsonl` is for.

    None when it cannot be computed honestly: an unknown window length, an
    unparseable or absent `resets_at`, or a window so freshly reset that
    the elapsed share rounds to nothing and the ratio explodes.
    """
    length = WINDOW_SECONDS.get(name)
    if not length or not resets_at:
        return None
    try:
        ends = datetime.datetime.fromisoformat(str(resets_at))
    except ValueError:
        return None
    if ends.tzinfo is None:
        ends = ends.replace(tzinfo=datetime.timezone.utc)
    elapsed = length - (ends.timestamp() - time.time())
    # Guard both ends: a clock skew that puts `resets_at` further out than
    # the window is long gives a negative elapsed, and the first seconds of
    # a window give a near-zero denominator.
    if elapsed < length * 0.01:
        return None
    return round(float(used_pct) / (elapsed / length * 100.0), 3)


def summarize(usage):
    """{"windows": [...], "tightest": {...}} -- or None if nothing usable.

    "tightest" is whichever tracked window has the least left, since that
    is the one that will actually stop the session. Both are kept because
    the two fail in different ways and reading only one is misleading: a
    fresh five-hour window says nothing about a seven-day window at 95%.

    `windows` also carries the per-model caps from `limits[]` (see
    `_scoped_windows`), which are reported but never eligible to be
    "tightest".
    """
    if not isinstance(usage, dict):
        return None
    windows = []
    for name in TRACKED_WINDOWS:
        block = usage.get(name)
        if not isinstance(block, dict):
            continue
        used = block.get("utilization")
        if not isinstance(used, (int, float)):
            continue
        windows.append({
            "window": name,
            "used_pct": float(used),
            # Clamped: the endpoint can report over 100 once a window is
            # spent, and a negative "remaining" reads as a bug to whoever
            # sees it in a warning.
            "remaining_pct": max(0.0, min(100.0, 100.0 - float(used))),
            "resets_at": block.get("resets_at") or "",
            "pace": _pace(name, block.get("resets_at"), used),
        })
    if not windows:
        return None
    return {
        "windows": windows + _scoped_windows(usage),
        "tightest": min(windows, key=lambda w: w["remaining_pct"]),
    }


def _scoped_windows(usage):
    """Per-model weekly caps from `limits[]`, named `weekly_scoped:<model>`.

    Recorded like any other window -- so the snapshot, the history file and
    anything charting them get it for free -- but deliberately NOT eligible
    to be "tightest". A scoped cap only stops the model it names, and this
    module has no idea which model the session it is watching runs on.
    Letting a spent Fable window become "tightest" would tell an Opus cycle
    to wrap up while nothing was actually blocking it, and a warning that
    cries wolf is worse than no warning at all.
    """
    windows = []
    for limit in usage.get("limits") or []:
        if not isinstance(limit, dict) or limit.get("kind") != SCOPED_LIMIT_KIND:
            continue
        model = ((limit.get("scope") or {}).get("model") or {}).get("display_name")
        used = limit.get("percent")
        if not model or not isinstance(used, (int, float)):
            continue
        windows.append({
            "window": f"{SCOPED_LIMIT_KIND}:{model}",
            "used_pct": float(used),
            "remaining_pct": max(0.0, min(100.0, 100.0 - float(used))),
            "resets_at": limit.get("resets_at") or "",
            "scoped_to": model,
        })
    return windows


def write_snapshot(summary, path=None):
    """Atomic, because a hook may read this file at any instant and a
    half-written one would parse as absent -- silently turning the warning
    off at random."""
    path = path or SNAPSHOT_FILE
    payload = dict(summary or {})
    payload["fetched_at"] = time.time()
    tmp = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)
        return True
    except Exception as exc:
        log(f"quota snapshot write failed: {type(exc).__name__}: {exc}")
        return False


def read_snapshot(path=None):
    """The last snapshot, or None. Used by the hook; never raises."""
    try:
        with open(path or SNAPSHOT_FILE) as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) and data.get("tightest") else None
    except Exception:
        return None


def history_path_for(snapshot_path=None):
    """The history file that belongs beside a given snapshot file. Derived
    rather than fixed so a test pointing the watcher at a tmp dir gets its
    history there too, instead of appending to the real one."""
    if not snapshot_path or snapshot_path == SNAPSHOT_FILE:
        return HISTORY_FILE
    return os.path.join(os.path.dirname(snapshot_path), "quota-history.jsonl")


def _history_row(summary, boundary=None):
    """One flat row: {"at": epoch, "five_hour": pct_used, ...}. Flat rather
    than the nested snapshot shape because the consumer is a chart, and a
    row per reading with a column per window is what plots.

    A boundary row also carries `"boundary": "start"|"end"`, so a chart can
    tell where a cycle began and ended from the file alone rather than
    inferring it from the size of the gap between rows.
    """
    row = {"at": round(time.time(), 3)}
    if boundary:
        row["boundary"] = boundary
    for window in (summary or {}).get("windows", []):
        name = window.get("window")
        if name:
            row[name] = window.get("used_pct")
            row[f"{name}_resets_at"] = window.get("resets_at") or ""
            if window.get("pace") is not None:
                row[f"{name}_pace"] = window["pace"]
    return row


def _reading_only(row):
    """The part of a history row that is a measurement: the utilizations,
    without the timestamp, the boundary marker, the server's per-request
    `resets_at`, or the derived `_pace`.

    `_pace` has to be stripped for the same reason `resets_at` does, and it
    would break this guard harder. Pace divides used share by *elapsed*
    share, so it moves on every single poll even when utilization has not
    budged -- including it would make no two rows ever compare equal and
    turn the dedup off entirely, which is the exact failure this function
    was written to fix. Nothing is lost: the row still carries pace, it
    just does not get a vote on whether the row is new.
    """
    return {k: v for k, v in (row or {}).items()
            if k not in ("at", "boundary")
            and not k.endswith("_resets_at") and not k.endswith("_pace")}


def append_history(summary, path=None, boundary=None):
    """Append a reading, unless it is identical to the previous one.

    Skipping unchanged readings is not a cap on the data -- an unchanged
    row carries no information a chart can use, and the poll runs once a
    minute for as long as a turn lasts. Nothing is truncated or rotated:
    the file grows for as long as the loop runs, which at ~15 rows per
    cycle is a rounding error against the PVC.

    "Identical" means the utilizations, not the whole row. The endpoint
    computes `resets_at` per request, so it comes back with fresh
    sub-second jitter every time -- 00:29:59.498355 then 00:29:59.027533,
    four seconds apart, same window. Comparing the whole row made this
    guard inert: 5 of the first 9 rows written on the live pod carried a
    reading already on the line above. The unit test did not catch it
    because its fixture held `resets_at` fixed, which real responses
    never do. A window actually rolling over still writes a row, because
    utilization drops when it does.

    A `boundary` reading is exempt, and has to be: the information in a
    cycle's first and last row is the *timestamp*, which is precisely
    what the comparison strips. The percentage moves about once every one
    to three minutes, so an end reading taken seconds after the last tick
    usually matches it and was being dropped -- Cycle 47's history ends at
    02:00:17, exactly three poll intervals after the row above it, while
    the cycle was still writing to the vault at 01:59. The final reading
    Cycle 46 added ran every time and landed nowhere.
    """
    path = path or HISTORY_FILE
    row = _history_row(summary, boundary)
    comparable = _reading_only(row)
    try:
        if not boundary and comparable == _last_history_reading(path):
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        return True
    except Exception as exc:
        log(f"quota history append failed: {type(exc).__name__}: {exc}")
        return False


def _last_history_reading(path):
    """The previous row minus its timestamp, or None. Reads the tail rather
    than the file: this is called once a minute and the file only grows."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            handle.seek(max(0, size - 4096))
            tail = handle.read().decode("utf-8", "replace").strip().splitlines()
        if not tail:
            return None
        return _reading_only(json.loads(tail[-1]))
    except Exception:
        return None


def refresh(path=None, boundary=None):
    """One fetch -> one snapshot (+ a history row if the reading moved, or
    always if this is a `boundary` reading). True if a usable snapshot was
    written."""
    summary = summarize(fetch_usage())
    if summary is None:
        return False
    written = write_snapshot(summary, path)
    # After the snapshot, and never gating on it: the history is the
    # nice-to-have and the snapshot is what the low-quota warning reads.
    append_history(summary, history_path_for(path), boundary)
    return written


HOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hooks")
HOOK_SCRIPT = os.path.join(HOOKS_DIR, "quota_notice.py")
# The bridge's other in-flight notice: how much wall-clock is left before
# the turn is killed (deadline.py). It rides the same settings file and
# the same two events because it solves the same problem on a different
# axis -- a cycle could see its quota but not its clock.
DEADLINE_HOOK_SCRIPT = os.path.join(HOOKS_DIR, "deadline_notice.py")
HOOK_SETTINGS_FILE = os.path.join(CLAUDE_HOME, ".claude", "bridge-hooks.settings.json")


def write_hook_settings(path=None):
    """Generate the --settings file that attaches the bridge's hooks, and
    return its path (or "" if it could not be written, which the caller
    treats as "run without the hooks").

    Passed per invocation with --settings rather than merged into
    ~/.claude/settings.json on purpose: that file is persistent user
    config on the PVC, and a service that edits it leaves state behind
    that outlives the code that wanted it. A flag is scoped to the call
    and disappears when this code does.
    """
    path = path or HOOK_SETTINGS_FILE
    # sys.executable, not "python3": the hooks import from bridge, so they
    # have to run under the same interpreter this service is running under,
    # not whatever "python3" happens to resolve to in the CLI's env.
    entry = [{"matcher": "*", "hooks": [
        {"type": "command", "command": f"{sys.executable} {script}", "timeout": 10}
        for script in (HOOK_SCRIPT, DEADLINE_HOOK_SCRIPT)
    ]}]
    settings = {"hooks": {"UserPromptSubmit": entry, "PostToolUse": entry}}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(settings, handle)
        return path
    except Exception as exc:
        log(f"quota hook settings write failed: {type(exc).__name__}: {exc}")
        return ""


class QuotaWatcher:
    """One background poller per CLI invocation, same shape and same
    best-effort contract as ActivityReporter -- start() before the stream
    is read, close() in the finally.

    Polling rather than reacting purely to `rate_limit_event` because that
    event's cadence is the CLI's business and not something this service
    should depend on: observed firing once on a one-turn session, which
    would leave a 45-minute cycle running on a single reading taken at
    minute zero. The event is still used, as an immediate trigger when it
    reports anything other than "allowed" (note_rate_limit_event).
    """

    def __init__(self, path=None, poll_seconds=POLL_SECONDS, publish_costs=False):
        self._path = path or SNAPSHOT_FILE
        self._poll_seconds = poll_seconds
        # Off unless the caller says this invocation is a cycle. Every CLI
        # invocation gets a watcher -- reply turns, probes, the whole test
        # suite -- and publishing is a full transcript scan plus a ~90KB
        # vault write, which none of those should be doing. Defaulting it
        # on is what broke 8 cli tests when this was first attempted: they
        # start a real watcher against a faked subprocess, so the scan ran
        # on the live PVC and the publish outlived the test that started it.
        self._publish_costs = publish_costs
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def note_rate_limit_event(self, info):
        """Called with a `rate_limit_event`'s rate_limit_info. Anything but
        "allowed" means the API itself has started warning, so take a fresh
        reading now instead of waiting out the poll interval."""
        status = (info or {}).get("status") if isinstance(info, dict) else None
        if status and status != "allowed":
            log(f"quota: rate_limit_event status={status!r}, refreshing snapshot")
            self._wake.set()

    def close(self):
        if self._thread is None:
            return
        self._stop.set()
        self._wake.set()
        # Joined, briefly. This comment used to claim "same trade as
        # ActivityReporter's short close wait" while doing no waiting at
        # all -- and the wait is the whole trade. Without it the thread's
        # final reading lands at an unknowable time after close() returns,
        # which makes it both untestable and, in the suite, unguarded.
        self._thread.join(timeout=CLOSE_WAIT_SECONDS)
        self._thread = None

    def _run(self):
        failures = 0
        wait = self._poll_seconds
        opened = False
        while not self._stop.is_set():
            try:
                ok = refresh(self._path, None if opened else "start")
            except Exception:
                ok = False
            if ok:
                opened = True
                wait = self._poll_seconds
                failures = 0
            else:
                wait = min(POLL_BACKOFF_START_SECONDS * 2 ** failures,
                           POLL_BACKOFF_MAX_SECONDS)
                failures += 1
                # First of each outage: an endpoint that has moved or a
                # token that cannot refresh fails every poll for the rest
                # of the session, and 45 identical lines would bury the
                # CLI's own logs (activity.py learned this the same way).
                # Counted per outage rather than per session because a
                # recovery resets `failures` -- so a flapping endpoint says
                # so once per outage, and a dead one still says it once.
                if failures == 1:
                    log("quota poll failed (further failures silenced until it recovers)")
            self._wake.wait(wait)
            self._wake.clear()
        # One last reading on the way out, because the interesting one is
        # the reading nobody was awake for. The poll only runs while a turn
        # runs, so without this the final row in the history is wherever
        # the last 60s tick happened to land -- never where the cycle
        # actually ended. The whole point of the history is to answer "what
        # did that cycle cost", and every row was systematically short of
        # the answer by up to a minute of the most expensive part of a run.
        # Best-effort like everything else here: this is a daemon thread
        # and nobody is waiting on it, so a slow endpoint delays no reply.
        # Marked as a boundary, because taking the reading was never the
        # hard part -- it ran, and then the dedup dropped it for matching
        # the tick before it, which is the normal case.
        try:
            refresh(self._path, "end")
        except Exception:
            pass
        # Then, for a cycle only, republish the cost record into the vault.
        # After the final reading rather than before it, so the document
        # carries the boundary row this cycle just wrote instead of missing
        # it by one poll -- which is the same off-by-one-tick the "end"
        # reading above exists to fix.
        #
        # Imported here rather than at module scope because publish_costs
        # imports this module: at the top it is a cycle, and the module that
        # loses the race gets a half-initialised partner. Nothing is waiting
        # on this thread, so the import cost is paid off the reply's path.
        #
        # Two honest properties of publishing from here, both fine and
        # neither obvious. The scan takes longer than close()'s 5s join, so
        # the join times out and this finishes after the turn returns --
        # which is why it must stay a daemon thread and must never raise.
        # And the publishing cycle's own row is read from a transcript still
        # being written, so it under-reports its own last few turns; the
        # next cycle's publish rewrites it complete. Every row but the last
        # is settled, which is the shape a chart of past cycles wants.
        if self._publish_costs:
            try:
                from bridge import publish_costs
                publish_costs.refresh()
            except Exception as exc:
                log(f"quota: cost publish failed: {type(exc).__name__}: {exc}")
