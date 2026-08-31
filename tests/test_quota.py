"""Covers the quota warning path: reading usage, turning it into a
remaining-percentage snapshot, and the hook that reports it to a running
session.

The thing actually being protected here is that a cycle finds out it is
nearly out of quota *while it can still act*. So the tests that matter
most are the ones about when the hook stays quiet -- a warning that fires
on all 300 tool calls is as useless as one that never fires, just
expensive instead of silent.
"""
import datetime
import os
import io
import json
import time
from unittest.mock import patch

import pytest

from bridge import publish_costs
from bridge import quota
from bridge.hooks import quota_notice


# A real response, trimmed: the several always-null experiment windows are
# kept because ignoring them correctly is one of the things under test.
USAGE_PAYLOAD = {
    "five_hour": {"utilization": 90.0, "resets_at": "2026-08-05T04:50:00.711083+00:00"},
    "seven_day": {"utilization": 57.0, "resets_at": "2026-08-05T17:00:00.711105+00:00"},
    "seven_day_opus": None,
    "tangelo": None,
}

# The same endpoint's response with nothing removed, captured from the
# live pod 2026-08-09 02:52 Oslo. Kept beside the trimmed one because the
# trimmed one is a statement about what we think matters, and this is a
# statement about what the server actually sends -- the difference is
# where the last two quota defects lived. Carries no credential: the
# whole body is percentages, reset timestamps and null feature flags.
LIVE_USAGE_PAYLOAD = {
    "five_hour": {"utilization": 2.0, "resets_at": "2026-08-09T05:40:00.084069+00:00",
                  "limit_dollars": None, "used_dollars": None, "remaining_dollars": None},
    "seven_day": {"utilization": 4.0, "resets_at": "2026-08-12T16:59:59.084090+00:00",
                  "limit_dollars": None, "used_dollars": None, "remaining_dollars": None},
    "seven_day_oauth_apps": None,
    "seven_day_opus": None,
    "seven_day_sonnet": None,
    "seven_day_cowork": None,
    "seven_day_omelette": None,
    "tangelo": None,
    "iguana_necktie": None,
    "omelette_promotional": None,
    "nimbus_quill": {"utilization": 0.0, "resets_at": None, "limit_dollars": None,
                     "used_dollars": None, "remaining_dollars": None},
    "cinder_cove": None,
    "amber_ladder": None,
    "extra_usage": {
        "is_enabled": False, "monthly_limit": None, "used_credits": None,
        "utilization": None, "currency": None, "decimal_places": None,
        "disabled_reason": None, "user_disabled": False,
        "spend_limit_reached": False, "credits_ever_enabled": False,
        "daily": None, "weekly": None,
    },
    "limits": [
        {"kind": "session", "group": "session", "percent": 2, "severity": "normal",
         "resets_at": "2026-08-09T05:40:00.084069+00:00", "scope": None, "is_active": False},
        {"kind": "weekly_all", "group": "weekly", "percent": 4, "severity": "normal",
         "resets_at": "2026-08-12T16:59:59.084090+00:00", "scope": None, "is_active": True},
        {"kind": "weekly_scoped", "group": "weekly", "percent": 0, "severity": "normal",
         "resets_at": None, "scope": {"model": {"id": None, "display_name": "Fable"},
                                      "surface": None}, "is_active": False},
    ],
    "spend": {
        "used": {"amount_minor": 0, "currency": "USD", "exponent": 2},
        "limit": None, "percent": 0, "severity": "normal", "enabled": False,
        "disabled_reason": None, "cap": None, "balance": None, "auto_reload": None,
        "disclaimer": "Usage credits cover you when you hit your plan limits.",
        "can_purchase_credits": False, "can_toggle": False,
    },
    "member_dashboard_available": False,
}


# ---------------------------------------------------------------------------
# summarize -- utilization (used) -> remaining, and which window binds
# ---------------------------------------------------------------------------

def test_summarize_converts_utilization_to_remaining():
    summary = quota.summarize(USAGE_PAYLOAD)
    by_name = {w["window"]: w for w in summary["windows"]}
    assert by_name["five_hour"]["remaining_pct"] == 10.0
    assert by_name["seven_day"]["remaining_pct"] == 43.0


def test_summarize_tightest_is_the_window_with_least_left():
    summary = quota.summarize(USAGE_PAYLOAD)
    assert summary["tightest"]["window"] == "five_hour"


def test_summarize_tightest_can_be_the_seven_day_window():
    """The five-hour window resets constantly; the seven-day one is what
    actually stops a run of cycles. Reading only the first would report
    "plenty left" on a nearly-spent week."""
    payload = {
        "five_hour": {"utilization": 5.0, "resets_at": ""},
        "seven_day": {"utilization": 96.0, "resets_at": ""},
    }
    assert quota.summarize(payload)["tightest"]["window"] == "seven_day"


def test_summarize_ignores_null_experiment_windows():
    assert {w["window"] for w in quota.summarize(USAGE_PAYLOAD)["windows"]} == {
        "five_hour", "seven_day"}


def test_summarize_returns_none_when_nothing_usable():
    assert quota.summarize({"seven_day_opus": None}) is None
    assert quota.summarize({"five_hour": {"utilization": None}}) is None
    assert quota.summarize(None) is None


def test_summarize_clamps_overspent_window_to_zero():
    """The endpoint reports over 100 once a window is blown, and a
    "-4% remaining" in a warning reads as a bug rather than as danger."""
    summary = quota.summarize({"five_hour": {"utilization": 104.0}})
    assert summary["tightest"]["remaining_pct"] == 0.0


def test_summarize_handles_the_endpoint_response_verbatim():
    """`USAGE_PAYLOAD` above is trimmed by hand, and a hand-trimmed fixture
    is exactly how two defects reached production here: bridge#20's dedup
    passed a mutation check against a fixture holding `resets_at` fixed,
    which no real response does. LIVE_USAGE_PAYLOAD is the untouched body
    of GET /api/oauth/usage, captured 2026-08-09 02:52 Oslo -- including
    the blocks the trimmed fixture drops entirely (`limits`, `spend`,
    `extra_usage`) and a non-null window that is not tracked
    (`nimbus_quill`, utilization 0.0, `resets_at: null`), which would
    otherwise summarize as the emptiest window and become "tightest"."""
    summary = quota.summarize(LIVE_USAGE_PAYLOAD)

    assert [w["window"] for w in summary["windows"]] == [
        "five_hour", "seven_day", "weekly_scoped:Fable"]
    assert summary["tightest"]["window"] == "seven_day"
    assert summary["tightest"]["remaining_pct"] == 96.0


def test_summarize_records_the_per_model_scoped_cap():
    """The `limits[]` array carries a weekly window scoped to one model,
    enforced on top of the shared weekly one. Measured live 2026-08-09:
    four Fable 5 calls moved `weekly_all` 11 -> 13 and this window 0 -> 5,
    so it is the cap that stops that model first -- and nothing recorded
    it, because summarize only ever read the two top-level windows."""
    scoped = [w for w in quota.summarize(LIVE_USAGE_PAYLOAD)["windows"]
              if w["window"] == "weekly_scoped:Fable"]

    assert len(scoped) == 1
    assert scoped[0]["scoped_to"] == "Fable"
    assert scoped[0]["used_pct"] == 0.0
    assert scoped[0]["remaining_pct"] == 100.0


def test_summarize_never_lets_a_scoped_cap_become_tightest():
    """A scoped cap only stops the model it names. This module cannot know
    which model the session it watches is running, so a spent Fable window
    must not make an Opus cycle wrap up with 90% of its own quota left."""
    payload = {
        "five_hour": {"utilization": 10.0},
        "seven_day": {"utilization": 20.0},
        "limits": [{"kind": "weekly_scoped", "percent": 99, "resets_at": "z",
                    "scope": {"model": {"id": None, "display_name": "Fable"}}}],
    }
    summary = quota.summarize(payload)

    assert summary["tightest"]["window"] == "seven_day"
    assert summary["tightest"]["remaining_pct"] == 80.0
    # ...and it is still reported, rather than dropped to keep it out.
    assert any(w["window"] == "weekly_scoped:Fable" and w["remaining_pct"] == 1.0
               for w in summary["windows"])


def test_summarize_skips_scoped_limits_it_cannot_name_or_read():
    """An unnamed model or a null percent must not summarize as a window
    at 0% used -- the same "plenty left" misreading TRACKED_WINDOWS avoids
    for the null experiment keys."""
    payload = {
        "five_hour": {"utilization": 10.0},
        "limits": [
            {"kind": "weekly_scoped", "percent": None,
             "scope": {"model": {"display_name": "Fable"}}},
            {"kind": "weekly_scoped", "percent": 5, "scope": None},
            {"kind": "weekly_all", "percent": 4, "scope": None},
            {"kind": "session", "percent": 2, "scope": None},
        ],
    }
    assert [w["window"] for w in quota.summarize(payload)["windows"]] == ["five_hour"]


def test_history_row_carries_a_scoped_cap_as_its_own_column():
    """The point of recording it is the series, not the snapshot -- the
    Fable window moving 2.7x faster than the shared one is only visible
    across readings."""
    summary = {"windows": [
        {"window": "seven_day", "used_pct": 13.0, "resets_at": "a"},
        {"window": "weekly_scoped:Fable", "used_pct": 5.0, "resets_at": "b",
         "scoped_to": "Fable"},
    ]}
    row = quota._history_row(summary)

    assert row["seven_day"] == 13.0
    assert row["weekly_scoped:Fable"] == 5.0
    assert row["weekly_scoped:Fable_resets_at"] == "b"
    # The dedup guard strips reset timestamps, and must strip this one too
    # or every reading looks changed and the history fills with duplicates.
    assert "weekly_scoped:Fable_resets_at" not in quota._reading_only(row)
    assert quota._reading_only(row)["weekly_scoped:Fable"] == 5.0


# ---------------------------------------------------------------------------
# fetch_usage -- best-effort, never raises into the turn it is watching
# ---------------------------------------------------------------------------

def test_fetch_usage_returns_none_without_a_token():
    with patch.object(quota, "_read_access_token", return_value=""):
        assert quota.fetch_usage() is None


def test_fetch_usage_swallows_network_errors():
    with patch.object(quota, "_read_access_token", return_value="tok"), \
         patch("urllib.request.urlopen", side_effect=OSError("no route to host")):
        assert quota.fetch_usage() is None


def test_read_access_token_is_reread_each_call(tmp_path):
    """The CLI rotates this file mid-session; a cached token goes 401
    exactly when the warning matters most."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    creds = home / ".claude" / ".credentials.json"
    creds.write_text(json.dumps({"claudeAiOauth": {"accessToken": "first"}}))
    with patch.object(quota, "CLAUDE_HOME", str(home)):
        assert quota._read_access_token() == "first"
        creds.write_text(json.dumps({"claudeAiOauth": {"accessToken": "second"}}))
        assert quota._read_access_token() == "second"


def test_read_access_token_survives_a_missing_file(tmp_path):
    with patch.object(quota, "CLAUDE_HOME", str(tmp_path)):
        assert quota._read_access_token() == ""


# ---------------------------------------------------------------------------
# snapshot file -- the handoff between the bridge process and the hook
# ---------------------------------------------------------------------------

def test_snapshot_roundtrip(tmp_path):
    path = str(tmp_path / "snap.json")
    assert quota.write_snapshot(quota.summarize(USAGE_PAYLOAD), path)
    loaded = quota.read_snapshot(path)
    assert loaded["tightest"]["remaining_pct"] == 10.0
    assert loaded["fetched_at"] > 0


def test_read_snapshot_returns_none_when_absent_or_junk(tmp_path):
    missing = str(tmp_path / "nope.json")
    assert quota.read_snapshot(missing) is None
    junk = tmp_path / "junk.json"
    junk.write_text("{not json")
    assert quota.read_snapshot(str(junk)) is None
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    assert quota.read_snapshot(str(empty)) is None


def test_write_snapshot_leaves_no_partial_file_behind(tmp_path):
    """Written via a temp file and os.replace: the hook can read at any
    instant, and a half-written snapshot parses as absent -- silently
    turning the warning off at random."""
    path = str(tmp_path / "snap.json")
    quota.write_snapshot(quota.summarize(USAGE_PAYLOAD), path)
    assert [p.name for p in tmp_path.iterdir()] == ["snap.json"]


def test_refresh_writes_nothing_when_the_endpoint_is_unusable(tmp_path):
    path = str(tmp_path / "snap.json")
    with patch.object(quota, "fetch_usage", return_value=None):
        assert quota.refresh(path) is False
    assert quota.read_snapshot(path) is None


# ---------------------------------------------------------------------------
# QuotaWatcher -- the rate_limit_event trigger
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,should_wake", [
    ("allowed", False),
    ("allowed_warning", True),
    ("rejected", True),
])
def test_rate_limit_event_wakes_the_poller_only_when_not_allowed(status, should_wake):
    watcher = quota.QuotaWatcher(path="/tmp/unused-snapshot.json")
    watcher.note_rate_limit_event({"status": status})
    assert watcher._wake.is_set() is should_wake


def test_rate_limit_event_tolerates_a_missing_or_odd_payload():
    watcher = quota.QuotaWatcher(path="/tmp/unused-snapshot.json")
    watcher.note_rate_limit_event(None)
    watcher.note_rate_limit_event("nonsense")
    watcher.note_rate_limit_event({})
    assert not watcher._wake.is_set()


# ---------------------------------------------------------------------------
# QuotaWatcher -- the poll loop: how fast it retries, and the last reading
# ---------------------------------------------------------------------------

class _FakeWake:
    """Stands in for the watcher's wake Event so `_run` can be driven
    synchronously: records the timeout of each wait, and trips the real
    stop flag once enough of the schedule has been observed. Without this
    a backoff test would have to actually sleep through the backoff."""

    def __init__(self, stop, stop_after):
        self.waits = []
        self._stop = stop
        self._stop_after = stop_after

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if len(self.waits) >= self._stop_after:
            self._stop.set()
        return False

    def clear(self):
        pass


def _drive(watcher, stop_after, refresh):
    watcher._wake = _FakeWake(watcher._stop, stop_after)
    with patch.object(quota, "refresh", refresh):
        watcher._run()
    return watcher._wake.waits


def _drive_for_real(watcher, stop_after, usage):
    """Same driver, but with `refresh` left alone -- only the HTTP fetch is
    stubbed. The tests below are about what reaches the history file, and
    patching `refresh` (which is what every test here used to do) skips
    both the write and the dedup guard that was swallowing it."""
    watcher._wake = _FakeWake(watcher._stop, stop_after)
    with patch.object(quota, "fetch_usage", lambda *a, **k: usage):
        watcher._run()


def _rows(path):
    return [json.loads(line) for line in open(path)]


def test_a_failing_poll_retries_in_seconds_rather_than_backing_off_from_a_minute():
    """The failure this actually meets is the first poll of a session
    racing the CLI's OAuth refresh, which clears in about three seconds.
    Doubling from the 60s poll interval turned that into a two-minute hole
    at the start of every cycle -- the exact window in which a cycle reads
    the snapshot to decide how much work to take on."""
    watcher = quota.QuotaWatcher(path="/tmp/unused-snapshot.json")

    waits = _drive(watcher, 4, lambda path, boundary=None: False)

    assert waits == [5, 10, 20, 40]


def test_a_poll_that_keeps_failing_still_backs_off_to_the_cap():
    """The short first retry must not turn a moved endpoint into a
    five-second hammer for the rest of the session."""
    watcher = quota.QuotaWatcher(path="/tmp/unused-snapshot.json")

    waits = _drive(watcher, 8, lambda path, boundary=None: False)

    assert waits == [5, 10, 20, 40, 80, 160, 300, 300]


def test_a_recovered_poll_goes_back_to_the_normal_interval():
    results = [False, True, True]
    watcher = quota.QuotaWatcher(path="/tmp/unused-snapshot.json")

    waits = _drive(watcher, 3, lambda path, boundary=None: results.pop(0))

    assert waits == [5, quota.POLL_SECONDS, quota.POLL_SECONDS]


def test_a_recovery_re_arms_the_failure_log():
    """`failures` resets on success, so a flapping endpoint reports each
    outage once instead of going quiet forever after the first."""
    results = [False, True, False]
    watcher = quota.QuotaWatcher(path="/tmp/unused-snapshot.json")

    with patch.object(quota, "log") as logged:
        _drive(watcher, 3, lambda path, boundary=None: results.pop(0))

    assert logged.call_count == 2


def test_the_last_reading_is_taken_after_the_watcher_is_told_to_stop():
    """The reading that matters most is the one nobody is awake for. The
    poll only runs while a turn runs, so without a reading on the way out
    the history's final row for a cycle is wherever the last 60s tick
    landed -- never where the cycle actually ended."""
    calls = []
    watcher = quota.QuotaWatcher(path="/tmp/unused-snapshot.json")

    _drive(watcher, 1, lambda path, boundary=None: calls.append((path, boundary)) or True)

    assert calls == [("/tmp/unused-snapshot.json", "start"),
                     ("/tmp/unused-snapshot.json", "end")]


def test_the_last_reading_reaches_the_history_even_when_the_number_has_not_moved(tmp_path):
    """This is the half the test above could not see, because it patches
    out `refresh` -- and therefore the dedup guard sitting underneath it.
    The reading was always taken; `append_history` then dropped it for
    matching the tick before it, which is the *normal* case, since the
    percentage only moves every one to three minutes. Live proof: Cycle
    47's history ends at 02:00:17, exactly three poll intervals after the
    row above it, while the cycle was still writing to the vault at 01:59.
    """
    snapshot = str(tmp_path / "quota-snapshot.json")
    history = str(tmp_path / "quota-history.jsonl")
    watcher = quota.QuotaWatcher(path=snapshot)

    _drive_for_real(watcher, 3, LIVE_USAGE_PAYLOAD)

    rows = _rows(history)
    assert [r.get("boundary") for r in rows] == ["start", "end"]
    assert rows[-1]["at"] >= rows[0]["at"]


def test_the_first_reading_of_a_session_is_kept_even_if_it_repeats_the_last(tmp_path):
    """The mirror image, and the one that would have re-broken what
    bridge#20 fixed: a new cycle waking into an unchanged seven-day
    window reads the same numbers the previous cycle ended on, so its
    opening row would be deduped against a row written hours earlier and
    the cycle would have no recorded start."""
    snapshot = str(tmp_path / "quota-snapshot.json")
    history = str(tmp_path / "quota-history.jsonl")
    quota.append_history(quota.summarize(LIVE_USAGE_PAYLOAD), history)
    watcher = quota.QuotaWatcher(path=snapshot)

    _drive_for_real(watcher, 1, LIVE_USAGE_PAYLOAD)

    assert [r.get("boundary") for r in _rows(history)] == [None, "start", "end"]


def test_the_start_boundary_marks_the_first_poll_that_works_not_the_first_attempt(tmp_path):
    """The first poll of a session races the CLI's own OAuth refresh and
    loses about a third of the time. Marking the attempt rather than the
    reading would put "start" on nothing and leave the real first reading
    indistinguishable from a tick."""
    snapshot = str(tmp_path / "quota-snapshot.json")
    history = str(tmp_path / "quota-history.jsonl")
    payloads = [None, LIVE_USAGE_PAYLOAD, LIVE_USAGE_PAYLOAD]
    watcher = quota.QuotaWatcher(path=snapshot)
    watcher._wake = _FakeWake(watcher._stop, 2)

    with patch.object(quota, "fetch_usage", lambda *a, **k: payloads.pop(0)):
        watcher._run()

    assert [r.get("boundary") for r in _rows(history)] == ["start", "end"]


def test_a_boundary_row_does_not_stop_the_next_tick_being_deduped(tmp_path):
    """The exemption is for boundaries only. An ordinary tick following a
    boundary row must still be compared on the numbers alone, or the
    marker would quietly turn the dedup off for the row after it."""
    history = str(tmp_path / "quota-history.jsonl")
    summary = quota.summarize(LIVE_USAGE_PAYLOAD)

    assert quota.append_history(summary, history, "start") is True
    assert quota.append_history(summary, history) is False


def test_close_does_not_return_until_the_final_reading_is_written(tmp_path):
    """Every other watcher test above calls `_run()` on the main thread, so
    none of them can see anything about the thread `start()` actually makes
    -- and the defect lived exactly there. `close()` set the stop flag and
    returned, so the final reading ran at an unknowable time afterwards,
    outside whatever scope the caller believed it was in.

    In the suite that scope is conftest's patch of `fetch_usage`, so the
    reading escaped the guard and went to the live endpoint on this box's
    real credentials. Measured 2026-08-09: one `pytest tests/` run appended
    three rows carrying the true live utilization to the production
    quota-history.jsonl, 19ms apart, and six identical strays from the
    previous cycle's run were already sitting in it.

    The stub sleeps so this fails on purpose rather than on timing: without
    the join, close() returns in microseconds and the row is provably not
    there yet.
    """
    snapshot = str(tmp_path / "quota-snapshot.json")
    history = str(tmp_path / "quota-history.jsonl")
    watcher = quota.QuotaWatcher(path=snapshot)

    def slow_fetch(*a, **k):
        time.sleep(0.2)
        return LIVE_USAGE_PAYLOAD

    with patch.object(quota, "fetch_usage", slow_fetch):
        watcher.start()
        watcher.close()
        # Inside the patch, deliberately: this is the assertion the live
        # rows in production were the counter-example to.
        assert [r.get("boundary") for r in _rows(history)][-1] == "end"


def test_close_leaves_no_thread_still_running(tmp_path):
    """The property conftest's session-scoped guard depends on. A watcher
    thread that outlives close() outlives the test that made it, and then
    reaches the network under whichever patches happen to be standing."""
    snapshot = str(tmp_path / "quota-snapshot.json")
    watcher = quota.QuotaWatcher(path=snapshot)

    with patch.object(quota, "fetch_usage", lambda *a, **k: LIVE_USAGE_PAYLOAD):
        watcher.start()
        thread = watcher._thread
        watcher.close()

    assert thread is not None
    assert not thread.is_alive()


def test_a_failing_last_reading_does_not_escape_the_thread():
    def explode(path, boundary=None):
        raise RuntimeError("endpoint gone")

    watcher = quota.QuotaWatcher(path="/tmp/unused-snapshot.json")

    _drive(watcher, 1, explode)  # must not raise


# ---------------------------------------------------------------------------
# hook -- bands, and when it stays quiet
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("remaining,band", [
    (100.0, 0), (50.0, 0), (10.1, 0),
    (10.0, 1), (5.1, 1),
    (5.0, 2), (0.6, 2),
    (0.5, 3), (0.0, 3),
])
def test_band_boundaries(remaining, band):
    assert quota_notice.band_for(remaining) == band


def drive_hook(stdin_text, tmp_path, snapshot):
    """Drive the hook exactly as the CLI does -- JSON on stdin, one line of
    JSON or nothing on stdout. Returns the additionalContext, or None.

    Real stdin rather than a patched json.load: the hook also reads its own
    dedupe state with json.load, and patching that globally quietly
    disables the very deduplication these tests exist to check.
    """
    printed = []
    with patch.object(quota_notice.quota, "read_snapshot", return_value=snapshot), \
         patch.object(quota_notice, "STATE_FILE", str(tmp_path / "announced.json")), \
         patch("sys.stdin", io.StringIO(stdin_text)), \
         patch("builtins.print", side_effect=lambda s: printed.append(s)):
        quota_notice.main()
    if not printed:
        return None
    return json.loads(printed[0])["hookSpecificOutput"]["additionalContext"]


def run_hook(tmp_path, remaining, event, session_id="sess-1", fetched_at=None, resets=""):
    snapshot = {
        "tightest": {"window": "five_hour", "remaining_pct": remaining, "resets_at": resets},
        "fetched_at": fetched_at if fetched_at is not None else time.time(),
    }
    stdin = json.dumps({"hook_event_name": event, "session_id": session_id})
    return drive_hook(stdin, tmp_path, snapshot)


def test_prompt_submit_always_reports_even_with_plenty_left(tmp_path):
    """A cycle gets exactly one of these, and knowing it starts at 43%
    changes what it should attempt."""
    out = run_hook(tmp_path, 43.0, "UserPromptSubmit")
    assert "43% of your 5-hour Claude quota remains" in out
    assert "QUOTA" not in out


def test_post_tool_use_is_silent_while_there_is_room(tmp_path):
    assert run_hook(tmp_path, 43.0, "PostToolUse") is None


def test_post_tool_use_warns_at_edvards_ten_percent(tmp_path):
    out = run_hook(tmp_path, 9.0, "PostToolUse")
    assert out.startswith("QUOTA LOW")
    # Was `"journal.md" in out` until Cycle 82, which pinned the wrong
    # file: journal.md is the frozen archive, and an entry appended there
    # is invisible to the site and to every later cycle.
    assert "nova/journal/" in out and "reply to the owner" in out


def test_a_band_is_announced_once_not_on_every_tool_call(tmp_path):
    """The reason this dedupe exists: ~60 tokens on each of ~300 tool
    calls is ~18k tokens spent restating one warning, in a session whose
    problem is that it is running out of budget."""
    assert run_hook(tmp_path, 9.0, "PostToolUse") is not None
    assert run_hook(tmp_path, 9.0, "PostToolUse") is None
    assert run_hook(tmp_path, 8.0, "PostToolUse") is None


def test_getting_worse_escalates_to_the_next_band(tmp_path):
    assert run_hook(tmp_path, 9.0, "PostToolUse").startswith("QUOTA LOW")
    assert run_hook(tmp_path, 4.0, "PostToolUse").startswith("QUOTA CRITICAL")
    assert run_hook(tmp_path, 0.0, "PostToolUse").startswith("QUOTA SPENT")


def test_a_new_session_hears_the_warning_again(tmp_path):
    """Bands announced to the previous cycle were heard by a process that
    no longer exists."""
    assert run_hook(tmp_path, 9.0, "PostToolUse", session_id="sess-1") is not None
    assert run_hook(tmp_path, 9.0, "PostToolUse", session_id="sess-2") is not None


def test_reset_time_is_reported_in_oslo_not_utc(tmp_path):
    """Rule 7. 04:50 UTC is 06:50 in Oslo on this date."""
    out = run_hook(tmp_path, 9.0, "PostToolUse", resets="2026-08-05T04:50:00.711083+00:00")
    assert "resets 06:50 Oslo" in out


def test_a_stale_reading_is_reported_with_its_age_not_suppressed(tmp_path):
    """Silence is the failure this whole feature exists to remove; a
    number 12 minutes old still says whether it is 80% or 4%."""
    stale = time.time() - (12 * 60)
    out = run_hook(tmp_path, 9.0, "PostToolUse", fetched_at=stale)
    assert "12 min old" in out


def test_hook_is_silent_when_there_is_no_snapshot(tmp_path):
    stdin = json.dumps({"hook_event_name": "PostToolUse", "session_id": "s"})
    assert drive_hook(stdin, tmp_path, None) is None


def test_hook_is_silent_on_unreadable_stdin(tmp_path):
    snapshot = {"tightest": {"window": "five_hour", "remaining_pct": 1.0, "resets_at": ""},
                "fetched_at": time.time()}
    assert drive_hook("not json at all", tmp_path, snapshot) is None


# ---------------------------------------------------------------------------
# hook settings -- what actually attaches the hook to the CLI invocation
# ---------------------------------------------------------------------------

def test_hook_settings_registers_both_events(tmp_path):
    path = quota.write_hook_settings(str(tmp_path / "s.json"))
    hooks = json.load(open(path))["hooks"]
    assert set(hooks) == {"UserPromptSubmit", "PostToolUse", "PreToolUse"}
    assert hooks["PostToolUse"][0]["matcher"] == "*"
    assert quota.HOOK_SCRIPT in hooks["PostToolUse"][0]["hooks"][0]["command"]


def test_hook_settings_pins_the_auto_memory_directory(tmp_path):
    """Without this key the CLI keys its memory on the cwd, and a concurrent
    turn's cwd is unstable across cycles -- so nothing survived.

    Asserted against the literal, not against the constant. Comparing the
    settings value to `quota.AUTO_MEMORY_DIR` compares the code to itself and
    passes under every mutation that keeps the value constant -- including
    "", "nova-memory" and "/", each of which the CLI's own validator drops
    *silently*, putting production back on the per-cwd default with no
    warning anywhere. Measured on CLI 2.1.245."""
    path = quota.write_hook_settings(str(tmp_path / "s.json"),
                                     memory_dir=quota.AUTO_MEMORY_DIR)
    settings = json.load(open(path))
    assert settings["autoMemoryDirectory"] == "/data/claude-home/nova-memory"
    # The three properties the CLI validator rejects on, stated separately so
    # a failure says which one broke.
    assert os.path.isabs(settings["autoMemoryDirectory"])
    assert len(settings["autoMemoryDirectory"]) >= 3
    assert os.path.join(".claude", "projects") not in settings["autoMemoryDirectory"]


def test_hook_settings_omits_the_memory_key_for_a_non_cycle_turn(tmp_path):
    """This file is written for every turn of every conversation the bridge
    serves. The CLI prepends MEMORY.md to the first user turn under an
    "instructions OVERRIDE any default behavior" framing, so a shared
    directory would hand Nova's notes to itself to the owner's own personas.
    A turn that is not a cycle gets no pin at all."""
    path = quota.write_hook_settings(str(tmp_path / "s.json"))
    assert "autoMemoryDirectory" not in json.load(open(path))


def test_cli_passes_the_memory_pin_per_identity():
    """The wiring mutation, which is the first one to run on a change whose
    whole content is "call X from Y": deleting the argument at the call site
    leaves every test above green, because they all call the function
    directly.

    This reads the source rather than driving `_run_cli_once`, and that is a
    real limit worth stating: it pins that the call site names `memory_dir`
    and gates it on `is_cycle_opening`, not that the resulting file is right
    for a real turn. It fails if the wire is cut, which is the failure this
    change can actually have."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "bridge", "cli.py")).read()
    call = src[src.index("hook_settings = write_hook_settings("):]
    call = call[:call.index("\n    )") + 6]
    assert "memory_dir=" in call
    assert "is_cycle_opening(message)" in call
    assert "quota.AUTO_MEMORY_DIR" in call
    # A turn that is not a cycle reaches the persona's own directory, and
    # `persona_id` is what makes it that persona's rather than a second
    # shared one -- a pin computed from anything else would be the
    # cross-contamination this whole split exists to prevent, renamed.
    assert "quota.persona_memory_dir(persona_id)" in call
    assert "else None" not in call


def test_auto_memory_directory_does_not_move_with_the_workspace(tmp_path):
    """The invariant, stated as the failure it prevents: two turns running
    from two different concurrent workspaces must be handed the same memory
    directory.

    This reloads the module under each cwd rather than just chdir-ing. The
    first draft did not, and it passed against a mutation that put the cwd
    back into the path -- because the constant is computed at import, so one
    process only ever sees one cwd. A test that cannot fail against the bug
    it names is worth less than no test."""
    import importlib

    seen = []
    try:
        for name in ("ws-a", "ws-b"):
            cwd = tmp_path / name
            cwd.mkdir()
            os.chdir(cwd)
            reloaded = importlib.reload(quota)
            path = reloaded.write_hook_settings(str(tmp_path / f"s.{name}.json"),
                                                memory_dir=reloaded.AUTO_MEMORY_DIR)
            seen.append(json.load(open(path))["autoMemoryDirectory"])
    finally:
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        importlib.reload(quota)
    assert seen[0] == seen[1] == "/data/claude-home/nova-memory"


def test_hook_settings_failure_degrades_to_no_hook(tmp_path):
    """cli.py treats "" as "run without the hook" -- an unwritable config
    dir must not stop a cycle from running at all."""
    with patch("json.dump", side_effect=OSError("read-only fs")):
        assert quota.write_hook_settings(str(tmp_path / "s.json")) == ""


def test_history_appends_a_row_when_the_reading_moves(tmp_path):
    path = str(tmp_path / "quota-history.jsonl")
    first = {"windows": [{"window": "five_hour", "used_pct": 10.0, "resets_at": "x"}]}
    second = {"windows": [{"window": "five_hour", "used_pct": 12.0, "resets_at": "x"}]}

    assert quota.append_history(first, path) is True
    assert quota.append_history(second, path) is True

    rows = [json.loads(line) for line in open(path)]
    assert [r["five_hour"] for r in rows] == [10.0, 12.0]
    assert all("at" in r for r in rows)


def test_history_skips_an_unchanged_reading(tmp_path):
    path = str(tmp_path / "quota-history.jsonl")
    summary = {"windows": [{"window": "seven_day", "used_pct": 1.0, "resets_at": "y"}]}

    assert quota.append_history(summary, path) is True
    assert quota.append_history(summary, path) is False

    assert len(open(path).read().strip().splitlines()) == 1


def test_history_row_carries_every_tracked_window(tmp_path):
    path = str(tmp_path / "quota-history.jsonl")
    summary = {"windows": [
        {"window": "five_hour", "used_pct": 12.0, "resets_at": "a"},
        {"window": "seven_day", "used_pct": 1.0, "resets_at": "b"},
    ]}

    quota.append_history(summary, path)

    row = json.loads(open(path).read().strip())
    assert row["five_hour"] == 12.0
    assert row["seven_day"] == 1.0
    assert row["seven_day_resets_at"] == "b"


def test_an_unchanged_reading_is_skipped_even_though_resets_at_jitters(tmp_path):
    """The endpoint computes `resets_at` per request, so it comes back with
    fresh sub-second jitter every poll. Comparing the whole row made the
    dedup inert -- 5 of the first 9 rows on the live pod repeated the
    reading above them. These two timestamps are the real ones, four
    seconds apart, from the same window."""
    path = str(tmp_path / "quota-history.jsonl")

    def reading(resets_at):
        return {"windows": [{"window": "five_hour", "used_pct": 29.0,
                             "resets_at": resets_at}]}

    assert quota.append_history(reading("2026-08-09T00:29:59.498355+00:00"), path) is True
    assert quota.append_history(reading("2026-08-09T00:29:59.027533+00:00"), path) is False

    assert len(open(path).read().strip().splitlines()) == 1


def test_a_window_that_rolls_over_still_writes_a_row(tmp_path):
    """Ignoring `resets_at` must not swallow a real window boundary. It
    doesn't: utilization drops when a window resets, and that is the
    change being compared."""
    path = str(tmp_path / "quota-history.jsonl")
    before = {"windows": [{"window": "five_hour", "used_pct": 96.0, "resets_at": "a"}]}
    after = {"windows": [{"window": "five_hour", "used_pct": 0.0, "resets_at": "b"}]}

    assert quota.append_history(before, path) is True
    assert quota.append_history(after, path) is True

    assert len(open(path).read().strip().splitlines()) == 2


def test_history_path_is_derived_from_the_snapshot_path(tmp_path):
    """A test pointing the watcher at a tmp dir must not append to the
    real history file on the PVC."""
    snapshot = str(tmp_path / "quota-snapshot.json")

    assert quota.history_path_for(snapshot) == str(tmp_path / "quota-history.jsonl")
    assert quota.history_path_for(None) == quota.HISTORY_FILE


# ---------------------------------------------------------------------------
# pace -- used share against elapsed share, so a cycle can size its own work
# ---------------------------------------------------------------------------

def _usage_at(window, used, seconds_into_window):
    """A payload whose `window` is `seconds_into_window` old, by placing
    `resets_at` the remaining distance in the future from *now*."""
    length = quota.WINDOW_SECONDS[window]
    ends = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        seconds=length - seconds_into_window)
    return {window: {"utilization": used, "resets_at": ends.isoformat()}}


def test_pace_is_one_when_spending_exactly_matches_elapsed_time():
    """Half the window gone, half the quota gone -- the definition of on
    the line, and the number a cycle compares against."""
    summary = quota.summarize(_usage_at("seven_day", 50.0, 3.5 * 86400))
    assert summary["windows"][0]["pace"] == pytest.approx(1.0, abs=0.01)


def test_pace_above_one_means_the_window_empties_early():
    """90% spent with half the week still to run: 1.8, and the window
    cannot survive its own remaining time at that rate."""
    hot = quota.summarize(_usage_at("seven_day", 90.0, 3.5 * 86400))
    assert hot["windows"][0]["pace"] == pytest.approx(1.8, abs=0.01)


def test_pace_is_window_to_date_and_hides_an_idle_stretch():
    """The live 2026-08-09 reading, and the reason pace alone is not the
    whole answer: 13% used 3.97 days into the week is a pace of 0.23,
    which reads as a very quiet week. It was not. The loop was simply
    near-idle from 08-05 to 08-08 and then ran 14 cycles in one day at
    14.76%/day -- above the 14.3%/day the window affords.

    Pace is an average over the whole window to date, so a burst is
    diluted by whatever came before it. It answers "will *this* window
    hold out", which is what a cycle sizing its own work needs. It does
    not answer "is the current cadence sustainable" -- that needs the
    slope between two recent readings, which is why `quota-history.jsonl`
    keeps the series and not just this number.
    """
    summary = quota.summarize(_usage_at("seven_day", 13.0, 3.97 * 86400))
    assert summary["windows"][0]["pace"] == pytest.approx(0.229, abs=0.01)


def test_pace_is_none_in_the_opening_moments_of_a_window():
    """Elapsed share near zero makes the ratio explode; a cycle waking
    into a fresh window must read "unknown", never "infinitely over"."""
    summary = quota.summarize(_usage_at("five_hour", 2.0, 10))
    assert summary["windows"][0]["pace"] is None


def test_pace_is_none_without_a_usable_resets_at():
    """Every degraded path here reports nothing rather than a guess."""
    assert quota._pace("seven_day", "", 50.0) is None
    assert quota._pace("seven_day", None, 50.0) is None
    assert quota._pace("seven_day", "not-a-date", 50.0) is None
    assert quota._pace("nimbus_quill", "2026-08-12T17:00:00+00:00", 50.0) is None


def test_history_row_carries_pace_but_it_never_votes_on_dedup(tmp_path):
    """The trap: pace moves every poll because elapsed time grows even when
    utilization does not. If it counted as part of the reading, no two rows
    would ever compare equal and the dedup guard would be silently off --
    the exact failure that once put duplicates in 5 of the first 9 live
    rows. So the row carries it and the comparison ignores it.
    """
    path = tmp_path / "history.jsonl"
    usage = _usage_at("seven_day", 40.0, 3.0 * 86400)
    assert quota.append_history(quota.summarize(usage), path=str(path)) is True

    row = json.loads(path.read_text().splitlines()[0])
    assert row["seven_day_pace"] == pytest.approx(0.933, abs=0.01)

    # Same utilization, a later moment -- so pace really has moved.
    later = quota.summarize(_usage_at("seven_day", 40.0, 3.2 * 86400))
    assert later["windows"][0]["pace"] != row["seven_day_pace"]
    assert quota.append_history(later, path=str(path)) is False
    assert len(path.read_text().splitlines()) == 1


# --- publishing the cost record on the way out of a cycle ------------------
#
# Every CLI invocation gets a watcher, and only some of them are cycles. The
# gate is what keeps a journal-card reply -- or the test suite -- from
# scanning every transcript on the PVC and pushing ~90KB to the vault.


def test_a_cycle_publishes_the_cost_record_on_the_way_out(tmp_path):
    """The whole point of arming it: the last thing a cycle does, after its
    own closing quota reading, is republish the record the site reads. A
    hand-refreshed snapshot is the failure this replaces -- the file it
    supersedes sat describing 45 cycles while the loop was on its 107th."""
    calls = []
    watcher = quota.QuotaWatcher(path=str(tmp_path / "snap.json"), publish_costs=True)

    with patch.object(publish_costs, "refresh", lambda *a, **k: calls.append(1) or True):
        _drive_for_real(watcher, 1, LIVE_USAGE_PAYLOAD)

    assert calls == [1]


def test_an_ordinary_turn_does_not_publish_anything(tmp_path):
    """The default, and it must stay the default. cli.py builds a watcher for
    every invocation -- reply turns, probes, and every test in this suite that
    exercises stream parsing -- and a scan-plus-vault-write on each of those is
    both wrong and slow. Arming it by default is exactly what broke 8 cli tests
    the first time this was attempted."""
    calls = []
    watcher = quota.QuotaWatcher(path=str(tmp_path / "snap.json"))

    # Asserted on the attribute as well as the behaviour, because the
    # behaviour alone pinned nothing: a reviewer pointed out that reverting
    # this whole change -- no parameter, no publish call anywhere -- leaves
    # `calls == []` true for the boring reason that nothing publishes at all.
    # A test whose negative result was guaranteed in advance is not evidence.
    # This line fails with AttributeError on that revert.
    assert watcher._publish_costs is False

    with patch.object(publish_costs, "refresh", lambda *a, **k: calls.append(1) or True):
        _drive_for_real(watcher, 1, LIVE_USAGE_PAYLOAD)

    assert calls == []


def test_a_publish_that_raises_still_leaves_the_closing_reading_written(tmp_path):
    """Best-effort, and the ordering is the reason it can be. The reading is
    the thing the next cycle reads to size its own work; publishing is a
    convenience for a chart. So the record goes last and its failure is a log
    line, never a lost boundary row."""
    snapshot = str(tmp_path / "quota-snapshot.json")
    history = str(tmp_path / "quota-history.jsonl")

    def boom(*a, **k):
        raise RuntimeError("vault unreachable")

    watcher = quota.QuotaWatcher(path=snapshot, publish_costs=True)
    with patch.object(publish_costs, "refresh", boom):
        _drive_for_real(watcher, 1, LIVE_USAGE_PAYLOAD)

    assert [r.get("boundary") for r in _rows(history)] == ["start", "end"]


def test_persona_memory_dir_is_one_directory_per_persona():
    """Idea #165: a chat persona had no memory across conversations because
    only a Nova cycle was ever handed a pinned directory. Two personas must
    get two directories, and both must sit under the persistent claude home
    rather than under a concurrent workspace."""
    a = quota.persona_memory_dir("08ffac94-7c4a-4506-897f-968c592358cb")
    b = quota.persona_memory_dir("11111111-2222-3333-4444-555555555555")
    assert a and b and a != b
    assert a.startswith(quota.CLAUDE_HOME + os.sep)
    assert a != quota.AUTO_MEMORY_DIR and b != quota.AUTO_MEMORY_DIR
    assert os.path.join(".claude", "projects") not in a


def test_persona_memory_dir_refuses_anything_that_is_not_a_plain_id():
    """The id arrives over HTTP and becomes a directory name. Anything that
    could escape the root, or that is empty, gets no memory directory at all
    -- never a sanitised one, because rewriting an id would hand two
    personas one directory."""
    for bad in ("", None, "..", "../../etc", "a/b", "/etc/passwd", "ab",
                "x" * 65, "with space", ".hidden"):
        assert quota.persona_memory_dir(bad) == "", bad
    escaped = quota.persona_memory_dir("../../../etc/shadow")
    assert escaped == ""


def test_a_persona_pin_is_written_into_the_settings_file(tmp_path):
    """End of the wire on this side: the directory persona_memory_dir
    returns is what lands in the --settings file the CLI reads."""
    d = quota.persona_memory_dir("08ffac94-7c4a-4506-897f-968c592358cb")
    path = quota.write_hook_settings(str(tmp_path / "s.json"), memory_dir=d)
    assert json.load(open(path))["autoMemoryDirectory"] == d


def test_server_forwards_persona_id_from_the_request_to_the_turn():
    """The wiring mutation on the HTTP side: the field can be read off the
    payload and then dropped before it reaches run_turn, which every test
    above would still pass."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "bridge", "server.py")).read()
    assert 'payload.get("persona_id")' in src
    # generate() has three call sites into the CLI -- stateless, normal, and
    # the SESSION_NOT_FOUND retry -- and a persona that loses its memory
    # only on the retry path is the kind of bug nobody reproduces.
    assert src.count("persona_id=persona_id,") == 4
