"""Covers the quota warning path: reading usage, turning it into a
remaining-percentage snapshot, and the hook that reports it to a running
session.

The thing actually being protected here is that a cycle finds out it is
nearly out of quota *while it can still act*. So the tests that matter
most are the ones about when the hook stays quiet -- a warning that fires
on all 300 tool calls is as useless as one that never fires, just
expensive instead of silent.
"""
import io
import json
import time
from unittest.mock import patch

import pytest

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
    assert "journal.md" in out and "reply to Edvard" in out


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
    assert set(hooks) == {"UserPromptSubmit", "PostToolUse"}
    assert hooks["PostToolUse"][0]["matcher"] == "*"
    assert quota.HOOK_SCRIPT in hooks["PostToolUse"][0]["hooks"][0]["command"]


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
