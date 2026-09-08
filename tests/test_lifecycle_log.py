"""The bridge's record of its own shutdowns.

These guard the one distinction the file exists to make: a process that
was asked to stop and did not finish draining, versus a process that was
never asked at all. Both look identical from outside once the Pod is
gone, and reading them off ReplicaSet timestamps got it wrong -- under
the `Recreate` strategy the new ReplicaSet is created after the old Pods
are already gone, so its creationTimestamp is the end of termination and
not the start."""
import json
import os
import signal
import threading
from unittest.mock import patch

import pytest

from bridge import lifecycle_log


@pytest.fixture
def ledger(tmp_path):
    path = tmp_path / "bridge-lifecycle.jsonl"
    with patch.object(lifecycle_log, "LIFECYCLE_FILE", str(path)):
        yield path


def test_record_appends_one_json_line_per_call(ledger):
    lifecycle_log.record("started", port=8090)
    lifecycle_log.record("signal", signal=15, in_flight=2)

    lines = ledger.read_text().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["event"] == "started" and first["port"] == 8090
    assert second["event"] == "signal" and second["in_flight"] == 2
    # Every row carries when and which process, or two lives cannot be told apart.
    for row in (first, second):
        assert row["pid"] == os.getpid()
        assert row["at"].startswith("20")


def test_record_never_raises_when_the_file_cannot_be_written(tmp_path, capsys):
    """It is called from inside a signal handler. An exception there
    escapes into whatever the main thread was executing."""
    unwritable = tmp_path / "no-such-dir" / "lifecycle.jsonl"
    with patch.object(lifecycle_log, "LIFECYCLE_FILE", str(unwritable)):
        lifecycle_log.record("signal", signal=15, in_flight=1)  # must not raise
    assert "could not record" in capsys.readouterr().err


def test_read_is_empty_rather_than_raising_on_a_missing_file(tmp_path):
    assert lifecycle_log.read(str(tmp_path / "never-written.jsonl")) == []


def test_read_skips_a_torn_final_line(ledger):
    lifecycle_log.record("started", port=8090)
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write('{"event": "signal", "in_fli')  # killed mid-write

    rows = lifecycle_log.read(str(ledger))
    assert [r["event"] for r in rows] == ["started"]


def _rows(*events):
    return [dict(e) for e in events]


def test_a_signal_followed_by_a_drain_is_a_clean_shutdown():
    lives = lifecycle_log.lives(_rows(
        {"event": "started"},
        {"event": "signal", "signal": 15, "in_flight": 1},
        {"event": "drained", "waited_seconds": 812.4},
        {"event": "started"},
    ))
    assert [life["verdict"] for life in lives] == ["clean", "running"]
    assert lives[0]["turns_lost"] == 0


def test_a_signal_with_no_drain_before_the_next_start_is_a_kill_mid_drain():
    """This is what a grace period expiring looks like, and the count of
    turns in flight at the signal is what it cost."""
    lives = lifecycle_log.lives(_rows(
        {"event": "started"},
        {"event": "signal", "signal": 15, "in_flight": 2},
        {"event": "started"},
    ))
    assert lives[0]["verdict"] == "killed_mid_drain"
    assert lives[0]["turns_lost"] == 2


def test_a_life_that_ended_with_no_signal_at_all_is_not_a_drain_failure():
    """A crash, an OOM kill, or a delete with no grace. Reported as its own
    verdict because fixing the drain would be fixing the wrong thing."""
    lives = lifecycle_log.lives(_rows(
        {"event": "started"},
        {"event": "started"},
    ))
    assert [life["verdict"] for life in lives] == ["no_signal", "running"]
    assert lives[0]["turns_lost"] == 0


def test_the_newest_life_is_never_judged_as_killed():
    """A process that is draining right now has no `drained` row yet, and
    calling that a kill would report every live shutdown as a failure."""
    lives = lifecycle_log.lives(_rows(
        {"event": "started"},
        {"event": "signal", "signal": 15, "in_flight": 1},
    ))
    assert [life["verdict"] for life in lives] == ["running"]
    assert lives[0]["turns_lost"] == 0


def test_rows_before_the_first_started_row_are_not_a_life():
    """A rotated or truncated file must yield fewer lives, never a verdict
    on a life whose start it cannot see."""
    lives = lifecycle_log.lives(_rows(
        {"event": "drained", "waited_seconds": 3.0},
        {"event": "started"},
    ))
    assert len(lives) == 1
    assert lives[0]["verdict"] == "running"


def test_a_real_sigterm_writes_the_signal_row_with_the_turns_in_flight(ledger):
    """End to end through the actual handler the kernel calls, because the
    value of this row is that it is written on the way out."""
    from bridge import server

    previous = signal.getsignal(signal.SIGTERM)
    saved_flag, saved_in_flight = server._shutdown_requested, server._in_flight
    try:
        server._shutdown_requested = False
        server._in_flight = 0
        server.install_signal_handlers()
        server._enter_turn()
        server._enter_turn()

        with patch.object(server, "log", lambda *a, **k: None):
            os.kill(os.getpid(), signal.SIGTERM)
            # The handler runs between bytecodes on the main thread; give the
            # interpreter a point at which to run it.
            threading.Event().wait(0.2)

        assert server.shutdown_requested() is True
        rows = lifecycle_log.read(str(ledger))
        assert [r["event"] for r in rows] == ["signal"]
        assert rows[0]["signal"] == int(signal.SIGTERM)
        assert rows[0]["in_flight"] == 2
    finally:
        signal.signal(signal.SIGTERM, previous)
        server._shutdown_requested, server._in_flight = saved_flag, saved_in_flight


def test_the_suite_never_writes_to_the_live_lifecycle_ledger():
    """Pins `isolated_lifecycle_log` in tests/conftest.py.

    The precondition is asserted first, because without it this passes for
    free the day someone renames CLAUDE_HOME or the file: a test that only
    checks two strings differ proves nothing about which two."""
    from bridge.config import CLAUDE_HOME

    live = os.path.join(CLAUDE_HOME, "bridge-lifecycle.jsonl")
    assert live.startswith("/data/") or CLAUDE_HOME != "/data/claude-home", (
        "the live path this guards is no longer under CLAUDE_HOME")
    assert lifecycle_log.LIFECYCLE_FILE != live
    lifecycle_log.record("started", port=0)
    assert not os.path.exists(live), (
        "a test just wrote a fake life into the ledger a cycle reads to find "
        "out why a real turn was lost")
