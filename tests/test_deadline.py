"""The turn clock: deadline.py and hooks/deadline_notice.py.

The behaviour under test is "a cycle can find out it is about to be
killed, and a cycle that is killed anyway still says something". Both
halves existed only as a silent `raise` before Cycle 82.
"""
import io
import json
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import pytest

from bridge import cli, deadline, quota
from bridge.hooks import deadline_notice

from tests.test_bridge import FakeProc, _stream_json_lines


@pytest.fixture(autouse=True)
def _reset_invocation_lock():
    cli._invocation_lock = threading.Lock()
    yield


# ---------------------------------------------------------------------------
# the record itself
# ---------------------------------------------------------------------------

def test_write_then_read_round_trips(tmp_path):
    path = str(tmp_path / "d.json")
    before = time.time()
    assert deadline.write(2700, path) is True
    record = deadline.read(path)
    assert record["timeout_seconds"] == 2700
    assert record["deadline_at"] == pytest.approx(record["started_at"] + 2700)
    assert record["started_at"] >= before


def test_seconds_left_counts_down_and_can_go_negative(tmp_path):
    path = str(tmp_path / "d.json")
    deadline.write(600, path)
    record = deadline.read(path)
    started = record["started_at"]
    assert deadline.seconds_left(record, now=started) == pytest.approx(600)
    assert deadline.seconds_left(record, now=started + 540) == pytest.approx(60)
    # Past the deadline is a real state: proc.wait() only fires once the
    # CLI's own work returns, so a session can still be running.
    assert deadline.seconds_left(record, now=started + 700) == pytest.approx(-100)


def test_clear_makes_read_return_none(tmp_path):
    """The file carries no session id, so a stale one would be read by the
    *next* turn as its own clock -- already expired, every time."""
    path = str(tmp_path / "d.json")
    deadline.write(2700, path)
    deadline.clear(path)
    assert deadline.read(path) is None
    deadline.clear(path)  # absent file is the normal resting state


def test_read_rejects_garbage_rather_than_raising(tmp_path):
    path = tmp_path / "d.json"
    path.write_text("not json")
    assert deadline.read(str(path)) is None
    path.write_text('{"deadline_at": "soon"}')
    assert deadline.read(str(path)) is None
    path.write_text('["not", "a", "dict"]')
    assert deadline.read(str(path)) is None


def test_write_failure_is_silent(tmp_path):
    with patch("json.dump", side_effect=OSError("read-only fs")):
        assert deadline.write(2700, str(tmp_path / "d.json")) is False


# ---------------------------------------------------------------------------
# the hook -- what actually reaches a session in flight
# ---------------------------------------------------------------------------

def drive_hook(tmp_path, minutes_left, event, session_id="sess-1", timeout_seconds=2700):
    """Drive the hook exactly as the CLI does -- JSON on stdin, one line of
    JSON or nothing on stdout. Real stdin rather than a patched json.load,
    because the hook reads its own dedupe state with json.load too and
    patching that globally disables the deduplication under test."""
    now = time.time()
    record = {
        "started_at": now - (timeout_seconds - minutes_left * 60),
        "deadline_at": now + minutes_left * 60,
        "timeout_seconds": timeout_seconds,
    }
    printed = []
    stdin = json.dumps({"hook_event_name": event, "session_id": session_id})
    with patch.object(deadline_notice.deadline, "read", return_value=record), \
         patch.object(deadline_notice, "STATE_FILE", str(tmp_path / "announced.json")), \
         patch("sys.stdin", io.StringIO(stdin)), \
         patch("builtins.print", side_effect=lambda s: printed.append(s)):
        deadline_notice.main()
    if not printed:
        return None
    return json.loads(printed[0])["hookSpecificOutput"]["additionalContext"]


def test_bands_are_the_measured_thresholds():
    assert deadline_notice.band_for(44) == 0
    assert deadline_notice.band_for(30.1) == 0
    assert deadline_notice.band_for(30) == deadline_notice.POSITION_THIRD
    assert deadline_notice.band_for(22.1) == deadline_notice.POSITION_THIRD
    assert deadline_notice.band_for(22) == deadline_notice.POSITION_HALF
    assert deadline_notice.band_for(15.1) == deadline_notice.POSITION_HALF
    assert deadline_notice.band_for(15) == deadline_notice.WARN_LOW
    assert deadline_notice.band_for(8.1) == deadline_notice.WARN_LOW
    assert deadline_notice.band_for(8) == deadline_notice.WARN_CRITICAL
    assert deadline_notice.band_for(3.1) == deadline_notice.WARN_CRITICAL
    assert deadline_notice.band_for(3) == deadline_notice.WARN_NEARLY_UP
    assert deadline_notice.band_for(-5) == deadline_notice.WARN_NEARLY_UP


def test_position_bands_sit_below_the_warnings_so_they_cannot_mute_one():
    """main() only announces a band strictly higher than the last one, so a
    position report numbered above TIME LOW would swallow it."""
    warnings = (
        deadline_notice.WARN_LOW,
        deadline_notice.WARN_CRITICAL,
        deadline_notice.WARN_NEARLY_UP,
    )
    for position in (deadline_notice.POSITION_THIRD, deadline_notice.POSITION_HALF):
        assert all(position < warning for warning in warnings)
    assert deadline_notice.POSITION_THIRD < deadline_notice.POSITION_HALF


def test_the_silent_first_two_thirds_now_reports_position(tmp_path):
    """Ten cycles (175-184) misjudged where they were in the turn, every one
    of them above the 15-minute line, where this hook used to say nothing
    between turn start and TIME LOW."""
    assert drive_hook(tmp_path, 40, "PostToolUse").startswith("Clock:")
    third = drive_hook(tmp_path, 28, "PostToolUse")
    assert third is not None and third.startswith("Time check:")
    assert "17 min gone" in third
    assert "about 28 of this turn's 45 minutes remain" in third
    # Announced once, like every other band.
    assert drive_hook(tmp_path, 25, "PostToolUse").startswith("Clock:")
    half = drive_hook(tmp_path, 20, "PostToolUse")
    assert half is not None and "25 min gone" in half
    # A position report must not read as a deadline, or it becomes the
    # thing it exists to prevent.
    for out in (third, half):
        assert "not a warning" in out
        assert "nothing needs to change" in out
        assert not out.startswith("TIME")
    # The warnings still fire underneath them.
    assert drive_hook(tmp_path, 12, "PostToolUse").startswith("TIME LOW")


def test_every_line_carries_the_oslo_wall_clock(tmp_path):
    """The misreadings are made in wall-clock terms ("certain it was 23:33
    when it was 23:09"); relative minutes give a drifted cycle nothing to
    notice the drift against."""
    with patch.object(deadline_notice, "oslo_clock", return_value="07:32 Oslo"):
        for minutes in (45, 28, 20, 12, 6, 2):
            event = "UserPromptSubmit" if minutes == 45 else "PostToolUse"
            out = drive_hook(tmp_path / f"m{minutes}", minutes, event)
            assert out is not None and "07:32 Oslo" in out


def test_no_clock_is_reported_rather_than_a_wrong_one(tmp_path):
    """A confidently stated wrong time is the exact failure this fixes, so
    an unresolvable zone degrades to no clock, not to UTC."""
    with patch.dict(sys.modules, {"zoneinfo": None}):
        assert deadline_notice.oslo_clock() == ""
    with patch.object(deadline_notice, "oslo_clock", return_value=""):
        out = drive_hook(tmp_path, 28, "PostToolUse")
    assert out is not None and out.startswith("Time check: 17 min gone")


def test_position_reports_without_a_start_time_still_report(tmp_path):
    """Records written before started_at existed must not silence the band."""
    record = {"deadline_at": time.time() + 28 * 60, "timeout_seconds": 45 * 60}
    printed = []
    stdin = json.dumps({"hook_event_name": "PostToolUse", "session_id": "s"})
    with patch.object(deadline_notice.deadline, "read", return_value=record), \
         patch.object(deadline_notice, "STATE_FILE", str(tmp_path / "a.json")), \
         patch("sys.stdin", io.StringIO(stdin)), \
         patch("builtins.print", side_effect=lambda s: printed.append(s)):
        deadline_notice.main()
    out = json.loads(printed[0])["hookSpecificOutput"]["additionalContext"]
    assert "about 28 of this turn's 45 minutes remain" in out
    assert "min gone" not in out


def test_first_prompt_always_reports_even_with_the_whole_turn_left(tmp_path):
    """A cycle gets one chance to hear its budget before it chooses what to
    attempt -- same reasoning as the quota hook's UserPromptSubmit."""
    out = drive_hook(tmp_path, 45, "UserPromptSubmit")
    assert "45 of this turn's 45 minutes remain" in out
    assert "killed with no reply posted" in out


def test_post_tool_use_stamps_the_clock_while_there_is_time(tmp_path):
    """It used to say nothing here. Silence between the bands is where the
    drift lives -- idea #72."""
    out = drive_hook(tmp_path, 40, "PostToolUse")
    assert out.startswith("Clock:")
    assert "5 min gone" in out and "40 min left of this turn" in out
    assert not out.startswith("TIME")


def test_the_stamp_carries_the_oslo_wall_clock(tmp_path):
    """The owner's ask is the wall clock specifically -- "getting the actual
    timestamp ... can get you to get a better hold on reality"."""
    with patch.object(deadline_notice, "oslo_clock", return_value="23:14 Oslo"):
        out = drive_hook(tmp_path, 40, "PostToolUse")
    assert out == "Clock: 23:14 Oslo · 5 min gone, 40 min left of this turn."


def test_the_stamp_degrades_to_no_clock_rather_than_a_wrong_one(tmp_path):
    """Same call oslo_clock() makes: a confidently stated wrong time is the
    failure being fixed. The elapsed figures come off the record, not the
    zone database, so they survive."""
    with patch.object(deadline_notice, "oslo_clock", return_value=""):
        out = drive_hook(tmp_path, 40, "PostToolUse")
    assert out == "Clock: 5 min gone, 40 min left of this turn."
    assert "Oslo" not in out


def test_the_stamp_repeats_where_a_warning_would_be_suppressed(tmp_path):
    """The deliberate design decision, pinned because it looks like waste to
    anyone optimising later. ~98 tool calls a cycle at ~20 tokens is ~2k
    tokens; the value is being in front of the model at the moment it
    reasons about time, which a once-a-minute stamp would miss.

    Driven *past a band crossing* on purpose. Before minute 15 the dedupe
    machinery is dormant -- band 0 is never "announced" -- so repeated
    stamps there prove only that nothing suppressed them, which is also
    what a broken dedupe would look like. Inside a band the suppression is
    live and demonstrably working on the warning, so a stamp that still
    fires on every call is the real evidence.
    """
    assert drive_hook(tmp_path, 40, "PostToolUse").startswith("Clock:")
    assert drive_hook(tmp_path, 12, "PostToolUse").startswith("TIME LOW")
    # Same band, three more calls: the warning is suppressed and the stamp
    # is not.
    for minutes in (11, 10, 9):
        out = drive_hook(tmp_path, minutes, "PostToolUse")
        assert out.startswith("Clock:"), out
        assert f"{minutes} min left of this turn" in out


def test_the_stamp_is_not_a_warning(tmp_path):
    """A position line that reads as a deadline becomes the thing it exists
    to prevent -- so it carries no advice and no wrap-up instructions."""
    out = drive_hook(tmp_path, 40, "PostToolUse")
    assert not out.startswith("TIME")
    assert "journal-digest.md" not in out
    assert "Start nothing new" not in out
    assert "size the work" not in out


def test_each_band_announces_once_and_only_escalates(tmp_path):
    """Repeating a full warning on every tool call costs context restating
    something already heard -- so a band fires once, and every call in
    between carries the bare stamp instead."""
    assert drive_hook(tmp_path, 40, "PostToolUse").startswith("Clock:")
    first = drive_hook(tmp_path, 12, "PostToolUse")
    assert first is not None and first.startswith("TIME LOW")
    # Same band again, and time still passing inside it: no second warning.
    assert drive_hook(tmp_path, 10, "PostToolUse").startswith("Clock:")
    assert drive_hook(tmp_path, 9, "PostToolUse").startswith("Clock:")
    second = drive_hook(tmp_path, 6, "PostToolUse")
    assert second is not None and second.startswith("TIME CRITICAL")
    assert drive_hook(tmp_path, 5, "PostToolUse").startswith("Clock:")
    third = drive_hook(tmp_path, 2, "PostToolUse")
    assert third is not None and third.startswith("TIME NEARLY UP")
    assert drive_hook(tmp_path, 1, "PostToolUse").startswith("Clock:")


def test_a_new_session_starts_from_a_clean_slate(tmp_path):
    """Bands announced to the previous cycle were heard by a process that
    no longer exists."""
    assert drive_hook(tmp_path, 12, "PostToolUse", session_id="sess-1").startswith("TIME LOW")
    assert drive_hook(tmp_path, 12, "PostToolUse", session_id="sess-1").startswith("Clock:")
    assert drive_hook(tmp_path, 12, "PostToolUse", session_id="sess-2").startswith("TIME LOW")


def test_wrap_up_instructions_name_what_actually_survives(tmp_path):
    low = drive_hook(tmp_path, 12, "PostToolUse")
    assert "journal-digest.md" in low and "Next cycle" in low
    # At 2 minutes the honest advice is not a task list.
    final = drive_hook(tmp_path, 2, "PostToolUse")
    assert "Only your reply is still saveable" in final
    assert "journal-digest.md" not in final


def test_hook_is_silent_when_no_turn_is_running(tmp_path):
    printed = []
    with patch.object(deadline_notice.deadline, "read", return_value=None), \
         patch.object(deadline_notice, "STATE_FILE", str(tmp_path / "a.json")), \
         patch("sys.stdin", io.StringIO(json.dumps({"hook_event_name": "PostToolUse"}))), \
         patch("builtins.print", side_effect=lambda s: printed.append(s)):
        deadline_notice.main()
    assert printed == []


def test_no_hook_tells_a_cycle_to_write_to_the_frozen_archive():
    """journal.md has been the frozen pre-2026-08-09 archive since the
    journal became one document per entry, and an entry appended there is
    invisible to the site and to every later cycle. Both hooks fire when a
    cycle is under pressure and least likely to check."""
    from bridge.hooks import quota_notice
    for module in (quota_notice, deadline_notice):
        assert "journal.md" not in module.WRAP_UP
        assert "nova/journal/" in module.WRAP_UP


def test_hook_settings_attach_both_bridge_hooks(tmp_path):
    """The deadline hook is useless if it is never registered -- the quota
    hook shipped correct and unattached once already."""
    path = quota.write_hook_settings(str(tmp_path / "s.json"))
    hooks = json.load(open(path))["hooks"]
    for event in ("UserPromptSubmit", "PostToolUse"):
        commands = " ".join(h["command"] for h in hooks[event][0]["hooks"])
        assert quota.HOOK_SCRIPT in commands
        assert quota.DEADLINE_HOOK_SCRIPT in commands


# ---------------------------------------------------------------------------
# the salvage -- a killed turn still says something
# ---------------------------------------------------------------------------

class TimingOutProc(FakeProc):
    """Streams its events, then never exits -- exactly what a turn that
    overruns looks like from cli.py's side."""

    def __init__(self, lines):
        super().__init__(lines)
        self._killed = False

    def wait(self, timeout=None):
        if timeout is not None and not self._killed:
            raise subprocess.TimeoutExpired("claude", timeout)

    def kill(self):
        self._killed = True


def run_until_timeout(tmp_path, lines, activity=None):
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", return_value=TimingOutProc(lines)):
        return cli.run_turn("hello", activity=activity)


def run_narrated_until_timeout(tmp_path, lines):
    """A killed turn in its real production shape.

    Every real call carries an activity block -- the runner's claude_cli
    provider sets one unconditionally -- so `reporter.enabled` is True on
    every actual cycle and False in every test that forgets it. Returns
    (reply, passages already narrated to Agora).
    """
    narrated = []
    with patch.object(cli.ActivityReporter, "report_text",
                      side_effect=lambda text: narrated.append(text), autospec=False), \
         patch.object(cli.ActivityReporter, "start", autospec=False), \
         patch.object(cli.ActivityReporter, "close", autospec=False), \
         patch.object(cli.ActivityReporter, "report", autospec=False), \
         patch.object(cli.ActivityReporter, "report_result", autospec=False):
        text, _, _ = run_until_timeout(
            tmp_path, lines, activity={"url": "http://agora.invalid", "token": "t"})
    return text, narrated


def test_salvage_does_not_repeat_what_was_already_narrated(tmp_path):
    """The reviewer's finding, and the one that would have reached his phone.

    `pending` is emptied by release_narrative() on every tool_use, so a
    turn killed mid-tool-call -- the normal shape of this failure -- left
    the reply falling through to "join every passage", duplicating the
    whole run underneath the narration he had already watched.
    """
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "reading the vault"}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "merged the PR"}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t2", "name": "Bash", "input": {}}]}},
    )
    text, narrated = run_narrated_until_timeout(tmp_path, lines)
    assert narrated == ["reading the vault", "merged the PR"]
    # The last passage, once -- not the transcript a second time.
    assert "merged the PR" in text
    assert "reading the vault" not in text
    assert text.count("merged the PR") == 1


def test_salvage_keeps_everything_when_nothing_was_narrated(tmp_path):
    """The /invoke path has no reporter, so nothing reached the caller
    live and the join is the only record there is."""
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "first"}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "second"}]}},
    )
    text, _, _ = run_until_timeout(tmp_path, lines)
    assert "first" in text and "second" in text


def test_timeout_returns_what_the_session_managed_to_say(tmp_path):
    """Cycle 81 was killed here having already merged its PR and written
    its journal entry, and the owner was told only "failed: timed out"."""
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "merged the PR"}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Now the digest, re-fetching first:"}]}},
    )
    text, _, _ = run_until_timeout(tmp_path, lines)
    assert "Now the digest, re-fetching first:" in text
    assert "45-minute limit" in text
    assert "survived" in text


def test_the_salvaged_reply_is_labelled_as_truncated(tmp_path):
    """Unlabelled it would arrive looking like a considered reply that
    simply stops -- which is worse than the error it replaces."""
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "half a thought"}]}},
    )
    text, _, _ = run_until_timeout(tmp_path, lines)
    assert text.startswith("_Cut off at this turn's 45-minute limit")
    assert text.rstrip().endswith("half a thought")


def test_timeout_with_nothing_written_still_fails_the_turn(tmp_path):
    """Nothing to salvage is the one case where the old behaviour was
    right: an empty reply is not worth posting."""
    lines = _stream_json_lines({"type": "system", "session_id": "s1", "subtype": "init"})
    with pytest.raises(cli.ClaudeCliError, match="timed out after 2700s"):
        run_until_timeout(tmp_path, lines)


def test_a_normal_turn_is_not_labelled(tmp_path):
    """The mutation that matters: the marker must key off the timeout, not
    be pasted onto every reply."""
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "the answer"}]}},
        {"type": "result", "session_id": "sess-new", "subtype": "success"},
    )
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        text, _, _ = cli.run_turn("hello")
    assert text == "the answer"
    assert "Cut off" not in text


def test_a_finished_turn_leaves_no_clock_behind(tmp_path):
    """Otherwise the next turn's hook reads this turn's expired deadline."""
    path = str(tmp_path / "d.json")
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
        {"type": "result", "session_id": "s", "subtype": "success"},
    )
    with patch.object(deadline, "DEADLINE_FILE", path), \
         patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        cli.run_turn("hello")
        assert deadline.read(path) is None


def test_the_clock_is_running_while_the_turn_is(tmp_path):
    """The complement of the test above: clearing it at the end is only
    correct if it was actually written at the start."""
    path = str(tmp_path / "d.json")
    seen = {}

    class Watching(FakeProc):
        def wait(self, timeout=None):
            seen["record"] = deadline.read(path)

    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
        {"type": "result", "session_id": "s", "subtype": "success"},
    )
    with patch.object(deadline, "DEADLINE_FILE", path), \
         patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", return_value=Watching(lines)):
        cli.run_turn("hello")

    assert seen["record"] is not None
    assert seen["record"]["timeout_seconds"] == cli.CLI_TIMEOUT_SECONDS
