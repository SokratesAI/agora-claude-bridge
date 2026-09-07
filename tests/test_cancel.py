"""The stop button, bridge half: a registry of in-flight CLI subprocesses
and a POST /cancel that reaches into it.

The registry is exercised against *real* subprocesses rather than mocks
wherever the thing under test is "does the process actually die". A mock
Popen would let `cancel()` pass while the signal never left the process,
which is the only failure mode that matters here.
"""
import json
import subprocess
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from bridge import cancel, cli, server
from bridge.cli import TurnCancelled
from tests.test_bridge import FakeProc, _stream_json_lines as _lines


@pytest.fixture(autouse=True)
def clean_registry():
    with cancel._lock:
        cancel._turns.clear()
    yield
    with cancel._lock:
        cancel._turns.clear()


def _sleeper(seconds=30):
    """A real child that exits on SIGTERM, like the CLI does."""
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])


def _ignores_sigterm(tmp_path):
    """A real child that does NOT exit on SIGTERM, so the SIGKILL escalation
    is exercised against something that genuinely needs it rather than
    against a mock that would have died either way.

    It touches `ready` only once SIG_IGN is installed, and the caller waits
    for that file. Without the handshake the SIGTERM can land in the
    interpreter's own startup, before the handler exists -- the child dies
    on the default disposition, the test passes, and it passes just as
    happily with the SIGKILL escalation deleted. Measured: that is exactly
    what happened, and the mutation SURVIVED until this wait went in."""
    ready = tmp_path / "ready"
    proc = subprocess.Popen([
        sys.executable, "-c",
        "import signal, pathlib, time, sys; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).touch(); time.sleep(30)",
        str(ready),
    ])
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.exists(), "the child never installed its SIGTERM handler"
    return proc


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

def test_cancel_kills_the_registered_process():
    proc = _sleeper()
    try:
        cancel.register("conv-1", proc)
        assert cancel.cancel("conv-1") == 1
        assert proc.poll() is not None
    finally:
        proc.kill()
        proc.wait()


def test_cancel_escalates_to_sigkill_when_sigterm_is_ignored(tmp_path):
    proc = _ignores_sigterm(tmp_path)
    try:
        cancel.register("conv-1", proc)
        assert cancel.cancel("conv-1", grace_seconds=0.5) == 1
        proc.wait(timeout=5)
        assert proc.poll() is not None
    finally:
        proc.kill()
        proc.wait()


def test_cancel_marks_the_turn_so_run_turn_can_tell_it_apart():
    proc = _sleeper()
    try:
        turn = cancel.register("conv-1", proc)
        assert turn.cancelled is False
        cancel.cancel("conv-1")
        assert turn.cancelled is True
    finally:
        proc.kill()
        proc.wait()


def test_cancel_with_nothing_in_flight_is_zero_not_an_error():
    assert cancel.cancel("conv-nobody-is-running") == 0


def test_cancel_only_touches_the_conversation_it_was_asked_for():
    mine, theirs = _sleeper(), _sleeper()
    try:
        cancel.register("conv-1", mine)
        cancel.register("conv-2", theirs)
        assert cancel.cancel("conv-1") == 1
        assert mine.poll() is not None
        assert theirs.poll() is None
    finally:
        for p in (mine, theirs):
            p.kill()
            p.wait()


def test_two_concurrent_turns_on_one_conversation_are_both_stopped():
    """allow_concurrent lets a second turn start for a conversation that
    already has one running. A dict of conversation -> one process would
    drop the first on registration and leave it unkillable."""
    first, second = _sleeper(), _sleeper()
    try:
        cancel.register("conv-1", first)
        cancel.register("conv-1", second)
        assert cancel.cancel("conv-1") == 2
        assert first.poll() is not None
        assert second.poll() is not None
    finally:
        for p in (first, second):
            p.kill()
            p.wait()


def test_unregister_removes_the_turn_so_a_later_cancel_finds_nothing():
    proc = MagicMock()
    turn = cancel.register("conv-1", proc)
    cancel.unregister(turn)
    assert cancel.cancel("conv-1") == 0
    proc.terminate.assert_not_called()


def test_unregister_leaves_a_sibling_turn_registered():
    first, second = MagicMock(), MagicMock()
    turn_one = cancel.register("conv-1", first)
    cancel.register("conv-1", second)
    cancel.unregister(turn_one)
    assert len(cancel.active("conv-1")) == 1
    assert cancel.active("conv-1")[0].proc is second


def test_a_turn_with_no_conversation_id_is_not_registered():
    """It could never be addressed by a cancel request, so filing it under
    "" would only build a bucket cancel() must never be asked for."""
    proc = MagicMock()
    turn = cancel.register("", proc)
    assert cancel.active("") == []
    cancel.unregister(turn)  # must not raise
    assert cancel.cancel("") == 0


def test_cancelling_one_turn_does_not_mark_the_next_turn_on_that_conversation():
    """The race the flag-on-the-process design exists to close: a cancel
    that lands as a turn ends must not stop the turn that follows it."""
    first = _sleeper()
    try:
        turn_one = cancel.register("conv-1", first)
        cancel.cancel("conv-1")
        cancel.unregister(turn_one)
        turn_two = cancel.register("conv-1", MagicMock())
        assert turn_two.cancelled is False
    finally:
        first.kill()
        first.wait()


# ---------------------------------------------------------------------------
# the endpoint
# ---------------------------------------------------------------------------

class _FakeHandler(server.BridgeHandler):
    """BridgeHandler without its socket, so the routing and the body
    handling can be driven directly. Same shape test_bridge.py uses."""

    def __init__(self, path, body, token=None):
        self.path = path
        self.headers = {"Content-Length": str(len(body))}
        if token is not None:
            self.headers["x-bridge-token"] = token
        self.rfile = MagicMock()
        self.rfile.read.return_value = body.encode()
        self.sent = []

    def _send(self, status, payload):
        self.sent.append((status, payload))


def _post(path, payload, token=None):
    handler = _FakeHandler(path, json.dumps(payload), token=token)
    with patch.object(server, "BRIDGE_TOKEN", ""):
        handler.do_POST()
    return handler.sent[-1]


def test_cancel_endpoint_stops_the_turn_and_reports_how_many():
    proc = _sleeper()
    try:
        cancel.register("conv-1", proc)
        status, payload = _post("/cancel", {"conversation_id": "conv-1"})
        assert status == 200
        assert payload == {"cancelled": 1}
        assert proc.poll() is not None
    finally:
        proc.kill()
        proc.wait()


def test_cancel_endpoint_answers_200_when_the_turn_already_finished():
    """Not a 404: the turn may have returned in the moment between the owner
    pressing stop and this call arriving, and that is nobody's failure."""
    status, payload = _post("/cancel", {"conversation_id": "conv-gone"})
    assert status == 200
    assert payload == {"cancelled": 0}


def test_cancel_endpoint_requires_a_conversation_id():
    status, payload = _post("/cancel", {})
    assert status == 400
    assert "conversation_id" in payload["error"]


def test_cancel_endpoint_is_token_guarded():
    proc = MagicMock()
    cancel.register("conv-1", proc)
    handler = _FakeHandler("/cancel", json.dumps({"conversation_id": "conv-1"}),
                           token="wrong")
    with patch.object(server, "BRIDGE_TOKEN", "right"):
        handler.do_POST()
    assert handler.sent[-1][0] == 401
    proc.terminate.assert_not_called()


def test_an_unknown_post_path_is_still_404():
    status, _ = _post("/nonsense", {})
    assert status == 404


# ---------------------------------------------------------------------------
# what the caller gets back
# ---------------------------------------------------------------------------

def test_a_cancelled_turn_answers_200_with_stopped_and_the_partial_text():
    """The whole point of SIGTERM-before-SIGKILL: a stopped turn that had
    already written something reports what it wrote, not nothing."""
    handler = _FakeHandler("/generate", json.dumps(
        {"conversation_id": "conv-1", "prompt": "hi"}))
    with patch.object(server, "BRIDGE_TOKEN", ""), \
            patch.object(server, "generate",
                         side_effect=TurnCancelled("half an answer", "some thinking")):
        handler.do_POST()
    status, payload = handler.sent[-1]
    assert status == 200
    assert payload == {"text": "half an answer", "thinking": "some thinking",
                       "stopped": True}


def test_a_cancelled_turn_is_not_reported_as_a_cli_error():
    """A 502 would render as a failure in the thread. The caller asked for
    this, so it is not one."""
    handler = _FakeHandler("/generate", json.dumps(
        {"conversation_id": "conv-1", "prompt": "hi"}))
    with patch.object(server, "BRIDGE_TOKEN", ""), \
            patch.object(server, "generate", side_effect=TurnCancelled("", "")):
        handler.do_POST()
    status, payload = handler.sent[-1]
    assert status == 200
    assert payload["stopped"] is True
    assert "error" not in payload


def test_an_ordinary_reply_carries_no_stopped_flag():
    """The negative control: `stopped` must be absent on a normal turn, or
    a caller keying on its presence would mark every turn stopped."""
    handler = _FakeHandler("/generate", json.dumps(
        {"conversation_id": "conv-1", "prompt": "hi"}))
    with patch.object(server, "BRIDGE_TOKEN", ""), \
            patch.object(server, "generate", return_value=("a real answer", "")):
        handler.do_POST()
    status, payload = handler.sent[-1]
    assert status == 200
    assert "stopped" not in payload


def test_the_drain_still_refuses_a_new_turn_while_shutting_down():
    """The cancel path added a second POST route; the shutdown guard must
    still sit on /generate and not have moved above the routing."""
    handler = _FakeHandler("/generate", json.dumps(
        {"conversation_id": "conv-1", "prompt": "hi"}))
    with patch.object(server, "BRIDGE_TOKEN", ""), \
            patch.object(server, "_enter_turn", return_value=False):
        handler.do_POST()
    assert handler.sent[-1][0] == 503


# ---------------------------------------------------------------------------
# run_turn: what a cancelled turn does with the words it already had
# ---------------------------------------------------------------------------

class _BlockingProc:
    """A CLI whose stream stops mid-turn and only ends when it is signalled,
    which is the real shape of this: the owner presses stop while the model
    is still writing. `terminate` releases the reader, exactly as closing a
    real child's stdout would."""

    def __init__(self, lines, trailing=()):
        self._lines = list(lines)
        self._trailing = list(trailing)
        self._released = threading.Event()
        self.streaming = threading.Event()
        self.returncode = -15
        self.stdout = self
        self.terminated = False

    def __iter__(self):
        for line in self._lines:
            yield line
        self.streaming.set()
        self._released.wait(timeout=10)
        for line in self._trailing:
            yield line

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self._released.set()

    def kill(self):
        self.terminate()

    def poll(self):
        return self.returncode if self._released.is_set() else None

    def wait(self, timeout=None):
        self._released.wait(timeout=timeout)
        return self.returncode

    def close(self):
        pass


def _run_turn_and_cancel(tmp_path, proc, conversation_id="conv-cancel-me"):
    """Start run_turn against `proc` on a thread, wait until it is really
    streaming, then cancel it the way POST /cancel does."""
    box = {}

    def run():
        try:
            box["result"] = cli.run_turn(
                "hello", session_id=None, conversation_id=conversation_id)
        except BaseException as exc:  # noqa: BLE001 -- the test inspects it
            box["error"] = exc

    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
            patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
            patch.object(cli.subprocess, "Popen", return_value=proc):
        thread = threading.Thread(target=run)
        thread.start()
        assert proc.streaming.wait(timeout=10), "run_turn never started streaming"
        stopped = cancel.cancel(conversation_id, grace_seconds=1)
        thread.join(timeout=15)
    assert not thread.is_alive()
    return box, stopped


def test_a_cancelled_run_turn_raises_with_the_text_it_had_already_written(tmp_path):
    proc = _BlockingProc(_lines(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "I was halfway through when you stopped me"}]}},
    ))
    box, stopped = _run_turn_and_cancel(tmp_path, proc)
    assert stopped == 1
    assert proc.terminated
    assert isinstance(box.get("error"), TurnCancelled), box
    assert box["error"].text == "I was halfway through when you stopped me"


def test_a_cancelled_run_turn_is_not_reported_as_a_cli_error(tmp_path):
    """A CLI shutting down under SIGTERM can emit an error_during_execution
    on its way out. Reporting that as a failure would lose the salvage and
    put a red error in the thread instead of the word "stopped"."""
    proc = _BlockingProc(
        _lines({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "partial work"}]}}),
        trailing=_lines({"type": "result", "subtype": "error_during_execution",
                         "errors": ["interrupted"]}),
    )
    box, _ = _run_turn_and_cancel(tmp_path, proc)
    assert isinstance(box.get("error"), TurnCancelled), box
    assert box["error"].text == "partial work"


def test_a_turn_that_was_not_cancelled_returns_normally(tmp_path):
    """The positive control. Without it every assertion above would also
    pass on a run_turn that raised TurnCancelled unconditionally."""
    proc = FakeProc(_lines(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "the whole answer"}]}},
        {"type": "result", "session_id": "sess-new", "subtype": "success"},
    ))
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
            patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
            patch.object(cli.subprocess, "Popen", return_value=proc):
        text, _, _ = cli.run_turn("hello", session_id=None, conversation_id="conv-fine")
    assert text == "the whole answer"
    assert cancel.active("conv-fine") == []


def test_run_turn_deregisters_the_turn_when_it_finishes(tmp_path):
    """A registry that only ever grows would let a cancel signal a process
    that exited hours ago -- and on a recycled pid, something else."""
    proc = FakeProc(_lines(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "done"}]}},
        {"type": "result", "session_id": "s", "subtype": "success"},
    ))
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
            patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
            patch.object(cli.subprocess, "Popen", return_value=proc):
        cli.run_turn("hello", session_id=None, conversation_id="conv-done")
    assert cancel.active("conv-done") == []


def test_run_turn_deregisters_the_turn_even_when_it_raises(tmp_path):
    proc = FakeProc(_lines(
        {"type": "result", "subtype": "error_during_execution",
         "errors": ["something broke"]},
    ))
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
            patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
            patch.object(cli.subprocess, "Popen", return_value=proc):
        with pytest.raises(Exception):
            cli.run_turn("hello", session_id=None, conversation_id="conv-broke")
    assert cancel.active("conv-broke") == []
