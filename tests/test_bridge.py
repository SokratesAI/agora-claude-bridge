import json
import os
import tempfile
import threading
from unittest.mock import patch, MagicMock

import pytest

from bridge import cli, sessions, server


# ---------------------------------------------------------------------------
# sessions.py -- JSON-file-backed conversation_id -> session_id mapping
# ---------------------------------------------------------------------------

@pytest.fixture
def sessions_file(tmp_path):
    path = str(tmp_path / "agora-sessions.json")
    with patch.object(sessions, "SESSIONS_FILE", path):
        yield path


def test_get_session_id_returns_none_when_file_missing(sessions_file):
    assert sessions.get_session_id("conv-1") is None


def test_set_then_get_session_id(sessions_file):
    sessions.set_session_id("conv-1", "sess-abc")
    assert sessions.get_session_id("conv-1") == "sess-abc"


def test_sessions_persist_across_reloads(sessions_file):
    """Simulates a pod restart -- a fresh read must still see prior writes."""
    sessions.set_session_id("conv-1", "sess-abc")
    sessions.set_session_id("conv-2", "sess-xyz")
    assert sessions.get_session_id("conv-1") == "sess-abc"
    assert sessions.get_session_id("conv-2") == "sess-xyz"
    with open(sessions_file) as f:
        data = json.load(f)
    assert data == {"conv-1": "sess-abc", "conv-2": "sess-xyz"}


def test_set_session_id_overwrites_existing(sessions_file):
    sessions.set_session_id("conv-1", "sess-old")
    sessions.set_session_id("conv-1", "sess-new")
    assert sessions.get_session_id("conv-1") == "sess-new"


def test_clear_session_id_removes_entry(sessions_file):
    sessions.set_session_id("conv-1", "sess-abc")
    sessions.clear_session_id("conv-1")
    assert sessions.get_session_id("conv-1") is None


def test_clear_session_id_on_unknown_conversation_is_a_noop(sessions_file):
    sessions.clear_session_id("never-existed")  # must not raise
    assert sessions.get_session_id("never-existed") is None


def test_get_session_id_survives_corrupt_file(sessions_file):
    with open(sessions_file, "w") as f:
        f.write("not json{{{")
    assert sessions.get_session_id("conv-1") is None


# ---------------------------------------------------------------------------
# cli.py -- subprocess invocation + stream-json parsing
# ---------------------------------------------------------------------------

class FakeStdout:
    def __init__(self, lines):
        self._iter = iter(lines)

    def __iter__(self):
        return self._iter

    def close(self):
        pass


class FakeProc:
    def __init__(self, lines, returncode=0):
        self.stdout = FakeStdout(lines)
        self.returncode = returncode
        self._waited = False

    def wait(self, timeout=None):
        self._waited = True


def _stream_json_lines(*events):
    return [json.dumps(e) + "\n" for e in events]


@pytest.fixture(autouse=True)
def _reset_invocation_lock():
    # Each test gets a fresh, unlocked lock -- a prior test raising inside
    # the `with` block would otherwise leave it held (it wouldn't, since
    # `with` always releases, but this keeps tests independent regardless).
    cli._invocation_lock = threading.Lock()
    yield


def test_run_turn_returns_text_and_thinking(tmp_path):
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "let me think..."},
            {"type": "text", "text": "the answer"},
        ]}},
        {"type": "result", "session_id": "sess-new", "subtype": "success"},
    )
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        text, thinking, session_id = cli.run_turn("hello", session_id=None, model="claude-haiku")
    assert text == "the answer"
    assert thinking == "let me think..."
    assert session_id == "sess-new"


def test_run_turn_with_no_thinking_block_returns_empty_thinking(tmp_path):
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "plain answer"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        text, thinking, _ = cli.run_turn("hello")
    assert text == "plain answer"
    assert thinking == ""


def test_run_turn_passes_resume_flag_when_session_id_given(tmp_path):
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc(lines)

    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", side_effect=fake_popen):
        cli.run_turn("hello", session_id="sess-existing")
    assert "--resume" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--resume") + 1] == "sess-existing"


def test_run_turn_omits_resume_flag_when_no_session_id(tmp_path):
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc(lines)

    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", side_effect=fake_popen):
        cli.run_turn("hello", session_id=None)
    assert "--resume" not in captured["cmd"]


def test_run_turn_always_disallows_filesystem_bash_tools(tmp_path):
    """Chat mode (v1) -- defense in depth even though there's no other
    reason for the model to reach for these tools."""
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc(lines)

    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", side_effect=fake_popen):
        cli.run_turn("hello")
    assert "--disallowedTools" in captured["cmd"]
    disallowed = captured["cmd"][captured["cmd"].index("--disallowedTools") + 1]
    assert "Bash" in disallowed and "Write" in disallowed


def test_run_turn_raises_claude_cli_error_when_no_text_produced(tmp_path):
    lines = _stream_json_lines({"type": "result", "session_id": "sess-1", "subtype": "success"})
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        with pytest.raises(cli.ClaudeCliError):
            cli.run_turn("hello")


def test_run_turn_raises_claude_cli_error_with_session_not_found_sentinel(tmp_path):
    lines = _stream_json_lines(
        {"type": "result", "subtype": "error_during_execution",
         "errors": ["No conversation found with session ID: sess-gone"]},
    )
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        with pytest.raises(cli.ClaudeCliError, match=cli.SESSION_NOT_FOUND):
            cli.run_turn("hello", session_id="sess-gone")


def test_run_turn_raises_usage_limit_error_on_usage_cap_text(tmp_path):
    lines = _stream_json_lines(
        {"type": "result", "subtype": "error_during_execution",
         "errors": ["You've hit your usage limit, resets in 4 hours"]},
    )
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        with pytest.raises(cli.UsageLimitError):
            cli.run_turn("hello")


def test_run_turn_raises_claude_cli_error_on_other_execution_errors(tmp_path):
    lines = _stream_json_lines(
        {"type": "result", "subtype": "error_during_execution",
         "errors": ["Something else went wrong entirely"]},
    )
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        with pytest.raises(cli.ClaudeCliError, match="Something else"):
            cli.run_turn("hello")


def test_run_turn_ignores_non_json_lines(tmp_path):
    lines = [
        "not json at all\n",
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "answer"}]}}) + "\n",
        json.dumps({"type": "result", "session_id": "sess-1", "subtype": "success"}) + "\n",
    ]
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        text, _, _ = cli.run_turn("hello")
    assert text == "answer"


def test_run_turn_serializes_via_module_level_lock(tmp_path):
    """The lock must actually be acquired around the real subprocess call --
    verified by patching Popen to assert the lock is held at call time."""
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    locked_during_call = {}

    def fake_popen(cmd, **kwargs):
        locked_during_call["held"] = cli._invocation_lock.locked()
        return FakeProc(lines)

    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", side_effect=fake_popen):
        cli.run_turn("hello")
    assert locked_during_call["held"] is True
    assert cli._invocation_lock.locked() is False  # released after


# ---------------------------------------------------------------------------
# server.py -- generate() orchestration + HTTP handler
# ---------------------------------------------------------------------------

def test_generate_prepends_system_prompt_on_first_turn():
    captured = {}

    def fake_run_turn(message, session_id=None, model=None):
        captured["message"] = message
        captured["session_id"] = session_id
        return "reply", "", "sess-new"

    with patch.object(server, "get_session_id", return_value=None), \
         patch.object(server, "set_session_id") as mock_set, \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        text, thinking = server.generate("conv-1", "You are helpful.", "hi there")

    assert captured["message"] == "You are helpful.\n\nhi there"
    assert captured["session_id"] is None
    assert text == "reply"
    mock_set.assert_called_once_with("conv-1", "sess-new")


def test_generate_sends_only_new_message_on_resumed_turn():
    captured = {}

    def fake_run_turn(message, session_id=None, model=None):
        captured["message"] = message
        captured["session_id"] = session_id
        return "reply2", "", "sess-existing"

    with patch.object(server, "get_session_id", return_value="sess-existing"), \
         patch.object(server, "set_session_id"), \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        server.generate("conv-1", "You are helpful.", "second message")

    assert captured["message"] == "second message"
    assert captured["session_id"] == "sess-existing"


def test_generate_retries_fresh_on_session_not_found():
    calls = []

    def fake_run_turn(message, session_id=None, model=None):
        calls.append((message, session_id))
        if session_id == "sess-gone":
            raise server.ClaudeCliError(server.SESSION_NOT_FOUND)
        return "recovered reply", "", "sess-brand-new"

    with patch.object(server, "get_session_id", return_value="sess-gone"), \
         patch.object(server, "clear_session_id") as mock_clear, \
         patch.object(server, "set_session_id") as mock_set, \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        text, _ = server.generate("conv-1", "system", "hi")

    assert text == "recovered reply"
    assert len(calls) == 2
    assert calls[0] == ("hi", "sess-gone")
    assert calls[1] == ("system\n\nhi", None)  # retried fresh, system re-sent
    mock_clear.assert_called_once_with("conv-1")
    mock_set.assert_called_once_with("conv-1", "sess-brand-new")


def test_generate_propagates_other_cli_errors_without_retry():
    def fake_run_turn(message, session_id=None, model=None):
        raise server.ClaudeCliError("a real bug")

    with patch.object(server, "get_session_id", return_value="sess-1"), \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        with pytest.raises(server.ClaudeCliError, match="a real bug"):
            server.generate("conv-1", "system", "hi")


class _FakeRequest:
    """Minimal socket-like object BaseHTTPRequestHandler needs at init."""
    def makefile(self, *a, **k):
        import io
        return io.BytesIO()


def _make_handler(body, path="/generate", headers=None):
    handler = server.BridgeHandler.__new__(server.BridgeHandler)
    handler.path = path
    handler.headers = headers or {}
    handler.rfile = __import__("io").BytesIO(json.dumps(body).encode())
    handler.headers = {"Content-Length": str(len(json.dumps(body).encode())), **(headers or {})}
    sent = {}

    def fake_send(status, payload):
        sent["status"] = status
        sent["payload"] = payload
    handler._send = fake_send
    return handler, sent


def test_do_post_requires_conversation_id_and_prompt():
    handler, sent = _make_handler({"prompt": "hi"})  # missing conversation_id
    with patch.object(server, "BRIDGE_TOKEN", ""):
        handler.do_POST()
    assert sent["status"] == 400


def test_do_post_rejects_wrong_bridge_token():
    handler, sent = _make_handler(
        {"conversation_id": "c1", "prompt": "hi"}, headers={"x-bridge-token": "wrong"},
    )
    with patch.object(server, "BRIDGE_TOKEN", "correct-token"):
        handler.do_POST()
    assert sent["status"] == 401


def test_do_post_success_returns_text_and_thinking():
    handler, sent = _make_handler({"conversation_id": "c1", "prompt": "hi", "system": "sys"})
    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", return_value=("the answer", "some thought")):
        handler.do_POST()
    assert sent["status"] == 200
    assert sent["payload"] == {"text": "the answer", "thinking": "some thought"}


def test_do_post_usage_limit_returns_429():
    handler, sent = _make_handler({"conversation_id": "c1", "prompt": "hi"})
    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", side_effect=server.UsageLimitError("resets in 3h")):
        handler.do_POST()
    assert sent["status"] == 429
    assert sent["payload"]["error"] == "usage_limit"


def test_do_post_cli_error_returns_502():
    handler, sent = _make_handler({"conversation_id": "c1", "prompt": "hi"})
    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", side_effect=server.ClaudeCliError("boom")):
        handler.do_POST()
    assert sent["status"] == 502
    assert sent["payload"]["error"] == "cli_error"


def test_do_get_health_returns_ok():
    handler, sent = _make_handler({}, path="/health")
    handler.do_GET()
    assert sent["status"] == 200
    assert sent["payload"] == {"status": "ok"}


def test_do_post_unknown_path_returns_404():
    handler, sent = _make_handler({}, path="/nonsense")
    with patch.object(server, "BRIDGE_TOKEN", ""):
        handler.do_POST()
    assert sent["status"] == 404
