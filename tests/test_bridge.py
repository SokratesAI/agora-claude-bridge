import json
import os
import signal
import stat as stat_module
import tempfile
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from bridge import activity, cli, sessions, server, credentials


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


def test_run_turn_is_unrestricted_by_default(tmp_path):
    """2026-08-01 design reversal: no --disallowedTools flag at all unless
    a caller explicitly asks for restriction -- the old always-on denylist
    was live-tested and found incomplete (the model found and used an
    unlisted tool, "Monitor", to run real shell commands anyway)."""
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
    assert "--disallowedTools" not in captured["cmd"]


def test_run_turn_applies_disallowed_tools_when_given(tmp_path):
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
        cli.run_turn("hello", disallowed_tools="Bash,Write")
    assert "--disallowedTools" in captured["cmd"]
    disallowed = captured["cmd"][captured["cmd"].index("--disallowedTools") + 1]
    assert disallowed == "Bash,Write"


def test_discovered_full_tool_roster_covers_the_tool_found_live():
    """Monitor is the specific tool the model used live to escape the old
    incomplete denylist -- must be in the roster anyone restricting a call
    would actually use."""
    assert "Monitor" in cli.DISCOVERED_FULL_TOOL_ROSTER
    assert "Bash" in cli.DISCOVERED_FULL_TOOL_ROSTER


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

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None):
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

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None):
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

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None):
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
    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None):
        raise server.ClaudeCliError("a real bug")

    with patch.object(server, "get_session_id", return_value="sess-1"), \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        with pytest.raises(server.ClaudeCliError, match="a real bug"):
            server.generate("conv-1", "system", "hi")


def test_generate_is_unrestricted_by_default():
    captured = {}

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None):
        captured["disallowed_tools"] = disallowed_tools
        return "reply", "", "sess-1"

    with patch.object(server, "get_session_id", return_value=None), \
         patch.object(server, "set_session_id"), \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        server.generate("conv-1", "system", "hi")

    assert captured["disallowed_tools"] is None


def test_generate_restricted_true_passes_the_full_tool_roster():
    captured = {}

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None):
        captured["disallowed_tools"] = disallowed_tools
        return "reply", "", "sess-1"

    with patch.object(server, "get_session_id", return_value=None), \
         patch.object(server, "set_session_id"), \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        server.generate("conv-1", "system", "hi", restricted=True)

    assert captured["disallowed_tools"] == server.DISCOVERED_FULL_TOOL_ROSTER


def test_generate_stateless_always_sends_full_system_and_no_resume():
    captured = {}

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None):
        captured["message"] = message
        captured["session_id"] = session_id
        return "reply", "", "sess-should-be-ignored"

    with patch.object(server, "get_session_id") as mock_get, \
         patch.object(server, "set_session_id") as mock_set, \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        text, _ = server.generate("conv-1", "system prompt", "hi", stateless=True)

    assert text == "reply"
    assert captured["message"] == "system prompt\n\nhi"
    assert captured["session_id"] is None
    mock_get.assert_not_called()  # never even looks up a stored session
    mock_set.assert_not_called()  # and never persists the new one


def test_generate_stateless_ignores_a_stored_session_for_the_same_conversation():
    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None):
        return "reply", "", "sess-x"

    with patch.object(server, "get_session_id", return_value="sess-existing") as mock_get, \
         patch.object(server, "set_session_id") as mock_set, \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        server.generate("conv-1", "system", "second call", stateless=True)

    mock_get.assert_not_called()
    mock_set.assert_not_called()


def test_generate_stateless_can_combine_with_restricted():
    captured = {}

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None):
        captured["disallowed_tools"] = disallowed_tools
        return "reply", "", "sess-1"

    with patch.object(server, "run_turn", side_effect=fake_run_turn):
        server.generate("conv-1", "system", "hi", restricted=True, stateless=True)

    assert captured["disallowed_tools"] == server.DISCOVERED_FULL_TOOL_ROSTER


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


def test_do_post_passes_restricted_flag_through_to_generate():
    handler, sent = _make_handler(
        {"conversation_id": "c1", "prompt": "hi", "system": "sys", "restricted": True},
    )
    captured = {}

    def fake_generate(conversation_id, system, prompt, model=None, restricted=False, stateless=False,
                      mcp=None,
                      activity=None):
        captured["restricted"] = restricted
        return "answer", ""

    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", side_effect=fake_generate):
        handler.do_POST()
    assert captured["restricted"] is True


def test_do_post_restricted_defaults_false_when_omitted():
    handler, sent = _make_handler({"conversation_id": "c1", "prompt": "hi"})
    captured = {}

    def fake_generate(conversation_id, system, prompt, model=None, restricted=False, stateless=False,
                      mcp=None,
                      activity=None):
        captured["restricted"] = restricted
        return "answer", ""

    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", side_effect=fake_generate):
        handler.do_POST()
    assert captured["restricted"] is False


def test_do_post_passes_stateless_flag_through_to_generate():
    handler, sent = _make_handler(
        {"conversation_id": "c1", "prompt": "hi", "system": "sys", "stateless": True},
    )
    captured = {}

    def fake_generate(conversation_id, system, prompt, model=None, restricted=False, stateless=False,
                      mcp=None,
                      activity=None):
        captured["stateless"] = stateless
        return "answer", ""

    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", side_effect=fake_generate):
        handler.do_POST()
    assert captured["stateless"] is True


def test_do_post_stateless_defaults_false_when_omitted():
    handler, sent = _make_handler({"conversation_id": "c1", "prompt": "hi"})
    captured = {}

    def fake_generate(conversation_id, system, prompt, model=None, restricted=False, stateless=False,
                      mcp=None,
                      activity=None):
        captured["stateless"] = stateless
        return "answer", ""

    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", side_effect=fake_generate):
        handler.do_POST()
    assert captured["stateless"] is False


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
    # 200 is the whole contract as far as the readiness probe is concerned;
    # "draining" was added alongside the SIGTERM drain below so the state is
    # queryable without shell access to the pod's logs.
    assert sent["payload"]["status"] == "ok"


def test_do_post_unknown_path_returns_404():
    handler, sent = _make_handler({}, path="/nonsense")
    with patch.object(server, "BRIDGE_TOKEN", ""):
        handler.do_POST()
    assert sent["status"] == 404


# ---------------------------------------------------------------------------
# credentials.py -- first-boot-only .credentials.json bootstrap, piping the
# claude-auth secret's raw JSON straight to disk unmodified. Deliberately
# never overwrites an existing file -- see the module's own docstring for
# why (a live token refresh only ever lands on the PVC, never back in the
# k8s Secret).
# ---------------------------------------------------------------------------

REAL_CREDS_JSON = json.dumps({
    "claudeAiOauth": {
        "accessToken": "sk-ant-oat-test",
        "refreshToken": "sk-ant-ort-test",
        "expiresAt": 1785608541000,
        "scopes": ["user:inference", "user:profile"],
        "subscriptionType": "max",
    }
})


def test_bootstrap_credentials_writes_raw_json_unmodified(tmp_path):
    with patch.object(credentials, "CLAUDE_HOME", str(tmp_path)), \
         patch.dict(os.environ, {"CLAUDE_CREDENTIALS_JSON": REAL_CREDS_JSON}, clear=False):
        credentials.bootstrap_credentials()

    dest = tmp_path / ".claude" / ".credentials.json"
    assert dest.exists()
    # Byte-identical, not re-serialized -- no field-by-field reconstruction
    # that could silently drop fields the CLI's own validation needs.
    assert dest.read_text() == REAL_CREDS_JSON
    assert json.loads(dest.read_text())["claudeAiOauth"]["scopes"] == ["user:inference", "user:profile"]


def test_bootstrap_credentials_sets_owner_only_permissions(tmp_path):
    with patch.object(credentials, "CLAUDE_HOME", str(tmp_path)), \
         patch.dict(os.environ, {"CLAUDE_CREDENTIALS_JSON": REAL_CREDS_JSON}, clear=False):
        credentials.bootstrap_credentials()

    dest = tmp_path / ".claude" / ".credentials.json"
    mode = stat_module.S_IMODE(os.stat(dest).st_mode)
    assert mode == 0o600


def test_bootstrap_credentials_never_overwrites_existing_file(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir(parents=True)
    dest = claude_dir / ".credentials.json"
    dest.write_text('{"claudeAiOauth": {"accessToken": "already-refreshed-by-cli"}}')

    with patch.object(credentials, "CLAUDE_HOME", str(tmp_path)), \
         patch.dict(os.environ, {"CLAUDE_CREDENTIALS_JSON": REAL_CREDS_JSON}, clear=False):
        credentials.bootstrap_credentials()

    data = json.loads(dest.read_text())
    assert data["claudeAiOauth"]["accessToken"] == "already-refreshed-by-cli"


def test_bootstrap_credentials_skips_when_env_var_missing(tmp_path):
    with patch.object(credentials, "CLAUDE_HOME", str(tmp_path)), \
         patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CLAUDE_CREDENTIALS_JSON", None)
        credentials.bootstrap_credentials()

    assert not (tmp_path / ".claude" / ".credentials.json").exists()


def test_bootstrap_credentials_skips_on_invalid_json(tmp_path):
    with patch.object(credentials, "CLAUDE_HOME", str(tmp_path)), \
         patch.dict(os.environ, {"CLAUDE_CREDENTIALS_JSON": "not valid json{{{"}, clear=False):
        credentials.bootstrap_credentials()

    assert not (tmp_path / ".claude" / ".credentials.json").exists()


# ---------------------------------------------------------------------------
# activity.py -- live tool-use narration back to agora-persona-runner
# ---------------------------------------------------------------------------

def test_summarize_uses_the_field_that_says_what_the_call_did():
    assert activity.summarize("Bash", {"command": "ls -la", "description": "list"}) == "ls -la"
    assert activity.summarize("Read", {"file_path": "/etc/hosts", "limit": 5}) == "/etc/hosts"
    assert activity.summarize("Grep", {"pattern": "TODO", "path": "."}) == "TODO"


def test_summarize_collapses_whitespace_so_a_chip_stays_one_line():
    """A heredoc or a formatted patch would otherwise blow out a chat bubble."""
    assert activity.summarize("Bash", {"command": "git commit -m 'a\n\nb'   c"}) == \
        "git commit -m 'a b' c"


def test_summarize_truncates_to_the_chip_budget():
    summary = activity.summarize("Bash", {"command": "x" * 5000})
    assert len(summary) == activity.DETAIL_CHARS_MAX


def test_summarize_falls_back_to_the_whole_input_for_an_unknown_tool():
    """New tools appear between CLI versions -- an unlisted one must still
    say something useful rather than rendering a blank chip."""
    summary = activity.summarize("SomeToolShippedNextVersion", {"target": "prod", "n": 3})
    assert "prod" in summary


def test_summarize_returns_empty_for_a_tool_with_no_input():
    assert activity.summarize("TodoWrite", {}) == ""
    assert activity.summarize("Bash", None) == ""


def test_reporter_is_disabled_without_an_activity_block():
    """An older runner sends no activity block at all -- that must be a
    silent no-op, not a crash and not a post to nowhere."""
    for block in (None, {}, {"url": "http://x/y"}, {"token": "t"}):
        reporter = activity.ActivityReporter(block)
        assert not reporter.enabled
        reporter.start()
        reporter.report("Bash", {"command": "ls"})
        reporter.close()  # must not hang or raise


def test_reporter_posts_each_tool_call_with_its_token():
    posted = []
    with patch.object(activity, "_post", lambda url, payload: posted.append((url, payload)) or True):
        reporter = activity.ActivityReporter({"url": "http://runner/tool-activity", "token": "tok"})
        reporter.start()
        reporter.report("Bash", {"command": "pytest"})
        reporter.report("Read", {"file_path": "/tmp/f"})
        reporter.close()

    assert [p[0] for p in posted] == ["http://runner/tool-activity"] * 2
    assert [p[1] for p in posted] == [
        {"token": "tok", "capability": "Bash", "detail": "pytest"},
        {"token": "tok", "capability": "Read", "detail": "/tmp/f"},
    ]


def test_reporter_preserves_tool_call_order():
    """Order is the entire point of narrating live rather than dumping at
    the end -- a thread per report would race and shuffle them."""
    posted = []
    with patch.object(activity, "_post", lambda url, payload: posted.append(payload["detail"]) or True):
        reporter = activity.ActivityReporter({"url": "http://runner/x", "token": "tok"})
        reporter.start()
        for i in range(50):
            reporter.report("Bash", {"command": f"step-{i}"})
        reporter.close()
    assert posted == [f"step-{i}" for i in range(50)]


def test_reporter_survives_a_runner_that_is_down():
    """The turn being narrated matters more than the narration."""
    with patch.object(activity, "_post", side_effect=RuntimeError("connection refused")):
        reporter = activity.ActivityReporter({"url": "http://runner/x", "token": "tok"})
        reporter.start()
        reporter.report("Bash", {"command": "ls"})
        reporter.close()  # must not raise


def test_post_returns_false_instead_of_raising_on_a_dead_runner():
    with patch.object(activity.urllib.request, "urlopen", side_effect=OSError("no route")):
        assert activity._post("http://runner/x", {"token": "t"}) is False


def test_reporter_keeps_narrating_after_one_post_fails():
    """A single bad post must not end the session's narration -- the worker
    dying quietly is the exact fails-invisibly shape this feature removes."""
    posted = []
    calls = {"n": 0}

    def flaky_post(url, payload):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("connection reset")
        posted.append(payload["detail"])
        return True

    with patch.object(activity, "_post", flaky_post):
        reporter = activity.ActivityReporter({"url": "http://runner/x", "token": "tok"})
        reporter.start()
        reporter.report("Bash", {"command": "first"})
        reporter.report("Bash", {"command": "second"})
        reporter.report("Bash", {"command": "third"})
        reporter.close()

    assert posted == ["second", "third"]


# ---------------------------------------------------------------------------
# cli.py + server.py -- activity plumbed through a real turn
# ---------------------------------------------------------------------------

def test_run_turn_reports_each_tool_use_block(tmp_path):
    posted = []
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "pytest tests/"}},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/app/x.py"}},
            {"type": "text", "text": "done"},
        ]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        text, _, _ = cli.run_turn(
            "hello", activity={"url": "http://runner/tool-activity", "token": "tok"})

    assert text == "done"
    assert [(p["capability"], p["detail"]) for p in posted] == [
        ("Bash", "pytest tests/"),
        ("Read", "/app/x.py"),
    ]


def test_run_turn_reports_nothing_when_no_activity_block_is_given(tmp_path):
    posted = []
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
            {"type": "text", "text": "ok"},
        ]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        text, _, _ = cli.run_turn("hello")

    assert text == "ok"
    assert posted == []


def test_tool_use_is_reported_while_the_session_is_still_running(tmp_path):
    """The whole point. A chip that lands after the turn returns is the
    'displayed after the process is finished... hindsight logging' Edvard
    complained about -- so assert the post happens before the CLI has even
    emitted its next event, not merely that it happens at some point."""
    posted = []
    first_post_landed = threading.Event()

    def fake_post(url, payload):
        posted.append(payload["detail"])
        first_post_landed.set()
        return True

    def lazy_stream():
        yield json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "slow-thing"}},
        ]}}) + "\n"
        assert first_post_landed.wait(timeout=10), "tool call was not reported live"
        yield json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "finished"},
        ]}}) + "\n"
        yield json.dumps({"type": "result", "session_id": "s", "subtype": "success"}) + "\n"

    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(activity, "_post", fake_post), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lazy_stream())):
        text, _, _ = cli.run_turn(
            "hello", activity={"url": "http://runner/x", "token": "tok"})

    assert text == "finished"
    assert posted == ["slow-thing"]


def test_a_broken_activity_endpoint_does_not_break_the_turn(tmp_path):
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
            {"type": "text", "text": "the answer"},
        ]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(activity.urllib.request, "urlopen", side_effect=OSError("refused")), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        text, _, session_id = cli.run_turn(
            "hello", activity={"url": "http://runner/x", "token": "tok"})

    assert text == "the answer"
    assert session_id == "sess-1"


def test_do_post_passes_activity_through_to_generate():
    handler, sent = _make_handler({
        "conversation_id": "c1", "prompt": "hi", "system": "sys",
        "activity": {"url": "http://runner/tool-activity", "token": "tok"},
    })
    captured = {}

    def fake_generate(conversation_id, system, prompt, model=None, restricted=False, mcp=None,
                      stateless=False, activity=None):
        captured["activity"] = activity
        return "answer", ""

    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", fake_generate):
        handler.do_POST()
    assert sent["status"] == 200
    assert captured["activity"] == {"url": "http://runner/tool-activity", "token": "tok"}


def test_do_post_activity_defaults_to_none_when_omitted():
    """An older runner sends no activity block; the bridge must still work."""
    handler, sent = _make_handler({"conversation_id": "c1", "prompt": "hi", "system": "sys"})
    captured = {}

    def fake_generate(conversation_id, system, prompt, model=None, restricted=False, mcp=None,
                      stateless=False, activity=None):
        captured["activity"] = activity
        return "answer", ""

    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", fake_generate):
        handler.do_POST()
    assert sent["status"] == 200
    assert captured["activity"] is None


# ---------------------------------------------------------------------------
# activity.py -- what a tool RETURNED (Edvard's issue 1, asked three times:
# "I need to see the command with all metadata and also the output from that
# command, such as the return of a echo command")
# ---------------------------------------------------------------------------

def test_result_text_reads_a_plain_string_result():
    assert activity.result_text({"content": "hello from echo\n"}) == "hello from echo\n"


def test_result_text_joins_the_blocks_of_a_structured_result():
    block = {"content": [
        {"type": "text", "text": "line one"},
        {"type": "text", "text": "line two"},
    ]}
    assert activity.result_text(block) == "line one\nline two"


def test_result_text_names_a_non_text_block_instead_of_dropping_it():
    """An empty result and a screenshot must not look identical -- dropping
    the block would make a tool that returned an image indistinguishable
    from one that returned nothing at all."""
    block = {"content": [{"type": "image", "source": {"data": "..."}}]}
    assert activity.result_text(block) == "[image]"


def test_result_text_truncates_to_agoras_own_content_ceiling():
    """Not a limit of ours: anything past this is sliced off by
    AuditStore.CONTENT_CHARS_MAX on arrival, so sending it would push bytes
    across two hops to be provably discarded."""
    out = activity.result_text({"content": "x" * 50_000})
    assert len(out) == activity.OUTPUT_CHARS_MAX


def test_result_text_is_empty_for_a_result_with_no_content():
    assert activity.result_text({}) == ""
    assert activity.result_text({"content": None}) == ""


def test_report_result_carries_the_output_and_its_correlation_id():
    posted = []
    with patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True):
        reporter = activity.ActivityReporter({"url": "http://runner/x", "token": "tok"})
        reporter.start()
        reporter.report_result("Bash", "toolu_1", "hello\n", is_error=False)
        reporter.close()
    assert posted == [{
        "token": "tok", "capability": "Bash", "toolUseId": "toolu_1",
        "output": "hello\n", "isError": False,
    }]


def test_report_result_marks_a_failed_tool_call():
    posted = []
    with patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True):
        reporter = activity.ActivityReporter({"url": "http://runner/x", "token": "tok"})
        reporter.start()
        reporter.report_result("Bash", "toolu_9", "command not found", is_error=True)
        reporter.close()
    assert posted[0]["isError"] is True


def test_report_result_without_a_correlation_id_is_dropped():
    """Nothing downstream could pair it with its call, so it would render as
    a second, orphaned chip with no label."""
    posted = []
    with patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True):
        reporter = activity.ActivityReporter({"url": "http://runner/x", "token": "tok"})
        reporter.start()
        reporter.report_result("Bash", "", "output", is_error=False)
        reporter.close()
    assert posted == []


def test_run_turn_pairs_each_tool_result_with_the_call_that_made_it(tmp_path):
    """The end-to-end shape: the call is reported when it starts (so the
    chip is live) and its output follows under the same toolUseId."""
    posted = []
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "toolu_a", "name": "Bash",
             "input": {"command": "echo hi"}},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "toolu_a", "content": "hi\n"},
        ]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        text, _, _ = cli.run_turn(
            "hello", activity={"url": "http://runner/x", "token": "tok"})

    assert text == "done"
    assert posted == [
        {"token": "tok", "capability": "Bash", "detail": "echo hi", "toolUseId": "toolu_a"},
        {"token": "tok", "capability": "Bash", "toolUseId": "toolu_a",
         "output": "hi\n", "isError": False},
    ]


def test_run_turn_ignores_a_tool_result_it_never_saw_the_call_for(tmp_path):
    """A resumed session replays results whose tool_use block was streamed
    to a previous process. With no name to label it, an orphan result would
    render as a blank chip."""
    posted = []
    lines = _stream_json_lines(
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "toolu_from_last_session",
             "content": "stale"},
        ]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        text, _, _ = cli.run_turn("hello", activity={"url": "http://runner/x", "token": "tok"})

    assert text == "ok"
    assert posted == []


def test_run_turn_reports_a_failed_tool_call_as_an_error(tmp_path):
    posted = []
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "toolu_b", "name": "Bash",
             "input": {"command": "nope"}},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "toolu_b",
             "content": "nope: command not found", "is_error": True},
        ]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        cli.run_turn("hello", activity={"url": "http://runner/x", "token": "tok"})

    assert posted[1]["isError"] is True
    assert posted[1]["output"] == "nope: command not found"


# ---------------------------------------------------------------------------
# The narrative between the tool calls (Edvard, 2026-08-04: "how would you
# like to be presented a story? ... first a narrative, then an action, then a
# narrative, then an action"). Every passage but the last is narration and
# goes out live; the last one is the reply.
# ---------------------------------------------------------------------------

def _story_lines():
    return _stream_json_lines(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "First I look at the pods."},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "toolu_a", "name": "Bash",
             "input": {"command": "kubectl get pods"}},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "toolu_a", "content": "all Running\n"},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "They are all up."},
        ]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )


def _run_story(tmp_path, posted, **kwargs):
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(_story_lines())):
        return cli.run_turn("hello", **kwargs)


def test_run_turn_streams_each_passage_in_the_order_it_was_written(tmp_path):
    posted = []
    _run_story(tmp_path, posted, activity={"url": "http://runner/x", "token": "tok"})
    assert [p["capability"] for p in posted] == ["assistant_text", "Bash", "Bash"]
    assert posted[0]["detail"] == "First I look at the pods."


def test_run_turn_returns_only_the_closing_passage_as_the_reply(tmp_path):
    """The rest is already in the conversation as narration -- returning the
    join too would print the whole run again inside the reply bubble."""
    posted = []
    text, _, _ = _run_story(tmp_path, posted, activity={"url": "http://runner/x", "token": "tok"})
    assert text == "They are all up."
    assert "First I look at the pods." not in text


def test_run_turn_does_not_narrate_the_closing_passage(tmp_path):
    """It is the reply, and it is about to be posted as one. Sending it as
    narration as well would show it twice, once in the drawer and once in
    the bubble underneath."""
    posted = []
    _run_story(tmp_path, posted, activity={"url": "http://runner/x", "token": "tok"})
    assert all(p.get("detail") != "They are all up." for p in posted)


def test_run_turn_returns_every_passage_when_nothing_is_narrated(tmp_path):
    """The /invoke path, or a runner too old to send an activity block. With
    nowhere to stream to, dropping the earlier passages would lose them."""
    posted = []
    text, _, _ = _run_story(tmp_path, posted)
    assert text == "First I look at the pods.\nThey are all up."
    assert posted == []


def test_run_turn_keeps_a_lone_passage_intact(tmp_path):
    """One text block and no tools: nothing precedes it, so nothing is
    narrated and the reply is exactly what it always was."""
    posted = []
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "just an answer"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        text, _, _ = cli.run_turn("hello", activity={"url": "http://runner/x", "token": "tok"})
    assert text == "just an answer"
    assert posted == []


def test_run_turn_falls_back_rather_than_replying_with_nothing(tmp_path):
    """A session that ends on a tool call has no closing passage. Every
    passage has then been narrated, so the join repeats them -- accepted,
    because the alternative is an empty reply, which fails the whole turn."""
    posted = []
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "on it"}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "toolu_z", "name": "Bash", "input": {"command": "ls"}},
        ]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        text, _, _ = cli.run_turn("hello", activity={"url": "http://runner/x", "token": "tok"})
    assert text == "on it"


def test_report_text_does_not_truncate_a_passage(tmp_path):
    """A chip label is clipped at DETAIL_CHARS_MAX because it is a label.
    This is prose, and clipping it mid-sentence is exactly the "block of
    text" complaint in a new place."""
    posted = []
    long_passage = "word " * 400
    with patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True):
        reporter = activity.ActivityReporter({"url": "http://runner/x", "token": "tok"})
        reporter.start()
        reporter.report_text(long_passage)
        reporter.close()
    assert posted[0]["detail"] == long_passage.strip()
    assert len(posted[0]["detail"]) > activity.DETAIL_CHARS_MAX


def test_report_text_skips_a_blank_passage():
    posted = []
    with patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True):
        reporter = activity.ActivityReporter({"url": "http://runner/x", "token": "tok"})
        reporter.start()
        reporter.report_text("   \n  ")
        reporter.close()
    assert posted == []


# --- Draining on SIGTERM (2026-08-05) -------------------------------------
# The runner learned this in #32; the bridge never did, and kept dying
# mid-turn whenever merging a bridge PR rolled the pod hosting the Claude
# Code session that merged it. Four cycles lost their reply that way.


@pytest.fixture
def drainable_server():
    """Restores the real signal handlers and the module's drain state, so a
    genuine SIGTERM fired at the pytest process can't leak into later tests
    or leave _shutdown_requested stuck on."""
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)
    server._shutdown_requested = False
    server._in_flight = 0
    try:
        yield server
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        server._shutdown_requested = False
        server._in_flight = 0


def test_start_server_drains_the_in_flight_turn_after_a_real_sigterm(drainable_server):
    """The actual regression, with a real signal rather than a faked flag.

    Without the handler, SIGTERM's default disposition kills the process
    outright and this test can't even run to its assertions.
    """
    drainable_server.install_signal_handlers()
    returned = threading.Event()

    def run():
        drainable_server.start_server()
        returned.set()

    # PORT 0 -> the OS picks a free port; the bridge's real 8090 is in use
    # by the pod these tests run inside.
    with patch.object(drainable_server, "PORT", 0), \
         patch.object(drainable_server, "log", lambda *a, **k: None):
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        time.sleep(0.3)  # let it bind and enter its wait loop
        assert not returned.is_set(), "server exited before any signal arrived"

        drainable_server._enter_turn()  # a turn is now mid-flight
        os.kill(os.getpid(), signal.SIGTERM)  # the pod is rolled

        time.sleep(1.0)
        assert drainable_server.shutdown_requested() is True
        assert not returned.is_set(), \
            "server exited while a turn was still in flight -- the reply would be lost"

        drainable_server._leave_turn()  # the turn finishes and returns its text
        assert returned.wait(timeout=10), "server did not exit after the drain completed"


def test_start_server_keeps_serving_when_no_signal_arrives(drainable_server):
    """The other half: the drain must not fire on its own."""
    returned = threading.Event()

    def run():
        drainable_server.start_server()
        returned.set()

    with patch.object(drainable_server, "PORT", 0), \
         patch.object(drainable_server, "log", lambda *a, **k: None):
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        time.sleep(1.0)
        assert not returned.is_set()
        drainable_server._shutdown_requested = True  # let the thread finish
        assert returned.wait(timeout=10)


def test_do_post_refuses_a_new_turn_while_draining(drainable_server):
    """Accepting a 45-minute turn on a pod that is already shutting down
    guarantees losing it. Fail fast instead."""
    handler, sent = _make_handler({"conversation_id": "c1", "prompt": "hi"})
    drainable_server._shutdown_requested = True
    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", side_effect=AssertionError("must not start a turn")):
        handler.do_POST()
    assert sent["status"] == 503
    assert sent["payload"]["error"] == "shutting_down"


def test_do_post_counts_a_turn_as_in_flight_only_while_generate_runs(drainable_server):
    seen = {}

    def fake_generate(conversation_id, system, prompt, **kwargs):
        seen["during"] = server._in_flight
        return "answer", ""

    handler, sent = _make_handler({"conversation_id": "c1", "prompt": "hi"})
    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", side_effect=fake_generate):
        handler.do_POST()

    assert seen["during"] == 1
    assert server._in_flight == 0
    assert sent["status"] == 200


def test_a_failing_turn_still_decrements_the_in_flight_count(drainable_server):
    """Otherwise one crashed turn makes every later drain hang until the
    grace period runs out and Kubernetes SIGKILLs the pod anyway."""
    handler, sent = _make_handler({"conversation_id": "c1", "prompt": "hi"})
    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", side_effect=server.ClaudeCliError("boom")):
        handler.do_POST()
    assert sent["status"] == 502
    assert server._in_flight == 0


def test_await_drain_returns_immediately_when_nothing_is_in_flight(drainable_server):
    started = time.monotonic()
    with patch.object(server, "log", lambda *a, **k: None):
        server._await_drain()
    assert time.monotonic() - started < 1.0


def test_health_reports_whether_the_pod_is_draining(drainable_server):
    handler, sent = _make_handler({}, path="/health")
    handler.do_GET()
    assert sent["payload"] == {"status": "ok", "draining": False}

    server._shutdown_requested = True
    handler, sent = _make_handler({}, path="/health")
    handler.do_GET()
    assert sent["payload"]["draining"] is True


def test_enter_turn_refuses_once_draining_has_started(drainable_server):
    """The check and the increment must be atomic. If a turn could register
    itself after the flag was set, _await_drain could already have seen zero
    and let the process exit on top of it."""
    server._shutdown_requested = True
    assert server._enter_turn() is False
    assert server._in_flight == 0


def test_enter_turn_admits_a_turn_while_serving_normally(drainable_server):
    assert server._enter_turn() is True
    assert server._in_flight == 1
    server._leave_turn()
    assert server._in_flight == 0


# ---------------------------------------------------------------------------
# --mcp-config: Agora's own capability tools, handed to the CLI session
# (2026-08-06). Edvard: "There are different tools for you and Gemini? That
# should not be the case." The runner hosts them; this side only has to
# render the flag, and above all has to never render a broken one.
# ---------------------------------------------------------------------------

MCP_BLOCK = {"url": "http://runner.agents.svc:8082/mcp", "token": "tok-abc"}


def _cli_run_capturing_cmd(tmp_path, **run_turn_kwargs):
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        # Read the config while the subprocess would still exist -- it is
        # deleted in the finally, so afterwards there is nothing to read.
        if "--mcp-config" in cmd:
            path = cmd[cmd.index("--mcp-config") + 1]
            captured["config_path"] = path
            with open(path) as handle:
                captured["config"] = json.load(handle)
        return FakeProc(lines)

    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli, "MCP_CONFIG_FILE", str(tmp_path / "home" / "mcp.json")), \
         patch.object(cli.subprocess, "Popen", side_effect=fake_popen):
        cli.run_turn("hello", **run_turn_kwargs)
    return captured


def test_run_turn_sends_no_mcp_flag_when_the_caller_asks_for_none(tmp_path):
    """Every caller before 2026-08-06 sent no `mcp` block, and must keep
    getting exactly the invocation it used to."""
    captured = _cli_run_capturing_cmd(tmp_path)
    assert "--mcp-config" not in captured["cmd"]
    assert "--strict-mcp-config" not in captured["cmd"]


def test_run_turn_passes_a_strict_mcp_config_when_given_a_block(tmp_path):
    captured = _cli_run_capturing_cmd(tmp_path, mcp=MCP_BLOCK)
    assert "--mcp-config" in captured["cmd"]
    # Without --strict, the CLI would ALSO load whatever is registered in
    # the PVC's ~/.claude.json -- persistent state no caller asked for.
    assert "--strict-mcp-config" in captured["cmd"]
    assert captured["config"] == {"mcpServers": {"agora": {
        "type": "http",
        "url": "http://runner.agents.svc:8082/mcp",
        "headers": {"Authorization": "Bearer tok-abc"},
    }}}


def test_run_turn_deletes_the_mcp_config_when_the_turn_ends(tmp_path):
    """It holds the turn's bearer token; the turn is over."""
    captured = _cli_run_capturing_cmd(tmp_path, mcp=MCP_BLOCK)
    assert not os.path.exists(captured["config_path"])


def test_write_mcp_config_is_always_valid_json(tmp_path):
    """The one hazard worth a test of its own. Measured on CLI 2.1.197: an
    unreachable MCP server is harmless (the turn completes, exit 0), but a
    --mcp-config file that is not valid JSON aborts the CLI before the
    model is called at all. A token carrying a quote or a newline must
    therefore never be able to break the file -- which is what building it
    with json.dump instead of interpolation buys."""
    path = str(tmp_path / "mcp.json")
    written = cli.write_mcp_config(
        {"url": 'http://x/mcp?a="b"', "token": 'tok"with\nnasty\\chars'}, path=path)
    assert written == path
    with open(path) as handle:
        config = json.load(handle)  # raises if this ever stops being valid JSON
    server = config["mcpServers"]["agora"]
    assert server["headers"]["Authorization"] == 'Bearer tok"with\nnasty\\chars'


@pytest.mark.parametrize("block", [
    None, {}, "not-a-dict",
    {"url": "http://x/mcp"},              # no token: would connect anonymously
    {"token": "tok"},                     # no url
    {"url": "  ", "token": "tok"},        # blank url
])
def test_write_mcp_config_returns_empty_for_anything_unusable(block, tmp_path):
    """Every rejection path returns "" (run without MCP) rather than
    raising or writing a half-formed server."""
    path = str(tmp_path / "mcp.json")
    assert cli.write_mcp_config(block, path=path) == ""
    assert not os.path.exists(path)


def test_write_mcp_config_returns_empty_when_it_cannot_write(tmp_path):
    """A failed write must degrade to "no MCP tools this turn", never to a
    flag pointing at a file that isn't there -- the CLI treats that as an
    invalid configuration and refuses to start."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    assert cli.write_mcp_config(MCP_BLOCK, path=str(blocker / "sub" / "mcp.json")) == ""


def test_run_turn_still_runs_when_the_mcp_config_cannot_be_written(tmp_path):
    """The degradation path, end to end: no flag, turn completes."""
    with patch.object(cli, "write_mcp_config", return_value=""):
        captured = _cli_run_capturing_cmd(tmp_path, mcp=MCP_BLOCK)
    assert "--mcp-config" not in captured["cmd"]
