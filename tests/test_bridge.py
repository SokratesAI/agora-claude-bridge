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


def test_run_turn_passes_system_as_append_system_prompt(tmp_path):
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
        cli.run_turn("hello", system="You are Nova.")
    cmd = captured["cmd"]
    assert cmd[cmd.index("--append-system-prompt") + 1] == "You are Nova."
    # and it must NOT have been smuggled into the user turn as well
    assert cmd[cmd.index("-p") + 1] == "hello"


def test_run_turn_still_passes_system_on_a_resumed_turn(tmp_path):
    """The CLI does not carry --append-system-prompt across --resume
    (measured 2026-08-08). Dropping it on resumed turns would leave every
    turn after the first running with no persona at all."""
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
        cli.run_turn("hello", session_id="sess-existing", system="You are Nova.")
    cmd = captured["cmd"]
    assert "--resume" in cmd
    assert cmd[cmd.index("--append-system-prompt") + 1] == "You are Nova."


def test_run_turn_omits_the_flag_entirely_when_no_system_given(tmp_path):
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
    assert "--append-system-prompt" not in captured["cmd"]


# ---------------------------------------------------------------------------
# Attachments -- 2026-08-10. `claude -p <text>` can only carry text, so a
# claude-cli persona silently dropped every image while the runner's other
# two providers built real image blocks. --input-format stream-json reads
# the same content-block shape off stdin. Verified live against CLI 2.1.226
# in this pod on the full production path (resume + --append-system-prompt +
# an image block): the model answered from the image and still knew its
# system-prompt codeword.
# ---------------------------------------------------------------------------

_PNG_B64 = "iVBORw0KGgo="


def test_write_stream_json_input_builds_text_then_image_blocks(tmp_path):
    path = cli.write_stream_json_input(
        "what is this?",
        [{"filename": "photo.png", "mimeType": "image/png", "data": _PNG_B64}],
        path=str(tmp_path / "in.jsonl"),
    )
    event = json.loads(open(path).read())
    assert event["type"] == "user"
    assert event["message"]["role"] == "user"
    assert event["message"]["content"] == [
        {"type": "text", "text": "what is this?"},
        {"type": "image",
         "source": {"type": "base64", "media_type": "image/png", "data": _PNG_B64}},
    ]


def test_write_stream_json_input_omits_the_text_block_for_an_uncaptioned_image(tmp_path):
    path = cli.write_stream_json_input(
        "", [{"filename": "photo.png", "mimeType": "image/png", "data": _PNG_B64}],
        path=str(tmp_path / "in.jsonl"),
    )
    blocks = json.loads(open(path).read())["message"]["content"]
    assert [b["type"] for b in blocks] == ["image"]


def test_write_stream_json_input_notes_an_attachment_that_has_no_data(tmp_path):
    """A non-image, or an image whose fetch failed -- the caller decides
    which, and this renders the same note the other two providers emit."""
    path = cli.write_stream_json_input(
        "read this", [{"filename": "notes.pdf", "mimeType": "application/pdf"}],
        path=str(tmp_path / "in.jsonl"),
    )
    blocks = json.loads(open(path).read())["message"]["content"]
    assert blocks[1] == {
        "type": "text",
        "text": "[attached file: notes.pdf (application/pdf) -- not loaded]",
    }


def test_run_turn_with_attachments_switches_to_stream_json_input(tmp_path):
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["stdin"] = kwargs.get("stdin")
        return FakeProc(lines)

    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli, "CLI_INPUT_FILE", str(tmp_path / "home" / "in.jsonl")), \
         patch.object(cli.subprocess, "Popen", side_effect=fake_popen):
        cli.run_turn("what is this?", attachments=[
            {"filename": "photo.png", "mimeType": "image/png", "data": _PNG_B64}])
    cmd = captured["cmd"]
    assert cmd[cmd.index("--input-format") + 1] == "stream-json"
    # The message moved to stdin, so it must NOT also be an argv prompt --
    # sending it both ways would deliver the user's text twice.
    assert cmd[cmd.index("-p") + 1] == "--input-format"
    assert "what is this?" not in cmd
    assert captured["stdin"] is not None


def test_run_turn_without_attachments_still_passes_the_prompt_as_argv(tmp_path):
    """Every ordinary chat turn and every Nova cycle runs this path. It has
    to stay byte-identical to what it was before attachments existed."""
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["stdin"] = kwargs.get("stdin")
        return FakeProc(lines)

    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", side_effect=fake_popen):
        cli.run_turn("hello")
    cmd = captured["cmd"]
    assert cmd[cmd.index("-p") + 1] == "hello"
    assert "--input-format" not in cmd
    assert captured["stdin"] is None


def test_run_turn_deletes_the_input_file_after_the_turn(tmp_path):
    """It holds whatever Edvard photographed, on a persistent volume, and
    the CLI has read it before the first event arrives."""
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    input_file = tmp_path / "home" / "in.jsonl"
    seen = {}

    def fake_popen(cmd, **kwargs):
        seen["existed_during_turn"] = input_file.exists()
        return FakeProc(lines)

    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli, "CLI_INPUT_FILE", str(input_file)), \
         patch.object(cli.subprocess, "Popen", side_effect=fake_popen):
        cli.run_turn("hi", attachments=[
            {"filename": "photo.png", "mimeType": "image/png", "data": _PNG_B64}])
    assert seen["existed_during_turn"] is True
    assert not input_file.exists()


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

def test_generate_sends_system_prompt_out_of_band_not_in_the_message():
    captured = {}

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None, system=None, attachments=None, allow_concurrent=False):
        captured["message"] = message
        captured["session_id"] = session_id
        captured["system"] = system
        return "reply", "", "sess-new"

    with patch.object(server, "get_session_id", return_value=None), \
         patch.object(server, "set_session_id") as mock_set, \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        text, thinking = server.generate("conv-1", "You are helpful.", "hi there")

    # The persona text must reach the model as an operator instruction, never
    # as something the user appears to have typed.
    assert captured["message"] == "hi there"
    assert captured["system"] == "You are helpful."
    assert captured["session_id"] is None
    assert text == "reply"
    mock_set.assert_called_once_with("conv-1", "sess-new")


def test_generate_resends_the_system_prompt_on_a_resumed_turn():
    """The CLI does not carry --append-system-prompt across --resume, so a
    resumed turn that omitted it would run with no constitution at all."""
    captured = {}

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None, system=None, attachments=None, allow_concurrent=False):
        captured["message"] = message
        captured["session_id"] = session_id
        captured["system"] = system
        return "reply2", "", "sess-existing"

    with patch.object(server, "get_session_id", return_value="sess-existing"), \
         patch.object(server, "set_session_id"), \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        server.generate("conv-1", "You are helpful.", "second message")

    assert captured["message"] == "second message"
    assert captured["system"] == "You are helpful."
    assert captured["session_id"] == "sess-existing"


def test_generate_retries_fresh_on_session_not_found():
    calls = []

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None, system=None, attachments=None, allow_concurrent=False):
        calls.append((message, session_id, system))
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
    assert calls[0] == ("hi", "sess-gone", "system")
    assert calls[1] == ("hi", None, "system")  # retried fresh, system still out of band
    mock_clear.assert_called_once_with("conv-1")
    mock_set.assert_called_once_with("conv-1", "sess-brand-new")


def test_generate_propagates_other_cli_errors_without_retry():
    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None, system=None, attachments=None, allow_concurrent=False):
        raise server.ClaudeCliError("a real bug")

    with patch.object(server, "get_session_id", return_value="sess-1"), \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        with pytest.raises(server.ClaudeCliError, match="a real bug"):
            server.generate("conv-1", "system", "hi")


def test_generate_is_unrestricted_by_default():
    captured = {}

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None, system=None, attachments=None, allow_concurrent=False):
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
                      mcp=None, system=None, attachments=None, allow_concurrent=False):
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
                      mcp=None, system=None, attachments=None, allow_concurrent=False):
        captured["message"] = message
        captured["session_id"] = session_id
        captured["system"] = system
        return "reply", "", "sess-should-be-ignored"

    with patch.object(server, "get_session_id") as mock_get, \
         patch.object(server, "set_session_id") as mock_set, \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        text, _ = server.generate("conv-1", "system prompt", "hi", stateless=True)

    assert text == "reply"
    assert captured["message"] == "hi"
    assert captured["system"] == "system prompt"
    assert captured["session_id"] is None
    mock_get.assert_not_called()  # never even looks up a stored session
    mock_set.assert_not_called()  # and never persists the new one


def test_generate_stateless_ignores_a_stored_session_for_the_same_conversation():
    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None, system=None, attachments=None, allow_concurrent=False):
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
                      mcp=None, system=None, attachments=None, allow_concurrent=False):
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
                      activity=None, attachments=None, allow_concurrent=False):
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
                      activity=None, attachments=None, allow_concurrent=False):
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
                      activity=None, attachments=None, allow_concurrent=False):
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
                      activity=None, attachments=None, allow_concurrent=False):
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
# credentials.py -- .credentials.json bootstrap, piping the claude-auth
# secret's raw JSON straight to disk unmodified. The rule is "newest wins",
# not "first boot only": a live token refresh only ever lands on the PVC and
# so always outranks the k8s Secret's frozen snapshot, but a Secret a human
# has just re-authed by hand is genuinely newer and must not be ignored --
# see the module's own docstring, and #60.
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


def test_bootstrap_credentials_never_overwrites_a_newer_file(tmp_path):
    """The invariant, with a fixture that can actually see it.

    This test used to seed an on-disk credential carrying no `expiresAt` at
    all, which no real credentials.json does. That routed it into the
    "cannot parse, leave it alone" branch, so it agreed with the author
    whether or not the comparison below existed -- it passed identically
    before #60, when nothing was ever overwritten, and after it. Review of
    #60 caught it. The seed is now shaped like the real file and dated after
    the Secret, which is what a live CLI refresh always produces.
    """
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir(parents=True)
    dest = claude_dir / ".credentials.json"
    on_disk = json.dumps({
        "claudeAiOauth": {
            "accessToken": "already-refreshed-by-cli",
            "refreshToken": "sk-ant-ort-refreshed",
            # REAL_CREDS_JSON expires at 1785608541000; the CLI's own refresh
            # is always later than the snapshot the Secret froze.
            "expiresAt": 1785608541000 + 86_400_000,
            "scopes": ["user:inference", "user:profile"],
            "subscriptionType": "max",
        }
    })
    dest.write_text(on_disk)

    with patch.object(credentials, "CLAUDE_HOME", str(tmp_path)), \
         patch.dict(os.environ, {"CLAUDE_CREDENTIALS_JSON": REAL_CREDS_JSON}, clear=False):
        credentials.bootstrap_credentials()

    data = json.loads(dest.read_text())
    assert data["claudeAiOauth"]["accessToken"] == "already-refreshed-by-cli"


def test_bootstrap_credentials_takes_a_hand_refreshed_secret(tmp_path):
    """The other half of "newest wins", and the recovery path: put a fresh
    credential in the Secret and restart the pod. Before #60 this silently
    did nothing whenever a file already existed."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir(parents=True)
    dest = claude_dir / ".credentials.json"
    dest.write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "dead",
            "expiresAt": 1785608541000 - 86_400_000,
            "scopes": ["user:inference"],
        }
    }))

    with patch.object(credentials, "CLAUDE_HOME", str(tmp_path)), \
         patch.dict(os.environ, {"CLAUDE_CREDENTIALS_JSON": REAL_CREDS_JSON}, clear=False):
        credentials.bootstrap_credentials()

    assert dest.read_text() == REAL_CREDS_JSON


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
                      stateless=False, activity=None, attachments=None, allow_concurrent=False):
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
                      stateless=False, activity=None, attachments=None, allow_concurrent=False):
        captured["activity"] = activity
        return "answer", ""

    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", fake_generate):
        handler.do_POST()
    assert sent["status"] == 200
    assert captured["activity"] is None


def _capture_generate_attachments(payload):
    handler, sent = _make_handler(payload)
    captured = {}

    def fake_generate(conversation_id, system, prompt, model=None, restricted=False, mcp=None,
                      stateless=False, activity=None, attachments=None, allow_concurrent=False):
        captured["attachments"] = attachments
        captured["prompt"] = prompt
        return "answer", ""

    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", fake_generate):
        handler.do_POST()
    return sent, captured


def test_do_post_passes_attachments_through_to_generate():
    att = [{"filename": "photo.png", "mimeType": "image/png", "data": "iVBORw0KGgo="}]
    sent, captured = _capture_generate_attachments(
        {"conversation_id": "c1", "prompt": "what is this?", "attachments": att})
    assert sent["status"] == 200
    assert captured["attachments"] == att


def test_do_post_attachments_default_to_empty_when_omitted():
    sent, captured = _capture_generate_attachments({"conversation_id": "c1", "prompt": "hi"})
    assert sent["status"] == 200
    assert captured["attachments"] == []


def test_do_post_accepts_an_image_with_no_caption():
    """An uncaptioned image is a real message. This used to 400 as an empty
    prompt, which is the empty-turn crash the runner's _gemini_parts
    documents on the other side."""
    sent, captured = _capture_generate_attachments({
        "conversation_id": "c1",
        "prompt": "",
        "attachments": [{"filename": "photo.png", "mimeType": "image/png", "data": "x"}],
    })
    assert sent["status"] == 200
    assert captured["prompt"] == ""


def test_do_post_normalises_a_null_prompt_to_empty_string():
    """A caller sending "prompt": null alongside an attachment gets past
    the emptiness check, and None then reaches cli.py's `message[:120]`
    as a TypeError -- a 500 for what is a perfectly valid message."""
    sent, captured = _capture_generate_attachments({
        "conversation_id": "c1",
        "prompt": None,
        "attachments": [{"filename": "photo.png", "mimeType": "image/png", "data": "x"}],
    })
    assert sent["status"] == 200
    assert captured["prompt"] == ""


def test_do_post_still_rejects_a_turn_carrying_neither_prompt_nor_attachments():
    handler, sent = _make_handler({"conversation_id": "c1", "prompt": ""})
    with patch.object(server, "BRIDGE_TOKEN", ""):
        handler.do_POST()
    assert sent["status"] == 400


def test_generate_passes_attachments_to_run_turn():
    att = [{"filename": "photo.png", "mimeType": "image/png", "data": "x"}]
    captured = {}

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None, system=None, attachments=None, allow_concurrent=False):
        captured["attachments"] = attachments
        return "reply", "", "sess-1"

    with patch.object(server, "get_session_id", return_value=None), \
         patch.object(server, "set_session_id"), \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        server.generate("conv-1", "system", "hi", attachments=att)
    assert captured["attachments"] == att


def test_generate_stateless_also_passes_attachments_to_run_turn():
    """The stateless branch is a separate call site and has been forgotten
    by a previous pass-through before (it is how the Nova workflow runs)."""
    att = [{"filename": "photo.png", "mimeType": "image/png", "data": "x"}]
    captured = {}

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None, system=None, attachments=None, allow_concurrent=False):
        captured["attachments"] = attachments
        return "reply", "", "sess-1"

    with patch.object(server, "run_turn", side_effect=fake_run_turn):
        server.generate("conv-1", "system", "hi", stateless=True, attachments=att)
    assert captured["attachments"] == att


# ---------------------------------------------------------------------------
# allow_concurrent, end to end through this module -- cli.run_turn's lane
# (tested further down) was merged inert on 2026-08-10 because nothing
# asked for it. These are the wiring that makes a journal-card reply
# actually skip a running Nova cycle instead of queueing 45 minutes behind
# it. Every one of them is a pass-through test, and every pass-through in
# this file exists because one of them was once forgotten.
# ---------------------------------------------------------------------------


def _capture_generate_allow_concurrent(payload):
    handler, sent = _make_handler(payload)
    captured = {}

    def fake_generate(conversation_id, system, prompt, model=None, restricted=False, mcp=None,
                      stateless=False, activity=None, attachments=None, allow_concurrent=False):
        captured["allow_concurrent"] = allow_concurrent
        return "answer", ""

    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", fake_generate):
        handler.do_POST()
    return sent, captured


def test_do_post_passes_allow_concurrent_through_to_generate():
    sent, captured = _capture_generate_allow_concurrent(
        {"conversation_id": "c1", "prompt": "hi", "allow_concurrent": True})
    assert sent["status"] == 200
    assert captured["allow_concurrent"] is True


def test_do_post_allow_concurrent_defaults_to_false_when_omitted():
    """Every existing caller sends a long turn and must keep the lock."""
    sent, captured = _capture_generate_allow_concurrent(
        {"conversation_id": "c1", "prompt": "hi"})
    assert sent["status"] == 200
    assert captured["allow_concurrent"] is False


def test_generate_passes_allow_concurrent_to_run_turn():
    captured = {}

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None, system=None, attachments=None, allow_concurrent=False):
        captured["allow_concurrent"] = allow_concurrent
        return "reply", "", "sess-1"

    with patch.object(server, "get_session_id", return_value=None), \
         patch.object(server, "set_session_id"), \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        server.generate("conv-1", "system", "hi", allow_concurrent=True)
    assert captured["allow_concurrent"] is True


def test_generate_stateless_also_passes_allow_concurrent_to_run_turn():
    """The stateless branch is the one that matters here: the journal-card
    reply is stateless, so a pass-through that only covered the resumed
    branch would leave the whole feature inert with green tests."""
    captured = {}

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None, system=None, attachments=None, allow_concurrent=False):
        captured["allow_concurrent"] = allow_concurrent
        return "reply", "", "sess-1"

    with patch.object(server, "run_turn", side_effect=fake_run_turn):
        server.generate("conv-1", "system", "hi", stateless=True, allow_concurrent=True)
    assert captured["allow_concurrent"] is True


def test_generate_keeps_allow_concurrent_on_the_session_not_found_retry():
    """Third call site. A retry that silently dropped the flag would take
    the lock and hang for the length of a cycle -- the exact wait this is
    meant to remove, on the one path a caller cannot predict."""
    seen = []

    def fake_run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
                      mcp=None, system=None, attachments=None, allow_concurrent=False):
        seen.append(allow_concurrent)
        if session_id is not None:
            raise server.ClaudeCliError(server.SESSION_NOT_FOUND)
        return "reply", "", "sess-2"

    with patch.object(server, "get_session_id", return_value="sess-old"), \
         patch.object(server, "clear_session_id"), \
         patch.object(server, "set_session_id"), \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        server.generate("conv-1", "system", "hi", allow_concurrent=True)
    assert seen == [True, True]


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


def test_write_mcp_config_is_not_world_readable(tmp_path):
    """For the length of the turn it is a live bearer token for the
    runner's tool endpoint, on a shared PVC."""
    path = str(tmp_path / "mcp.json")
    cli.write_mcp_config(MCP_BLOCK, path=path)
    assert stat_module.S_IMODE(os.stat(path).st_mode) == 0o600


# ---------------------------------------------------------------------------
# cli.py -- subagent forwarding (--forward-subagent-text)
#
# Every event shape below was captured from a real CLI 2.1.226 run on
# 2026-08-10, not invented: the ordering in the reply-protection tests is the
# ordering a background subagent actually produced.
# ---------------------------------------------------------------------------

ACTIVITY_BLOCK = {"url": "http://runner/tool-activity", "token": "tok"}


def _run_with_reporter(tmp_path, lines):
    """Run a stream and return (text, posted payloads)."""
    posted = []
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        text, thinking, _ = cli.run_turn("hello", activity=ACTIVITY_BLOCK)
    return text, thinking, posted


def test_run_turn_asks_the_cli_to_forward_subagent_events(tmp_path):
    captured = _cli_run_capturing_cmd(tmp_path)
    assert "--forward-subagent-text" in captured["cmd"]


def test_a_subagent_finishing_last_does_not_become_the_reply(tmp_path):
    """The failure this whole change is built around.

    A backgrounded subagent finishes whenever it finishes -- routinely after
    the persona has written its closing passage. If its text reached
    `pending`, the reply posted to Edvard's phone would be the subagent's
    words instead of the persona's. This ordering is copied from a real run.
    """
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Agent", "id": "toolu_agent",
             "input": {"description": "Gather state"}},
        ]}},
        {"type": "system", "subtype": "task_started", "task_id": "task_1",
         "tool_use_id": "toolu_agent", "description": "Gather state",
         "subagent_type": "Explore"},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "the persona's real answer"},
        ]}},
        {"type": "assistant", "parent_tool_use_id": "toolu_agent",
         "message": {"content": [{"type": "text", "text": "the subagent's report"}]}},
        {"type": "result", "session_id": "s", "subtype": "success"},
    )
    text, _, posted = _run_with_reporter(tmp_path, lines)

    assert text == "the persona's real answer"
    assert "subagent" not in text
    # ...and it is not silently dropped either -- it is narrated, attributed.
    narrated = [p["detail"] for p in posted if p["capability"] == activity.NARRATION_TEXT]
    assert any("the subagent's report" in d and "Gather state" in d for d in narrated)


def test_a_subagent_tool_call_does_not_flush_the_pending_reply(tmp_path):
    """The second, subtler half. `release_narrative()` empties `pending`; if a
    background child's tool call could trigger it after the persona's final
    passage, `pending` would be empty at the end and the reply would fall
    through to the join of EVERY passage -- the wall of text that fallback
    exists to prevent."""
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "first passage"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "final answer"}]}},
        {"type": "assistant", "parent_tool_use_id": "toolu_agent", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "id": "toolu_child", "input": {"command": "ls"}},
        ]}},
        {"type": "result", "session_id": "s", "subtype": "success"},
    )
    text, _, _ = _run_with_reporter(tmp_path, lines)
    assert text == "final answer"
    assert "first passage" not in text


def test_subagent_tool_calls_are_reported_and_attributed(tmp_path):
    lines = _stream_json_lines(
        {"type": "system", "subtype": "task_started", "task_id": "task_1",
         "tool_use_id": "toolu_agent", "description": "Gather state",
         "subagent_type": "Explore"},
        {"type": "assistant", "parent_tool_use_id": "toolu_agent", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "id": "toolu_child",
             "input": {"command": "echo hi"}},
        ]}},
        {"type": "user", "parent_tool_use_id": "toolu_agent", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "toolu_child", "content": "hi"},
        ]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
        {"type": "result", "session_id": "s", "subtype": "success"},
    )
    text, _, posted = _run_with_reporter(tmp_path, lines)

    assert text == "done"
    call = [p for p in posted if p["capability"] == "Bash" and "detail" in p]
    result = [p for p in posted if p["capability"] == "Bash" and "output" in p]
    # The tool keeps its own name -- a subagent's Bash call is a Bash call --
    # and only the label says who ran it.
    assert call[0]["detail"] == "↳ Gather state · echo hi"
    # The result still pairs back by tool_use_id, across the parent/child line.
    assert result[0]["output"] == "hi"
    assert result[0]["toolUseId"] == "toolu_child"


def test_subagent_start_and_finish_fold_into_one_chip(tmp_path):
    lines = _stream_json_lines(
        {"type": "system", "subtype": "task_started", "task_id": "task_1",
         "tool_use_id": "toolu_agent", "description": "Gather state",
         "subagent_type": "Explore"},
        {"type": "system", "subtype": "task_notification", "task_id": "task_1",
         "status": "completed", "summary": "found three things",
         "usage": {"total_tokens": 94183, "tool_uses": 37, "duration_ms": 215028}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
        {"type": "result", "session_id": "s", "subtype": "success"},
    )
    _, _, posted = _run_with_reporter(tmp_path, lines)

    chips = [p for p in posted if p["capability"] == activity.SUBAGENT]
    assert len(chips) == 2
    start, finish = chips
    assert start["detail"] == "Explore · Gather state"
    # Same id on both halves is what makes Agora's client render one chip.
    assert start["toolUseId"] == finish["toolUseId"] == "task_1"
    assert finish["isError"] is False
    assert "completed · 94,183 tokens · 37 tool calls · 215.0s" in finish["output"]
    assert "found three things" in finish["output"]


def test_a_failed_subagent_is_marked_as_an_error(tmp_path):
    lines = _stream_json_lines(
        {"type": "system", "subtype": "task_notification", "task_id": "task_1",
         "status": "failed", "summary": "it broke", "usage": {}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
        {"type": "result", "session_id": "s", "subtype": "success"},
    )
    _, _, posted = _run_with_reporter(tmp_path, lines)
    finish = [p for p in posted if p["capability"] == activity.SUBAGENT][0]
    assert finish["isError"] is True


def test_subagent_thinking_is_never_mistaken_for_the_personas_own(tmp_path):
    """`thinking` is returned to the runner as the persona's reasoning. A
    subagent's must not end up in it."""
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "the persona's reasoning"},
        ]}},
        {"type": "assistant", "parent_tool_use_id": "toolu_agent", "message": {"content": [
            {"type": "thinking", "thinking": "the subagent's reasoning"},
            {"type": "text", "text": "child says hello"},
        ]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
        {"type": "result", "session_id": "s", "subtype": "success"},
    )
    text, thinking, _ = _run_with_reporter(tmp_path, lines)
    assert text == "done"
    assert thinking == "the persona's reasoning"


def test_an_unnamed_subagent_still_gets_attributed(tmp_path):
    """A child event can arrive before its `task_started` -- or after a
    resume, where this process never saw one. It must still be marked as a
    subagent rather than silently reading as the persona."""
    lines = _stream_json_lines(
        {"type": "assistant", "parent_tool_use_id": "toolu_unknown",
         "message": {"content": [{"type": "text", "text": "orphan report"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
        {"type": "result", "session_id": "s", "subtype": "success"},
    )
    _, _, posted = _run_with_reporter(tmp_path, lines)
    narrated = [p["detail"] for p in posted if p["capability"] == activity.NARRATION_TEXT]
    assert any(d.startswith("↳ subagent") and "orphan report" in d for d in narrated)


# ---------------------------------------------------------------------------
# cli.py -- the opt-in concurrent lane past _invocation_lock
# ---------------------------------------------------------------------------


def _clear_creds(tmp_path, seconds_out):
    path = tmp_path / ".credentials.json"
    path.write_text(json.dumps(
        {"claudeAiOauth": {"expiresAt": (time.time() + seconds_out) * 1000}}))
    return str(path)


def test_refresh_window_clear_only_when_token_is_far_from_expiry(tmp_path):
    assert cli.refresh_window_clear(path=_clear_creds(tmp_path, 8 * 3600)) is True
    # Inside the margin the refresh the module docstring worries about could
    # fire, so the lane has to shut.
    assert cli.refresh_window_clear(path=_clear_creds(tmp_path, 60)) is False


@pytest.mark.parametrize("body", ["not json at all", "{}", '{"claudeAiOauth": {}}'])
def test_refresh_window_clear_is_false_on_anything_unreadable(tmp_path, body):
    """An unreadable credential is not evidence that concurrency is safe."""
    path = tmp_path / ".credentials.json"
    path.write_text(body)
    assert cli.refresh_window_clear(path=str(path)) is False
    assert cli.refresh_window_clear(path=str(tmp_path / "does-not-exist.json")) is False


def _lock_probe(tmp_path, **run_turn_kwargs):
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    seen = {}

    def fake_popen(cmd, **kwargs):
        seen["held"] = cli._invocation_lock.locked()
        seen["cmd"] = cmd
        return FakeProc(lines)

    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli.subprocess, "Popen", side_effect=fake_popen):
        cli.run_turn("hello", **run_turn_kwargs)
    return seen


def test_allow_concurrent_runs_outside_the_lock_when_the_window_is_clear(tmp_path):
    with patch.object(cli, "refresh_window_clear", return_value=True):
        assert _lock_probe(tmp_path, allow_concurrent=True)["held"] is False


def test_allow_concurrent_falls_back_to_the_lock_inside_the_refresh_window(tmp_path):
    """A caller asks for the lane; the token decides whether it opens."""
    with patch.object(cli, "refresh_window_clear", return_value=False):
        assert _lock_probe(tmp_path, allow_concurrent=True)["held"] is True


def test_default_callers_never_reach_the_concurrent_lane(tmp_path):
    """Not opting in must not even consult the credential file."""
    with patch.object(cli, "refresh_window_clear", side_effect=AssertionError("consulted")):
        assert _lock_probe(tmp_path)["held"] is True


def test_slotted_isolates_the_per_turn_files_only_for_a_concurrent_turn():
    """The mcp config carries this turn's bearer token and is deleted in the
    finally, so two overlapping turns sharing that one path can hand a turn
    the other's token. A serialized turn keeps its historical filename."""
    assert cli._slotted("/x/.claude/bridge-mcp.config.json", "") == \
        "/x/.claude/bridge-mcp.config.json"
    assert cli._slotted("/x/.claude/bridge-mcp.config.json", "7-9") == \
        "/x/.claude/bridge-mcp.config.7-9.json"
    assert cli._slotted("/x/.claude/bridge-input.jsonl", "7-9") == \
        "/x/.claude/bridge-input.7-9.jsonl"


def test_concurrent_turns_do_not_share_their_mcp_config_path(tmp_path):
    """Two concurrent turns must not write the same file -- the regression
    this whole change would otherwise introduce."""
    paths = []
    real_write = cli.write_mcp_config

    def spy(mcp, path=None):
        paths.append(path)
        return real_write(mcp, path)

    shared = str(tmp_path / "bridge-mcp.config.json")
    with patch.object(cli, "refresh_window_clear", return_value=True), \
         patch.object(cli, "MCP_CONFIG_FILE", shared), \
         patch.object(cli, "write_mcp_config", side_effect=spy):
        mcp = {"url": "http://runner/mcp", "token": "t"}
        _lock_probe(tmp_path, allow_concurrent=True, mcp=mcp)
        _lock_probe(tmp_path, allow_concurrent=True, mcp=mcp)
        _lock_probe(tmp_path, mcp=mcp)
    assert paths[0] != paths[2], "concurrent turn reused the shared path"
    assert paths[2] == shared, "serialized turn changed path"


def test_workspace_for_isolates_only_a_concurrent_turn():
    """A real git checkout, not a disposable per-turn artifact like the MCP
    config/input file _slotted isolates -- two turns sharing it would race
    on the same working tree. A serialized turn (no slot) must get the
    exact same shared directory every turn has always used."""
    with patch.object(cli, "CLAUDE_WORKSPACE", "/data/workspace"):
        assert cli._workspace_for("") == "/data/workspace"
        assert cli._workspace_for("7-9") == "/data/workspace/concurrent/7-9"


def test_serialized_turn_uses_the_shared_workspace_unchanged(tmp_path):
    workspace = str(tmp_path / "workspace")
    seen = _lock_probe_cwd(tmp_path, workspace)
    assert seen["cwd"] == workspace


def test_concurrent_turn_gets_an_isolated_workspace_that_exists_at_call_time(tmp_path):
    workspace = str(tmp_path / "workspace")
    with patch.object(cli, "refresh_window_clear", return_value=True):
        seen = _lock_probe_cwd(tmp_path, workspace, allow_concurrent=True)
    assert seen["cwd"] != workspace
    assert seen["cwd"].startswith(workspace)
    assert "concurrent" in seen["cwd"]
    assert seen["cwd_existed_at_call_time"] is True


def test_concurrent_workspace_is_removed_after_the_turn_but_the_shared_one_is_not(tmp_path):
    workspace = str(tmp_path / "workspace")
    with patch.object(cli, "refresh_window_clear", return_value=True):
        seen = _lock_probe_cwd(tmp_path, workspace, allow_concurrent=True)
    assert not os.path.exists(seen["cwd"]), "a concurrent turn must not leave its workspace behind"

    seen = _lock_probe_cwd(tmp_path, workspace)
    assert os.path.isdir(seen["cwd"]), "the shared workspace must survive a serialized turn"


def test_concurrent_turn_clears_a_slot_a_killed_turn_left_behind(tmp_path):
    """The `finally` cleanup is the normal path, not a guarantee: it does
    not run when the process is killed (pod eviction, OOM, SIGKILL), and
    `shutil.rmtree(..., ignore_errors=True)` accepts a partial removal in
    silence. Slots collide in ordinary operation -- `slot` is
    pid+thread-ident, the pid is fixed for the pod's life and CPython
    reuses a thread ident once that thread exits -- so a later turn really
    can be handed the directory an earlier one left behind. The invariant
    has to be enforced where it is needed, at the start of the turn."""
    workspace = str(tmp_path / "workspace")
    with patch.object(cli, "refresh_window_clear", return_value=True):
        first = _lock_probe_cwd(tmp_path, workspace, allow_concurrent=True)

    # Same test, same thread, so the next concurrent turn gets the same
    # slot -- stage exactly what a killed turn would have stranded there.
    stranded = first["cwd"]
    os.makedirs(os.path.join(stranded, ".git"), exist_ok=True)
    with open(os.path.join(stranded, "half-written-branch.txt"), "w") as fh:
        fh.write("a checkout from a turn that never got to its finally")

    with patch.object(cli, "refresh_window_clear", return_value=True):
        second = _lock_probe_cwd(tmp_path, workspace, allow_concurrent=True)

    assert second["cwd"] == stranded, "same thread must reuse the same slot for this to test anything"
    assert second["cwd_entries_at_call_time"] == [], (
        "a concurrent turn inherited a previous turn's checkout: "
        f"{second['cwd_entries_at_call_time']}")


def _lock_probe_cwd(tmp_path, workspace, **run_turn_kwargs):
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "result", "session_id": "sess-1", "subtype": "success"},
    )
    seen = {}

    def fake_popen(cmd, **kwargs):
        seen["cwd"] = kwargs.get("cwd")
        seen["cwd_existed_at_call_time"] = os.path.isdir(kwargs.get("cwd") or "")
        seen["cwd_entries_at_call_time"] = sorted(os.listdir(kwargs.get("cwd") or "."))
        return FakeProc(lines)

    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", workspace), \
         patch.object(cli.subprocess, "Popen", side_effect=fake_popen):
        cli.run_turn("hello", **run_turn_kwargs)
    return seen


# --- which invocations publish the cost record ----------------------------

# The real strings, copied from agora_runner/heartbeats.py:288-290, not
# invented. Checked live 2026-08-11 against this loop's own transcript, which
# analytics classifies `cycle manual` off exactly this text.
_REAL_MANUAL_TRIGGER = ("[Manual heartbeat trigger — Edvard started this run himself. "
                        "Address Edvard directly.]")
_REAL_AUTOMATIC_TRIGGER = "[Automatic heartbeat trigger — address Edvard directly.]"


def _watcher_armed_for(message, tmp_path, **kwargs):
    """Run one turn and report whether its watcher was told to publish."""
    lines = _stream_json_lines(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "result", "session_id": "sess-new", "subtype": "success"},
    )
    made = MagicMock()
    with patch.object(cli, "CLAUDE_HOME", str(tmp_path / "home")), \
         patch.object(cli, "CLAUDE_WORKSPACE", str(tmp_path / "workspace")), \
         patch.object(cli, "QuotaWatcher", made), \
         patch.object(cli.subprocess, "Popen", return_value=FakeProc(lines)):
        cli.run_turn(message, **kwargs)
    return made.call_args.kwargs["publish_costs"]


def test_a_heartbeat_cycle_arms_the_cost_publish(tmp_path):
    """The end of the chain: a cycle's own invocation is what republishes the
    record the site charts. Asserted against the runner's literal trigger
    strings rather than a stand-in, because the whole gate is one prefix match
    against text written in another repo -- a fixture that says "a cycle" would
    pass with the markers spelled wrong."""
    assert _watcher_armed_for(_REAL_MANUAL_TRIGGER, tmp_path) is True
    assert _watcher_armed_for(_REAL_AUTOMATIC_TRIGGER, tmp_path) is True


def test_a_trigger_that_carries_a_message_from_edvard_still_arms_it(tmp_path):
    """heartbeats.py appends his most recent message to the trigger when one
    is pending, so the opening turn is routinely longer than the marker. An
    equality check here would silently stop publishing on exactly the cycles
    he had spoken to."""
    carried = _REAL_MANUAL_TRIGGER + " Edvard's most recent message in this conversation: hi"
    assert _watcher_armed_for(carried, tmp_path) is True


def test_a_journal_card_reply_does_not_arm_it(tmp_path):
    """The turn this gate exists for. A reply is a short, concurrent turn on
    the same bridge, and publishing from it would scan every transcript on the
    PVC and push ~90KB to the vault for a two-sentence answer.

    The opening line is the real one, from `nova_replies.build_prompt` in
    agora-persona-runner. Writing this fixture by hand is what made it fail
    first time round: the invented version opened "You are Nova", which is a
    legacy `CYCLE_MARKER` from the pre-heartbeat shape where the constitution
    arrived as the user turn -- so it armed. The live reply does not, because
    `SYSTEM` goes to --append-system-prompt and the *message* starts with the
    entry. The marker stays as wide as it is on purpose: analytics still has
    to classify those old transcripts, and a second, narrower tuple here is
    the drift this shares one predicate to avoid. Over-firing costs one
    idempotent republish; under-firing is the silent staleness being fixed.
    """
    reply = ("Here is the journal entry you are answering a comment on -- cycle 107, "
             "2026-08-11 12:09.\n\n<entry>\n### Cycle 107\n...\n</entry>")
    assert _watcher_armed_for(reply, tmp_path, allow_concurrent=True) is False


def test_an_ordinary_turn_does_not_arm_it(tmp_path):
    """The default every other caller gets, including this suite."""
    assert _watcher_armed_for("hello", tmp_path) is False


# --- GET /health/database ---------------------------------------------
#
# The bridge is the vault client every Nova cycle actually reads through,
# and it holds its own copy of the routing rule. nova-site's /api/health
# answers for the runner's copy; nothing answered for this one, so
# confirming it agreed meant importing VaultClient and calling `db_for` by
# hand -- a worse instrument than the write-probe that replaced.

def _health_payload(reachable=True):
    return {
        "routing_enabled": True,
        "databases": {
            "main": {"name": "obsidian", "reachable": True,
                     "doc_count": 13196, "error": None},
            "nova": {"name": "nova", "reachable": reachable,
                     "doc_count": 713 if reachable else None,
                     "error": None if reachable else "HTTP 500"},
        },
        "routes": [{"path": "projects/sokrates/projects/agora/journal-digest.md",
                    "database": "nova"}],
    }


def _patch_client(payload=None, exc=None):
    from bridge import vault_tool

    class FakeClient:
        def __init__(self):
            if exc is not None:
                raise exc

        def database_health(self):
            return payload

    return patch.object(vault_tool, "VaultClient", FakeClient)


def test_health_database_reports_routing_and_reachability():
    handler, sent = _make_handler({}, path="/health/database")
    with _patch_client(_health_payload()):
        handler.do_GET()
    assert sent["status"] == 200
    assert sent["payload"]["ok"] is True
    assert sent["payload"]["databases"]["nova"]["doc_count"] == 713
    assert sent["payload"]["routes"][0]["database"] == "nova"


def test_health_database_is_503_when_a_database_is_unreachable():
    handler, sent = _make_handler({}, path="/health/database")
    with _patch_client(_health_payload(reachable=False)):
        handler.do_GET()
    assert sent["status"] == 503
    assert sent["payload"]["ok"] is False
    # The half that still works has to survive the failure of the other --
    # "which database is broken" is the whole question.
    assert sent["payload"]["databases"]["main"]["reachable"] is True


def test_health_database_reports_an_unconstructable_client_rather_than_raising():
    """The real failure mode, not a stand-in for it.

    This test used to inject `RuntimeError`, which `except Exception`
    catches -- so it was green whether or not the handler dealt with what
    `VaultClient()` actually raises. `_env()` raises **SystemExit** for a
    missing variable, and SystemExit derives from BaseException, so the
    original `except Exception` did not catch it: the request thread
    unwound without calling _send and the caller got a dropped connection
    with no status. So no fake client here -- the real one is constructed
    with the variable genuinely absent.
    """
    import os
    handler, sent = _make_handler({}, path="/health/database")
    with patch.dict(os.environ):
        os.environ.pop("CDB_BASE", None)
        handler.do_GET()
    assert sent["status"] == 503, "a missing env var must be reported, not raised"
    assert "CDB_BASE" in sent["payload"]["error"]


def test_readiness_health_stays_green_when_couchdb_is_unreachable():
    """The reason /health/database is a separate path at all.

    /health is this pod's readiness probe. Folding the database check into
    it would pull the platform's only bridge out of its Service over a
    CouchDB blip -- an outage caused by the monitoring, not the fault. The
    bridge can still serve /generate with CouchDB down.
    """
    handler, sent = _make_handler({}, path="/health")
    with _patch_client(_health_payload(reachable=False)):
        handler.do_GET()
    assert sent["status"] == 200
    assert sent["payload"] == {"status": "ok", "draining": False}
