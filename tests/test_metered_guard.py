"""The owner's rule 9 as a harness decision rather than a paragraph.

Cycle 500 measured that the config-only version of idea #104 does not
work: a `permissions.deny` rule with a parameter specifier on an MCP tool
is silently ignored under `--dangerously-skip-permissions`, which is how
every cycle runs. These tests pin the hook that replaced it, and the
registration -- a correct hook that nothing attaches has already shipped
once in this repo.
"""
import io
import json

from bridge import quota
from bridge.hooks import metered_guard


def _payload(tool, tool_input):
    return {"tool_name": tool, "tool_input": tool_input}


def test_metered_persona_is_refused():
    reason = metered_guard.decide(
        _payload("mcp__agora__create_persona", {"model": "anthropic:claude-sonnet-5"}))
    assert reason
    assert "anthropic:claude-sonnet-5" in reason
    assert "claude-cli:" in reason


def test_subscription_persona_is_untouched():
    assert metered_guard.decide(
        _payload("mcp__agora__create_persona",
                 {"model": "claude-cli:claude-sonnet-5"})) == ""


def test_a_metered_string_nested_anywhere_in_the_arguments_is_found():
    """The guard walks the arguments instead of reading a `model` key.

    Note what this does and does not prove. `create_workflow`'s schema
    carries no per-step model today (`steps[].{prompt, loopCount}` only),
    so no metered string can currently reach it by this route -- its place
    in GUARDED_TOOLS is a forward guard against a schema that grows one,
    not a hole being plugged. What the walk does close today is any
    argument shape on the other three tools that is not a top-level
    `model` key."""
    reason = metered_guard.decide(_payload(
        "mcp__agora__create_workflow",
        {"steps": [{"name": "a", "config": {"model": "anthropic:claude-opus-5"}}]}))
    assert reason
    assert "anthropic:claude-opus-5" in reason


class _Answer:
    """Enough of urlopen's context-manager contract for json.load."""

    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *args):
        return self._body


def _opener(body):
    def open_it(url, timeout=None):
        return _Answer(body)
    return open_it


def _raiser(exc):
    def open_it(url, timeout=None):
        raise exc
    return open_it


def test_a_conversation_on_an_already_metered_persona_is_refused():
    """The bypass Cycle 500's reviewer found: `personaId` and no model, so
    the metered value is never in the arguments at all -- tools_dispatch
    fills it in from the persona after the hook has already answered."""
    reason = metered_guard.decide(
        _payload("mcp__agora__create_conversation", {"name": "x", "personaId": "p-1"}),
        opener=_opener({"persona": {"id": "p-1", "model": "anthropic:claude-opus-5"}}))
    assert reason
    assert "anthropic:claude-opus-5" in reason


def test_a_heartbeat_on_a_subscription_persona_is_untouched():
    assert metered_guard.decide(
        _payload("mcp__agora__create_heartbeat", {"personaId": "p-2"}),
        opener=_opener({"persona": {"id": "p-2", "model": "claude-cli:claude-sonnet-5"}})) == ""


def test_an_unresolvable_persona_fails_closed():
    """Allowing on a failed lookup would make the guard advisory exactly
    when the system is unhealthy, which is when nobody is watching."""
    reason = metered_guard.decide(
        _payload("mcp__agora__create_heartbeat", {"personaId": "p-3"}),
        opener=_raiser(OSError("connection refused")))
    assert reason
    assert "fails closed" in reason
    assert "connection refused" in reason


def test_an_answer_without_a_persona_record_also_fails_closed():
    reason = metered_guard.decide(
        _payload("mcp__agora__create_conversation", {"name": "x", "personaId": "p-4"}),
        opener=_opener({"error": "not found"}))
    assert reason
    assert "p-4" in reason


def test_no_lookup_happens_when_the_arguments_already_name_a_model():
    """A persona created with an explicit metered model must be refused on
    the arguments alone -- there is no persona to look up yet."""
    reason = metered_guard.decide(
        _payload("mcp__agora__create_persona", {"model": "anthropic:claude-opus-5"}),
        opener=_raiser(AssertionError("must not be called")))
    assert "anthropic:claude-opus-5" in reason


def test_main_reads_stdin_and_prints_the_deny_envelope(monkeypatch, capsys):
    """main() is the entrypoint the CLI actually runs, and Cycle 500's
    tests asserted on a dict they had built themselves instead."""
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_payload(
        "mcp__agora__create_persona", {"model": "anthropic:claude-opus-5"}))))
    assert metered_guard.main() == 0
    block = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert block["hookEventName"] == "PreToolUse"
    assert block["permissionDecision"] == "deny"
    assert "anthropic:claude-opus-5" in block["permissionDecisionReason"]


def test_main_prints_nothing_when_the_call_is_allowed(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_payload(
        "mcp__agora__create_persona", {"model": "claude-cli:claude-sonnet-5"}))))
    assert metered_guard.main() == 0
    assert capsys.readouterr().out == ""


def test_main_stays_out_of_the_way_on_unparseable_input(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert metered_guard.main() == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    # but it does not go dark in silence -- a spend guard that switched
    # itself off on a payload-shape change would leave no trace at all
    assert "metered_guard" in captured.err


def test_an_unguarded_tool_carrying_the_string_is_not_blocked():
    """Reading a metered model back out of the store is not spending it.
    A guard that fires on any mention would make the store unreadable."""
    assert metered_guard.decide(
        _payload("mcp__agora__list_personas",
                 {"filter": "anthropic:claude-sonnet-5"})) == ""


def test_the_decision_is_the_shape_the_cli_honours():
    """Measured against CLI 2.1.245: this exact envelope denies the call
    under --dangerously-skip-permissions and the reason reaches the model."""
    out = json.loads(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": metered_guard.decide(
            _payload("mcp__agora__create_conversation",
                     {"model": "anthropic:claude-opus-5"})),
    }}))
    block = out["hookSpecificOutput"]
    assert block["hookEventName"] == "PreToolUse"
    assert block["permissionDecision"] == "deny"
    assert block["permissionDecisionReason"]


def test_the_guard_is_actually_registered(tmp_path):
    """A correct hook that nothing attaches is the failure this repo has
    already had once (test_hook_settings_attach_both_bridge_hooks)."""
    path = quota.write_hook_settings(str(tmp_path / "s.json"))
    hooks = json.load(open(path))["hooks"]
    assert "PreToolUse" in hooks
    assert hooks["PreToolUse"][0]["matcher"] == "*"
    commands = " ".join(h["command"] for h in hooks["PreToolUse"][0]["hooks"])
    assert quota.METERED_GUARD_SCRIPT in commands
    # and it does not ride the two notice events, where a "deny" envelope
    # means nothing
    for event in ("UserPromptSubmit", "PostToolUse"):
        attached = " ".join(h["command"] for h in hooks[event][0]["hooks"])
        assert quota.METERED_GUARD_SCRIPT not in attached
