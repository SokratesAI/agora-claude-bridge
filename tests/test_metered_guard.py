"""The owner's rule 9 as a harness decision rather than a paragraph.

Cycle 500 measured that the config-only version of idea #104 does not
work: a `permissions.deny` rule with a parameter specifier on an MCP tool
is silently ignored under `--dangerously-skip-permissions`, which is how
every cycle runs. These tests pin the hook that replaced it, and the
registration -- a correct hook that nothing attaches has already shipped
once in this repo.
"""
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


def test_a_metered_string_nested_in_a_workflow_is_still_found():
    """The guard walks the arguments instead of reading a `model` key --
    the three guarded tools do not agree on where the model sits."""
    reason = metered_guard.decide(_payload(
        "mcp__agora__create_workflow",
        {"steps": [{"name": "a", "config": {"model": "anthropic:claude-opus-5"}}]}))
    assert reason
    assert "anthropic:claude-opus-5" in reason


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
