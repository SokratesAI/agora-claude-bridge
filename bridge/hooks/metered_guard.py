"""Claude Code hook: refuse to build a durable Agora object on the metered API.

The owner's hard rule (identity.md rule 9, his words in issues.md
2026-08-10): production never spends the prepaid `anthropic:` balance,
because there is about $16 on it and he has said he will not refill it.
`claude-cli:` is the same set of models through the flat subscription, so
there is never a model to lose here -- only a provider to switch.

Until now that rule lived in two places a cycle can reason past: a
paragraph in its own constitution, and `agora_runner.reply`'s check, which
is code a cycle edits every day. This is his idea #104 -- put it in the
harness instead -- and the shape it takes was measured rather than
assumed (Cycle 500, 2026-08-26, CLI 2.1.245):

- A `permissions.deny` rule **does** fire under
  `--dangerously-skip-permissions`, which is how every cycle runs.
  `Bash(echo:*)` refused `echo NOVA_CANARY_1`.
- A deny rule naming an MCP tool **whole** works too: with
  `mcp__canary__make_thing` denied the tool left the roster entirely.
- **A deny rule with a parameter specifier on an MCP tool is silently
  ignored.** `mcp__canary__make_thing(model:anthropic:*)` did not refuse
  a call made with exactly that value; it returned the tool's real
  output. No warning, no error. So the obvious implementation of #104 is
  a guard that reports itself working while guarding nothing, which is
  the one failure shape this loop has paid for three times.
- A PreToolUse hook returning `permissionDecision: "deny"` **does** stop
  the call under the same flag, and the reason string reaches the model:
  the metered call was refused and the `claude-cli:` one on the next run
  went through untouched.

Scope, deliberately narrow. This refuses **creating an Agora object that
carries a metered model** -- a persona, a conversation, a workflow. It
does not touch a direct API probe: `curl`, `python`, a one-off script
against the Messages API are all still available, because he explicitly
allows testing and research on that balance ("You can ofcourse use it for
testing and research"). What he forbids is anything *scheduled* or
*defaulted* onto it, and in this system that means a durable object in
Agora, which is exactly and only what this blocks.

The known gap, written down rather than papered over: `create_heartbeat`
takes a `personaId` and no model, so a heartbeat attached to a metered
persona that already exists is not visible from the arguments and is not
caught here. Closing it needs a lookup against the Agora API from inside
a hook with a 10s budget; it is a separate change, not a silent
limitation of this one.
"""
import json
import sys

# `agora_runner.reply.METERED_PROVIDERS` is the same list, deliberately
# duplicated rather than imported: the runner package is not installed in
# the bridge pod, and this hook has to be readable by the CLI process
# with no imports beyond the stdlib.
METERED_PREFIXES = ("anthropic:",)

# Only the tools that persist something. A read tool that happens to echo
# a model string back is not a spender.
GUARDED_TOOLS = (
    "mcp__agora__create_persona",
    "mcp__agora__create_conversation",
    "mcp__agora__create_workflow",
)


def metered_values(tool_input):
    """Every string anywhere in the arguments that names a metered provider.

    Walks the whole structure rather than reading a `model` key: the tools
    above do not agree on where the model sits (a workflow carries steps),
    and a guard that reads one key is a guard that a new tool shape walks
    around.
    """
    found = []
    stack = [tool_input]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if item.startswith(METERED_PREFIXES):
                found.append(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return found


def decide(payload):
    """Return the deny reason, or "" to stay out of the way.

    Staying out of the way means printing nothing at all: a PreToolUse
    hook that emits no decision leaves the call exactly as it was, which
    is the right default for every tool this does not guard.
    """
    tool = payload.get("tool_name") or ""
    if tool not in GUARDED_TOOLS:
        return ""
    hits = metered_values(payload.get("tool_input") or {})
    if not hits:
        return ""
    named = ", ".join(sorted(set(hits)))
    return (
        f"Refused: {tool} would create a durable Agora object on the metered "
        f"Anthropic API ({named}). identity.md rule 9 -- production never spends "
        "that balance. Every metered model has an identical 'claude-cli:' twin on "
        "the flat subscription; use that instead. Testing the metered API by hand "
        "is still allowed, this only stops a persisted object from being built on it."
    )


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        # A hook that cannot parse its input must not block the turn. It
        # also must not pretend to have checked -- but there is nothing to
        # report to, so the honest failure here is silence plus exit 0,
        # the same contract quota_notice.py and deadline_notice.py use.
        return 0
    reason = decide(payload if isinstance(payload, dict) else {})
    if reason:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
