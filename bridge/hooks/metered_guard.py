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
would run on a metered model** -- a persona, a conversation, a heartbeat,
a workflow. It does not touch a direct API probe: `curl`, `python`, a
one-off script against the Messages API are all still available, because
he explicitly allows testing and research on that balance ("You can
ofcourse use it for testing and research"). What he forbids is anything
*scheduled* or *defaulted* onto it, and in this system that means a
durable object in Agora.

**The model is not always in the arguments, and the first version of this
file said it was.** Cycle 500 shipped a guard that only walked
`tool_input`, and wrote into its own docstring that a heartbeat was "the
known gap". Its reviewer found a second one of exactly the same shape and
it was not a gap in a corner: `create_conversation(name=...,
personaId=<an existing metered persona>)` carries no model at all in its
arguments, and `tools_dispatch` fills one in from the persona afterwards.
So the sentence claiming this was "exactly and only" what blocks a
durable metered object was false when it was written -- a completeness
claim on a check I had not enumerated the inputs of. Both holes close the
same way, so both are closed here rather than one being filed.

`resolve_persona` is the second half: for a tool whose argument is a
`personaId`, ask Agora what model that persona runs on. **It fails
closed**, and that is a real choice with a cost. An unreachable Agora
means a heartbeat or a conversation cannot be created until it is back,
and a cycle hitting that is told exactly why. The alternative -- allow on
a failed lookup -- makes the guard advisory precisely when the system is
unhealthy, which is the state in which nobody is watching. Rule 9 is the
one rule on this estate with a hard money ceiling behind it and no
mechanical backstop, so it gets the direction that errs toward refusing.
Measured Cycle 500, from the bridge pod: the endpoint answers in well
under the hook's 10s budget, and no persona in the live store is metered
today, so this refuses nothing that exists.

The remaining hole, and this time it is enumerated rather than asserted:
`create_conversation` with **neither** `personaId` nor `model` inherits
the model of the persona making the call. That value is not in the
arguments and not resolvable from them -- but a metered persona making
the call is already a metered persona running, which is a spend this hook
was never the thing standing in front of.
"""
import json
import sys
import urllib.error
import urllib.request

# `agora_runner.reply.METERED_PROVIDERS` is `("anthropic",)` -- the bare
# provider, because that module splits on the colon itself. This one keeps
# the colon so it can match with `startswith` and never fire on prose that
# merely mentions a metered model mid-sentence. Same set, different shape,
# duplicated rather than imported: the runner package is not installed in
# the bridge pod, and this hook has to run with nothing but the stdlib.
METERED_PREFIXES = ("anthropic:",)

# Only the tools that persist something. A read tool that happens to echo
# a model string back is not a spender.
GUARDED_TOOLS = (
    "mcp__agora__create_persona",
    "mcp__agora__create_conversation",
    "mcp__agora__create_heartbeat",
    "mcp__agora__create_workflow",
)

# The two that address a persona by id instead of naming a model. The
# public app on :8080 rather than the internal API on :8081, for the same
# reason `tools.heartbeat_health` uses it: it is the one that serves the
# full persona record.
PERSONA_URL = "http://agora.agents.svc.cluster.local:8080/personas/{id}"
LOOKUP_TIMEOUT = 5


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


def resolve_persona(persona_id, opener=None):
    """Return (model, error). Exactly one of them is truthy.

    `opener` is injectable so the tests never touch the network -- the
    real call is one GET and the failure path is the interesting half.
    """
    opener = opener or urllib.request.urlopen
    url = PERSONA_URL.format(id=persona_id)
    try:
        with opener(url, timeout=LOOKUP_TIMEOUT) as response:
            body = json.load(response)
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"
    persona = (body or {}).get("persona") if isinstance(body, dict) else None
    if not isinstance(persona, dict):
        return "", f"Agora answered without a persona record for {persona_id}"
    return str(persona.get("model") or ""), ""


def decide(payload, opener=None):
    """Return the deny reason, or "" to stay out of the way.

    Staying out of the way means printing nothing at all: a PreToolUse
    hook that emits no decision leaves the call exactly as it was, which
    is the right default for every tool this does not guard.
    """
    tool = payload.get("tool_name") or ""
    if tool not in GUARDED_TOOLS:
        return ""
    tool_input = payload.get("tool_input") or {}
    hits = metered_values(tool_input)
    persona_id = tool_input.get("personaId") if isinstance(tool_input, dict) else None
    if not hits and persona_id:
        model, error = resolve_persona(str(persona_id), opener=opener)
        if error:
            return (
                f"Refused: {tool} names persona {persona_id}, and this guard could not "
                f"ask Agora which model that persona runs on ({error}). It fails closed "
                "on purpose -- identity.md rule 9 is the one rule here with a hard money "
                "ceiling and no other backstop, so an unverifiable call is refused rather "
                "than allowed. Retry when Agora answers."
            )
        if model.startswith(METERED_PREFIXES):
            hits = [model]
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
    except Exception as exc:
        # A hook that cannot parse its input must not block the turn --
        # the same exit-0 contract quota_notice.py and deadline_notice.py
        # use. But those two fail open on a *notice*, and this one fails
        # open on a spend guard, so it says so on stderr rather than going
        # dark in silence. A payload-shape change in a future CLI would
        # otherwise switch this off with nothing anywhere recording it.
        print(f"metered_guard: unreadable hook payload, allowing "
              f"({type(exc).__name__}: {exc})", file=sys.stderr)
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
