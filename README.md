# agora-claude-bridge

Persistent Claude Code CLI bridge for Agora -- lets personas use real Claude
models (subscription-authenticated) via a model provider, same shape as the
existing Anthropic API/Gemini providers in `agora-persona-runner`.

## Why a separate service

The old `claude-auth-refresher` CronJob died because multiple pods shared
one OAuth token file: the CLI's own refresh token is single-use, so
whichever pod refreshed first invalidated it for the others. This service
is the *only* process anywhere that holds the credential -- one pod, one
CLI session/token. Every real subprocess invocation is additionally
serialized within that one process (`bridge/cli.py`'s lock), since a
concurrent second invocation could still race the same underlying refresh
even inside one process -- unverified either way until live-tested against
the real subscription.

## API

`POST /generate` (auth: `x-bridge-token` header, if `BRIDGE_TOKEN` is set)

```json
{"conversation_id": "...", "system": "...", "prompt": "...", "model": "claude-..."}
```

Returns `{"text": "...", "thinking": "..."}`. `thinking` is empty when the
model didn't produce a visible thought block for that turn.

One persistent Claude Code session per `conversation_id`, resumed via
`--resume` across turns and pod restarts (session ids persisted to a JSON
file on the `CLAUDE_HOME` PVC). The system/persona prompt is only sent on
a conversation's first turn -- the resumed session already has it as
context after that.

Errors: `429 {"error": "usage_limit"}` for a real subscription cap (don't
retry immediately -- wait for the interval named in `detail`, or a few
hours if none was parseable); `502 {"error": "cli_error"}` for anything
else that prevented a usable reply.

## Local dev

```
pip install pytest
pytest tests/ -q
```

No `requirements.txt` -- the bridge itself is stdlib-only at runtime. The
`claude` CLI is a separate Node package, installed in the Docker image
only; local tests mock every subprocess call.
