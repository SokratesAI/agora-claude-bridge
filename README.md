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

## Credentials

`bridge/credentials.py` bootstraps `~/.claude/.credentials.json` from the
`CLAUDE_CREDENTIALS_JSON` env var, once, on first boot only -- it never
overwrites an existing file. Set that env var to the *complete, unmodified*
contents of a real `~/.claude/.credentials.json` from a machine where
`claude` is logged in (e.g. `agents/claude-auth`'s `credentials_json` key
in the cluster). Deliberately no field-by-field reconstruction: an earlier
version assembled the file from three separate fields (access token,
refresh token, expiry) and the CLI's own client-side validation rejected
it instantly ("Not logged in") -- the real file carries additional fields
(`scopes`, etc.) that reconstruction had silently dropped. Piping the whole
original file through sidesteps needing to know its exact schema.

Once the CLI does its own real token refresh, the fresh credentials live
*only* on the `CLAUDE_HOME` PVC -- the source secret's refresh token
becomes single-use-stale at that point (same reason the old
`claude-auth-refresher` CronJob died, see above). If credentials ever need
a hard reset, delete `$CLAUDE_HOME/.claude/.credentials.json` from the PVC
first, then restart the pod -- restarting alone never re-bootstraps.

## Local dev

```
pip install pytest
pytest tests/ -q
```

No `requirements.txt` -- the bridge itself is stdlib-only at runtime. The
`claude` CLI is a separate Node package, installed in the Docker image
only; local tests mock every subprocess call.
