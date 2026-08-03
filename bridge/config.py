"""Environment configuration -- no requirements.txt, stdlib-only at runtime."""
import os

PORT = int(os.environ.get("PORT", "8090"))

# BRIDGE_TOKEN gates the HTTP API the same way AGORA_TOKEN gates
# agora-persona-runner's own internal app -- shared-secret, not per-caller
# identity (Decisions/0007 in the agora vault project: one trust domain,
# theater to do more given the topology).
BRIDGE_TOKEN = os.environ.get("BRIDGE_TOKEN", "")

# CLAUDE_HOME is expected to be a PVC mount -- credentials AND every
# conversation's session file must survive pod restarts, or every restart
# silently starts a fresh session for every conversation (defeating the one
# thing this service exists to provide over the raw Anthropic API).
CLAUDE_HOME = os.environ.get("CLAUDE_HOME", "/data/claude-home")
SESSIONS_FILE = os.path.join(CLAUDE_HOME, "agora-sessions.json")

# Per the interview with Edvard (2026-07-31): one persistent CLI session per
# Agora conversation, resumed via --resume across turns AND pod restarts.
CLAUDE_WORKSPACE = os.environ.get("CLAUDE_WORKSPACE", "/data/workspace")

# 300s was the original default; live-tested 2026-08-01 with the Evolve
# workflow's Coder step (real git clone + explore + edit + pytest + push +
# gh pr create in one turn) and it timed out on round 1 -- a real coding
# task routinely needs more than 5 minutes. 900s covered that one step.
# Bumped 900 -> 2700 on 2026-08-03: since the v2 single-session redesign,
# one call now does the Coder step's work PLUS reading state, deciding,
# reviewing its own diff, merging, health-checking, and journaling --
# Cycle 8 hit the 900s wall with no PR ever pushed (nothing lost, but a
# wasted cycle). 2700s (45min) leaves real headroom for that whole arc
# without letting one truly stuck call hold the bridge's single-instance
# lock for the better part of an hour.
CLI_TIMEOUT_SECONDS = int(os.environ.get("CLI_TIMEOUT_SECONDS", "2700"))
