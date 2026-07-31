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

CLI_TIMEOUT_SECONDS = int(os.environ.get("CLI_TIMEOUT_SECONDS", "300"))
