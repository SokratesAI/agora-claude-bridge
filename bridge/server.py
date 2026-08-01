"""POST /generate -- the only real endpoint. Mirrors agora-persona-runner's
own invoke_server.py: BaseHTTPRequestHandler + ThreadingHTTPServer, a _send
helper, x-agora-token-style header auth. Threaded server is fine even
though cli.run_turn serializes internally (see cli.py) -- request parsing/
auth/session lookup don't need to wait on each other, only the actual
subprocess invocation does.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bridge.config import BRIDGE_TOKEN, PORT
from bridge.log import log
from bridge.sessions import clear_session_id, get_session_id, set_session_id
from bridge.cli import (
    ClaudeCliError, UsageLimitError, SESSION_NOT_FOUND, DISCOVERED_FULL_TOOL_ROSTER, run_turn,
)


def generate(conversation_id, system, prompt, model=None, restricted=False, stateless=False):
    """One turn for one conversation. First turn (no stored session)
    prepends the system/persona prompt to the message -- a resumed session
    already has that context from turn 1, so later turns send just the new
    message, same as the old Slack bridge did.

    restricted (2026-08-01, off by default): when true, blocks the full
    known tool roster (DISCOVERED_FULL_TOOL_ROSTER) for this call --
    Edvard's call is that this service should be as capable as an
    interactive Claude Code session by default, with restriction an
    explicit per-persona opt-in rather than a silent default.

    stateless (2026-08-01, off by default): when true, never reads or
    writes this conversation's stored session_id -- every call gets the
    full system+prompt and starts a fresh CLI session with no --resume.
    Built for the Evolve workflow: its steps are deliberately bounded,
    single-purpose invocations that should only carry the context their
    own prompt gives them, not an ever-growing CLI-side memory of every
    prior cycle (that's what the vault journal is for -- see identity.md).
    An ordinary chat persona wants the opposite (continuity across turns),
    which is why this is opt-in, not the new default."""
    if stateless:
        message = f"{system}\n\n{prompt}"
        text, thinking, _ = run_turn(
            message, session_id=None, model=model,
            disallowed_tools=DISCOVERED_FULL_TOOL_ROSTER if restricted else None,
        )
        return text, thinking

    session_id = get_session_id(conversation_id)
    message = f"{system}\n\n{prompt}" if not session_id else prompt
    disallowed_tools = DISCOVERED_FULL_TOOL_ROSTER if restricted else None

    try:
        text, thinking, new_session_id = run_turn(
            message, session_id=session_id, model=model, disallowed_tools=disallowed_tools,
        )
    except ClaudeCliError as e:
        if str(e) == SESSION_NOT_FOUND:
            log(f"conversation={conversation_id}: stored session gone, retrying fresh")
            clear_session_id(conversation_id)
            message = f"{system}\n\n{prompt}"
            text, thinking, new_session_id = run_turn(
                message, session_id=None, model=model, disallowed_tools=disallowed_tools,
            )
        else:
            raise

    if new_session_id:
        set_session_id(conversation_id, new_session_id)
    return text, thinking


class BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet default request logging
        pass

    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, {"error": "not found"})
            return
        if BRIDGE_TOKEN and self.headers.get("x-bridge-token") != BRIDGE_TOKEN:
            self._send(401, {"error": "invalid bridge token"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            conversation_id = payload.get("conversation_id")
            prompt = payload.get("prompt")
            if not conversation_id or not prompt:
                self._send(400, {"error": "conversation_id and prompt are required"})
                return
            system = payload.get("system", "")
            model = payload.get("model")
            restricted = bool(payload.get("restricted", False))
            stateless = bool(payload.get("stateless", False))

            text, thinking = generate(
                conversation_id, system, prompt, model=model, restricted=restricted, stateless=stateless,
            )
            self._send(200, {"text": text, "thinking": thinking})
        except UsageLimitError as e:
            self._send(429, {"error": "usage_limit", "detail": str(e)[:300]})
        except ClaudeCliError as e:
            self._send(502, {"error": "cli_error", "detail": str(e)[:300]})
        except Exception as e:
            log(f"/generate failed: {e}")
            self._send(500, {"error": str(e)[:300]})


def start_server():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), BridgeHandler)
    log(f"agora-claude-bridge listening on :{PORT}")
    server.serve_forever()


def start_server_background():
    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()
    return thread
