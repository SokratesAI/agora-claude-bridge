"""POST /generate -- the only real endpoint. Mirrors agora-persona-runner's
own invoke_server.py: BaseHTTPRequestHandler + ThreadingHTTPServer, a _send
helper, x-agora-token-style header auth. Threaded server is fine even
though cli.run_turn serializes internally (see cli.py) -- request parsing/
auth/session lookup don't need to wait on each other, only the actual
subprocess invocation does.
"""
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bridge.config import BRIDGE_TOKEN, PORT
from bridge.log import log
from bridge.sessions import clear_session_id, get_session_id, set_session_id
from bridge.cli import (
    ClaudeCliError, UsageLimitError, SESSION_NOT_FOUND, DISCOVERED_FULL_TOOL_ROSTER, run_turn,
)


# Set by the SIGTERM/SIGINT handler, read by start_server's main thread and
# by do_POST. A plain module bool rather than a threading.Event for the same
# reason agora_runner/main.py uses one: a signal handler runs on the main
# thread between bytecodes, so touching a lock there could re-enter a lock
# that same thread already holds. A bool assignment cannot deadlock.
_shutdown_requested = False

# Turns currently inside generate(). The drain waits on this reaching zero.
_in_flight = 0
_in_flight_changed = threading.Condition()


def shutdown_requested():
    return _shutdown_requested


def _request_shutdown(signum, _frame):
    """Deliberately does NOT exit -- that is the whole point.

    Python's default SIGTERM disposition kills this process immediately,
    and a /generate call is minutes of real work whose result only reaches
    the caller when it returns. agora-persona-runner posts the persona's
    reply after that; kill the bridge mid-turn and the reply is destroyed
    even though the runner itself drains correctly (agora_runner/main.py,
    2026-08-03).

    That is not hypothetical here: merging a PR into this repo rolls the
    pod hosting the Claude Code session that merged it, and it has now
    ended four Nova cycles mid-sentence (2026-08-04/05, cycles 20-21).
    The runner got this fix in #32 and the matching grace period in
    agora-persona-runner-config#3; the bridge never did.

    Handling the signal turns the kill into a drain: stop accepting new
    turns, let the one in flight finish and return its text, then exit 0.
    Only works if terminationGracePeriodSeconds outlasts a real turn --
    it was 30s against a 2700s CLI timeout; raised alongside this in
    agora-claude-bridge-config."""
    global _shutdown_requested
    _shutdown_requested = True
    log(f"received signal {signum}, draining: finishing the in-flight turn, then exiting")


def install_signal_handlers():
    """Called from run.py, on the main thread -- signal.signal() raises
    ValueError anywhere else, which is why this is not inside
    start_server() (start_server_background would break)."""
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)


def _enter_turn():
    """Registers a turn as in flight; returns False if the pod is already
    draining and the caller must not start one.

    The check and the increment happen under one lock deliberately. Split
    into two steps they leave a window where a request that read the flag
    just before SIGTERM arrived increments *after* _await_drain has already
    seen zero and let the process exit -- losing exactly the turn this
    whole change exists to protect."""
    global _in_flight
    with _in_flight_changed:
        if _shutdown_requested:
            return False
        _in_flight += 1
        return True


def _leave_turn():
    global _in_flight
    with _in_flight_changed:
        _in_flight -= 1
        _in_flight_changed.notify_all()


def _await_drain():
    """Block until no turn is in flight.

    No timeout on purpose. run_turn already bounds itself at
    CLI_TIMEOUT_SECONDS (2700s), and Kubernetes bounds the whole drain at
    terminationGracePeriodSeconds and then SIGKILLs. A third limit here
    would be a number guarding nothing either of those already covers,
    and getting it wrong means throwing away the reply this drain exists
    to save."""
    with _in_flight_changed:
        while _in_flight:
            log(f"draining: {_in_flight} turn(s) still in flight")
            _in_flight_changed.wait(timeout=30)


def generate(conversation_id, system, prompt, model=None, restricted=False, stateless=False,
             activity=None):
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
    which is why this is opt-in, not the new default.

    activity (2026-08-03, absent by default): {"url", "token"} the caller
    wants every tool call reported to as it happens, so a session that runs
    for 45 minutes isn't a black box until it returns. See activity.py."""
    if stateless:
        message = f"{system}\n\n{prompt}"
        text, thinking, _ = run_turn(
            message, session_id=None, model=model,
            disallowed_tools=DISCOVERED_FULL_TOOL_ROSTER if restricted else None,
            activity=activity,
        )
        return text, thinking

    session_id = get_session_id(conversation_id)
    message = f"{system}\n\n{prompt}" if not session_id else prompt
    disallowed_tools = DISCOVERED_FULL_TOOL_ROSTER if restricted else None

    try:
        text, thinking, new_session_id = run_turn(
            message, session_id=session_id, model=model, disallowed_tools=disallowed_tools,
            activity=activity,
        )
    except ClaudeCliError as e:
        if str(e) == SESSION_NOT_FOUND:
            log(f"conversation={conversation_id}: stored session gone, retrying fresh")
            clear_session_id(conversation_id)
            message = f"{system}\n\n{prompt}"
            text, thinking, new_session_id = run_turn(
                message, session_id=None, model=model, disallowed_tools=disallowed_tools,
                activity=activity,
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
            # Still 200 while draining: the readiness probe going red would
            # pull this pod out of the Service, and with replicas=1 and
            # strategy Recreate there is nothing to fail over to -- it would
            # only turn the honest 503 below into a connection refused.
            self._send(200, {"status": "ok", "draining": _shutdown_requested})
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
            activity = payload.get("activity")

            if not _enter_turn():
                # Starting a 45-minute turn on a pod that is already
                # shutting down guarantees losing it. Fail fast so the
                # caller can retry against the replacement pod instead.
                self._send(503, {"error": "shutting_down"})
                return
            try:
                text, thinking = generate(
                    conversation_id, system, prompt, model=model, restricted=restricted,
                    stateless=stateless, activity=activity,
                )
            finally:
                _leave_turn()
            self._send(200, {"text": text, "thinking": thinking})
        except UsageLimitError as e:
            self._send(429, {"error": "usage_limit", "detail": str(e)[:300]})
        except ClaudeCliError as e:
            self._send(502, {"error": "cli_error", "detail": str(e)[:300]})
        except Exception as e:
            log(f"/generate failed: {e}")
            self._send(500, {"error": str(e)[:300]})


def start_server():
    """Serves until SIGTERM/SIGINT, then drains and returns.

    serve_forever runs on a worker thread rather than this one because
    BaseServer.shutdown() blocks until the serve_forever loop has exited --
    calling it from the thread running that loop deadlocks. So this thread
    stays free to notice the flag and do the shutdown."""
    server = ThreadingHTTPServer(("0.0.0.0", PORT), BridgeHandler)
    log(f"agora-claude-bridge listening on :{PORT}")
    serving = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.5},
                               daemon=True)
    serving.start()

    while not _shutdown_requested:
        time.sleep(0.5)

    server.shutdown()  # stop accepting; in-flight handler threads keep running
    _await_drain()
    server.server_close()
    log("drain complete, exiting")


def start_server_background():
    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()
    return thread
