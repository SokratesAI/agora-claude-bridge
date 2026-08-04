"""Narrates a live CLI session's tool calls back to agora-persona-runner,
which turns each one into an inline Activity chip in the conversation.

Edvard, on whether he wanted every tool call or only the ones that change
something (2026-08-03): "All. I want to know whats going on. It takes away
my feeling of control if everything is hidden."

Why we report to the runner and not to Agora directly: the write path that
renders a chip is Agora's INTERNAL /audit, behind AGORA_TOKEN, and this
pod's ServiceAccount deliberately cannot read Secrets in `agents`. The
runner hands us a single-purpose token per /generate call instead, scoped
to that one conversation and revoked when the call returns -- see that
repo's agora_runner/tool_activity.py for the full reasoning.

Best-effort throughout: a report that fails must never disturb the turn it
is describing. The CLI is mid-session doing real work someone is waiting
on, and a chip is worth strictly less than that.
"""
import json
import queue
import threading
import urllib.error
import urllib.request

from bridge.log import log

POST_TIMEOUT_SECONDS = 5

# How long close() waits for chips queued at the very end of a session.
# Short on purpose: the caller has a finished reply in hand and returning
# it matters more than the last few labels.
CLOSE_WAIT_SECONDS = 5

# The chip is a one-line label in a chat bubble, not a transcript. The
# runner truncates at 500 server-side anyway (agora_runner/audit.py).
DETAIL_CHARS_MAX = 300

# Per tool, the input fields that actually say what the call DID, in
# preference order. A tool that isn't listed -- or a call that has none of
# its listed fields -- falls back to a compact dump of the whole input,
# which is the right default for tools we have never seen: new ones appear
# between CLI versions (see cli.py's DISCOVERED_FULL_TOOL_ROSTER note).
_DETAIL_FIELDS = {
    "Bash": ("command",),
    "Read": ("file_path",),
    "Write": ("file_path",),
    "Edit": ("file_path",),
    "NotebookEdit": ("notebook_path",),
    "Glob": ("pattern",),
    "Grep": ("pattern",),
    "WebFetch": ("url",),
    "WebSearch": ("query",),
    "Task": ("description",),
    "Agent": ("description",),
    "Skill": ("skill",),
    "ToolSearch": ("query",),
    "SendMessage": ("message",),
}


def summarize(name, tool_input):
    """One line describing what this call did, for the chip's label."""
    if not isinstance(tool_input, dict) or not tool_input:
        return ""
    for field in _DETAIL_FIELDS.get(name, ()):
        value = tool_input.get(field)
        if isinstance(value, str) and value.strip():
            # Collapse whitespace: a multi-line heredoc or a formatted
            # patch would otherwise blow out a one-line chat bubble.
            return " ".join(value.split())[:DETAIL_CHARS_MAX]
    try:
        return json.dumps(tool_input, ensure_ascii=False)[:DETAIL_CHARS_MAX]
    except (TypeError, ValueError):
        return str(tool_input)[:DETAIL_CHARS_MAX]


def _post(url, payload):
    """True if the runner accepted the report. Never raises."""
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=POST_TIMEOUT_SECONDS) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


class ActivityReporter:
    """One background sender per CLI invocation.

    A thread per report would be simpler and wrong: chips would land out of
    order, and the order is the entire point of showing them live rather
    than dumping them at the end. A single worker draining a queue keeps
    them ordered without blocking the stdout reader, whose only job is to
    keep the CLI's stream moving.

    Disabled (every method a no-op) when the caller passed no activity
    block -- an older runner that doesn't send one, or the /invoke path
    where there is no conversation to post into.
    """

    def __init__(self, activity):
        block = activity if isinstance(activity, dict) else {}
        self._url = str(block.get("url") or "")
        self._token = str(block.get("token") or "")
        self._queue = queue.Queue()
        self._thread = None

    @property
    def enabled(self):
        return bool(self._url and self._token)

    def start(self):
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def report(self, name, tool_input):
        if self.enabled and name:
            self._queue.put((name, summarize(name, tool_input)))

    def close(self):
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=CLOSE_WAIT_SECONDS)
        self._thread = None

    def _drain(self):
        failures = 0
        while True:
            item = self._queue.get()
            if item is None:
                return
            name, detail = item
            # _post swallows its own errors, but an unexpected one from
            # anywhere here would kill this thread and silently end the
            # narration for the rest of the session -- the exact
            # fails-invisibly shape this whole feature exists to remove.
            try:
                ok = _post(self._url, {"token": self._token, "capability": name, "detail": detail})
            except Exception:
                ok = False
            if not ok:
                failures += 1
                # Log the first one only. A runner that is down or has
                # revoked the token fails for every remaining call in the
                # session, and one broken chip must not bury the CLI's own
                # logs under hundreds of identical lines.
                if failures == 1:
                    log(f"activity report failed for {name!r} "
                        f"(further failures this session silenced)")
