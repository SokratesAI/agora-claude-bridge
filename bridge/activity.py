"""Narrates a live CLI session's tool calls back to agora-persona-runner,
which turns each one into an inline Activity chip in the conversation.

The owner, on whether he wanted every tool call or only the ones that change
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
from bridge.redact import redact

POST_TIMEOUT_SECONDS = 5

# How long close() waits for chips queued at the very end of a session.
# Short on purpose: the caller has a finished reply in hand and returning
# it matters more than the last few labels.
CLOSE_WAIT_SECONDS = 5

# The chip is a one-line label in a chat bubble, not a transcript. The
# runner truncates at 500 server-side anyway (agora_runner/audit.py).
DETAIL_CHARS_MAX = 300

# What a tool RETURNED, which is a transcript and is meant to be read in the
# expandable detail view, not in the chip label -- so it gets a budget three
# orders of magnitude larger than DETAIL_CHARS_MAX and no whitespace
# collapsing.
#
# This is not a new limit, it is Agora's existing one applied one hop
# earlier: AuditStore.CONTENT_CHARS_MAX (agora/src/chat/audit-store.ts) is
# 20_000 and slices anything longer on arrival. Sending more than this would
# push bytes across two HTTP hops to be provably discarded at the far end --
# and unlike a vault file (before/after, bounded and human-written), tool
# output is routinely enormous: one `cat` of a log, one unbounded `git log`.
# Keep this in step with that constant if it ever moves.
OUTPUT_CHARS_MAX = 20_000

# The capability name a written passage travels under. Not a tool, and Agora
# renders it as prose rather than as a chip -- see that repo's public/app.js
# NARRATION_TEXT, which has to agree with this string.
NARRATION_TEXT = "assistant_text"

# A subagent's whole life as one chip. The Agent call that launched it is
# already narrated like any other tool, but everything the subagent then did
# used to be invisible: the CLI drops child events from the parent stream
# unless --forward-subagent-text is passed (cli.py).
#
# Start and finish travel under the same id so Agora's client folds them into
# a single chip, exactly as it already does for a tool call and its output.
# That id is the CLI's `task_id`, NOT the Agent call's `tool_use_id` -- the
# latter is already the id of the Agent chip itself, and reusing it would put
# two different calls in the client's `callsById` map under one key.
SUBAGENT = "subagent"

# Marks a line as a subagent's work rather than the persona's own. Everything
# a subagent does arrives on the same stream as the persona's own actions and
# is otherwise indistinguishable from it -- so a passage a subagent wrote
# would read, in the drawer, as if the persona had written it.
SUBAGENT_PREFIX = "↳"

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


def result_text(block):
    """The text a tool returned, from one CLI `tool_result` content block.

    The CLI gives `content` either as a plain string or as a list of typed
    blocks (text, and for screenshot-style tools, images). Only text is
    readable as output; a non-text block is named rather than dropped, so a
    screenshot reads as `[image]` instead of as an empty result, which would
    be indistinguishable from a tool that genuinely returned nothing.
    """
    content = block.get("content")
    if isinstance(content, str):
        return content[:OUTPUT_CHARS_MAX]
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        elif item.get("type"):
            parts.append(f"[{item['type']}]")
    return "\n".join(parts)[:OUTPUT_CHARS_MAX]


# The fields carrying text a human will read. `token` is deliberately not
# among them: it is this post's own credential, issued by the runner for
# this one conversation, and redacting it would break the report rather
# than protect anything.
_READABLE_FIELDS = ("detail", "output")


def _scrubbed(payload):
    """`payload` with credentials stripped out of its human-read fields.

    Done here, at the one point everything leaves for the runner, rather
    than in each report method -- narration and results both had to be
    covered, and a third call site added later would otherwise ship
    unfiltered by simply not knowing about this (redact.py).
    """
    for field in _READABLE_FIELDS:
        if field in payload:
            payload[field] = redact(payload[field])
    return payload


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

    def report(self, name, tool_input, tool_use_id="", subagent=""):
        """The call, at the moment it starts.

        Still posted before the tool has run, which is what makes the
        narration live -- a `pytest` that takes four minutes must show up
        when it starts, not when it finishes. `tool_use_id` is what lets
        report_result() below catch up with this chip afterwards.

        `subagent`: the description of the subagent that made this call, when
        it wasn't the persona itself. Only the label changes -- the chip is
        still the tool's own, and its result still pairs with it by
        tool_use_id, because a subagent's `Bash` call is a real `Bash` call.
        """
        if self.enabled and name:
            detail = summarize(name, tool_input)
            if subagent:
                detail = f"{SUBAGENT_PREFIX} {subagent} · {detail}" if detail \
                    else f"{SUBAGENT_PREFIX} {subagent}"
            payload = {"capability": name, "detail": detail}
            if tool_use_id:
                payload["toolUseId"] = tool_use_id
            self._queue.put(payload)

    def report_text(self, text):
        """A passage the persona wrote on its way to the answer.

        The reply used to be every text block the session produced, joined
        together and handed over at the end -- so the owner's phone got the
        whole internal monologue in one lump, after the fact, with the tool
        chips it belonged between already sitting above it. His words
        (2026-08-04): "how would you like to be presented a story? One does
        not describe all actions in the story first, and then the narrative.
        They are in between each other, first a narrative, then an action,
        then a narrative, then an action."

        So each passage goes down the same queue as the chips, in the same
        order the CLI emitted it, and lands in the conversation where it
        actually happened. No truncation here, unlike a chip label: this is
        prose meant to be read, and the runner stopped clipping narration
        text at 500 characters to match (agora-persona-runner#41).
        """
        passage = (text or "").strip()
        if self.enabled and passage:
            self._queue.put({"capability": NARRATION_TEXT, "detail": passage})

    def report_subagent_text(self, description, text):
        """A passage a SUBAGENT wrote, on its way to its own answer.

        Narration, never the reply -- see cli.py for why that distinction is
        load-bearing rather than tidy. It goes down the same queue as the
        persona's own passages and renders as prose in the same drawer, but
        it is attributed on its first line, because Agora's client uses that
        line as the collapsed label and an unattributed one would read as the
        persona's own voice.
        """
        passage = (text or "").strip()
        if self.enabled and passage:
            label = description or "subagent"
            self._queue.put({
                "capability": NARRATION_TEXT,
                "detail": f"{SUBAGENT_PREFIX} {label}\n\n{passage}",
            })

    def report_subagent_start(self, task_id, subagent_type, description):
        """A subagent was launched. Posted from the CLI's `task_started`
        event rather than from the Agent tool call, because that event is
        what carries the subagent's type and brief."""
        if self.enabled and task_id:
            detail = " · ".join(p for p in (subagent_type, description) if p)
            self._queue.put({
                "capability": SUBAGENT,
                "toolUseId": task_id,
                "detail": detail or "subagent",
            })

    def report_subagent_finish(self, task_id, status, summary, usage):
        """A subagent finished. Pairs with report_subagent_start by task_id,
        so the client folds the two into one chip whose expanded body is what
        the subagent actually reported back.

        The cost line is here because it is the one number that is otherwise
        unknowable from outside: a delegated read is charged to the cycle's
        own quota, and until now nothing said how much.
        """
        if not (self.enabled and task_id):
            return
        counts = usage if isinstance(usage, dict) else {}
        parts = [status or "finished"]
        tokens = counts.get("total_tokens")
        if isinstance(tokens, int):
            parts.append(f"{tokens:,} tokens")
        calls = counts.get("tool_uses")
        if isinstance(calls, int):
            parts.append(f"{calls} tool call{'' if calls == 1 else 's'}")
        duration = counts.get("duration_ms")
        if isinstance(duration, (int, float)):
            parts.append(f"{duration / 1000:.1f}s")
        body = (summary or "").strip()
        self._queue.put({
            "capability": SUBAGENT,
            "toolUseId": task_id,
            "output": f"{' · '.join(parts)}\n\n{body}" if body else " · ".join(parts),
            "isError": status not in ("", "completed", None),
        })

    def report_result(self, name, tool_use_id, output, is_error=False):
        """What the call returned, once it has.

        A second report rather than an amendment of the first, because the
        first is already sent and already on screen. The reader sees one
        chip: Agora's client pairs the two by `toolUseId` at render time.
        Nothing is correlated here, and nothing needs to be -- if the pair's
        other half never arrives (a failed post, a killed session), each
        half still stands on its own.
        """
        if self.enabled and name and tool_use_id:
            self._queue.put({
                "capability": name,
                "toolUseId": tool_use_id,
                "output": output or "",
                "isError": bool(is_error),
            })

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
            # _post swallows its own errors, but an unexpected one from
            # anywhere here would kill this thread and silently end the
            # narration for the rest of the session -- the exact
            # fails-invisibly shape this whole feature exists to remove.
            try:
                ok = _post(self._url, _scrubbed(dict(item, token=self._token)))
            except Exception:
                ok = False
            if not ok:
                failures += 1
                # Log the first one only. A runner that is down or has
                # revoked the token fails for every remaining call in the
                # session, and one broken chip must not bury the CLI's own
                # logs under hundreds of identical lines.
                if failures == 1:
                    log(f"activity report failed for {item.get('capability')!r} "
                        f"(further failures this session silenced)")
