"""Wraps the `claude` CLI as a single-invocation-at-a-time subprocess call.

Why single-invocation-at-a-time, system-wide (see _lock below), not just
per-conversation: the old claude-auth-refresher CronJob died because
multiple PODS shared one OAuth token file and each refreshed independently
-- whichever refreshed first invalidated the token for the others. This
service avoids that by being the only process anywhere that holds the
credential, but a second risk remains even within one pod: the CLI's own
token refresh is a side effect of *any* invocation, so two concurrent
invocations (different conversations, same pod) could still race each
other on the same underlying refresh. Unverified until live-tested against
the real subscription -- the lock is the deliberately conservative default
until that's confirmed safe to relax. Synchronous/blocking throughout,
matching agora-persona-runner's own style (no asyncio anywhere in that
codebase) -- serializing on a plain threading.Lock is simpler than
asyncio.Lock across ThreadingHTTPServer's per-request threads, and there's
no upside to async here since we want serialization, not parallelism.
"""
import json
import os
import subprocess
import threading
import time

from bridge.activity import ActivityReporter, result_text
from bridge.config import CLAUDE_HOME, CLAUDE_WORKSPACE, CLI_TIMEOUT_SECONDS
from bridge.log import log
from bridge.quota import QuotaWatcher, write_hook_settings

SESSION_NOT_FOUND = "\x00SESSION_NOT_FOUND"

# 2026-08-01 design reversal: v1 shipped with a hardcoded, always-on
# --disallowedTools restriction (the 8 "obvious" tools -- Bash/Read/Write/
# Edit/Glob/Grep/WebFetch/WebSearch). Live-tested and found genuinely
# incomplete: Claude Code ships a much larger built-in tool roster
# (confirmed live via a real session's own system.init event -- the exact
# list is DISCOVERED_FULL_TOOL_ROSTER below), and the model found and used
# an unlisted one ("Monitor") to run real shell commands anyway. Edvard's
# call: this service is meant to be as capable as an interactive Claude
# Code session, same as this very session building it -- restriction should
# be an explicit per-call opt-in, not a silent, incomplete default. Pass
# disallowed_tools to run_turn/_run_cli_once to restrict a specific call;
# omit it (the default) for full, unrestricted access.
#
# Kept here as a reference for anyone who DOES want to restrict a call --
# this is the complete roster observed live on CLI version 2.1.197, not a
# guess. Verify against a fresh `system.init` event if the CLI version
# changes; new tools can be added between versions.
DISCOVERED_FULL_TOOL_ROSTER = (
    "Task,CronCreate,CronDelete,CronList,DesignSync,EnterWorktree,ExitWorktree,"
    "Monitor,NotebookEdit,PushNotification,ReportFindings,ScheduleWakeup,SendMessage,"
    "Skill,TaskCreate,TaskGet,TaskList,TaskOutput,TaskStop,TaskUpdate,ToolSearch,"
    "Workflow,Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch"
)


class UsageLimitError(Exception):
    """Real, hours-long subscription usage cap -- distinct from a transient
    per-call error. Callers should NOT retry immediately."""


class ClaudeCliError(Exception):
    """Any other CLI failure -- transient, a real bug, or unparseable output."""


def _detect_usage_limit(text):
    """Best-effort text match on the CLI's own error output. Unverified
    against a real usage-cap response -- the raw-API equivalent (old
    agent-runtime's providers/claude.py) matched "You've hit your limit" in
    the Messages API's JSON error body, but the CLI's own wording may
    differ. Update this the first time a real cap is actually hit."""
    lowered = text.lower()
    return "usage limit" in lowered or "you've hit your limit" in lowered or "rate limit" in lowered


def _run_cli_once(message, session_id, model, disallowed_tools, activity=None):
    os.makedirs(CLAUDE_WORKSPACE, exist_ok=True)
    claude_dir = os.path.join(CLAUDE_HOME, ".claude")
    os.makedirs(claude_dir, exist_ok=True)
    env = {**os.environ, "HOME": CLAUDE_HOME, "CLAUDE_CONFIG_DIR": claude_dir}

    cmd = [
        "claude", "-p", message,
        "--output-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
    ]
    # Lets the session see its own remaining quota while it runs, so it can
    # wrap up deliberately instead of being cut off mid-sentence (quota.py).
    hook_settings = write_hook_settings()
    if hook_settings:
        cmd.extend(["--settings", hook_settings])
    if disallowed_tools:
        cmd.extend(["--disallowedTools", disallowed_tools])
    if session_id:
        cmd.extend(["--resume", session_id])
    if model:
        cmd.extend(["--model", model])

    log(f"CLI start: session={session_id} model={model} msg={message[:120]!r}")
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=CLAUDE_WORKSPACE,
        env=env,
        text=True,
    )

    text_parts = []
    thinking_parts = []
    new_session_id = session_id or ""
    saw_error = None

    # Narrates each tool call to the caller as it happens (activity.py).
    # A no-op when the caller didn't ask for it, which is why there's no
    # branch around every .report() below.
    reporter = ActivityReporter(activity)
    reporter.start()

    # Polls remaining quota into a snapshot file the hook above reads.
    watcher = QuotaWatcher()
    watcher.start()

    # tool_use_id -> tool name, for the `user` branch below. Entries are
    # removed as their results arrive, so this holds only calls currently in
    # flight (normally one) rather than the whole session's history.
    tool_names = {}

    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                log(f"CLI non-JSON stdout: {line[:300]!r}")
                continue

            t = event.get("type", "")
            if t == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "thinking":
                        thought = block.get("thinking", "").strip()
                        if thought:
                            thinking_parts.append(thought)
                    elif block.get("type") == "text":
                        chunk = block.get("text", "")
                        if chunk:
                            text_parts.append(chunk)
                    elif block.get("type") == "tool_use":
                        name = block.get("name", "")
                        tool_use_id = str(block.get("id", ""))
                        if tool_use_id:
                            tool_names[tool_use_id] = name
                        reporter.report(name, block.get("input"), tool_use_id)
            elif t == "user":
                # What each tool RETURNED. Edvard has asked for this three
                # times -- "I need to see the command with all metadata and
                # also the output from that command, such as the return of a
                # echo command" -- and until now this branch did not exist at
                # all: the CLI streams results on `user` events, and the loop
                # only ever looked at `assistant` ones, so every result was
                # read off the pipe and dropped.
                #
                # The tool's name is only on the `tool_use` block that opened
                # the call, so it is carried across in tool_names. Popped, not
                # read: one result per call, and a cycle makes hundreds.
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") != "tool_result":
                        continue
                    tool_use_id = str(block.get("tool_use_id", ""))
                    name = tool_names.pop(tool_use_id, "")
                    if name:
                        reporter.report_result(
                            name, tool_use_id, result_text(block),
                            is_error=bool(block.get("is_error")),
                        )
            elif t == "rate_limit_event":
                # The API's own view of the cap, carried on the stream for
                # free. No percentage in it -- it triggers a fresh reading
                # rather than being reported directly (quota.py).
                watcher.note_rate_limit_event(event.get("rate_limit_info"))
            elif t in ("result", "system"):
                sid = event.get("session_id", "")
                if sid:
                    new_session_id = sid
                subtype = event.get("subtype", "")
                if subtype == "error_during_execution":
                    errors = event.get("errors", [])
                    error_text = " ".join(str(e) for e in errors)
                    log(f"CLI error_during_execution: {error_text[:300]}")
                    if any("No conversation found" in str(e) for e in errors):
                        saw_error = (SESSION_NOT_FOUND, "")
                    elif _detect_usage_limit(error_text):
                        saw_error = ("usage_limit", error_text[:300])
                    else:
                        saw_error = ("error", error_text[:300])

        try:
            proc.wait(timeout=CLI_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise ClaudeCliError(f"CLI timed out after {CLI_TIMEOUT_SECONDS}s")
    finally:
        if proc.stdout:
            proc.stdout.close()
        reporter.close()
        watcher.close()

    elapsed = time.monotonic() - t0
    log(f"CLI done: exit={proc.returncode} elapsed={elapsed:.1f}s "
        f"text_len={len(''.join(text_parts))} thinking_len={len(''.join(thinking_parts))} "
        f"new_session={new_session_id}")

    if saw_error is not None:
        kind, detail = saw_error
        if kind == SESSION_NOT_FOUND:
            raise ClaudeCliError(SESSION_NOT_FOUND)
        if kind == "usage_limit":
            raise UsageLimitError(detail)
        raise ClaudeCliError(detail)

    text = "\n".join(text_parts).strip()
    thinking = "\n\n".join(thinking_parts).strip()
    if not text:
        raise ClaudeCliError("CLI produced no text output")
    return text, thinking, new_session_id


# Serializes every real subprocess invocation across the whole process --
# see this module's own docstring for why.
_invocation_lock = threading.Lock()


def run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None):
    """One turn. Returns (text, thinking, new_session_id).

    disallowed_tools: comma-separated tool names to block for this call
    (see DISCOVERED_FULL_TOOL_ROSTER above), or None/empty for full,
    unrestricted access -- the default.

    activity: optional {"url", "token"} the caller wants each tool call
    reported to, live, while this turn runs (see activity.py). Omit it and
    nothing is reported.

    Raises UsageLimitError on a real subscription cap, ClaudeCliError on
    anything else that prevented a usable reply (including the
    SESSION_NOT_FOUND sentinel message -- callers should clear their stored
    session_id and retry once with session_id=None on that specific case).
    """
    with _invocation_lock:
        return _run_cli_once(message, session_id, model, disallowed_tools, activity)
