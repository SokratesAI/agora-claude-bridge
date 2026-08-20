"""Wraps the `claude` CLI as a single-invocation-at-a-time subprocess call.

Why single-invocation-at-a-time, system-wide (see _lock below), not just
per-conversation: the old claude-auth-refresher CronJob died because
multiple PODS shared one OAuth token file and each refreshed independently
-- whichever refreshed first invalidated the token for the others. This
service avoids that by being the only process anywhere that holds the
credential, but a second risk remains even within one pod: the CLI's own
token refresh is a side effect of *any* invocation, so two concurrent
invocations (different conversations, same pod) could still race each
other on the same underlying refresh. Half of that is now measured
(2026-08-10, this pod, against the real subscription): two invocations
started simultaneously on one credential, while a third was mid-turn,
all returned cleanly and left `.credentials.json` byte-identical. So
concurrency itself is fine; what stays unverified is the narrow window
where a refresh is actually due, and run_turn's allow_concurrent lane is
gated on staying out of it rather than on the whole lock. The lock
remains the default for every caller that does not opt in.
Synchronous/blocking throughout,
matching agora-persona-runner's own style (no asyncio anywhere in that
codebase) -- serializing on a plain threading.Lock is simpler than
asyncio.Lock across ThreadingHTTPServer's per-request threads, and there's
no upside to async here since we want serialization, not parallelism.
"""
import json
import os
import shutil
import stat
import subprocess
import threading
import time

from bridge.activity import ActivityReporter, result_text
from bridge.analytics import is_cycle_opening
from bridge import deadline
from bridge.config import CLAUDE_HOME, CLAUDE_WORKSPACE, CLI_TIMEOUT_SECONDS
from bridge.log import log
from bridge import quota
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


# Where the per-call --mcp-config file is written, alongside the quota
# hook's --settings file and for the same reason (quota.py:190): a flag is
# scoped to the invocation, while registering the server in ~/.claude.json
# would leave it on the PVC long after the turn that wanted it is gone.
MCP_CONFIG_FILE = os.path.join(CLAUDE_HOME, ".claude", "bridge-mcp.config.json")

# What the runner's MCP server is called on the CLI side. Its tools reach
# the model as mcp__agora__<tool_name>.
MCP_SERVER_NAME = "agora"

# Where a turn carrying attachments writes its stream-json user message.
# Same directory and same lifecycle as MCP_CONFIG_FILE above, deleted in
# _run_cli_once's finally. A fixed path is safe because _invocation_lock
# serializes every real invocation in this process (see run_turn).
CLI_INPUT_FILE = os.path.join(CLAUDE_HOME, ".claude", "bridge-input.jsonl")


def write_stream_json_input(message, attachments, path=None):
    """Render one turn as a stream-json `user` event and return its path.

    `claude -p <text>` can only carry text, which is why a claude-cli
    persona silently lost every image sent to it while the other two
    providers built real image blocks (agora-persona-runner's
    _anthropic_content / _gemini_parts). `--input-format stream-json`
    reads the same content-block shape those two send to their APIs, so
    the fix is a different way of handing over the same message rather
    than a new capability.

    Measured against CLI 2.1.226 in this pod, 2026-08-10, because the
    combination is what matters and not the flag alone: a resumed session
    (`--resume`) reading an image block off stdin still answered from the
    image AND still had its `--append-system-prompt` codeword. That is
    exactly the bridge's production path, so all three work together.

    `attachments` entries are {filename, mimeType, data} with `data`
    base64. An entry with no `data` becomes the same "[attached file:
    ...]" text the other two providers emit for a non-image or a fetch
    that failed -- the caller decides what is renderable, and this stays
    a transport."""
    blocks = []
    if message:
        blocks.append({"type": "text", "text": message})
    for att in attachments:
        mime = att.get("mimeType", "")
        data = att.get("data")
        if data:
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": data},
            })
        else:
            blocks.append({"type": "text", "text": (
                f"[attached file: {att.get('filename', '?')} "
                f"({mime or 'unknown type'}) -- not loaded]")})
    event = {"type": "user", "message": {"role": "user", "content": blocks}}
    path = path or CLI_INPUT_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(json.dumps(event) + "\n")
    return path

def write_mcp_config(mcp, path=None):
    """Render the caller's {"url", "token"} into a --mcp-config file and
    return its path, or "" meaning "run this turn without it".

    Every failure mode here returns "" rather than raising, and that is
    the entire job of this function. Measured against CLI 2.1.197 on
    2026-08-06, because the two halves are not symmetric:

      * An MCP server that cannot be reached is harmless -- the CLI logs
        the failed server and completes the turn normally, exit 0. So a
        runner that stops serving /mcp costs a session its capability
        tools and nothing else.
      * A --mcp-config file that is not valid JSON is NOT harmless. The
        CLI refuses to start at all ("Error: Invalid MCP configuration:
        MCP config is not a valid JSON", exit 1), before the model is
        ever called. Every turn would die, including ones that would have
        been perfectly fine with no MCP server at all.

    That asymmetry is why the config is built with json.dump from a dict
    instead of interpolated, why the write is guarded, and why an
    incomplete `mcp` block (missing url or token) returns "" rather than
    writing a server the CLI would then connect to unauthenticated.
    """
    if not isinstance(mcp, dict) or not mcp:
        return ""
    url = str(mcp.get("url", "")).strip()
    token = str(mcp.get("token", "")).strip()
    if not url or not token:
        log(f"mcp config skipped: need both url and token (url={bool(url)} token={bool(token)})")
        return ""
    config = {"mcpServers": {MCP_SERVER_NAME: {
        "type": "http",
        "url": url,
        "headers": {"Authorization": f"Bearer {token}"},
    }}}
    path = path or MCP_CONFIG_FILE
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(config, handle)
        # 600 -- for the length of the turn this file is a live bearer
        # token for the runner's tool endpoint, so it gets the same
        # treatment credentials.py gives .credentials.json rather than the
        # 644 the quota hook's settings file is fine with.
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        return path
    except Exception as exc:
        log(f"mcp config write failed: {type(exc).__name__}: {exc}")
        return ""


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


def _report_subagent_event(event, subagent, reporter, tool_names, subagent_names):
    """Narrate one event a subagent produced.

    Deliberately a separate function from the main loop's own branches rather
    than an `if` inside them: nothing here may touch `pending`, `text_parts`
    or `thinking_parts`, and keeping it out of the scope that owns them is a
    stronger guarantee than remembering not to.

    A subagent's tool calls are reported as the tools they are -- a `Bash`
    call is a `Bash` call whoever made it -- and only the label says who ran
    it. Its results pair back by tool_use_id exactly like the persona's,
    which is why they share `tool_names`: the CLI's ids are unique across the
    whole session, so there is nothing to keep apart.
    """
    label = subagent_names.get(subagent, "")
    for block in event.get("message", {}).get("content", []):
        kind = block.get("type")
        if kind == "text":
            # Only on `assistant`. A child `user` event carries the brief the
            # subagent was handed, which `task_started` has already reported.
            if event.get("type") == "assistant":
                reporter.report_subagent_text(label, block.get("text", ""))
        elif kind == "tool_use":
            name = block.get("name", "")
            tool_use_id = str(block.get("id", ""))
            if tool_use_id:
                tool_names[tool_use_id] = name
            reporter.report(name, block.get("input"), tool_use_id, subagent=label)
        elif kind == "tool_result":
            tool_use_id = str(block.get("tool_use_id", ""))
            name = tool_names.pop(tool_use_id, "")
            if name:
                reporter.report_result(
                    name, tool_use_id, result_text(block),
                    is_error=bool(block.get("is_error")),
                )


def _slotted(path, slot):
    """`/x/.claude/bridge-mcp.config.json` -> `...config.<slot>.json`.

    Only a turn running outside _invocation_lock passes a slot; everything
    else keeps the exact filenames it has always used, so the serialized
    path is byte-for-byte unchanged.
    """
    if not slot:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}.{slot}{ext}"


def _workspace_for(slot):
    """CLAUDE_WORKSPACE unchanged for the default (locked) path, or an
    isolated subdirectory for a turn running outside _invocation_lock.

    Unlike the MCP config/CLI input file _slotted isolates, the workspace
    is a real git checkout: two turns sharing it would race on the same
    working tree, `.git/index`, and whatever branch either has checked
    out. A turn with no slot is either running alone under the lock, or an
    allow_concurrent turn whose refresh-window gate failed and fell back
    to the lock -- either way it gets the exact same shared directory
    every serialized turn has always used, unchanged.

    A concurrent turn always starts from an empty directory (see its
    cleanup in _run_cli_once's finally) rather than a workspace kept
    around for reuse: `slot` is a pid+thread-id, and thread idents get
    reused once a thread exits, so keeping the directory around would
    risk a later, unrelated turn silently inheriting an earlier one's
    checkout. The cost is a fresh clone on every concurrent turn that
    touches git -- acceptable today because allow_concurrent is only used
    for short turns (comment replies) that mostly don't.
    """
    if not slot:
        return CLAUDE_WORKSPACE
    return os.path.join(CLAUDE_WORKSPACE, "concurrent", slot)


def _run_cli_once(message, session_id, model, disallowed_tools, activity=None, mcp=None,
                  system=None, attachments=None, slot=""):
    workspace = _workspace_for(slot)
    os.makedirs(workspace, exist_ok=True)
    claude_dir = os.path.join(CLAUDE_HOME, ".claude")
    os.makedirs(claude_dir, exist_ok=True)
    env = {**os.environ, "HOME": CLAUDE_HOME, "CLAUDE_CONFIG_DIR": claude_dir}

    # Only a turn that actually carries attachments switches to stdin. The
    # text path is what every Nova cycle and every ordinary chat turn runs,
    # and moving all of them onto a second mechanism to avoid having two
    # would put the whole service behind a change nothing was asking for.
    input_file = (write_stream_json_input(message, attachments, _slotted(CLI_INPUT_FILE, slot))
                  if attachments else "")

    cmd = ["claude", "-p"]
    if input_file:
        cmd.extend(["--input-format", "stream-json"])
    else:
        cmd.append(message)
    cmd.extend([
        "--output-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        # Everything a subagent does, on the parent's stream. Without this the
        # CLI reports a Task call and then nothing until it returns, which for
        # a delegated read is several minutes of apparent silence -- Edvard's
        # issue #4, "it looks like it does nothing for a long time".
        #
        # Measured on 2.1.226 rather than read off --help, because what it
        # actually adds is narrower than the name suggests: `system/task_*`
        # events are on the stream with or without it, and only the child
        # `assistant` events (a subagent's text and its own tool calls) are
        # new. Every child event carries `parent_tool_use_id`; that field is
        # the only thing distinguishing a subagent's work from the persona's,
        # and the loop below depends on it.
        "--forward-subagent-text",
    ])
    # Lets the session see its own remaining quota and its own remaining
    # wall-clock while it runs, so it can wrap up deliberately instead of
    # being cut off mid-sentence (quota.py, deadline.py).
    # Slotted for the same reason as the two above, with a different
    # failure: this file is never deleted and every turn writes identical
    # content, so sharing it looks harmless -- but `open(path, "w")`
    # truncates, and a concurrent turn's CLI can read the empty window
    # between that truncate and the json.dump. Losing the quota and
    # deadline hooks is silent; the turn just stops being able to see its
    # own budget.
    hook_settings = write_hook_settings(_slotted(quota.HOOK_SETTINGS_FILE, slot))
    if hook_settings:
        cmd.extend(["--settings", hook_settings])
    # Agora's own capability tools (vault_read, kubectl_read, create_pr,
    # ...), served by the caller over MCP so a claude-cli persona runs the
    # same toolset every other provider does. --strict-mcp-config keeps
    # this to exactly the caller's server: without it the CLI would also
    # pick up whatever is registered in the PVC's ~/.claude.json, which is
    # persistent state no caller asked for and nobody audits.
    #
    # Expect the `system`/`init` event to report this server as
    # "status": "pending" with zero mcp__agora__* tools in the roster. That
    # is NOT a failure and does not need diagnosing again: init is emitted
    # before the handshake finishes. Measured in this pod, 2026-08-06, over
    # three runs -- initialize lands at 1.20/1.28/1.34s and tools/list at
    # 1.38/1.44/1.60s after the process starts, so the tools are live from
    # roughly 1.5s in. The only turn that can miss them is one whose first
    # and last action both happen inside that window; a persona cycle runs
    # for tens of minutes and cannot.
    mcp_config = write_mcp_config(mcp, _slotted(MCP_CONFIG_FILE, slot))
    if mcp_config:
        cmd.extend(["--mcp-config", mcp_config, "--strict-mcp-config"])
    if disallowed_tools:
        cmd.extend(["--disallowedTools", disallowed_tools])
    # The persona's constitution belongs in the operator channel, not in the
    # user turn. Until 2026-08-08 server.py concatenated it into the message
    # ("{system}\n\n{prompt}"), so a persona read its own identity as if the
    # human had typed it, and only on turn 1.
    #
    # Must be re-passed on EVERY invocation, including resumed ones -- the CLI
    # does not carry it in the session. Measured in this pod, 2026-08-08, with
    # a codeword planted in the system prompt only: fresh session reveals it;
    # --resume without the flag answers "NONE"; --resume with the flag
    # re-passed reveals it again. (A first attempt at that third case also
    # said "NONE" and nearly went into the record as "the flag is ignored on
    # resume" -- the session had been polluted by the second probe's own
    # "NONE" answer, and the model stayed consistent with its own transcript.
    # Re-run on a clean session before believing otherwise.)
    if system:
        cmd.extend(["--append-system-prompt", system])
    if session_id:
        cmd.extend(["--resume", session_id])
    if model:
        cmd.extend(["--model", model])

    log(f"CLI start: session={session_id} model={model} msg={message[:120]!r} "
        f"attachments={len(attachments or [])}")
    t0 = time.monotonic()
    # Starts the clock the deadline hook reports off. Written here rather
    # than at the top of the function so it measures the same span
    # proc.wait() enforces, and cleared in the finally below so a finished
    # turn never leaves an expired clock for the next one to read.
    deadline.write(CLI_TIMEOUT_SECONDS)
    # Handed over as a file rather than written to a pipe: an image is far
    # larger than a pipe buffer, and writing it to stdin while the child is
    # already writing to the stdout we have not started reading yet is the
    # classic way to deadlock two processes. The child gets its own fd, so
    # this copy closes immediately.
    stdin_handle = open(input_file) if input_file else None
    timed_out = False
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=stdin_handle,
            cwd=workspace,
            env=env,
            text=True,
        )
    finally:
        if stdin_handle:
            stdin_handle.close()

    text_parts = []
    thinking_parts = []
    new_session_id = session_id or ""
    saw_error = None

    # Narrates each tool call to the caller as it happens (activity.py).
    # A no-op when the caller didn't ask for it, which is why there's no
    # branch around every .report() below.
    reporter = ActivityReporter(activity)
    reporter.start()

    # Polls remaining quota into a snapshot file the hook above reads, and --
    # on a cycle, not on a reply turn or a probe -- republishes the cost
    # record into the vault on the way out. `message` is this invocation's
    # opening user turn, which is the same string analytics.py classifies a
    # finished transcript by, so the two agree by construction.
    watcher = QuotaWatcher(publish_costs=is_cycle_opening(message))
    watcher.start()

    # tool_use_id -> tool name, for the `user` branch below. Entries are
    # removed as their results arrive, so this holds only calls currently in
    # flight (normally one) rather than the whole session's history.
    tool_names = {}

    # The Agent call's tool_use_id -> that subagent's description, learned
    # from `task_started`. Child events name their parent by tool_use_id and
    # carry nothing else identifying, so this is what turns an anonymous
    # forwarded passage into "the opening-read subagent said this".
    subagent_names = {}

    # The newest text block, held back by exactly one event.
    #
    # A passage the persona writes is narration if more work follows it and
    # the reply if nothing does, and which one it is is not knowable when it
    # arrives -- only when the next thing does. So the newest one waits here,
    # and the one before it is released as narration the moment anything else
    # shows up. Whatever is still sitting here when the stream ends is the
    # last thing written, which is the reply.
    pending = []

    def release_narrative():
        """Send the held passage, now that we know it wasn't the last."""
        if pending:
            reporter.report_text(pending.pop())

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
            # Set on every event a subagent produced, naming the Agent call it
            # belongs to. Empty for the persona's own events.
            #
            # This is the single most load-bearing line in the loop, and it
            # guards two failures that both put a subagent's words in front of
            # Edvard as if the persona had said them:
            #
            #  1. The reply is `pending[-1]` -- the last passage written. A
            #     subagent launched in the background finishes whenever it
            #     finishes, which is routinely AFTER the persona has written
            #     its closing passage (measured: the persona's "OK", then the
            #     child's report, then the persona's real answer). Let child
            #     text into `pending` and the reply posted to his phone is
            #     whatever some subagent happened to say last.
            #  2. `release_narrative()` empties `pending`. A background child's
            #     tool call arriving after the persona's final passage would
            #     release it, leaving `pending` empty -- and the reply then
            #     falls through to the join of every passage in the session,
            #     which is the wall-of-text regression that fallback exists to
            #     avoid.
            #
            # So child events are narrated and never contribute to the reply.
            subagent = str(event.get("parent_tool_use_id") or "")
            if subagent and t in ("assistant", "user"):
                _report_subagent_event(event, subagent, reporter, tool_names, subagent_names)
                continue

            if t == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "thinking":
                        # Reachable, and always empty. Measured on 2.1.226,
                        # 2026-08-10: the block IS emitted on every turn, but
                        # as {"thinking": "", "signature": "Ep..."} -- the
                        # plaintext is stripped in --print mode and only the
                        # signature survives. `system/thinking_tokens` events
                        # on the same stream report a real, non-zero token
                        # count, so the model is thinking and the CLI is
                        # simply not handing the text over.
                        #
                        # So `thinking` returns "" for every claude-cli turn.
                        # That is not reasoning being dropped on the floor
                        # here -- there is none to drop. Kept rather than
                        # deleted because it costs nothing and is the branch
                        # that would start working if the CLI ever forwards
                        # the text; do not re-investigate without re-probing
                        # a real stream first.
                        thought = block.get("thinking", "").strip()
                        if thought:
                            thinking_parts.append(thought)
                    elif block.get("type") == "text":
                        chunk = block.get("text", "")
                        if chunk:
                            release_narrative()
                            pending.append(chunk)
                            text_parts.append(chunk)
                    elif block.get("type") == "tool_use":
                        release_narrative()
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
                # A subagent starting and finishing. Both were already on the
                # stream before --forward-subagent-text existed and this loop
                # dropped them, because the only subtype it looked at was the
                # error one -- so the "how long did that delegated read take
                # and what did it cost" question was answerable all along.
                if subtype == "task_started":
                    subagent_names[str(event.get("tool_use_id", ""))] = \
                        event.get("description", "")
                    reporter.report_subagent_start(
                        str(event.get("task_id", "")),
                        event.get("subagent_type", ""),
                        event.get("description", ""),
                    )
                elif subtype == "task_notification":
                    reporter.report_subagent_finish(
                        str(event.get("task_id", "")),
                        event.get("status", ""),
                        event.get("summary", ""),
                        event.get("usage"),
                    )
                elif subtype == "error_during_execution":
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
            # Falling through rather than raising, because this process is
            # holding every word the session wrote and used to throw all of
            # it away. Cycle 81 was killed here having already merged its
            # PR and written its journal entry, and the only thing Edvard
            # was told is "failed: timed out" -- which reads as "the cycle
            # achieved nothing", the opposite of what happened. The reply
            # built below is now the last thing the session managed to say,
            # labelled as truncated so it cannot be mistaken for a finished
            # one. If it wrote nothing at all there is genuinely nothing to
            # salvage, and the raise below still fires.
            timed_out = True
            log(f"CLI timed out after {CLI_TIMEOUT_SECONDS}s; "
                f"salvaging {len(''.join(text_parts))} chars of text")
    finally:
        if proc.stdout:
            proc.stdout.close()
        reporter.close()
        watcher.close()
        deadline.clear()
        # The config file holds this turn's bearer token, and the turn is
        # over -- the caller revokes the grant as its own call returns, so
        # what is left on the PVC is inert, but there is no reason to leave
        # a credential-shaped file lying on a persistent volume. The
        # subprocess read it at startup and has already exited.
        if mcp_config:
            try:
                os.remove(mcp_config)
            except OSError as exc:
                log(f"mcp config cleanup failed: {type(exc).__name__}: {exc}")
        # Same reasoning one step further: this one holds whatever Edvard
        # photographed, and the CLI read it before the first event arrived.
        if input_file:
            try:
                os.remove(input_file)
            except OSError as exc:
                log(f"cli input cleanup failed: {type(exc).__name__}: {exc}")
        # The shared CLAUDE_WORKSPACE (slot == "") is never torn down --
        # it's the persistent checkout every serialized turn reuses, same
        # as before this existed. Only a concurrent turn's own isolated
        # directory gets removed, and always, win or lose: _workspace_for's
        # own reasoning is that a slot must never be found populated by an
        # earlier, unrelated turn.
        if slot:
            shutil.rmtree(workspace, ignore_errors=True)

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

    # Everything the persona wrote except the last passage has already been
    # sent as narration and is already in the conversation, in order, where
    # it happened. Returning the joined whole on top of that would print the
    # entire run a second time inside the reply bubble -- which is what it
    # used to do, and is why Edvard's phone kept buzzing with a wall of "let
    # me check the deploy first" instead of an answer. So the reply is the
    # last passage: the thing written once there was nothing left to do.
    #
    # Both fallbacks matter and both mean "nothing was narrated, so nothing
    # is duplicated": no reporter (the /invoke path, or a runner too old to
    # send an activity block), and a session that ended on a tool call with
    # no closing passage at all. In the second case every passage HAS been
    # sent, so the join does repeat them -- accepted deliberately, because
    # the alternative is an empty reply, and an empty reply raises below and
    # fails the whole turn.
    if reporter.enabled and pending:
        text = pending[-1].strip()
    else:
        text = "\n".join(text_parts).strip()
    thinking = "\n\n".join(thinking_parts).strip()
    if timed_out:
        # A killed turn never reached its closing passage, so the salvage is
        # mid-run narration -- "now the digest, re-fetching first" was
        # Cycle 81's. That is worth sending and worth labelling: unlabelled
        # it would arrive looking like a considered reply that simply stops.
        #
        # It has to be picked here rather than reusing the selection above,
        # and the reviewer caught this: `pending` is emptied by
        # release_narrative() on every tool_use, and a turn killed mid-tool-
        # call is the normal shape of this failure -- Cycle 81 died three
        # tool calls into rewriting the digest. So `pending` is empty
        # exactly when this path runs, the selection above falls through to
        # joining every passage, and Edvard's phone gets the entire
        # transcript a second time on top of the narration he already
        # watched. That is the wall-of-text regression the comment above
        # exists to prevent, arriving through a different door. The last
        # passage is what the label promises and all it should send.
        text = text_parts[-1].strip() if (reporter.enabled and text_parts) else text
        if not text:
            raise ClaudeCliError(f"CLI timed out after {CLI_TIMEOUT_SECONDS}s")
        return (
            f"_Cut off at this turn's {CLI_TIMEOUT_SECONDS // 60}-minute limit "
            "before I could write a proper reply. Anything I wrote to the vault "
            "survived; anything else did not. The last thing I was doing:_\n\n"
            f"{text}"
        ), thinking, new_session_id
    if not text:
        raise ClaudeCliError("CLI produced no text output")
    return text, thinking, new_session_id


# Serializes every real subprocess invocation across the whole process --
# see this module's own docstring for why.
_invocation_lock = threading.Lock()

# How far from expiry the OAuth access token has to be before a caller is
# allowed past the lock (see run_turn's allow_concurrent).
#
# Measured in this pod on 2026-08-10, because the design this implements
# was written against a wrong number: `.credentials.json` carried an
# `expiresAt` **eight hours** out, not the ~60 minutes an earlier reading
# recorded, and the file's mtime matched the start of that 8-hour span.
# The CLI's own refresh is therefore a roughly-8-hourly event, not an
# hourly one, so a 15-minute margin leaves this lane open ~97% of the
# time and shut exactly across the window where the refresh race the
# module docstring describes could actually happen.
CONCURRENT_REFRESH_MARGIN_SECONDS = 900

CREDENTIALS_FILE = os.path.join(CLAUDE_HOME, ".claude", ".credentials.json")


def refresh_window_clear(margin=CONCURRENT_REFRESH_MARGIN_SECONDS, path=None, now=None):
    """True when the OAuth token is far enough from expiry that no
    invocation started right now should trigger a refresh.

    Every failure answers False. An unreadable, malformed or
    `expiresAt`-less credential file is not evidence that concurrency is
    safe -- it is the absence of evidence, and the lock is the thing that
    is correct in the absence of evidence.
    """
    try:
        with open(path or CREDENTIALS_FILE) as handle:
            expires_at = json.load(handle)["claudeAiOauth"]["expiresAt"]
        return (float(expires_at) / 1000.0) - (now or time.time()) > margin
    except Exception as exc:
        log(f"refresh window check failed, staying serialized: {type(exc).__name__}: {exc}")
        return False


def run_turn(message, session_id=None, model=None, disallowed_tools=None, activity=None,
             mcp=None, system=None, attachments=None, allow_concurrent=False):
    """One turn. Returns (text, thinking, new_session_id).

    attachments: optional [{filename, mimeType, data}] with `data` base64,
    sent as real content blocks alongside the text (write_stream_json_input).
    Omit it and the CLI is invoked exactly as it was before this existed.

    system: optional persona/system prompt, passed to the CLI as
    --append-system-prompt so it reaches the model as an operator
    instruction rather than as part of the user's message. Must be supplied
    on every turn, resumed or not -- the CLI does not persist it in the
    session. Omit it and no --append-system-prompt flag is passed at all.

    disallowed_tools: comma-separated tool names to block for this call
    (see DISCOVERED_FULL_TOOL_ROSTER above), or None/empty for full,
    unrestricted access -- the default.

    activity: optional {"url", "token"} the caller wants each tool call
    reported to, live, while this turn runs (see activity.py). Omit it and
    nothing is reported.

    mcp: optional {"url", "token"} for an HTTP MCP server the caller wants
    this turn's model to have (see write_mcp_config). Omit it and the CLI
    is invoked with no MCP configuration at all, exactly as before.

    allow_concurrent: let this turn run alongside one already in flight
    instead of queueing behind the process-wide lock. Opt-in per call, for
    short turns that a caller would rather drop than have wait out a
    45-minute Nova cycle -- a journal-card reply is the case it was built
    for. It is a request, not a guarantee: the lane opens only while
    refresh_window_clear() holds, and falls back to the lock otherwise, so
    a caller never has to reason about the token itself.

    Raises UsageLimitError on a real subscription cap, ClaudeCliError on
    anything else that prevented a usable reply (including the
    SESSION_NOT_FOUND sentinel message -- callers should clear their stored
    session_id and retry once with session_id=None on that specific case).
    """
    # Measured live in the bridge pod, 2026-08-10 (Cycle 84): two `claude
    # -p` invocations started simultaneously, on one credential, while a
    # third -- the Nova cycle running this very test -- was mid-turn. All
    # three returned is_error:false with distinct session ids and
    # `.credentials.json` was byte-identical afterwards. That retires the
    # "unverified" in this module's docstring for the no-refresh-due case,
    # and only for that case: the token was 8 hours from expiry, so
    # nothing in that test refreshed anything. The refresh race is still
    # unmeasured, which is exactly what the gate below exists to avoid.
    if allow_concurrent and refresh_window_clear():
        # The three files a turn writes into ~/.claude are on fixed paths
        # (MCP_CONFIG_FILE, CLI_INPUT_FILE), and the comments there say
        # plainly that this is safe *because* the lock serializes
        # invocations. Take the lock away and it stops being true: the mcp
        # config holds this turn's bearer token for the runner's tool
        # endpoint and is deleted in _run_cli_once's finally, so two
        # overlapping turns can hand one turn the other's token, or delete
        # a file the other's subprocess has not read yet. A concurrent turn
        # therefore gets its own paths rather than sharing.
        slot = f"{os.getpid()}-{threading.get_ident()}"
        return _run_cli_once(message, session_id, model, disallowed_tools, activity, mcp,
                             system, attachments, slot=slot)
    with _invocation_lock:
        return _run_cli_once(message, session_id, model, disallowed_tools, activity, mcp,
                             system, attachments)
