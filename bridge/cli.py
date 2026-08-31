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

# A slot directory older than this belongs to a turn that is definitely
# gone: the CLI itself is killed at CLI_TIMEOUT_SECONDS. See
# _sweep_stale_slots.
STALE_SLOT_SECONDS = CLI_TIMEOUT_SECONDS + 600

SESSION_NOT_FOUND = "\x00SESSION_NOT_FOUND"

# A turn that lost the OAuth refresh-token race: another invocation
# refreshed first and this one's token was invalidated mid-turn. Distinct
# from SESSION_NOT_FOUND (the *conversation's* session is gone) and from
# UsageLimitError (a real, hours-long subscription cap) -- this is neither,
# and the right response is neither "start fresh" nor "stop retrying". The
# winner has already written the new token to .credentials.json by the
# time anything downstream of the race can act on this, so a short wait
# and a plain retry (see server.generate's _run_turn_with_auth_retry)
# should just pick it up. See cli.py's own module docstring for why the
# lock is still the default and this is a backstop, not a replacement.
AUTH_EXPIRED = "\x00AUTH_EXPIRED"

# 2026-08-01 design reversal: v1 shipped with a hardcoded, always-on
# --disallowedTools restriction (the 8 "obvious" tools -- Bash/Read/Write/
# Edit/Glob/Grep/WebFetch/WebSearch). Live-tested and found genuinely
# incomplete: Claude Code ships a much larger built-in tool roster
# (confirmed live via a real session's own system.init event -- the exact
# list is DISCOVERED_FULL_TOOL_ROSTER below), and the model found and used
# an unlisted one ("Monitor") to run real shell commands anyway. The owner's
# call: this service is meant to be as capable as an interactive Claude
# Code session, same as this very session building it -- restriction should
# be an explicit per-call opt-in, not a silent, incomplete default. Pass
# disallowed_tools to run_turn/_run_cli_once to restrict a specific call;
# omit it (the default) for full, unrestricted access. Pass restricted=True
# with it for the CLI's own --restricted as well -- see the note on
# DISCOVERED_FULL_TOOL_ROSTER for why a hand-kept list needs the flag beside
# it, and the cmd construction below for why the two cannot be combined with
# --dangerously-skip-permissions.
#
# 2026-08-31: the paragraph above warned that this list goes stale between
# CLI versions, and it had. Measured on 2.1.251 from a real `system.init`
# event in this pod -- `claude -p ... --output-format stream-json --verbose`,
# both with and without --restricted, because the two report different
# rosters -- the CLI had grown `ListAgents` and `RemoteTrigger` since 2.1.197
# and neither was named here. So `restricted=True` was handing a persona two
# tools it was meant not to have, which is v1's incomplete-denylist failure
# happening again by drift rather than by design.
#
# The list is still hand-maintained and will drift again. What changed is
# that it is no longer the only thing standing there: a restricted call now
# also passes the CLI's own `--restricted` (2.1.248), which removes the
# command- and code-running tools and WebFetch upstream, so a tool added in
# 2.1.260 that this list has never heard of is still blocked from running a
# shell. Keep both -- this one says "no built-in tools at all", the flag says
# "and nothing that runs code, whatever it is called".
DISCOVERED_FULL_TOOL_ROSTER = (
    "Task,CronCreate,CronDelete,CronList,DesignSync,EnterWorktree,ExitWorktree,"
    "ListAgents,Monitor,NotebookEdit,PushNotification,RemoteTrigger,ReportFindings,"
    "ScheduleWakeup,SendMessage,Skill,TaskCreate,TaskGet,TaskList,TaskOutput,"
    "TaskStop,TaskUpdate,ToolSearch,Workflow,Bash,Read,Write,Edit,Glob,Grep,"
    "WebFetch,WebSearch"
)


# Where the per-call --mcp-config file is written, alongside the quota
# hook's --settings file and for the same reason (quota.py:190): a flag is
# scoped to the invocation, while registering the server in ~/.claude.json
# would leave it on the PVC long after the turn that wanted it is gone.
MCP_CONFIG_FILE = os.path.join(CLAUDE_HOME, ".claude", "bridge-mcp.config.json")

# What the runner's MCP server is called on the CLI side. Its tools reach
# the model as mcp__agora__<tool_name>.
MCP_SERVER_NAME = "agora"

# How much of a tool result reaches the model before the CLI cuts it, in
# characters. Both are the CLI's own ceiling on 2.1.245, measured rather than
# read off documentation -- `claude doctor` with 999999 in each prints
# "Capped from 999999 to 150000" and "... to 160000", and prints nothing at
# all for the two values below. So the CLI reads both variables, these are
# exactly its limits, and asking for more is a no-op rather than a risk.
#
#   BASH_MAX_OUTPUT_LENGTH  default  30_000, upper limit 150_000
#   TASK_MAX_OUTPUT_LENGTH  default  32_000, upper limit 160_000
#
# The two failures are not the same and only one of them loses data. Over
# the Bash limit the CLI writes the whole result to a file under the session
# directory and shows a ~2KB preview with the path, so the bytes survive and
# a second call gets them. Over the Task limit a subagent's report is cut to
# its *last* N characters and stamped "the earlier part of the report is not
# retrievable" -- the beginning of the report, which is where a brief's first
# heading sits, is gone for good.
#
# Set only when the environment does not already carry one, unlike the four
# values beside them below: HOME and CLAUDE_CONFIG_DIR are things this bridge
# has to control, while these two are a tuning knob, and clobbering a number
# somebody put in the Deployment on purpose would make that number a lie with
# nothing anywhere saying so.
#
# Left alone deliberately: MAX_MCP_OUTPUT_TOKENS (default 25_000 tokens) and
# the per-tool `_meta["anthropic/maxResultSizeChars"]` an MCP server can
# declare (default 100_000 chars for every tool). An MCP result passes both
# gates, so raising one without the other moves nothing, and the runner --
# not this repo -- is what would have to declare the `_meta`.
BASH_MAX_OUTPUT_LENGTH = 150_000
TASK_MAX_OUTPUT_LENGTH = 160_000

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


def _detect_auth_expired(text):
    """Best-effort text match for a turn that lost the OAuth refresh race
    (see AUTH_EXPIRED). Unverified against a real race caught live in this
    pod -- publicly reported symptoms (anthropics/claude-code#24317) are an
    `invalid_grant` from the token endpoint and a CLI prompt to run
    `/login`, both distinct enough from ordinary error text to be worth
    matching narrowly rather than broadly. Update this the first time a
    real race is actually caught here."""
    lowered = text.lower()
    return "invalid_grant" in lowered or "please run /login" in lowered or "please run claude /login" in lowered


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
    isolated directory for a turn running outside _invocation_lock.

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
    checkout. What used to make that expensive -- a fresh `git clone` per
    turn -- is what _provision_workspace removes: the checkout is a
    worktree off the shared clone's own object store, so it costs a
    checkout rather than a network fetch.

    Slots sit in a *sibling* of CLAUDE_WORKSPACE, not inside it. They were
    inside it until 2026-08-23, which put a live turn's working tree
    exactly where another turn's `for d in /data/workspace/*/` sweep would
    find it -- and nothing in that directory says whether it belongs to a
    cycle still running or a cycle that died an hour ago.
    """
    if not slot:
        return CLAUDE_WORKSPACE
    return os.path.join(_concurrent_root(), slot)


def _concurrent_root():
    """Where a concurrent turn's private workspace lives -- a sibling of the
    shared one, never a child of it. Derived on each call rather than fixed
    at import so it follows CLAUDE_WORKSPACE wherever that points.
    """
    return (os.environ.get("CLAUDE_CONCURRENT_ROOT")
            or CLAUDE_WORKSPACE.rstrip("/") + "-concurrent")


def _git(args, cwd, timeout=120):
    """`HOME` is not decoration. `bootstrap_git` writes the HTTPS
    credential helper into `$CLAUDE_HOME/.gitconfig` (a PVC, so it
    survives restarts), and the bridge process's own HOME is
    `/home/bridge`, which has no gitconfig at all. Without the override a
    network git call here authenticates as nobody: three of the four
    shared repos are public and fetch fine anonymously, and
    `platform-config` answers `fatal: could not read Username for
    'https://github.com'` in 0.4s. Measured on the live clones.
    """
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True, timeout=timeout,
                          env={**os.environ, "HOME": CLAUDE_HOME})


def _shared_repos():
    """Every git checkout sitting directly in the shared workspace."""
    try:
        names = sorted(os.listdir(CLAUDE_WORKSPACE))
    except OSError:
        return []
    return [n for n in names
            if os.path.isdir(os.path.join(CLAUDE_WORKSPACE, n, ".git"))]


def _sweep_stale_slots():
    """Drop slot directories a killed turn never cleaned up.

    _run_cli_once's `finally` removes a slot on the way out, but it does
    not run at all if the process dies (pod eviction, OOM, SIGKILL) -- and
    a worktree outlives the directory it was checked out into, as an entry
    in the shared clone's `.git/worktrees`. Both halves get cleared here,
    at the start of a concurrent turn, which is the only moment anything
    is guaranteed to be looking.

    STALE_SLOT_SECONDS is not a guess: a turn is killed at
    CLI_TIMEOUT_SECONDS, so a slot older than that plus a margin cannot
    belong to a turn that is still running.
    """
    cutoff = time.time() - STALE_SLOT_SECONDS
    try:
        slots = os.listdir(_concurrent_root())
    except OSError:
        slots = []
    for name in slots:
        path = os.path.join(_concurrent_root(), name)
        try:
            if os.path.getmtime(path) > cutoff:
                continue
        except OSError:
            continue
        shutil.rmtree(path, ignore_errors=True)
        log(f"swept stale concurrent slot: {name}")
    for repo in _shared_repos():
        try:
            _git(["worktree", "prune"], os.path.join(CLAUDE_WORKSPACE, repo))
        except Exception as exc:
            log(f"worktree prune failed for {repo}: {type(exc).__name__}: {exc}")


def _start_point(src):
    """Which commit a concurrent turn's worktree should be cut from.

    `HEAD` -- what this used until 2026-08-24 -- is whatever branch the
    shared checkout happens to be parked on, and nothing ever parks it
    back. A serialized turn runs `git checkout -b nova/<thing>` in that
    directory and leaves it there when it ends. On 2026-08-24 the shared
    `agora-persona-runner` sat on `nova/status-word-back-on-the-card`, a
    branch whose work had already merged in refined form two cycles
    earlier and which is not an ancestor of `origin/main`, so every
    concurrent cycle woke up on abandoned code and had to diff its own
    work out onto a fresh branch before it could push.

    So: fetch, and start from `origin/main` when there is one. The fetch
    writes remote-tracking refs and objects into the shared clone's
    `.git`; it touches neither its working tree nor its index, so it is
    safe to run while another turn is sitting in that checkout -- which
    is exactly what made a `git checkout`-based fix here unusable.

    Every path falls back to `HEAD`, which is the old behaviour: no
    remote default branch, a fetch that fails because the network is down
    (it was, twice this morning), or `git` itself blowing up still gets a
    checkout rather than none. The whole body is inside the `try` for
    that reason -- an unguarded call here does not degrade to the old
    behaviour, it makes `_provision_workspace` skip the repo and hand the
    turn no checkout at all.

    The timeouts are not about the turn's 45-minute cap: this runs before
    the CLI starts, so it is charged to the caller and is invisible to the
    session's own clock. They bound what a hung network call can add on
    top -- once per shared repo, so four repos is the unit to think in.
    """
    name = os.path.basename(src)
    try:
        res = _git(["fetch", "--quiet", "origin"], src, timeout=60)
        if res.returncode != 0:
            err = (res.stderr or "").strip()
            if "cannot lock ref" in err:
                # Two turns provisioning at once race on the same
                # remote-tracking ref. Measured: 34 of 45 rounds of three
                # simultaneous fetches hit this, every one benign -- the
                # winner has already written the newer value, so the
                # rev-parse below still reads a fresh ref, and no `.lock`
                # is left behind. Logging it as a failure would put an
                # alarming line in the log on most concurrent turns.
                log(f"fetch for {name} raced another turn (harmless)")
            else:
                log(f"fetch failed for {name}: {err[:200]}")
        # `origin/HEAD` first because it is the general answer -- it is the
        # remote's own default branch, so a repo on `master` works without
        # this function knowing about it. Every clone here has one; the
        # `origin/main` line is the fallback for a clone that does not.
        for ref in ("refs/remotes/origin/HEAD", "refs/remotes/origin/main"):
            res = _git(["rev-parse", "--verify", "--quiet", ref], src, timeout=30)
            if res.returncode == 0:
                return ref
    except Exception as exc:
        log(f"start point lookup failed for {name}: {type(exc).__name__}: {exc}")
    # Say so. A silent fall-through here is the original bug returning in
    # full, and "which commit did this turn wake up on" is the exact
    # question that produced this function.
    log(f"no origin default branch for {name} -- starting from the shared checkout's HEAD")
    return "HEAD"


def _provision_workspace(workspace):
    """Check every shared repo out into this turn's own workspace, as a
    git worktree off that repo's existing object store.

    An empty directory was the old contract, and it is the reason
    heartbeat cycles could not use this lane: step 1 of a cycle reads its
    own source, and a cycle that wakes up with no `agora-persona-runner`
    checkout has to clone one over the network before it can do anything.
    A worktree gives it the same files against the same objects, with a
    private working tree and a private index -- which is the whole
    isolation the lane exists for.

    Detached on purpose. Two worktrees may not have the same branch
    checked out, so `--detach` is what makes N of these coexist off one
    clone; a cycle that wants to commit still starts its own branch, from
    the start point _start_point picked -- freshly fetched `origin/main`
    where there is one, rather than whatever the shared checkout is
    parked on.

    One thing that costs, and it is the price of not starting on an
    abandoned branch: unmerged work a serialized turn left on a branch in
    the shared checkout is no longer where this turn wakes up, and
    `git checkout <that-branch>` cannot reach it either -- git refuses a
    branch another worktree holds. `git checkout -b <new> <that-branch>`
    does work. Say that rather than let a turn conclude the work is gone.

    A repo that fails to provision is logged and skipped rather than
    failing the turn: a turn with three of four checkouts can still do
    most jobs, and a turn that refuses to start can do none.
    """
    provisioned = []
    for repo in _shared_repos():
        src = os.path.join(CLAUDE_WORKSPACE, repo)
        dest = os.path.join(workspace, repo)
        try:
            res = _git(["worktree", "add", "--detach", "--force", dest,
                        _start_point(src)], src)
        except Exception as exc:
            log(f"worktree add failed for {repo}: {type(exc).__name__}: {exc}")
            continue
        if res.returncode != 0:
            log(f"worktree add failed for {repo}: {(res.stderr or '').strip()[:200]}")
            continue
        provisioned.append(repo)
    log(f"provisioned {len(provisioned)} worktree(s): {', '.join(provisioned) or 'none'}")
    return provisioned


def _run_cli_once(message, session_id, model, disallowed_tools, activity=None, mcp=None,
                  system=None, attachments=None, slot="", conversation_id="",
                  persona_id="", restricted=False):
    workspace = _workspace_for(slot)
    if slot:
        # _workspace_for's invariant is "a concurrent turn always starts
        # empty", and the `finally` below cannot carry that on its own: it
        # does not run at all if the process is killed (pod eviction, OOM,
        # SIGKILL), and `ignore_errors=True` accepts a partial removal in
        # silence. Slots collide in normal operation -- `slot` is
        # pid+thread-ident, the pid is fixed for the pod's life and CPython
        # reuses thread idents once a thread exits -- so a later, unrelated
        # turn really can be handed a directory an earlier one left behind.
        # Clear it at the point of use, where the invariant is actually
        # needed, instead of trusting the previous turn's exit path.
        shutil.rmtree(workspace, ignore_errors=True)
    os.makedirs(workspace, exist_ok=True)
    if slot:
        # After makedirs, so this turn's own slot has a fresh mtime and the
        # sweep's age test can never pick it up.
        _sweep_stale_slots()
        _provision_workspace(workspace)
    claude_dir = os.path.join(CLAUDE_HOME, ".claude")
    os.makedirs(claude_dir, exist_ok=True)
    # NOVA_WORKSPACE is the same directory as cwd, named so a prompt can
    # say where a file goes without hardcoding /data/workspace -- which is
    # the shared checkout, and therefore the one path a concurrent turn
    # must not write to. `cd "$NOVA_WORKSPACE/agora-persona-runner"` is
    # correct in both lanes; the literal path is only correct in one.
    # AGORA_CONVERSATION_ID is which conversation this turn is *for*, and it
    # is the only thing inside the CLI that can tell one concurrent turn from
    # another. Nova reads its own cycle number off the conversation name, and
    # until this existed it had no way to ask for its own -- so
    # `cycle_number.current_number` took the highest number that existed and
    # called it "mine". That is true only while one cycle runs at a time. On
    # 2026-08-24 three overlapping cycles each read the highest and all three
    # wrote a journal entry headed "Cycle 380"; the owner reported it, having
    # commented on one card and been answered from another, because comments
    # are keyed by cycle number and the number stopped identifying a cycle.
    # Empty string for a caller that passes nothing, so os.environ never gains
    # a None and a turn that does not care is byte-identical to before.
    env = {**os.environ, "HOME": CLAUDE_HOME, "CLAUDE_CONFIG_DIR": claude_dir,
           "NOVA_WORKSPACE": workspace,
           "AGORA_CONVERSATION_ID": conversation_id or ""}
    # A blank value is *not* an override, and the difference is silent in
    # both directions: `setdefault` treats "" as present and leaves it, while
    # the CLI treats "" as unset and falls back to its own 30_000 -- so an
    # empty string in the Deployment would put the cap back at the default
    # while the constant, the comment above and the tests all read 150_000.
    # Measured on 2.1.245 with `claude doctor`: "" prints nothing at all,
    # i.e. unset; "abc" prints `Invalid value "abc" (using default: 30000)`.
    for name, ceiling in (("BASH_MAX_OUTPUT_LENGTH", BASH_MAX_OUTPUT_LENGTH),
                          ("TASK_MAX_OUTPUT_LENGTH", TASK_MAX_OUTPUT_LENGTH)):
        if not env.get(name, "").strip():
            env[name] = str(ceiling)
    # Logged because this is the one value here somebody can change from
    # outside the image. An override that took effect and an override that
    # was ignored look identical from anywhere else.
    log("tool result caps: BASH={} TASK={}".format(
        env["BASH_MAX_OUTPUT_LENGTH"], env["TASK_MAX_OUTPUT_LENGTH"]))

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
    ])
    # --restricted and --dangerously-skip-permissions cannot both be passed:
    # the CLI exits with "bypassPermissions not supported in restricted mode"
    # before the turn starts (measured on 2.1.251, this pod). A restricted turn
    # therefore runs on the default permission mode instead -- which is not a
    # downgrade, because --restricted has already taken the tools that would
    # have needed bypassing. Measured the same way: nothing stalls -- a
    # restricted `-p` turn asked to Read a file uses Read and answers, and one
    # asked to Write gets an immediate permission_denied rather than a hang.
    # Read the second half of that: on the default permission mode `-p` denies
    # instead of prompting, which is why the MCP grant below is not optional.
    if not restricted:
        cmd.append("--dangerously-skip-permissions")
    cmd.extend([

        # Everything a subagent does, on the parent's stream. Without this the
        # CLI reports a Task call and then nothing until it returns, which for
        # a delegated read is several minutes of apparent silence -- the owner's
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
    # The memory pin is per identity, never shared. `write_hook_settings` is
    # called for every turn of every conversation this bridge serves, and the
    # CLI injects MEMORY.md into the first user turn as an override
    # instruction -- so one directory for all of them would make a note Nova
    # wrote to itself a standing instruction to the owner's chat personas,
    # and the reverse. A Nova cycle gets Nova's directory; any other turn
    # gets its own persona's, keyed on the id the caller sent, and no
    # directory at all when the caller sent none (quota.persona_memory_dir).
    # Until idea #165 the second branch was `None`, so a chat persona had no
    # memory across conversations at all -- the CLI's default is keyed on the
    # working directory, which is a fresh concurrent slot every turn.
    hook_settings = write_hook_settings(
        _slotted(quota.HOOK_SETTINGS_FILE, slot),
        memory_dir=(quota.AUTO_MEMORY_DIR if is_cycle_opening(message)
                    else quota.persona_memory_dir(persona_id)),
    )
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
    if restricted:
        cmd.append("--restricted")
        if mcp_config:
            # Without this a restricted persona has no working tools at all,
            # which is not what restricted means and is worse than the hole
            # --restricted was added to close. Measured in this pod on 2.1.251
            # against a stub MCP server: dropping --dangerously-skip-permissions
            # puts the turn on permissionMode "default", and in `-p` there is no
            # prompt and no --permission-prompt-tool, so anything not auto-allowed
            # is hard-denied. The denylist takes every built-in, so the only tools
            # left are this server's -- and an MCP tool is exactly the class that
            # needs approval. The turn returns 200 with the model apologising
            # about a permission nobody can grant it, and nothing logs a failure.
            #
            # A server-wide grant rather than a tool list on purpose: the runner
            # decides which capability tools a persona gets and hands them over
            # in the mcp config, so naming them again here would be a second copy
            # of that decision, drifting from the first. --permission-mode
            # bypassPermissions is not the alternative -- restricted mode refuses
            # it, which is the whole reason the bypass came off above.
            cmd.extend(["--allowedTools", f"mcp__{MCP_SERVER_NAME}"])
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
    # A lost auth race can plausibly fail before the CLI's stream-json
    # format is even up -- stderr merged onto the same stdout the JSON
    # parser reads (Popen's stderr=STDOUT above), printed as plain text
    # rather than a structured error_during_execution event. Buffered
    # separately so it can still be checked for AUTH_EXPIRED once the
    # process exits, without treating ordinary non-JSON chatter as a
    # reason to raise on its own -- see the check just above "CLI produced
    # no text output" below.
    non_json_lines = []

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
                non_json_lines.append(line)
                continue

            t = event.get("type", "")
            # Set on every event a subagent produced, naming the Agent call it
            # belongs to. Empty for the persona's own events.
            #
            # This is the single most load-bearing line in the loop, and it
            # guards two failures that both put a subagent's words in front of
            # the owner as if the persona had said them:
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
                # What each tool RETURNED. The owner has asked for this three
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
                    elif _detect_auth_expired(error_text):
                        saw_error = (AUTH_EXPIRED, error_text[:300])
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
            # PR and written its journal entry, and the only thing the owner
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
        # Same reasoning one step further: this one holds whatever the owner
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
            # Removing the directory leaves the shared clone still listing
            # a worktree that is no longer there, and `git worktree add`
            # refuses a path an existing entry claims. Prune on the way
            # out so the common case never depends on the next turn's
            # sweep; the sweep stays for the turns that never get here.
            for repo in _shared_repos():
                try:
                    _git(["worktree", "prune"], os.path.join(CLAUDE_WORKSPACE, repo))
                except Exception as exc:
                    log(f"worktree prune failed for {repo}: {type(exc).__name__}: {exc}")

    elapsed = time.monotonic() - t0
    log(f"CLI done: exit={proc.returncode} elapsed={elapsed:.1f}s "
        f"text_len={len(''.join(text_parts))} thinking_len={len(''.join(thinking_parts))} "
        f"new_session={new_session_id}")

    # A lost auth race can plausibly fail before stream-json is even up,
    # landing as plain non-JSON text rather than a structured
    # error_during_execution event -- see non_json_lines' own comment.
    # Scoped to "produced nothing at all": a turn that wrote real text and
    # merely logged an unrelated line containing this wording (e.g. a tool
    # reading a doc that mentions /login) must not be reclassified as a
    # lost race on the strength of that line alone.
    if saw_error is None and not text_parts and non_json_lines:
        raw = " ".join(non_json_lines)
        if _detect_auth_expired(raw):
            saw_error = (AUTH_EXPIRED, raw[:300])

    if saw_error is not None:
        kind, detail = saw_error
        if kind == SESSION_NOT_FOUND:
            raise ClaudeCliError(SESSION_NOT_FOUND)
        if kind == AUTH_EXPIRED:
            raise ClaudeCliError(AUTH_EXPIRED)
        if kind == "usage_limit":
            raise UsageLimitError(detail)
        raise ClaudeCliError(detail)

    # Everything the persona wrote except the last passage has already been
    # sent as narration and is already in the conversation, in order, where
    # it happened. Returning the joined whole on top of that would print the
    # entire run a second time inside the reply bubble -- which is what it
    # used to do, and is why the owner's phone kept buzzing with a wall of "let
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
        # joining every passage, and the owner's phone gets the entire
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
             mcp=None, system=None, attachments=None, allow_concurrent=False,
             conversation_id="", persona_id="", restricted=False):
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

    persona_id: which persona is speaking, used only to pin that persona's
    own auto-memory directory (quota.persona_memory_dir). Omit it and the
    turn runs with no memory pin at all, which is what every caller did
    before idea #165.

    conversation_id: which conversation this turn is for, exported to the
    CLI as AGORA_CONVERSATION_ID. Omit it and the variable is exported
    empty, which is what every caller that does not care wants -- the CLI
    sees a variable it will not read.

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
                             system, attachments, slot=slot,
                             conversation_id=conversation_id, persona_id=persona_id,
                             restricted=restricted)
    with _invocation_lock:
        return _run_cli_once(message, session_id, model, disallowed_tools, activity, mcp,
                             system, attachments, conversation_id=conversation_id,
                             persona_id=persona_id, restricted=restricted)
