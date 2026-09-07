"""A registry of the CLI subprocesses currently running, keyed on the
conversation each one is answering, so a `POST /cancel` can reach in and
stop the turn a caller no longer wants.

Nothing tracked an in-flight turn before this. `cli.run_turn` held its
`Popen` in a local, so the only thing that could end a turn early was the
2700-second timeout it sets on itself -- and the owner watching a turn burn
minutes on the wrong answer had no way to say so beyond sending a follow-up
message, which is answered on the *next* turn and does not stop the current
one.

Two decisions worth reading before changing anything here.

**A turn is marked cancelled on the process object, never on the
conversation.** A `cancelled` set keyed by conversation id looks simpler and
is wrong in one specific way: a cancel that lands microseconds after the
turn returned would sit in that set until something cleared it, and the
*next* turn for that conversation would read it and stop itself for no
reason. `Turn.cancelled` cannot outlive the process it describes, so that
race has nowhere to leave a mark.

**One conversation can hold more than one turn.** `server.generate` takes
`allow_concurrent`, so two turns for the same conversation can genuinely be
in flight at once. A dict of conversation id -> single process would drop
the older one on registration and leave it unkillable, which is the exact
failure this module exists to end, so the value is a list and `cancel`
stops every turn in it.
"""
import threading
import time

from bridge.log import log

# How long a SIGTERM'd CLI gets to exit on its own before SIGKILL. The CLI
# writes its stream-json to a pipe cli.py is reading, so a clean exit closes
# that pipe and lets the reader loop finish normally -- which is what makes
# the partial answer salvageable. 5s is well past the round trip for a
# process that only has to flush and close, and short enough that the HTTP
# caller waiting on /cancel is not left hanging.
KILL_GRACE_SECONDS = 5


class Turn:
    """One in-flight CLI invocation. `cancelled` is set by cancel() and read
    by run_turn after its process exits, which is how a killed turn tells
    itself apart from one that timed out or simply finished."""

    def __init__(self, conversation_id, proc):
        self.conversation_id = conversation_id
        self.proc = proc
        self.cancelled = False


_turns = {}
_lock = threading.Lock()


def register(conversation_id, proc):
    """Track `proc` as the turn answering `conversation_id`, and return the
    Turn to hand back to unregister().

    An empty conversation id is tracked under no key at all rather than
    under "": there is no way to address such a turn in a cancel request,
    so filing them all together would only build a bucket that cancel()
    could never legitimately be asked for."""
    turn = Turn(conversation_id, proc)
    if not conversation_id:
        return turn
    with _lock:
        _turns.setdefault(conversation_id, []).append(turn)
    return turn


def unregister(turn):
    """Stop tracking a turn whose process has exited. Safe to call for a
    turn register() never filed (empty conversation id) and safe to call
    twice, because the finally block that calls it runs on every path out
    of run_turn, including the ones that already raised."""
    if not turn.conversation_id:
        return
    with _lock:
        remaining = [t for t in _turns.get(turn.conversation_id, []) if t is not turn]
        if remaining:
            _turns[turn.conversation_id] = remaining
        else:
            _turns.pop(turn.conversation_id, None)


def active(conversation_id):
    with _lock:
        return list(_turns.get(conversation_id, []))


def cancel(conversation_id, grace_seconds=KILL_GRACE_SECONDS):
    """Stop every turn running for `conversation_id`. Returns how many were
    signalled -- 0 means there was nothing in flight, which is a normal
    answer and not an error: the turn may have finished between the owner
    pressing stop and this call arriving.

    SIGTERM first, SIGKILL only for a process still alive after the grace.
    The order matters for what the owner gets back: a CLI that exits on
    SIGTERM closes its stdout, cli.py's reader loop ends normally, and the
    text the session had already written is still in hand to be sent as a
    truncated reply. A straight SIGKILL would work just as reliably and
    throw that away."""
    turns = active(conversation_id)
    if not turns:
        return 0
    for turn in turns:
        turn.cancelled = True
        try:
            turn.proc.terminate()
        except (OSError, ProcessLookupError) as exc:
            log(f"cancel: terminate failed for {conversation_id}: "
                f"{type(exc).__name__}: {exc}")
    deadline = time.monotonic() + grace_seconds
    for turn in turns:
        while turn.proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if turn.proc.poll() is None:
            log(f"cancel: {conversation_id} ignored SIGTERM for {grace_seconds}s, killing")
            try:
                turn.proc.kill()
            except (OSError, ProcessLookupError) as exc:
                log(f"cancel: kill failed for {conversation_id}: "
                    f"{type(exc).__name__}: {exc}")
    log(f"cancel: stopped {len(turns)} turn(s) for conversation={conversation_id}")
    return len(turns)
