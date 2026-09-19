"""The turn's time limit has to hold while the CLI is still streaming, and a
CLI that has exited must not keep the turn open just because something it
started still holds its stdout (idea #303).

Real processes rather than fakes: both failures live in how a pipe behaves
when its writer dies or never finishes, which a fake iterable cannot show.
"""
import os
import signal
import subprocess
import sys
import time

from bridge import cli


def _popen(script):
    return subprocess.Popen([sys.executable, "-c", script],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def test_a_cli_that_exits_normally_yields_every_line_and_no_flag():
    """The positive control: without it the two tests below would pass on a
    reader that gave up on every process."""
    proc = _popen("print('one'); print('two')")
    state = {}
    lines = [l.strip() for l in cli._stdout_lines(proc, timeout=30, state=state, grace=1)]
    assert lines == ["one", "two"]
    assert state == {}
    assert proc.wait(timeout=5) == 0


def test_a_hung_cli_is_killed_at_the_limit_while_it_holds_stdout_open():
    proc = _popen("import time; print('started', flush=True); time.sleep(60)")
    state = {}
    began = time.monotonic()
    lines = [l.strip() for l in cli._stdout_lines(proc, timeout=2, state=state, grace=30)]
    assert time.monotonic() - began < 10
    assert lines == ["started"]
    assert state.get("timed_out") is True
    assert proc.wait(timeout=5) == -signal.SIGKILL


def test_an_exited_cli_whose_child_still_holds_the_pipe_ends_the_turn():
    """The child inherits stdout and outlives its parent: the pipe never
    reaches EOF while it lives, and before this the reader waited on it."""
    proc = _popen(
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print('answer', flush=True)\n"
        "print(child.pid, flush=True)\n")
    state = {}
    began = time.monotonic()
    lines = [l.strip() for l in cli._stdout_lines(proc, timeout=60, state=state, grace=1)]
    try:
        assert time.monotonic() - began < 15
        assert lines[0] == "answer"
        assert state.get("orphaned") is True
        assert "timed_out" not in state
        assert proc.wait(timeout=5) == 0
    finally:
        os.kill(int(lines[1]), signal.SIGKILL)
        time.sleep(0.5)  # let the reader thread see EOF so the leak guard stays quiet


def test_a_cli_that_never_stops_printing_is_still_killed_at_the_limit():
    """A deadline checked only when the pipe goes quiet never fires on a CLI
    that prints faster than the reader's tick."""
    proc = _popen("import time\nwhile True:\n    print('x', flush=True)\n    time.sleep(0.05)")
    state = {}
    began = time.monotonic()
    for _ in cli._stdout_lines(proc, timeout=2, state=state, grace=30):
        assert time.monotonic() - began < 10
    assert state.get("timed_out") is True
    assert proc.wait(timeout=5) == -signal.SIGKILL
