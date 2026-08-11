"""The leak check in conftest.py is only worth having if it actually fires.

Everything else in this suite is protected by the guard; the guard itself
is protected by nothing, and deleting it would go green. So this runs the
*real* `tests/conftest.py` -- copied, not reimplemented -- over a throwaway
test that leaks a thread on purpose, in a subprocess, and asserts the run
comes back red.

The twin of `agora-persona-runner/tests/test_thread_leak_guard.py`. Both
repos start background threads that outlived a test's patches and reached
something real; keeping the guard in one and not the other is how the
copies drift.
"""
import os
import shutil
import subprocess
import sys
import textwrap

CONFTEST = os.path.join(os.path.dirname(__file__), "conftest.py")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEAKY = """
import threading, time

def test_leaves_a_thread_running():
    threading.Thread(
        target=lambda: time.sleep(60), name="stray-worker", daemon=True
    ).start()


def test_starts_and_joins_a_thread():
    thread = threading.Thread(target=lambda: None, name="tidy-worker")
    thread.start()
    thread.join()
"""


def _run_pytest_on(tmp_path):
    """Subprocess, not an in-process run: the guard is a session-wide hook
    and installing a second copy of it inside this session would have it
    judging the tests that are already running."""
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120,
    )


def test_a_leaked_thread_fails_its_own_test(tmp_path):
    """Named after the test that leaked, not the one that ran next.

    That attribution is the whole point: the quota watcher's stray reading
    landed on whatever test happened to be running, so the symptom and the
    cause were never in the same place.
    """
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "test_leaky.py").write_text(textwrap.dedent(LEAKY), encoding="utf-8")

    result = _run_pytest_on(tmp_path)

    assert result.returncode != 0, result.stdout
    assert "test_leaves_a_thread_running" in result.stdout
    assert "stray-worker" in result.stdout
    assert "tidy-worker" not in result.stdout, "a thread that was joined is not a leak"


def test_a_clean_test_is_not_flagged(tmp_path):
    """The other half, and the one that decides whether anyone keeps the
    guard: a check that fires on tests doing nothing wrong gets deleted."""
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "test_clean.py").write_text(
        "def test_does_nothing():\n    pass\n", encoding="utf-8"
    )

    result = _run_pytest_on(tmp_path)

    assert result.returncode == 0, result.stdout
