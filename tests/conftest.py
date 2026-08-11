"""Keeps the quota watcher off the network during tests.

cli.py starts a real QuotaWatcher for every invocation, so the existing
cli tests -- which only mean to exercise stream parsing against a faked
subprocess -- would otherwise fire live requests at
/api/oauth/usage on this box's real subscription credentials. That is
slow, order-dependent, and it earned a real 429 the first time the suite
ran. Tests that want usage data patch fetch_usage themselves.

Session-scoped, not per-test, because the thing it is guarding is a
background thread and a per-test patch is only as good as the thread's
willingness to finish inside the test that started it. It wasn't: the
watcher's final reading ran after close() returned, the patch came off
with the test, and the reading went to the live endpoint and appended to
the real quota-history.jsonl. close() joins now, which fixes the common
case -- but that join has a timeout, so a slow endpoint would still walk
out of a function-scoped patch. Holding the patch open for the whole
session removes the race instead of narrowing it. Tests that stub
fetch_usage themselves nest inside this one and still win.
"""
from unittest.mock import patch
import threading

import pytest

from bridge import deadline, quota


LEAKED_MESSAGE = (
    "this test left {count} background thread(s) still running: {names}. A "
    "thread that outlives the test has also outlived the test's patches, so "
    "whatever it does next it does against the references the patches were "
    "hiding -- the real usage endpoint, the real deadline file, the real "
    "clock -- and it does it while some later test is running, which is "
    "where the blame lands. Its exception, if it raises one, goes to "
    "threading's default excepthook and fails nothing: the quota watcher's "
    "stray reading did not fail anything either, it just appended to the "
    "live history file. Either do not start the thread (patch whatever "
    "starts it) or join it before the test ends."
)

_threads_at_setup = {}


def pytest_runtest_setup(item):
    # The Thread objects rather than their `ident`s: an ident is the OS
    # thread id and the OS reuses those, so a leaked thread that happened
    # to land on a dead one's id would look like it had been here all
    # along. Holding the object costs nothing -- it does not keep the OS
    # thread alive, and the entry is dropped again in teardown.
    _threads_at_setup[item.nodeid] = set(threading.enumerate())


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Fail a test that leaves a thread running behind it.

    This is the general form of the incident the docstring above describes,
    and it is the check that would have caught it in the cycle that caused
    it rather than after a live 429. The two session-scoped fixtures are a
    fix for the two escapes we know about; this one is what notices the
    next one, because it needs no foresight about which thread or which
    module.

    The same shape has now cost the runner twice as well -- its reply
    worker escaped a site test in about one run in three, and its journal
    refresh thread raced the cache reset -- so this file and
    `agora-persona-runner/tests/conftest.py` carry the same guard on
    purpose.

    Checked after the wrapped hook, so fixture finalizers have already run
    -- a fixture that starts a thread and joins it in teardown is doing the
    right thing and must not be flagged for it. No grace period, because
    the bug is the escape and not the duration: a thread still alive once
    the test and its fixtures are done has already outlived the patches,
    whether it finishes a millisecond later or not.
    """
    result = yield
    before = _threads_at_setup.pop(item.nodeid, None)
    if before is None:
        return result
    leaked = sorted(
        t.name for t in threading.enumerate()
        if t not in before and t.is_alive()
    )
    if leaked:
        pytest.fail(LEAKED_MESSAGE.format(count=len(leaked), names=", ".join(leaked)))
    return result


@pytest.fixture(autouse=True, scope="session")
def no_live_quota_fetches():
    with patch.object(quota, "fetch_usage", return_value=None):
        yield


@pytest.fixture(autouse=True, scope="session")
def isolated_turn_deadline(tmp_path_factory):
    """Keeps the cli tests off the real turn clock.

    Same class of problem as the quota fixture above: cli.py writes a
    deadline record for every invocation and clears it in its finally, so
    a suite run on the bridge pod itself would delete the live clock of
    whatever cycle is running -- and the deadline hook would then go quiet
    for the rest of that cycle. Tests that care about the file's contents
    patch DEADLINE_FILE themselves and nest inside this one.
    """
    path = str(tmp_path_factory.mktemp("deadline") / "turn-deadline.json")
    with patch.object(deadline, "DEADLINE_FILE", path):
        yield
