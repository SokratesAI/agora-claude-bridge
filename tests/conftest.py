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

import pytest

from bridge import quota


@pytest.fixture(autouse=True, scope="session")
def no_live_quota_fetches():
    with patch.object(quota, "fetch_usage", return_value=None):
        yield
