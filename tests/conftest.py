"""Keeps the quota watcher off the network during tests.

cli.py starts a real QuotaWatcher for every invocation, so the existing
cli tests -- which only mean to exercise stream parsing against a faked
subprocess -- would otherwise fire live requests at
/api/oauth/usage on this box's real subscription credentials. That is
slow, order-dependent, and it earned a real 429 the first time the suite
ran. Tests that want usage data patch fetch_usage themselves.
"""
from unittest.mock import patch

import pytest

from bridge import quota


@pytest.fixture(autouse=True)
def no_live_quota_fetches():
    with patch.object(quota, "fetch_usage", return_value=None):
        yield
