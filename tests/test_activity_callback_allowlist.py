"""The activity callback URL is an allowlist, not a free field.

CodeQL alert #5 on this repo, `py/full-ssrf`, critical: the URL the bridge
POSTs live tool-use chips to arrives in the /generate request body and
reached `urllib.request.urlopen` with nothing looking at it. /generate is a
plain in-cluster HTTP endpoint, so any pod that can reach this one could
name any host and have the bridge fetch it.

The real caller only ever sends the runner's or the site's own Service
address, so these pin the shape that has always been used rather than the
attacks that were not.
"""
import urllib.error

import pytest

from bridge.activity import ActivityReporter, callback_allowed


ALLOWED = [
    "http://agora-persona-runner.agents.svc.cluster.local:8082/tool-activity",
    "http://nova-site.agents.svc.cluster.local:8083/tool-activity",
    "https://agora.agents.svc.cluster.local:8080/tool-activity",
    "http://localhost:8082/tool-activity",
    "http://127.0.0.1:8082/tool-activity",
]

REFUSED = [
    # The plain external host -- what a full SSRF is for.
    "http://evil.example.com/collect",
    "https://evil.example.com/collect",
    # The cloud metadata service, which is the reason this class of bug is
    # rated critical rather than medium.
    "http://169.254.169.254/latest/meta-data/",
    # A `user@` prefix: `netloc` reads as the allowed suffix and the host
    # this actually resolves is the one after the `@`.
    "http://agora-persona-runner.agents.svc.cluster.local@evil.example.com/x",
    # Suffix matching without the leading dot would accept this.
    "http://notsvc.cluster.local/x",
    # Schemes that are not an HTTP POST at all.
    "file:///etc/passwd",
    "gopher://evil.example.com:70/x",
    "ftp://evil.example.com/x",
    # An allowed HOST with a scheme that is not an HTTP POST. Without the
    # scheme check these pass the host test and are the only inputs that
    # do -- every other refused row above fails on the host alone, so
    # deleting the scheme check leaves the rest of this list green.
    "file://localhost/etc/passwd",
    "gopher://agora-persona-runner.agents.svc.cluster.local:70/x",
    "",
]


@pytest.mark.parametrize("url", ALLOWED)
def test_in_cluster_callbacks_are_allowed(url):
    assert callback_allowed(url) is True


@pytest.mark.parametrize("url", REFUSED)
def test_everything_else_is_refused(url):
    assert callback_allowed(url) is False


def test_reporter_disables_itself_on_a_refused_url():
    """A bad URL costs the narration, never the turn."""
    reporter = ActivityReporter({
        "url": "http://evil.example.com/collect", "token": "t"})
    assert reporter.enabled is False
    # Still a working object -- report/close must not raise on a disabled
    # reporter, which is the whole "best effort" contract of this module.
    reporter.report("Bash", {"command": "ls"})
    reporter.close()


def test_reporter_keeps_a_cluster_url():
    reporter = ActivityReporter({
        "url": "http://agora-persona-runner.agents.svc.cluster.local:8082/tool-activity",
        "token": "t"})
    assert reporter.enabled is True


def test_post_refuses_before_it_opens_a_socket(monkeypatch):
    """The guard is in _post too, so a URL that reached the queue by any
    other path still cannot leave the pod.

    Recorded rather than raised: `_post` swallows every exception by
    design, so an `AssertionError` from inside a fake urlopen is caught by
    the module under test and the assertion below passes whether or not the
    guard exists -- a test that guards nothing.
    """
    from bridge import activity

    calls = []

    def record(*args, **kwargs):
        calls.append(args)
        raise urllib.error.URLError("no network in a test")

    monkeypatch.setattr(activity.urllib.request, "urlopen", record)
    assert activity._post("http://evil.example.com/collect", {"a": 1}) is False
    assert calls == [], "urlopen was reached for a URL the allowlist refuses"

    # The control: the same call with an allowed URL does reach urlopen, so
    # the empty list above is a refusal rather than a fake that never runs.
    assert activity._post(
        "http://agora-persona-runner.agents.svc.cluster.local:8082/x",
        {"a": 1}) is False
    assert len(calls) == 1
