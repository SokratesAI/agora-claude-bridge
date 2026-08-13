"""A database that will not answer must not read as an empty slot to write into.

The read side lost this conflation in bridge#49 (see
`test_vault_unreadable_is_not_missing.py`). The write side kept it, in two
places: `VaultClient.write` and the fallback lookup inside `_put_raw` both
did `existing if status == 200 else None`, so a 500, a 503 or a 401 on the
pre-write lookup made a live document look absent.

It was filed as degrading safely, on the grounds that the resulting PUT
carries no `_rev` and 409s against the live document. That is true of
exactly one of the two shapes it takes, and it is the shape this client
uses least:

- **With a real `if_rev`** -- which is every `put --if-rev-file`, so every
  journal entry and every digest write Nova makes -- the revision comes
  from the caller, so the PUT *succeeds*. The only thing `existing` still
  carries at that point is `ctime`, which silently becomes "now". The write
  lands and quietly rewrites the file's creation time, and nothing anywhere
  says so.
- **Unconditional, or `if_rev=None`**, it does 409 -- but reports
  `FAILED(409 conflict: <path> changed since it was read)`, which is a
  false statement about what happened. Nothing changed. The database
  refused. That string is load bearing: `_write_exit` turns it into
  `CONFLICT_EXIT`, `append` retries on it, and Nova's own instructions say
  exit 3 means re-read and write again. Retrying is the one wrong response
  to a 500.

`agora_runner/vault.py` in agora-persona-runner carries the same client
against the same database and its half of this landed in the same cycle.
The two are hand-synced with nothing detecting drift, so this file is
deliberately the same shape as its twin -- a diff between them should read
as "class method vs module function" and nothing else.
"""
import pytest

from bridge import vault_tool
from tests.test_vault_conditional_writes import FakeCouch

PATH = "notes/issues.md"
BEFORE = "# Issues\n\n- one\n"
MINE = "# Issues\n\n- mine\n"


@pytest.fixture
def couch(monkeypatch):
    """A client wired to a revision-enforcing fake, and the fake itself."""
    monkeypatch.setenv("CDB_BASE", "http://couchdb.obsidian.svc.cluster.local:5984")
    monkeypatch.setenv("CDB_USER", "u")
    monkeypatch.setenv("CDB_PASS", "p")
    monkeypatch.setenv("CDB_DB", "obsidian")
    client = vault_tool.VaultClient()
    fake = FakeCouch()
    monkeypatch.setattr(vault_tool, "_req", fake.req)
    fake.client = client
    return fake


def _seeded_and_unreadable(couch, status=500):
    couch.seed(couch.client, PATH, BEFORE)
    couch.unreadable[PATH] = status
    return couch


def test_an_unreadable_document_is_not_overwritten_as_if_it_were_absent(couch):
    _seeded_and_unreadable(couch)
    result = couch.client.write(PATH, MINE)
    assert "unreadable" in result, result
    assert "500" in result, result
    assert couch.text(PATH) == BEFORE


def test_the_refusal_does_not_masquerade_as_a_conflict(couch):
    """The consequence, not the wording. `409 conflict` in a write result
    means "someone else wrote first, re-read and retry" to `_write_exit`
    (`CONFLICT_EXIT`), to `append`, and to Nova's own instructions. A 500
    answered with that string sends every one of them into a retry against
    a database that is refusing."""
    _seeded_and_unreadable(couch)
    assert "409 conflict" not in couch.client.write(PATH, MINE)


def test_a_conditional_write_no_longer_lands_with_the_ctime_wiped(couch):
    """The half that was never safe. The caller supplies `_rev`, so the PUT
    succeeds whatever the lookup said -- and `existing` is the only source
    of `ctime`, which becomes `now_ms` when it is None. Old behaviour:
    "written", creation time silently replaced. This is the real
    interleaving too: the caller's own read succeeds, and the pre-write
    lookup a moment later does not."""
    couch.seed(couch.client, PATH, BEFORE)
    _content, rev = couch.client.read_rev(PATH)
    couch.unreadable[PATH] = 500
    result = couch.client.write(PATH, MINE, if_rev=rev)
    assert "unreadable" in result, result
    assert couch.text(PATH) == BEFORE
    assert couch.docs[PATH]["ctime"] == 1


def test_the_inner_lookup_refuses_on_its_own(couch):
    """`_put_raw` re-looks-up whenever it is handed `existing=None`, which
    is every call from `write` against a file that really is absent, plus
    any direct caller. Fixing only the outer site would leave the same
    conflation one frame down."""
    _seeded_and_unreadable(couch)
    result = couch.client._put_raw(PATH, MINE, existing=None)
    assert "unreadable" in result, result
    assert couch.text(PATH) == BEFORE


@pytest.mark.parametrize("status", [401, 403, 500, 502, 503])
def test_every_non_404_failure_is_refused_not_just_500(couch, status):
    _seeded_and_unreadable(couch, status=status)
    result = couch.client.write(PATH, MINE)
    assert "unreadable" in result and str(status) in result, result
    assert couch.text(PATH) == BEFORE


def test_a_genuine_404_still_creates_the_file(couch):
    """The negative control, and the reason this is a narrowing rather than
    a new gate. 404 is a real answer -- a journal entry that has not been
    written yet, an archive not yet rolled -- and creating the file is the
    correct response to it. A fix that refused here would break every first
    write this client makes."""
    assert couch.client.write("notes/brand-new.md", MINE) == "written"
    assert couch.text("notes/brand-new.md") == MINE


def test_the_helper_raises_rather_than_returning_a_sentinel(couch):
    """Absent and unreadable share one vocabulary with the read side --
    `VaultUnreadableDocument` -- and the string contract lives at the write
    entry points, which catch it. Both halves matter: a raise that escaped
    `write` would break every caller that branches on the returned string,
    which is the finding runner#148's reviewer caught about this same
    exception class."""
    _seeded_and_unreadable(couch)
    with pytest.raises(vault_tool.VaultUnreadableDocument) as excinfo:
        couch.client._doc_to_overwrite(PATH)
    assert couch.client._doc_to_overwrite("notes/nothing-here.md") is None
    assert "500" in str(excinfo.value)
