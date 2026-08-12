"""Optimistic concurrency on the vault write path (idea #63, slice 1).

Every write through this client is a read-modify-write, and until this
change the revision the caller read at was thrown away: `write` looked up
a *fresh* `_rev` immediately before the PUT. Two writers overlapping did
not conflict, did not error and did not retry -- the second simply won,
and the first one's work was gone with no trace anywhere.

The fake below is a real revision store rather than a stub, because that
is the only way to test this honestly. A fake that returns whatever status
the test wants proves the code branches on 409; it does not prove CouchDB
would ever send one. This one applies CouchDB's actual rule -- a PUT whose
`_rev` does not match the stored one is rejected -- so the tests fail if
the client stops carrying the revision, which is exactly the bug.

`agora_runner/vault.py` in agora-persona-runner carries the same client
against the same database; its half of this landed as runner #118 with
the matching `tests/test_vault_conditional_writes.py` there. The two
clients are hand-synced with nothing detecting drift, so this file is
deliberately the same shape as that one -- a diff between them should
read as "class method vs module function" and nothing else.
"""
import urllib.parse
from unittest.mock import patch

import pytest

from bridge import vault_tool


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("CDB_BASE", "http://couchdb.obsidian.svc.cluster.local:5984")
    monkeypatch.setenv("CDB_USER", "u")
    monkeypatch.setenv("CDB_PASS", "p")
    monkeypatch.setenv("CDB_DB", "obsidian")


class FakeCouch:
    """A CouchDB that enforces `_rev`, and counts the writes it accepted."""

    def __init__(self, docs=None):
        self.docs = dict(docs or {})
        self.rejected = 0
        self.accepted = 0
        self.reads = 0
        #: {nth file-doc read: fn(couch)} -- another writer landing just
        #: before that read is served. Read 1 is the caller's own; read 2
        #: is the lookup inside `write`. **Two is the one that matters,
        #: and getting this wrong is why the first version of these tests
        #: passed against the bug.** An interloper that lands after read 2
        #: is caught either way, because the unconditional path has
        #: already taken its revision by then -- so a test that
        #: interleaves there proves nothing about `if_rev`.
        self.interleave = {}

    def _next_rev(self, doc_id):
        current = self.docs.get(doc_id, {}).get("_rev", "0-x")
        return f"{int(current.split('-')[0]) + 1}-x"

    def store(self, doc_id, doc):
        """Write bypassing the revision check -- 'the other writer'."""
        doc = dict(doc)
        doc["_rev"] = self._next_rev(doc_id)
        self.docs[doc_id] = doc
        return doc["_rev"]

    def req(self, method, _base, _db, _auth, path, body=None, timeout=60):
        rest = urllib.parse.unquote(path)
        if method == "POST" and rest.startswith("_all_docs"):
            return 200, {"rows": [
                {"key": k, "id": k, "value": {"rev": self.docs[k]["_rev"]},
                 "doc": dict(self.docs[k])}
                if k in self.docs else {"key": k, "error": "not_found"}
                for k in body["keys"]
            ]}
        if method == "GET":
            if not rest.startswith("h:"):
                self.reads += 1
                hook = self.interleave.pop(self.reads, None)
                if hook is not None:
                    hook(self)
            if rest in self.docs:
                return 200, dict(self.docs[rest])
            return 404, {"error": "not_found"}
        if method == "PUT":
            sent = (body or {}).get("_rev")
            held = self.docs.get(rest, {}).get("_rev")
            if sent != held:
                self.rejected += 1
                return 409, {"error": "conflict"}
            stored = {k: v for k, v in body.items() if k != "_rev"}
            stored["_rev"] = self._next_rev(rest)
            self.docs[rest] = stored
            self.accepted += 1
            return 201, {"ok": True}
        raise AssertionError(f"unexpected {method} {path}")

    def text(self, doc_id):
        doc = self.docs[doc_id]
        return "".join(self.docs[c]["data"] for c in doc.get("children", []))

    def seed(self, client, doc_id, content):
        """Put `content` at `doc_id` the way another writer would --
        bypassing the revision check, and *advancing* the revision. Not
        advancing it is the one mistake that makes every test here pass for
        the wrong reason: a conflict is a moved revision, so a fake
        other-writer that leaves the revision alone conflicts with nobody."""
        ids = []
        for text in vault_tool._split_chunks(content):
            chunk_id = client._chunk_id_for(text.encode("utf-8"))
            self.docs.setdefault(
                chunk_id, {"_id": chunk_id, "data": text, "_rev": "1-x"})
            ids.append(chunk_id)
        self.store(doc_id, {"_id": doc_id, "path": doc_id, "children": ids,
                            "ctime": 1})


PATH = "notes/issues.md"


@pytest.fixture
def couch(env, monkeypatch):
    """A client wired to a revision-enforcing fake, and the fake itself."""
    client = vault_tool.VaultClient()
    fake = FakeCouch()
    monkeypatch.setattr(vault_tool, "_req", fake.req)
    fake.client = client
    return fake


def test_read_rev_hands_back_the_revision_it_read_at(couch):
    couch.seed(couch.client, PATH, "# Issues\n\n- one\n")
    content, rev = couch.client.read_rev(PATH)
    assert content == "# Issues\n\n- one\n"
    assert rev == "1-x"


def test_read_still_returns_only_the_content(couch):
    """`read` is called all over the CLI and the server; it must not start
    handing back a tuple."""
    couch.seed(couch.client, PATH, "# Issues\n\n- one\n")
    assert couch.client.read(PATH) == "# Issues\n\n- one\n"
    assert couch.client.read("notes/never.md") is None


def test_a_missing_file_and_a_tombstone_are_not_the_same_answer(couch):
    """Both have no content. Only one has a revision, and writing over a
    tombstone has to carry it or the write 409s forever."""
    couch.seed(couch.client, PATH, "gone\n")
    couch.docs[PATH]["deleted"] = True
    assert couch.client.read_rev(PATH) == (None, "1-x")
    assert couch.client.read_rev("notes/never.md") == (None, None)


def test_a_write_carrying_a_stale_revision_loses_instead_of_winning(couch):
    """The whole point. Read, someone else writes, our write must fail --
    and must not have touched their text."""
    couch.seed(couch.client, PATH, "# Issues\n\n- one\n")
    _content, rev = couch.client.read_rev(PATH)
    couch.seed(couch.client, PATH, "# Issues\n\n- one\n- theirs\n")  # the other writer
    result = couch.client.write(PATH, "# Issues\n\n- mine\n", if_rev=rev)
    assert "409 conflict" in result, result
    assert couch.text(PATH) == "# Issues\n\n- one\n- theirs\n"


def test_the_default_write_is_still_an_unconditional_overwrite(couch):
    """Every existing caller passes no `if_rev` and must be unaffected --
    a conditional-by-default write would start failing writes that have
    always succeeded, which is a worse bug than the one being fixed."""
    couch.seed(couch.client, PATH, "# Issues\n\n- one\n")
    assert couch.client.write(PATH, "# Issues\n\n- mine\n") == "written"
    assert couch.text(PATH) == "# Issues\n\n- mine\n"


def test_if_rev_none_means_this_file_should_not_exist_yet(couch):
    assert couch.client.write(PATH, "fresh\n", if_rev=None) == "written"
    clash = couch.client.write(PATH, "clobber\n", if_rev=None)
    assert "409 conflict" in clash, clash
    assert couch.text(PATH) == "fresh\n"


def test_append_that_loses_a_race_re_reads_and_keeps_both_lines(couch):
    """A retry that resent the same body would restore the clobber in a
    loop. The merge has to be redone against the text that won."""
    couch.seed(couch.client, PATH, "# Issues\n\n## Entries\n\n- old\n")

    def other_writer(c):
        c.seed(c.client, PATH, "# Issues\n\n## Entries\n\n- theirs\n- old\n")

    couch.interleave = {2: other_writer}
    result = couch.client.append(PATH, "- mine", "## Entries")

    assert result == "written", result
    assert couch.rejected == 1, "the first attempt should have been rejected"
    final = couch.text(PATH)
    assert "- theirs" in final and "- mine" in final, final


def test_append_gives_up_after_a_bounded_number_of_attempts(couch):
    """A writer that always wins must not spin forever, and the caller has
    to hear that it lost rather than that it succeeded."""
    couch.seed(couch.client, PATH, "# Issues\n\n## Entries\n\n- old\n")
    counter = {"n": 0}

    def always_lose(c):
        counter["n"] += 1
        c.seed(c.client, PATH, f"# Issues\n\n## Entries\n\n- theirs {counter['n']}\n")

    couch.interleave = {2: always_lose, 4: always_lose, 6: always_lose}
    result = couch.client.append(PATH, "- mine", "## Entries")

    assert "409 conflict" in result, result
    assert couch.rejected == vault_tool.APPEND_ATTEMPTS == 3
    assert "- mine" not in couch.text(PATH)


def test_a_missing_marker_still_fails_before_writing_anything(couch):
    """The retry loop must not turn a caller error into three of them."""
    couch.seed(couch.client, PATH, "# Issues\n\n- old\n")
    result = couch.client.append(PATH, "- mine", "## Nope")
    assert "after_marker not found" in result
    assert couch.accepted == 0


def test_an_append_to_a_missing_file_is_still_reported_not_created(couch):
    """`write` would happily create it; a capture file that appeared from
    nowhere would be a silent second copy of the backlog."""
    result = couch.client.append("notes/never.md", "- mine")
    assert "not found" in result
    assert couch.accepted == 0
