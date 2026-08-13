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
import io
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
        #: {doc_id: status} -- a GET of that *file* doc is answered with this
        #: status instead of the document. Chunk reads and `_all_docs` are
        #: untouched, which is the point: the failure this models is one
        #: document's read failing while the database is otherwise up, so the
        #: chunk writes still succeed and the file PUT still reaches CouchDB.
        #: A fake that took the whole database down instead would fail at the
        #: chunk stage and never exercise the branch under test.
        #: Used by `test_vault_unreadable_is_not_absent_on_write.py`.
        self.unreadable = {}

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
                status = self.unreadable.get(rest)
                if status is not None:
                    return status, {"error": "server_error"}
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


# ---------------------------------------------------------------------------
# The CLI half (2026-08-12, second slice). The client above has been able to
# do a conditional write since #47 and *no caller passed one*: `prompt.md`
# step 7 writes the journal and the digest through `vault_tool put`, which
# had no way to send a revision and no way to report a lost one. The
# mechanism existed everywhere except the two writes it was built for.
# ---------------------------------------------------------------------------


def _cli(argv, capsys):
    """`main(argv)`, returning `(exit_code, stdout, stderr)`.

    All three, because `capsys` can only be drained once: a test that reads
    stdout here and then asks for stderr gets an empty string and passes
    whatever the tool actually printed.
    """
    code = vault_tool.main(argv)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_get_records_the_revision_it_was_served(couch, tmp_path, capsys):
    couch.seed(couch.client, PATH, "# Issues\n\n- one\n")
    rev_file = tmp_path / "d.rev"
    code, out, _err = _cli(["get", PATH, "--rev-file", str(rev_file)], capsys)
    assert code == 0
    assert out == "# Issues\n\n- one\n\n"
    assert rev_file.read_text().strip() == "1-x"


def test_a_paired_get_and_put_refuses_to_overwrite_a_writer_in_between(
        couch, tmp_path, capsys):
    """The whole reason this exists, in the shape `prompt.md` actually uses:
    read the digest to a file, edit it, write it back. Somebody lands in the
    gap -- which with four cycles an hour is an ordinary Tuesday."""
    couch.seed(couch.client, PATH, "line A\n")
    rev_file, body = tmp_path / "d.rev", tmp_path / "live.md"
    _cli(["get", PATH, "--rev-file", str(rev_file)], capsys)
    body.write_text("line A\nmine\n")

    couch.seed(couch.client, PATH, "line A\ntheirs\n")   # the other writer

    code, out, _err = _cli(["put", PATH, str(body), "--if-rev-file", str(rev_file)], capsys)
    assert code == vault_tool.CONFLICT_EXIT == 3
    assert "409 conflict" in out
    assert couch.text(PATH) == "line A\ntheirs\n"        # theirs is intact
    assert "mine" not in couch.text(PATH)


def test_the_same_pair_writes_normally_when_nobody_interferes(couch, tmp_path, capsys):
    """A guard that also blocks the ordinary path is not a guard."""
    couch.seed(couch.client, PATH, "line A\n")
    rev_file, body = tmp_path / "d.rev", tmp_path / "live.md"
    _cli(["get", PATH, "--rev-file", str(rev_file)], capsys)
    body.write_text("line A\nmine\n")

    code, out, _err = _cli(["put", PATH, str(body), "--if-rev-file", str(rev_file)], capsys)
    assert code == 0
    assert out.startswith("written:")
    assert couch.text(PATH) == "line A\nmine\n"


def test_an_unpaired_put_is_still_an_unconditional_overwrite(couch, tmp_path, capsys):
    """Every existing caller in the vault and in `prompt.md` is unpaired.
    None of them may start failing today."""
    couch.seed(couch.client, PATH, "line A\n")
    body = tmp_path / "live.md"
    body.write_text("replaced\n")
    code, _out, err = _cli(["put", PATH, str(body)], capsys)
    assert code == 0
    assert couch.text(PATH) == "replaced\n"


def test_a_rev_file_for_a_missing_path_means_it_must_still_be_missing(
        couch, tmp_path, capsys):
    """The journal-entry case: two cycles pick the same sequence number, both
    `get` a 404, both `put`. One of them has to lose, and the loser must not
    be the one whose entry is already there."""
    rev_file, body = tmp_path / "j.rev", tmp_path / "entry.md"
    code, out, _err = _cli(["get", "j/071-cycle-71.md", "--rev-file", str(rev_file)], capsys)
    assert code == 0 and "[not found:" in out
    assert rev_file.read_text().strip() == vault_tool.ABSENT_REV

    couch.seed(couch.client, "j/071-cycle-71.md", "the other cycle's entry\n")
    body.write_text("my entry\n")
    code, out, _err = _cli(
        ["put", "j/071-cycle-71.md", str(body), "--if-rev-file", str(rev_file)], capsys)
    assert code == 3, out
    assert couch.text("j/071-cycle-71.md") == "the other cycle's entry\n"


def test_a_missing_rev_file_is_an_error_not_an_unconditional_write(
        couch, tmp_path, capsys):
    """The dangerous fallback. A caller that asked for a conditional write
    and silently got a clobber is worse off than one that never asked."""
    couch.seed(couch.client, PATH, "line A\n")
    body = tmp_path / "live.md"
    body.write_text("mine\n")
    code, _out, err = _cli(
        ["put", PATH, str(body), "--if-rev-file", str(tmp_path / "gone.rev")], capsys)
    assert code == 1
    assert couch.accepted == 0
    assert couch.text(PATH) == "line A\n"
    assert "Refusing to fall back" in err


def test_an_empty_rev_file_is_an_error_too(couch, tmp_path, capsys):
    """A truncated or half-written rev file reads as 'no expectation' if you
    let it, which is the same clobber one step removed."""
    couch.seed(couch.client, PATH, "line A\n")
    body, rev_file = tmp_path / "live.md", tmp_path / "d.rev"
    body.write_text("mine\n")
    rev_file.write_text("\n")
    code, _out, err = _cli(
        ["put", PATH, str(body), "--if-rev-file", str(rev_file)], capsys)
    assert code == 1
    assert couch.text(PATH) == "line A\n"


def test_puts_takes_the_same_pairing(couch, tmp_path, capsys, monkeypatch):
    """`puts` is the stdin form and is what the server-side helpers reach
    for; leaving it unconditional would have left a second unguarded door."""
    couch.seed(couch.client, PATH, "line A\n")
    rev_file = tmp_path / "d.rev"
    _cli(["get", PATH, "--rev-file", str(rev_file)], capsys)
    couch.seed(couch.client, PATH, "line A\ntheirs\n")

    monkeypatch.setattr(vault_tool.sys, "stdin", io.StringIO("mine\n"))
    code, out, _err = _cli(["puts", PATH, "--if-rev-file", str(rev_file)], capsys)
    assert code == 3, out
    assert couch.text(PATH) == "line A\ntheirs\n"


def test_a_failed_write_exits_non_zero(couch, tmp_path, capsys):
    """It exited 0 while printing `FAILED`, and this loop's own instructions
    compensated with "read the output" -- a shell running four chained vault
    writes does not read output."""
    body = tmp_path / "live.md"
    body.write_text("mine\n")
    with patch.object(vault_tool.VaultClient, "write", return_value="FAILED(503)"):
        code, out, _err = _cli(["put", PATH, str(body)], capsys)
    assert code == 1
    assert "FAILED(503)" in out


def test_a_failed_append_exits_non_zero_too(couch, tmp_path, capsys):
    """Same trap on the capture files: a missing marker printed a failure and
    exited 0, so a wrapper script counted it as filed."""
    couch.seed(couch.client, PATH, "# Issues\n\n- old\n")
    body = tmp_path / "cap.md"
    body.write_text("- mine\n")
    code, out, _err = _cli(["append", PATH, str(body), "## Nope"], capsys)
    assert code == 1
    assert "after_marker not found" in out
    assert couch.accepted == 0


def test_a_successful_write_still_exits_zero(couch, tmp_path, capsys):
    couch.seed(couch.client, PATH, "# Issues\n\n## Entries\n\n- old\n")
    body = tmp_path / "cap.md"
    body.write_text("- mine\n")
    assert _cli(["append", PATH, str(body), "## Entries"], capsys)[0] == 0


def test_an_option_is_not_mistaken_for_a_path(couch, tmp_path, capsys):
    """`--rev-file` is pulled out before the positionals are read, so the
    path is still argv[1] wherever the option was written. `append` and
    `appends` take no options, so their markers -- which may begin with
    `-` -- never reach this parser at all."""
    couch.seed(couch.client, PATH, "# Issues\n\n## Entries\n\n- old\n")
    rev_file = tmp_path / "d.rev"
    code, out, _err = _cli(["get", "--rev-file", str(rev_file), PATH], capsys)
    assert code == 0 and "- old" in out
    assert rev_file.read_text().strip() == couch.docs[PATH]["_rev"]


def test_a_rev_file_option_with_no_value_is_a_usage_error(couch, capsys):
    code, _out, err = _cli(["get", PATH, "--rev-file"], capsys)
    assert code == 1
    assert "needs a filename" in err


def test_a_tombstone_rev_is_recorded_so_the_overwrite_can_carry_it(
        couch, tmp_path, capsys):
    """`get` prints `[not found]` for a tombstone as well as for a missing
    file. Recording ABSENT_REV for the tombstone would make every write over
    a deleted note 409 forever."""
    couch.seed(couch.client, PATH, "gone\n")
    couch.docs[PATH]["deleted"] = True
    rev_file, body = tmp_path / "d.rev", tmp_path / "live.md"
    _cli(["get", PATH, "--rev-file", str(rev_file)], capsys)
    assert rev_file.read_text().strip() == "1-x"

    body.write_text("back again\n")
    code, out, _err = _cli(["put", PATH, str(body), "--if-rev-file", str(rev_file)], capsys)
    assert code == 0, out
    assert couch.text(PATH) == "back again\n"
