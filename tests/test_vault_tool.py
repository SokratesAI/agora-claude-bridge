import json
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

    # _client_with_fake_req only replaces VaultClient._doc; anything calling
    # the module-level _req directly (delete's DELETE, recent's _find) went
    # to the network unfaked. That is harmless on a laptop and not harmless
    # here -- the bridge pod resolves CDB_BASE, so such a test runs against
    # the REAL vault. Found 2026-08-06 when a new delete test passed in-pod
    # and failed in CI, where there is no DNS. Fail loudly instead.
    def _no_network(method, base, db, auth, path, body=None):
        raise AssertionError(
            f"test made a real _req call ({method} {path}) -- patch "
            "vault_tool._req for this test instead"
        )
    monkeypatch.setattr(vault_tool, "_req", _no_network)


def _client_with_fake_req(responses):
    """responses: dict of (method, doc_id) -> (status, body), consumed via a
    fake bound to VaultClient._doc/_req-shaped calls."""
    client = vault_tool.VaultClient()
    calls = []

    def fake_doc(method, doc_id, body=None):
        calls.append((method, doc_id, body))
        return responses.get((method, doc_id), (404, {"error": "not_found"}))

    client._doc = fake_doc
    return client, calls


def test_read_returns_none_when_doc_missing(env):
    client, _ = _client_with_fake_req({})
    assert client.read("projects/sokrates/foo.md") is None


def test_read_lowercases_path(env):
    client, calls = _client_with_fake_req({
        ("GET", "projects/foo.md"): (200, {"data": "hello", "children": []}),
    })
    content = client.read("Projects/Foo.md")
    assert content == "hello"
    assert calls[0] == ("GET", "projects/foo.md", None)


def test_read_assembles_chunked_children(env):
    client, _ = _client_with_fake_req({
        ("GET", "note.md"): (200, {"children": ["h:1", "h:2"]}),
        ("GET", "h:1"): (200, {"data": "part one "}),
        ("GET", "h:2"): (200, {"data": "part two"}),
    })
    assert client.read("note.md") == "part one part two"


def test_write_does_not_copy_previous_content_into_the_vault(env):
    """Overwriting used to write a second document under agora/backups/
    holding the old content. Edvard asked for that to stop (2026-08-05):
    it doubled the write cost of every edit and left 272 files behind.
    The daily GitHub snapshot is the recovery path instead."""
    client, calls = _client_with_fake_req({
        ("GET", "note.md"): (200, {"data": "old content", "children": [], "_rev": "1-abc"}),
    })
    client.write("note.md", "new content")

    put_calls = [c for c in calls if c[0] == "PUT"]
    assert [c for c in put_calls if c[1].startswith("agora/backups/")] == []
    # The old content must not be re-persisted under any other path either.
    assert [c for c in put_calls if c[2].get("data") == "old content"] == []


def test_delete_does_not_copy_content_into_the_vault(env):
    """Deleting made a backup copy too -- which meant deleting the
    backups folder itself created new backups of it."""
    client, calls = _client_with_fake_req({
        ("GET", "doomed.md"): (200, {"data": "content", "children": [], "_rev": "1-abc"}),
    })
    sent = []

    def fake_req(method, base, db, auth, path, body=None):
        sent.append((method, path))
        return 200, {"ok": True}

    with patch.object(vault_tool, "_req", fake_req):
        assert client.delete("doomed.md") == "deleted"

    assert [c for c in calls if c[0] == "PUT"] == []
    assert [m for m, _ in sent] == ["DELETE"]


def test_append_fails_when_file_does_not_exist(env):
    client, _ = _client_with_fake_req({})
    result = client.append("missing.md", "new entry")
    assert result.startswith("FAILED")


def test_append_inserts_after_marker(env):
    existing = "# Journal\n\n## Entries\n\n## [old] Cycle 1\nstuff\n"
    client, calls = _client_with_fake_req({
        ("GET", "journal.md"): (200, {"data": existing, "children": [], "_rev": "1-abc"}),
    })
    client.append("journal.md", "## [new] Cycle 2\nnew stuff", after_marker="## Entries")
    chunk_put = [c for c in calls if c[0] == "PUT" and c[2].get("type") == "leaf"][-1]
    assert "## [new] Cycle 2" in chunk_put[2]["data"]
    assert "## [old] Cycle 1" in chunk_put[2]["data"]
    assert chunk_put[2]["data"].index("## [new] Cycle 2") < chunk_put[2]["data"].index("## [old] Cycle 1")


def test_append_appends_at_end_when_no_marker_given(env):
    existing = "line one\nline two\n"
    client, calls = _client_with_fake_req({
        ("GET", "note.md"): (200, {"data": existing, "children": [], "_rev": "1-abc"}),
    })
    client.append("note.md", "line three")
    chunk_put = [c for c in calls if c[0] == "PUT" and c[2].get("type") == "leaf"][-1]
    assert chunk_put[2]["data"].strip().endswith("line three")


def test_append_fails_loudly_when_explicit_marker_not_found(env):
    """This used to fall through to a silent append at the END of the file.
    journal.md asks for insertion after '## Entries' to keep newest-first
    order; when the marker didn't match, entries landed at the bottom
    instead, for several cycles, with no error -- so the journal's order
    silently scrambled and its newest entries became unfindable."""
    existing = "line one\nline two\n"
    client, calls = _client_with_fake_req({
        ("GET", "note.md"): (200, {"data": existing, "children": [], "_rev": "1-abc"}),
    })
    result = client.append("note.md", "line three", after_marker="## Nonexistent")
    assert result.startswith("FAILED")
    assert "## Nonexistent" in result
    # Nothing may be written at all.
    assert [c for c in calls if c[0] == "PUT"] == []


def test_delete_returns_absent_when_missing(env):
    client, _ = _client_with_fake_req({})
    assert client.delete("gone.md") == "absent"


def _fake_find(docs, captured=None):
    """Stand in for the module-level _req, answering the _find POST."""
    def fake_req(method, base, db, auth, path, body=None):
        assert (method, path) == ("POST", "_find")
        if captured is not None:
            captured.append(body)
        return 200, {"docs": docs}
    return fake_req


def test_recent_drops_livesync_internals_and_backups_newest_first(env):
    docs = [
        {"_id": "h:deadbeef", "mtime": 5000},
        {"_id": "agora/backups/20260803-114439 journal.md", "mtime": 4000},
        {"_id": "projects/a.md", "mtime": 1000},
        {"_id": "projects/b.md", "mtime": 3000},
    ]
    client = vault_tool.VaultClient()
    with patch.object(vault_tool, "_req", _fake_find(docs)):
        rows, truncated = client.recent(hours=6)
    assert rows == [(3000, "projects/b.md", False), (1000, "projects/a.md", False)]
    assert truncated is False


def test_recent_includes_backups_only_when_prefix_asks_for_them(env):
    docs = [
        {"_id": "agora/backups/20260803-114439 journal.md", "mtime": 4000},
        {"_id": "projects/a.md", "mtime": 1000},
    ]
    client = vault_tool.VaultClient()
    with patch.object(vault_tool, "_req", _fake_find(docs)):
        rows, _ = client.recent(hours=6, prefix="agora/backups/")
    assert [p for _, p, _d in rows] == ["agora/backups/20260803-114439 journal.md"]


def test_recent_selector_window_follows_hours(env):
    captured = []
    client = vault_tool.VaultClient()
    with patch.object(vault_tool, "_req", _fake_find([], captured)):
        client.recent(hours=6)
    since_six = captured[0]["selector"]["mtime"]["$gt"]
    captured.clear()
    with patch.object(vault_tool, "_req", _fake_find([], captured)):
        client.recent(hours=12)
    since_twelve = captured[0]["selector"]["mtime"]["$gt"]
    # A wider window must reach further back -- ~6h more, allowing for the
    # clock moving between the two calls.
    assert 5.9 * 3600_000 < since_six - since_twelve < 6.1 * 3600_000


def test_recent_reports_truncation_because_the_subset_is_arbitrary(env):
    docs = [{"_id": f"projects/{i}.md", "mtime": i} for i in range(3)]
    client = vault_tool.VaultClient()
    with patch.object(vault_tool, "_req", _fake_find(docs)):
        rows, truncated = client.recent(hours=6, limit=3)
    assert len(rows) == 3
    assert truncated is True


def test_main_recent_warns_loudly_when_the_list_is_incomplete(env, capsys):
    with patch.object(vault_tool, "VaultClient") as MockClient:
        MockClient.return_value.recent.return_value = ([(0, "projects/a.md", False)], True)
        vault_tool.main(["recent", "6"])
    out = capsys.readouterr().out
    assert "INCOMPLETE" in out
    assert "projects/a.md" in out


def test_main_recent_says_so_when_nothing_changed(env, capsys):
    with patch.object(vault_tool, "VaultClient") as MockClient:
        MockClient.return_value.recent.return_value = ([], False)
        vault_tool.main(["recent", "6"])
    assert "[nothing modified in the last 6h]" in capsys.readouterr().out


def test_local_stamp_is_oslo_not_utc(env):
    # 2026-08-03 11:39:15Z -- the moment PR #34 merged. Oslo is UTC+2 in August.
    assert vault_tool._local_stamp(1785757155000) == "2026-08-03 13:39"


def test_main_appends_treats_leading_dash_as_stdin_not_as_the_marker(env):
    """The usage line used to read `appends <path> - [after_marker]`, so a
    caller following it passed "-" through as after_marker. Nothing matches
    "-", so the entry silently went to the end of the file -- which is how
    journal.md's newest entries ended up buried at the bottom."""
    with patch.object(vault_tool, "VaultClient") as MockClient, \
         patch.object(vault_tool.sys, "stdin") as stdin:
        stdin.read.return_value = "new entry"
        vault_tool.main(["appends", "journal.md", "-", "## Entries"])
    MockClient.return_value.append.assert_called_once_with(
        "journal.md", "new entry", "## Entries")


def test_main_appends_without_the_dash_still_reads_the_marker(env):
    with patch.object(vault_tool, "VaultClient") as MockClient, \
         patch.object(vault_tool.sys, "stdin") as stdin:
        stdin.read.return_value = "new entry"
        vault_tool.main(["appends", "journal.md", "## Entries"])
    MockClient.return_value.append.assert_called_once_with(
        "journal.md", "new entry", "## Entries")


def test_main_appends_with_no_marker_at_all_appends_at_the_end(env):
    with patch.object(vault_tool, "VaultClient") as MockClient, \
         patch.object(vault_tool.sys, "stdin") as stdin:
        stdin.read.return_value = "new entry"
        vault_tool.main(["appends", "notes.md"])
    MockClient.return_value.append.assert_called_once_with(
        "notes.md", "new entry", "")


def test_main_get_prints_not_found_marker(env, capsys):
    with patch.object(vault_tool, "VaultClient") as MockClient:
        MockClient.return_value.read.return_value = None
        vault_tool.main(["get", "missing.md"])
    assert "[not found: missing.md]" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# LiveSync tombstones (2026-08-07). Deleting a note in Obsidian does not
# remove the CouchDB document -- it sets `deleted: true` and leaves the
# content chunks attached, which is how other clients learn to drop their
# local copy. Nothing here knew the flag existed, so `ls` listed deleted
# files and `get` rebuilt their text.
#
# Measured on the live vault the day this was found: 309 of 897 documents
# were tombstones. Most were pre-move copies left by a vault reorganisation,
# sitting one prefix away from their live replacement. One was `kanban.md`,
# deleted outright with no replacement -- while `prompt.md` still told every
# cycle to read it as "Edvard's own real backlog", handing four cycles a day
# a board frozen on 2026-07-29 with no way to tell it was gone.
#
# `list`/`read` treat deleted as gone. `recent` deliberately does not: it
# answers "what changed", and a deletion is the change that invalidates
# every path still pointing at it. Not seeing it is what made this take a
# day to notice.
# ---------------------------------------------------------------------------
def _fake_all_docs(docs, seen=None):
    """Fake _req covering the two calls file_docs makes: the id sweep and
    the batched include_docs re-fetch.

    The sweep **honours `startkey`/`endkey`**, the way CouchDB does. A fake
    that ignored them would serve the whole database to a caller asking for
    one folder and every test here would still pass -- which is exactly the
    bug that lived in `file_docs` for two days, and a fake is the one place
    it could hide again. `seen` collects the sweep paths so a test can
    assert the range was asked for at all."""
    def fake(method, base, db, auth, path, body=None):
        if path.startswith("_all_docs?startkey=") and method == "GET":
            if seen is not None:
                seen.append(path)
            query = urllib.parse.parse_qs(path.split("?", 1)[1])
            start = json.loads(query["startkey"][0])
            end = json.loads(query["endkey"][0])
            return 200, {"rows": [{"id": d["_id"]} for d in docs
                                  if start <= d["_id"] <= end]}
        if path.startswith("_all_docs?include_docs=true") and method == "POST":
            by_id = {d["_id"]: d for d in docs}
            return 200, {"rows": [{"id": k, "doc": by_id[k]}
                                  for k in body["keys"] if k in by_id]}
        raise AssertionError(f"unexpected request: {method} {path}")
    return fake


def test_listing_a_folder_asks_couchdb_for_that_folder_only(env):
    """The id sweep used to fetch `_all_docs` unrestricted and filter the
    rows in Python -- every listing paid for the whole vault. Measured end
    to end on the live vault 2026-08-11: `ls` of the 103-file journal
    folder went 3.3s to 1.0s, byte-identical output. `list` is the only
    caller -- `recent` goes through Mango `_find` and never paid this."""
    docs = [
        {"_id": "projects/agora/journal/104-cycle-95.md"},
        {"_id": "projects/agora/journal/103-cycle-94.md"},
        {"_id": "projects/other/big.md"},
        {"_id": "zzz/last.md"},
    ]
    seen = []
    client = vault_tool.VaultClient()
    with patch.object(vault_tool, "_req", _fake_all_docs(docs, seen)):
        listed = client.list("projects/agora/journal/")
    assert listed == ["projects/agora/journal/103-cycle-94.md",
                      "projects/agora/journal/104-cycle-95.md"]
    query = urllib.parse.parse_qs(seen[0].split("?", 1)[1])
    assert json.loads(query["startkey"][0]) == "projects/agora/journal/"
    assert json.loads(query["endkey"][0]).startswith("projects/agora/journal/")


def test_a_mixed_case_prefix_still_finds_the_folder(env):
    """Paths are case-insensitive here -- `read` lowercases, the tool
    description promises it, and Nova's own docs write `Projects/Sokrates/`
    with capitals. The old sweep lowercased the row ids in Python *after*
    fetching everything, so case never reached CouchDB. A range does reach
    it, and `"Projects/" <= "projects/foo.md"` is false: get this wrong and
    `ls 'Projects/...'` returns an empty folder rather than an error."""
    docs = [{"_id": "projects/agora/journal/104-cycle-95.md"}]
    client = vault_tool.VaultClient()
    with patch.object(vault_tool, "_req", _fake_all_docs(docs)):
        assert client.list("Projects/Agora/Journal/") == [
            "projects/agora/journal/104-cycle-95.md"
        ]


def test_the_endkey_sentinel_is_above_every_code_point_not_just_most(env):
    """`"\\uffff"` is the idiom people reach for and it is wrong: it is the
    top of the BMP, not the top of Unicode, so any filename containing an
    emoji or any other astral character sorts *above* it and drops out of
    its own folder. The vault has no such filename today, which is exactly
    why this needs a test rather than a measurement -- the failure arrives
    the day Edvard names a note with an emoji, silently, as a folder that
    is missing one file."""
    astral = "notes/\U0001F600 idea.md"
    docs = [{"_id": "notes/plain.md"}, {"_id": astral}]
    client = vault_tool.VaultClient()
    with patch.object(vault_tool, "_req", _fake_all_docs(docs)):
        assert client.list("notes/") == sorted(["notes/plain.md", astral])


def test_an_empty_prefix_still_sweeps_the_whole_vault(env):
    """`recent()` asks for everything, and a range that excluded anything
    would silently shrink what a cycle sees changed since it last ran."""
    docs = [{"_id": "a/first.md"}, {"_id": "zzz/last.md"}]
    client = vault_tool.VaultClient()
    with patch.object(vault_tool, "_req", _fake_all_docs(docs)):
        assert client.list("") == ["a/first.md", "zzz/last.md"]


def test_read_returns_none_for_a_deleted_file(env):
    """The live case is the control -- without it this passes on a fake
    that simply serves nothing."""
    client, _ = _client_with_fake_req({
        ("GET", "notes/live.md"): (200, {"data": "still here", "children": []}),
        ("GET", "notes/gone.md"): (200, {"data": "old text", "children": [],
                                         "deleted": True}),
    })
    assert client.read("notes/live.md") == "still here"
    assert client.read("notes/gone.md") is None


def test_list_omits_deleted_files(env):
    docs = [
        {"_id": "notes/live.md"},
        {"_id": "notes/gone.md", "deleted": True},
        {"_id": "h:chunk", "data": "x"},
    ]
    client = vault_tool.VaultClient()
    with patch.object(vault_tool, "_req", _fake_all_docs(docs)):
        assert client.list("notes/") == ["notes/live.md"]


def test_append_to_a_deleted_file_refuses_instead_of_reviving_it(env):
    """Before the flag was understood, append read the tombstone's text,
    glued the new content on, and wrote the note back to life."""
    client, calls = _client_with_fake_req({
        ("GET", "notes/gone.md"): (200, {"data": "old text", "children": [],
                                         "deleted": True}),
    })
    result = client.append("notes/gone.md", "new line")

    assert result.startswith("FAILED(not found")
    assert not [c for c in calls if c[0] == "PUT"], (
        f"a refused append still wrote: {calls}"
    )


def test_recent_keeps_deleted_files_and_flags_them(env):
    """The opposite rule from list/read, on purpose: a deletion is news."""
    docs = [
        {"_id": "projects/a.md", "mtime": 3000},
        {"_id": "projects/kanban.md", "mtime": 4000, "deleted": True},
    ]
    client = vault_tool.VaultClient()
    with patch.object(vault_tool, "_req", _fake_find(docs)):
        rows, _ = client.recent(hours=6)
    assert rows == [(4000, "projects/kanban.md", True), (3000, "projects/a.md", False)]


def test_recent_asks_couch_for_the_deleted_field(env):
    """It can only report the flag if it projects it -- _find returns only
    the fields named, so dropping it from the projection would silently
    make every row read as not-deleted."""
    captured = []

    def capture(method, base, db, auth, path, body=None):
        captured.append(body)
        return 200, {"docs": []}

    client = vault_tool.VaultClient()
    with patch.object(vault_tool, "_req", capture):
        client.recent(hours=6)
    assert "deleted" in captured[0]["fields"]


def test_main_recent_marks_a_deleted_file_in_its_output(env, capsys):
    with patch.object(vault_tool, "VaultClient") as MockClient:
        MockClient.return_value.recent.return_value = (
            [(0, "projects/kanban.md", True), (0, "projects/a.md", False)], False,
        )
        vault_tool.main(["recent", "6"])
    out = capsys.readouterr().out
    assert "projects/kanban.md  [DELETED]" in out
    assert "projects/a.md" in out and "projects/a.md  [DELETED]" not in out


# ---------------------------------------------------------------------------
# Missing content chunks (2026-08-10). LiveSync stores a note as an ordered
# list of chunk documents. `assemble` substituted "" for any chunk CouchDB
# did not return, so a note with a hole came back as its surviving pieces
# concatenated -- mid-word, no marker at the seam, parses fine.
#
# It happened to Edvard's `ideas.md`: a LiveSync client re-chunked it from 1
# chunk into 184 and 6 never reached the database. `get` printed the other
# 178. 1238 characters were gone -- the `## Board` heading, its table header,
# rows #57 to #50, and the tail of the capture sentence he had typed 83
# seconds earlier -- and a cycle read the result and believed it.
#
# Silence is what makes this unsurvivable rather than merely annoying:
# `append` reads before it writes, so a truncated read written back makes
# the truncation permanent, with no record the file was ever bigger.
# ---------------------------------------------------------------------------


def test_read_raises_when_a_content_chunk_is_missing(env):
    client, _ = _client_with_fake_req({
        ("GET", "note.md"): (200, {"children": ["h:1", "h:gone", "h:2"]}),
        ("GET", "h:1"): (200, {"data": "one two "}),
        ("GET", "h:2"): (200, {"data": "|five six"}),
    })
    with pytest.raises(vault_tool.VaultIncompleteDocument) as excinfo:
        client.read("note.md")
    message = str(excinfo.value)
    assert "note.md" in message
    assert "h:gone" in message
    assert "1 of 3" in message


def test_the_spliced_text_never_escapes_as_a_value(env):
    """`"one two |five six"` is the danger stated as itself: plausible text
    that a caller would go on to edit and write back."""
    client, _ = _client_with_fake_req({
        ("GET", "note.md"): (200, {"children": ["h:1", "h:gone", "h:2"]}),
        ("GET", "h:1"): (200, {"data": "one two "}),
        ("GET", "h:2"): (200, {"data": "|five six"}),
    })
    with pytest.raises(vault_tool.VaultIncompleteDocument):
        client.read("note.md")


def test_assemble_fetches_each_chunk_once(env):
    """It used to call get_doc twice per chunk -- once for the status, once
    for the data -- so the real 184-chunk file cost 368 round trips."""
    client, calls = _client_with_fake_req({
        ("GET", "note.md"): (200, {"children": ["h:1", "h:2"]}),
        ("GET", "h:1"): (200, {"data": "a"}),
        ("GET", "h:2"): (200, {"data": "b"}),
    })
    assert client.read("note.md") == "ab"
    assert [c[1] for c in calls] == ["note.md", "h:1", "h:2"]


def test_main_get_reports_an_incomplete_document_on_stderr_and_exits_nonzero(env, capsys):
    """Exit code and stream both matter: `get x.md > file.md` must leave an
    empty file and a visible error, never a confident partial one."""
    with patch.object(vault_tool, "VaultClient") as MockClient:
        MockClient.return_value.read.side_effect = vault_tool.VaultIncompleteDocument(
            "ideas.md: 6 of 184 content chunks missing from the vault (h:vucrv1sugciv)"
        )
        code = vault_tool.main(["get", "ideas.md"])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "INCOMPLETE DOCUMENT" in captured.err
    assert "6 of 184" in captured.err
