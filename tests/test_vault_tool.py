from unittest.mock import patch

import pytest

from bridge import vault_tool


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("CDB_BASE", "http://couchdb.obsidian.svc.cluster.local:5984")
    monkeypatch.setenv("CDB_USER", "u")
    monkeypatch.setenv("CDB_PASS", "p")
    monkeypatch.setenv("CDB_DB", "obsidian")


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
        ("DELETE", "doomed.md"): (200, {}),
    })
    client.delete("doomed.md")
    assert [c for c in calls if c[0] == "PUT"] == []


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
    assert rows == [(3000, "projects/b.md"), (1000, "projects/a.md")]
    assert truncated is False


def test_recent_includes_backups_only_when_prefix_asks_for_them(env):
    docs = [
        {"_id": "agora/backups/20260803-114439 journal.md", "mtime": 4000},
        {"_id": "projects/a.md", "mtime": 1000},
    ]
    client = vault_tool.VaultClient()
    with patch.object(vault_tool, "_req", _fake_find(docs)):
        rows, _ = client.recent(hours=6, prefix="agora/backups/")
    assert [p for _, p in rows] == ["agora/backups/20260803-114439 journal.md"]


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
        MockClient.return_value.recent.return_value = ([(0, "projects/a.md")], True)
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
