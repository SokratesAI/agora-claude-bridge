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


def test_write_backs_up_previous_content_first(env):
    client, calls = _client_with_fake_req({
        ("GET", "note.md"): (200, {"data": "old content", "children": [], "_rev": "1-abc"}),
    })
    client.write("note.md", "new content")

    put_calls = [c for c in calls if c[0] == "PUT"]
    backup_calls = [c for c in put_calls if c[1].startswith("agora/backups/")]
    assert len(backup_calls) == 1
    backup_chunk = next(c for c in put_calls if c[2].get("type") == "leaf" and c[2].get("data") == "old content")
    assert backup_chunk is not None


def test_write_skips_backup_when_file_is_new(env):
    client, calls = _client_with_fake_req({})
    client.write("brand-new.md", "content")
    backup_calls = [c for c in calls if c[1].startswith("agora/backups/")]
    assert backup_calls == []


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
    # The LAST leaf PUT is the real write -- append's own backup-before-write
    # (unchanged content, no marker match needed) writes an earlier leaf chunk first.
    chunk_put = [c for c in calls if c[0] == "PUT" and c[2].get("type") == "leaf"][-1]
    assert "## [new] Cycle 2" in chunk_put[2]["data"]
    assert "## [old] Cycle 1" in chunk_put[2]["data"]
    assert chunk_put[2]["data"].index("## [new] Cycle 2") < chunk_put[2]["data"].index("## [old] Cycle 1")


def test_append_falls_back_to_end_when_marker_not_found(env):
    existing = "line one\nline two\n"
    client, calls = _client_with_fake_req({
        ("GET", "note.md"): (200, {"data": existing, "children": [], "_rev": "1-abc"}),
    })
    client.append("note.md", "line three", after_marker="## Nonexistent")
    chunk_put = [c for c in calls if c[0] == "PUT" and c[2].get("type") == "leaf"][-1]
    assert chunk_put[2]["data"].strip().endswith("line three")


def test_delete_returns_absent_when_missing(env):
    client, _ = _client_with_fake_req({})
    assert client.delete("gone.md") == "absent"


def test_main_get_prints_not_found_marker(env, capsys):
    with patch.object(vault_tool, "VaultClient") as MockClient:
        MockClient.return_value.read.return_value = None
        vault_tool.main(["get", "missing.md"])
    assert "[not found: missing.md]" in capsys.readouterr().out
