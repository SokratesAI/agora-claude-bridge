"""Routing between Edvard's vault and Nova's own database.

The runner's copy of this client (agora_runner/vault.py) has had these
rules since 2026-08-11; this is the same rule in the bridge's copy, which
is the client every Nova cycle actually runs. Turning the switch on with
only one of the two routing is the paired-repo bug the prompt warns about
-- the runner writes a file to `nova` and the bridge reads it from
`obsidian` and reports it missing.
"""
import json
import urllib.parse

import pytest

from bridge import vault_tool


NOVA_FILE = "projects/sokrates/projects/agora/nova/resources/issues.md"
DIGEST = "projects/sokrates/projects/agora/journal-digest.md"
HIS_FILE = "projects/sokrates/projects/agora/issues.md"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("CDB_BASE", "http://couchdb.invalid:5984")
    monkeypatch.setenv("CDB_USER", "u")
    monkeypatch.setenv("CDB_PASS", "p")
    monkeypatch.setenv("CDB_DB", "obsidian")
    monkeypatch.setenv("CDB_NOVA_DB", "nova")

    def _no_network(method, base, db, auth, path, body=None, timeout=60):
        raise AssertionError(
            f"test made a real _req call ({method} {db}/{path}) -- patch "
            "vault_tool._req for this test instead"
        )
    monkeypatch.setattr(vault_tool, "_req", _no_network)


@pytest.fixture
def env_off(env, monkeypatch):
    """The switch not yet flipped -- which is how this ships."""
    monkeypatch.delenv("CDB_NOVA_DB")


def _recording_client(responses):
    """A client whose every CouchDB call is recorded as (method, db, path)."""
    client = vault_tool.VaultClient()
    calls = []

    def fake_req(method, base, db, auth, path, body=None, timeout=60):
        # _doc percent-encodes the doc id, so keys are written unquoted
        # here and the fake undoes it -- otherwise every expectation is a
        # hand-encoded string nobody can read.
        path = urllib.parse.unquote(path)
        calls.append((method, db, path))
        return responses.get((method, db, path), (404, {"error": "not_found"}))

    vault_tool._req = fake_req
    return client, calls


class TestDbFor:
    def test_nova_folder_routes_to_nova(self, env):
        client = vault_tool.VaultClient()
        assert client.db_for(NOVA_FILE) == "nova"
        assert client.db_for(
            "projects/sokrates/projects/agora/nova/journal/135-cycle-118.md") == "nova"

    def test_digest_routes_to_nova(self, env):
        assert vault_tool.VaultClient().db_for(DIGEST) == "nova"

    def test_edvards_files_stay_in_his_vault(self, env):
        client = vault_tool.VaultClient()
        for path in (HIS_FILE,
                     "projects/sokrates/projects/agora/ideas.md",
                     "projects/sokrates/projects/agora/architecture.md"):
            assert client.db_for(path) == "obsidian", path

    def test_single_file_targets_match_exactly_not_by_prefix(self, env):
        """`journal-digest.md.bak` is Edvard's, not Nova's.

        The review finding on the runner's copy (#104). A tuple tested
        wholly with startswith answers a file merely *beginning* with a
        Nova filename out of the wrong database.
        """
        client = vault_tool.VaultClient()
        for impostor in (DIGEST + ".bak", DIGEST + "x",
                         "projects/sokrates/projects/agora/nova-notes.md"):
            assert client.db_for(impostor) == "obsidian", impostor

    def test_case_is_irrelevant(self, env):
        assert vault_tool.VaultClient().db_for(NOVA_FILE.upper()) == "nova"

    def test_unset_switch_means_one_database(self, env_off):
        client = vault_tool.VaultClient()
        for path in (NOVA_FILE, DIGEST, HIS_FILE):
            assert client.db_for(path) == "obsidian", path


class TestDbsForPrefix:
    def test_prefix_inside_nova_asks_only_nova(self, env):
        client = vault_tool.VaultClient()
        assert client.dbs_for_prefix(
            "projects/sokrates/projects/agora/nova/journal/") == ["nova"]

    def test_ancestor_prefix_asks_both(self, env):
        """A whole-vault listing that asks one database loses the other."""
        client = vault_tool.VaultClient()
        for prefix in ("", "projects/", "projects/sokrates/projects/agora/"):
            assert set(client.dbs_for_prefix(prefix)) == {"obsidian", "nova"}, prefix

    def test_unrelated_prefix_asks_only_his(self, env):
        assert vault_tool.VaultClient().dbs_for_prefix("newspaper/") == ["obsidian"]


class TestChunksFollowTheirDocument:
    def test_reading_a_nova_file_fetches_its_chunks_from_nova(self, env):
        """The failure this guards is silent: a chunk looked up in the
        wrong database is indistinguishable from one never written, so an
        intact file reports itself as corrupt (or comes back empty)."""
        client, calls = _recording_client({
            ("GET", "nova", NOVA_FILE): (
                200, {"_id": NOVA_FILE, "children": ["h:aa", "h:bb"]}),
            ("POST", "nova", "_all_docs"): (200, {"rows": [
                {"key": "h:aa", "doc": {"data": "one "}},
                {"key": "h:bb", "doc": {"data": "two"}},
            ]}),
        })
        assert client.read(NOVA_FILE) == "one two"
        assert ("POST", "obsidian", "_all_docs") not in calls

    def test_writing_a_nova_file_puts_chunks_and_doc_in_nova(self, env):
        client, calls = _recording_client({
            ("GET", "nova", NOVA_FILE): (404, {}),
            ("POST", "nova", "_all_docs"): (200, {"rows": []}),
        })
        client.write(NOVA_FILE, "hello")
        assert calls, "no CouchDB calls made"
        assert all(db == "nova" for _, db, _ in calls), calls
        assert any(m == "PUT" and p.startswith("h:") for m, _, p in calls), calls

    def test_writing_his_file_never_touches_nova(self, env):
        client, calls = _recording_client({
            ("GET", "obsidian", HIS_FILE): (404, {}),
            ("POST", "obsidian", "_all_docs"): (200, {"rows": []}),
        })
        client.write(HIS_FILE, "hello")
        assert all(db == "obsidian" for _, db, _ in calls), calls

    def test_deleting_a_nova_file_deletes_it_from_nova(self, env):
        """Asserting only the database left this test passing under the
        pre-2026-08-12 hard delete, which also routed to `nova` and
        never wrote a tombstone at all -- and under a tombstone written
        to the wrong id. It has to see the write land, not just see it
        aimed at the right database."""
        client, calls = _recording_client({
            ("GET", "nova", NOVA_FILE): (200, {"_id": NOVA_FILE, "_rev": "1-x"}),
            ("PUT", "nova", NOVA_FILE): (200, {}),
        })
        assert client.delete(NOVA_FILE) == "deleted"
        assert all(db == "nova" for _, db, _ in calls), calls
        assert ("PUT", "nova", NOVA_FILE) in [(m, db, p) for m, db, p in calls], calls


class TestListingAcrossBothDatabases:
    # Built the same way file_docs builds it, then unquoted to match the
    # fake -- hand-encoding this string is how the test drifts from the code.
    SWEEP = "_all_docs?" + urllib.parse.unquote(urllib.parse.urlencode({
        "startkey": json.dumps(""), "endkey": json.dumps(vault_tool._ID_MAX)}))

    def _both(self, nova_rows, obsidian_rows, nova_status=200, obs_status=200):
        return {
            ("GET", "obsidian", self.SWEEP): (obs_status, {"rows": obsidian_rows}),
            ("GET", "nova", self.SWEEP): (nova_status, {"rows": nova_rows}),
            ("POST", "obsidian", "_all_docs?include_docs=true"): (200, {"rows": [
                {"id": HIS_FILE, "doc": {"_id": HIS_FILE, "data": "his"}}]}),
            ("POST", "nova", "_all_docs?include_docs=true"): (200, {"rows": [
                {"id": NOVA_FILE, "doc": {"_id": NOVA_FILE, "data": "mine"}}]}),
        }

    def test_whole_vault_listing_includes_both(self, env):
        client, _ = _recording_client(
            self._both([{"id": NOVA_FILE}], [{"id": HIS_FILE}]))
        assert set(client.file_docs("")) == {NOVA_FILE, HIS_FILE}

    def test_a_returned_doc_remembers_which_database_it_came_from(self, env):
        """Not where db_for predicts it should be. The two agree in steady
        state and disagree during a migration, which is exactly when the
        chunk lookup matters (runner review finding, #104)."""
        client, _ = _recording_client(
            self._both([{"id": NOVA_FILE}], [{"id": HIS_FILE}]))
        docs = client.file_docs("")
        assert docs[NOVA_FILE][vault_tool._SRC_DB_KEY] == "nova"
        assert docs[HIS_FILE][vault_tool._SRC_DB_KEY] == "obsidian"

    def test_assemble_prefers_the_source_database_over_the_predicted_one(self, env):
        """A Nova file still sitting in Edvard's database mid-migration.

        db_for says "nova"; it was really read from "obsidian", and that
        is where its chunks are. Deriving from the path here is how an
        intact file reports itself corrupt.
        """
        client, calls = _recording_client({
            ("POST", "obsidian", "_all_docs"): (200, {"rows": [
                {"key": "h:aa", "doc": {"data": "still here"}}]}),
        })
        doc = {"_id": NOVA_FILE, "children": ["h:aa"],
               vault_tool._SRC_DB_KEY: "obsidian"}
        assert client.assemble(doc, NOVA_FILE) == "still here"
        assert ("POST", "nova", "_all_docs") not in calls

    def test_one_database_failing_is_reported_not_swallowed(self, env, capsys):
        """With one database a failure gave an obviously-empty listing.
        With two, one failing hands back a partial answer that looks
        completely healthy -- so it has to name the database."""
        client, _ = _recording_client(
            self._both([{"id": NOVA_FILE}], [{"id": HIS_FILE}], nova_status=500))
        docs = client.file_docs("")
        assert set(docs) == {HIS_FILE}
        err = capsys.readouterr().err
        assert "nova" in err and "500" in err

    def test_recent_asks_both_databases(self, env):
        client, calls = _recording_client({
            ("POST", "obsidian", "_find"): (200, {"docs": [
                {"_id": HIS_FILE, "mtime": 1000}]}),
            ("POST", "nova", "_find"): (200, {"docs": [
                {"_id": NOVA_FILE, "mtime": 2000}]}),
        })
        rows, truncated = client.recent(24)
        assert [p for _, p, _ in rows] == [NOVA_FILE, HIS_FILE]
        assert not truncated
        assert ("POST", "nova", "_find") in calls

    def test_recent_truncation_in_either_database_is_reported(self, env):
        """`limit` is applied per query, so one database truncating makes
        the merged list an arbitrary subset -- the exact thing the flag
        exists to prevent a cycle from believing."""
        client, _ = _recording_client({
            ("POST", "obsidian", "_find"): (200, {"docs": []}),
            ("POST", "nova", "_find"): (200, {"docs": [
                {"_id": NOVA_FILE, "mtime": 1}] * 5}),
        })
        _, truncated = client.recent(24, limit=5)
        assert truncated

    def test_recent_failure_on_one_database_is_fatal(self, env):
        client, _ = _recording_client({
            ("POST", "obsidian", "_find"): (200, {"docs": []}),
            ("POST", "nova", "_find"): (500, {"error": "boom"}),
        })
        with pytest.raises(SystemExit, match="nova"):
            client.recent(24)


def test_switch_off_behaves_exactly_as_before(env_off):
    """The state this actually ships in. Every call goes to `obsidian`."""
    client, calls = _recording_client({
        ("GET", "obsidian", NOVA_FILE): (200, {"_id": NOVA_FILE, "data": "x"}),
    })
    assert client.read(NOVA_FILE) == "x"
    assert [db for _, db, _ in calls] == ["obsidian"]


class TestDatabaseHealth:
    """`database_health()` -- what this process resolved, and what it can
    actually reach.

    The bridge is the client every Nova cycle reads the vault through, and
    until this existed the only way to ask it which database it had picked
    was to import VaultClient and call `db_for` by hand. That mattered on
    2026-08-12, with an irreversible delete of 165 files from `obsidian`
    waiting on the answer.
    """

    def _health_client(self, info_by_db):
        client = vault_tool.VaultClient()
        seen = []

        def fake_req(method, base, db, auth, path, body=None, timeout=60):
            seen.append({"method": method, "db": db, "path": path, "timeout": timeout})
            return info_by_db.get(urllib.parse.unquote(db), (404, {"error": "not_found"}))

        vault_tool._req = fake_req
        return client, seen

    def test_reports_the_database_each_probe_path_resolves_to(self, env):
        client, _ = self._health_client({
            "obsidian": (200, {"doc_count": 13196}),
            "nova": (200, {"doc_count": 713}),
        })
        routes = {r["path"]: r["database"] for r in client.database_health()["routes"]}
        assert routes == {
            "projects/sokrates/projects/agora/nova/journal/138-cycle-121.md": "nova",
            "projects/sokrates/projects/agora/journal-digest.md": "nova",
            # The two regressions rather than examples. A `.bak` beside the
            # digest is Edvard's file and must not follow it, and the Nova
            # folder he asked to keep is in *his* vault -- everything under
            # `agora/nova/` routes away, and this one does not live there.
            "projects/sokrates/projects/agora/journal-digest.md.bak": "obsidian",
            "projects/sokrates/projects/nova/nova.md": "obsidian",
            "projects/sokrates/projects/agora/issues.md": "obsidian",
        }

    def test_probes_cover_both_databases(self, env):
        """A probe list that only ever names one database proves nothing --
        it would look identical to routing being switched off."""
        client, _ = self._health_client({"obsidian": (200, {}), "nova": (200, {})})
        resolved = {r["database"] for r in client.database_health()["routes"]}
        assert resolved == {"obsidian", "nova"}

    def test_reports_names_and_doc_counts_when_reachable(self, env):
        client, _ = self._health_client({
            "obsidian": (200, {"doc_count": 13196}),
            "nova": (200, {"doc_count": 713}),
        })
        health = client.database_health()
        assert health["routing_enabled"] is True
        assert health["databases"] == {
            "main": {"name": "obsidian", "reachable": True,
                     "doc_count": 13196, "error": None},
            "nova": {"name": "nova", "reachable": True,
                     "doc_count": 713, "error": None},
        }

    def test_an_unreachable_database_is_reported_not_raised(self, env):
        """Reachability is the answer, so failing to reach one cannot be an
        exception -- that would lose the half of the report that says which
        database is fine."""
        client, _ = self._health_client({
            "obsidian": (200, {"doc_count": 13196}),
            "nova": (500, {"error": "boom"}),
        })
        dbs = client.database_health()["databases"]
        assert dbs["main"]["reachable"] is True
        assert dbs["nova"]["reachable"] is False
        assert dbs["nova"]["error"] == "HTTP 500"

    def test_a_raising_transport_is_reported_not_raised(self, env):
        client = vault_tool.VaultClient()

        def boom(*a, **kw):
            raise OSError("connection refused")

        vault_tool._req = boom
        dbs = client.database_health()["databases"]
        assert dbs["main"]["reachable"] is False
        assert "connection refused" in dbs["main"]["error"]

    def test_probes_use_the_short_timeout_not_the_default(self, env):
        """Two unreachable databases at the 60s default is a two-minute
        wait from the one instrument built to replace a slow uncertain
        wait."""
        client, seen = self._health_client({"obsidian": (200, {}), "nova": (200, {})})
        client.database_health()
        assert [c["timeout"] for c in seen] == [vault_tool.HEALTH_TIMEOUT_SECONDS] * 2
        assert vault_tool.HEALTH_TIMEOUT_SECONDS < 60

    def test_switch_off_reports_one_database_and_no_routing(self, env_off):
        client, _ = self._health_client({"obsidian": (200, {"doc_count": 13196})})
        health = client.database_health()
        assert health["routing_enabled"] is False
        assert list(health["databases"]) == ["main"]
        assert {r["database"] for r in health["routes"]} == {"obsidian"}
