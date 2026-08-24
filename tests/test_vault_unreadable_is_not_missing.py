"""A database that will not answer must not read as a file that is not there.

`read_rev` collapsed every non-200 from CouchDB into `(None, None)`. 404
means the document does not exist; 500, 503 and 401 mean the database did
not answer. One return value for both, and `get` printed
`[not found: <path>]` for either.

The reader that makes this expensive is Nova itself. This CLI is how a
cycle reads its own instructions, its journal and the owner's capture files
at the start of every run, and `[not found]` is a fact a cycle acts on --
it has written a missing file into the permanent record before. The
listing half of this was fixed in runner#117, and two comments in
`vault_tool.py` say in those words that "that folder is empty" and "that
file does not exist" are things a cycle writes into the journal as fact.
The single-document read was the copy left behind.

`read_rev`'s own docstring described this bug accurately and deferred the
fix "because the fix belongs in both clients at once". Both are changed
now; the runner's half is runner#148 with the matching test file there.
The two clients are hand-synced with nothing detecting drift, so this
file is deliberately the same shape as that one -- a diff between them
should read as "class method vs module function" and nothing else.

404 deliberately still returns `(None, None)`: `append` distinguishes it
from a tombstone by the `rev` beside it, and `put` needs it to create a
file that does not exist yet.
"""
import pytest

from bridge import vault_tool

PATH = "projects/sokrates/projects/agora/nova/resources/prompt.md"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("CDB_BASE", "http://couchdb.obsidian.svc.cluster.local:5984")
    monkeypatch.setenv("CDB_USER", "u")
    monkeypatch.setenv("CDB_PASS", "p")
    monkeypatch.setenv("CDB_DB", "obsidian")


@pytest.fixture
def client(env, monkeypatch):
    """A client whose GET answers with whatever status the test asks for.

    Faked at `get_doc` rather than at the socket because the question here
    is only what `read_rev` does with a status it is handed. The routing
    and chunk-assembly seams below it have their own files.
    """
    def install(status, doc=None):
        c = vault_tool.VaultClient()
        monkeypatch.setattr(c, "get_doc", lambda doc_id, db=None: (status, doc or {}))
        return c

    return install


def test_a_readable_document_still_reads(client):
    """The control. Without it every assertion below would pass against a
    client that failed on everything."""
    c = client(200, {"_id": PATH, "path": PATH, "data": "hello", "_rev": "3-abc"})
    assert c.read_rev(PATH) == ("hello", "3-abc")
    assert c.read(PATH) == "hello"


def test_a_missing_document_is_still_absent_not_an_error(client):
    """404 must keep working: `put` creates a file that does not exist yet
    and `append` reports it as `FAILED(not found:...)` on purpose."""
    assert client(404).read_rev(PATH) == (None, None)


def test_a_tombstone_is_still_absent_with_its_revision(client):
    """The boundary a careless fix breaks. A deleted note is a 200 with a
    flag, not a 404, so it must not start raising -- and it has to keep
    handing back the revision, because overwriting a tombstone carries it.
    """
    c = client(200, {"_id": PATH, "path": PATH, "deleted": True, "_rev": "9-z"})
    assert c.read_rev(PATH) == (None, "9-z")


@pytest.mark.parametrize("status", [500, 502, 503, 401, 403])
def test_an_unreadable_document_raises_instead_of_reading_as_missing(client, status):
    """The four that actually happen: an overloaded or compacting CouchDB
    (500/503), a proxy in the way (502), and credentials that rotated
    without this process noticing (401/403)."""
    with pytest.raises(vault_tool.VaultUnreadableDocument) as excinfo:
        client(status).read(PATH)
    message = str(excinfo.value)
    # Both, because the next thing a human does is decide whether the file
    # or the database is the problem.
    assert PATH in message
    assert str(status) in message


def test_append_no_longer_calls_a_live_file_not_found(client):
    """`append` reads first and returns `FAILED(not found: ...)` when the
    read comes back empty. On an unreadable database that sentence was
    false and told a cycle to create a file that already existed."""
    with pytest.raises(vault_tool.VaultUnreadableDocument):
        client(503).append(PATH, "- a capture\n", "## Entries")


def test_the_cli_exits_non_zero_and_says_why(client, monkeypatch, capsys):
    """What a cycle actually sees. `get` printed `[not found: <path>]` and
    exited 0, which is a confident wrong answer about the vault; it now
    prints the reason to stderr and exits 1, so `&&` stops."""
    # Built before the name is patched, or the fixture's own
    # `VaultClient()` call resolves to the patch and recurses.
    c = client(500)
    monkeypatch.setattr(vault_tool, "VaultClient", lambda *a, **k: c)
    code = vault_tool.main(["get", PATH])
    captured = capsys.readouterr()
    assert code == 1
    assert "[UNREADABLE DOCUMENT]" in captured.err
    assert "500" in captured.err
    # The old wrong answer must not also be on stdout -- a caller
    # redirecting stdout to a file would otherwise still capture it.
    assert "[not found" not in captured.out


def test_the_error_is_a_runtimeerror(client):
    """Matches `VaultIncompleteDocument` beside it, so a caller that
    already handles one failed-read class handles this one."""
    assert issubclass(vault_tool.VaultUnreadableDocument, RuntimeError)
