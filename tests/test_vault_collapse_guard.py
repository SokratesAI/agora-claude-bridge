"""A blind read must not read as a short file, and a short write must not
silently replace a long document.

On 2026-08-15 a cycle read Edvard's 123,586-byte `issues.md` through this
client, got an empty body and exit 0, and wrote the empty result back over
the document -- which was still fully intact underneath. It was recovered
from a local copy within the minute, but nothing in the client objected at
any point. The revision guard did not fire and could not: the write
carried the correct `_rev` and was, to CouchDB, an ordinary edit by the
only writer in the room.

Two independent guards, because they fail in opposite directions and
either one alone leaves the door open.

`_size_checked` is the read half. A LiveSync file doc records `size`, the
byte length of the text it stands for, and every writer sets it -- this
client at `_put_raw` and Obsidian itself. Measured 2026-08-15 across 37
documents (Edvard's phone-written captures, this loop's journal entries,
the JSON ledgers, the 291KB frozen archive): `size` equalled
`len(content.encode())` exactly 37 times out of 37, and no document
lacked the field. It is a length checksum the vault has always carried
and nothing has ever read. `VaultIncompleteDocument` already catches a
chunk that is *absent*; this catches a document that assembles to the
wrong length for any other reason, and the shortest such case -- no
children, no data -- is exactly the one that took the boards out.

`_collapse_refusal` is the write half, and it is the one that would have
stopped the actual loss. It does not care why the body is short.

The two clients are hand-synced with nothing detecting drift, so this
file is deliberately the same shape as the runner's copy -- a diff
between them should read as "class method vs module function" and
nothing else.
"""
import pytest

from bridge import vault_tool

PATH = "projects/sokrates/projects/nova/issues.md"
#: The real file, at the size it actually was when it was lost.
REAL_SIZE = 123586


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("CDB_BASE", "http://couchdb.obsidian.svc.cluster.local:5984")
    monkeypatch.setenv("CDB_USER", "u")
    monkeypatch.setenv("CDB_PASS", "p")
    monkeypatch.setenv("CDB_DB", "obsidian")


@pytest.fixture
def client(env):
    return vault_tool.VaultClient()


def doc(**kw):
    base = {"_id": PATH, "path": PATH, "_rev": "7-abc", "children": [], "data": ""}
    base.update(kw)
    return base


# --------------------------------------------------------------------- read


def test_a_document_matching_its_own_size_still_reads(client):
    """The control. Without it every assertion below would pass against a
    client that had simply started raising on everything."""
    assert client.assemble(doc(data="hello", size=5), path=PATH) == "hello"


def test_the_blind_read_that_lost_the_boards_now_raises(client):
    """The exact shape: no children, no data, and a `size` saying 123,586
    bytes should be there. This returned `""` and exit 0."""
    with pytest.raises(vault_tool.VaultIncompleteDocument) as excinfo:
        client.assemble(doc(size=REAL_SIZE), path=PATH)
    message = str(excinfo.value)
    assert PATH in message
    # Both numbers, because the reader's next question is how much is gone.
    assert str(REAL_SIZE) in message
    assert "0 bytes" in message


def test_a_genuinely_empty_document_is_not_an_error(client):
    """The boundary the fix above must not break. An empty file records
    `size: 0`, agrees with itself, and is a legitimate read."""
    assert client.assemble(doc(size=0), path=PATH) == ""


def test_a_document_with_no_size_field_still_reads(client):
    """`size` is present on all 37 documents measured, but a client that
    refused to read anything without it would turn a field this code has
    never depended on into a hard requirement, on a vault that three
    separate writers write to."""
    assert client.assemble(doc(data="hello"), path=PATH) == "hello"


def test_size_is_bytes_not_characters(client):
    """The measurement said bytes. A non-ASCII document is where those two
    diverge, and getting it backwards would raise on every file Edvard
    writes an em-dash into -- which is most of them."""
    text = "Skøyen — Oslo"
    assert len(text.encode("utf-8")) != len(text)
    assert client.assemble(doc(data=text, size=len(text.encode("utf-8"))),
                           path=PATH) == text


def test_a_chunked_document_that_assembles_short_raises(client, monkeypatch):
    """The same guard on the path that actually serves large files. No
    chunk is missing here, so `VaultIncompleteDocument`'s existing check
    passes and only the length disagrees."""
    monkeypatch.setattr(client, "_fetch_chunks",
                        lambda ids, db: {"c1": "ab", "c2": "cd"})
    assert client.assemble(doc(children=["c1", "c2"], size=4), path=PATH) == "abcd"
    with pytest.raises(vault_tool.VaultIncompleteDocument):
        client.assemble(doc(children=["c1", "c2"], size=4000), path=PATH)


def test_the_cli_get_exits_non_zero_and_writes_nothing_to_stdout(
        client, monkeypatch, capsys):
    """What a cycle actually sees, and the half that matters: a caller
    redirecting `get` to a file must end up with an empty file and a
    visible error, never a short body it will go on to write back."""
    monkeypatch.setattr(client, "get_doc",
                        lambda doc_id, db=None: (200, doc(size=REAL_SIZE)))
    monkeypatch.setattr(vault_tool, "VaultClient", lambda *a, **k: client)
    code = vault_tool.main(["get", PATH])
    captured = capsys.readouterr()
    assert code == 1
    assert "[INCOMPLETE DOCUMENT]" in captured.err
    assert captured.out == ""


# -------------------------------------------------------------------- write


def test_an_ordinary_edit_is_allowed():
    """The control for the write half. Rolling one digest line, appending a
    capture, striking a bullet -- none of these may need a flag, or the
    flag becomes the habit and the guard stops meaning anything."""
    assert _refusal(REAL_SIZE, REAL_SIZE - 200) is None
    assert _refusal(35859, 35400) is None


def test_the_write_that_lost_the_boards_is_refused():
    refusal = _refusal(REAL_SIZE, 0)
    assert refusal is not None
    assert refusal.startswith("FAILED(collapse:")
    assert str(REAL_SIZE) in refusal
    # The message has to say what to do. A cycle reading it is mid-shell
    # sequence and its first instinct will be to retry the write.
    assert "--allow-shrink" in refusal
    assert "re-reading" in refusal


def test_a_short_but_non_empty_write_is_also_refused():
    """The version nobody would notice. A partially-assembled read gives a
    plausible small body, not an empty one, so a guard that only checked
    for emptiness would pass the quieter half of this failure."""
    assert _refusal(REAL_SIZE, 10_000) is not None


def test_allow_shrink_lets_a_deliberate_truncation_through():
    assert _refusal(REAL_SIZE, 0, allow_shrink=True) is None


def test_a_small_document_may_still_be_rewritten_whole():
    """Below the floor the small JSON ledgers live, where replacing most of
    the file is the normal operation and the blast radius is a few rows."""
    assert _refusal(vault_tool.COLLAPSE_FLOOR - 1, 0) is None


def test_creating_a_file_that_does_not_exist_is_not_a_collapse():
    """Every journal entry is this write. If it needed a flag the guard
    would be in the way of the loop's most common write."""
    assert vault_tool._collapse_refusal(PATH, None, 0, False) is None


def test_a_document_with_no_size_field_is_not_guessed_at():
    assert vault_tool._collapse_refusal(PATH, {"_rev": "1-a"}, 0, False) is None
    assert vault_tool._collapse_refusal(
        PATH, {"size": "123586"}, 0, False) is None


def test_the_ratio_boundary_is_inclusive_on_the_permitted_side():
    """Exactly at the ratio is allowed, just under it is not -- stated
    because 'a quarter' is ambiguous and the next reader will assume the
    other one."""
    old = 40_000
    exact = int(old * vault_tool.COLLAPSE_RATIO)
    assert _refusal(old, exact) is None
    assert _refusal(old, exact - 1) is not None


def _refusal(old, new, allow_shrink=False):
    return vault_tool._collapse_refusal(PATH, {"size": old}, new, allow_shrink)


# ------------------------------------------------------------- CLI plumbing


def test_allow_shrink_flag_does_not_swallow_the_path(monkeypatch, capsys, tmp_path):
    """`_take_flag` exists because `_take_option` consumes the next
    argument as a value. A flag that ate the local filename would be a
    worse bug than the one being guarded, and it would look like a
    successful write of the wrong file."""
    seen = {}

    class FakeClient:
        def write(self, path, content, if_rev=None, allow_shrink=False):
            seen.update(path=path, content=content, allow_shrink=allow_shrink)
            return "written"

    monkeypatch.setattr(vault_tool, "VaultClient", lambda *a, **k: FakeClient())
    body = "the whole file\n"
    local = _tmpfile(tmp_path, body)
    assert vault_tool.main(["put", PATH, local, "--allow-shrink"]) == 0
    assert seen == {"path": PATH, "content": body, "allow_shrink": True}
    capsys.readouterr()


def test_put_without_the_flag_still_guards(monkeypatch, capsys, tmp_path):
    seen = {}

    class FakeClient:
        def write(self, path, content, if_rev=None, allow_shrink=False):
            seen.update(allow_shrink=allow_shrink)
            return "written"

    monkeypatch.setattr(vault_tool, "VaultClient", lambda *a, **k: FakeClient())
    local = _tmpfile(tmp_path, "x\n")
    assert vault_tool.main(["put", PATH, local]) == 0
    assert seen == {"allow_shrink": False}
    capsys.readouterr()


def test_a_refused_write_exits_non_zero(monkeypatch, capsys, tmp_path):
    """The whole point of the guard from a shell's side. `prompt.md` chains
    these writes, so a refusal that exited 0 would stop nothing."""
    class FakeClient:
        def write(self, path, content, if_rev=None, allow_shrink=False):
            return "FAILED(collapse: ...)"

    monkeypatch.setattr(vault_tool, "VaultClient", lambda *a, **k: FakeClient())
    local = _tmpfile(tmp_path, "")
    assert vault_tool.main(["put", PATH, local]) == 1
    assert "collapse" in capsys.readouterr().out


def _tmpfile(tmp_path, body):
    local = tmp_path / "body.md"
    local.write_text(body, encoding="utf-8")
    return str(local)
