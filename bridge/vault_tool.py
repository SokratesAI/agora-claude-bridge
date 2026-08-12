#!/usr/bin/env python3
"""vault_tool -- CouchDB (Obsidian LiveSync) read/write, for use from
inside a claude-cli session's own Bash tool. Same wire format as
agora_runner/vault.py (the tool belt agora-persona-runner's
gemini/anthropic personas already use) and ~/vault-tools/vault_tool.py
(the interactive-session equivalent) -- kept as a third, independent
copy rather than an import because this image has no dependency on
either of those repos and shouldn't grow one just for this.

Usage (from Bash inside the bridge pod):
  python3 -m bridge.vault_tool get    <path>
  python3 -m bridge.vault_tool put    <path> <local_file>
  python3 -m bridge.vault_tool puts   <path> -              # content from stdin
  python3 -m bridge.vault_tool append <path> <local_file> [after_marker]
  python3 -m bridge.vault_tool appends <path> [after_marker]    # content from stdin
  python3 -m bridge.vault_tool delete <path>
  python3 -m bridge.vault_tool ls     [prefix]
  python3 -m bridge.vault_tool recent [hours] [prefix]   # what changed lately

Env: CDB_BASE, CDB_USER, CDB_PASS, CDB_DB (default "obsidian"),
CDB_NOVA_DB (unset = one database, exactly as before routing existed;
set = Nova's own files are read and written there instead -- see
NOVA_DB_TARGETS below).

All paths are lowercased before use, always (standing vault-wide
convention -- see CLAUDE.md/memory `feedback-always-use-lowercase-vault-paths`).

Overwrites and deletes are NOT backed up into the vault. They used to
be, under `agora/backups/<timestamp> <basename>`, which cost one extra
document per write and grew to 272 of them before Edvard asked for it
to stop (2026-08-05): "since the switch to Nova, this is just noise."
The real safety net was never this copy -- it is the daily snapshot of
the whole vault into the `SokratesAI/vault` GitHub repo, which keeps
every version in git history rather than beside the original. Use
`vault_git_revision_history` to recover a previous version.
"""
import base64
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


# Obsidian LiveSync's own bookkeeping docs -- chunks, file/index/version
# entries. Never files a human wrote.
INTERNAL_PREFIXES = ("_", "h:", "f:", "i:", "v:")
# The largest code point there is, so `prefix + _ID_MAX` sorts above every
# id that starts with `prefix` and below the next one that does not.
_ID_MAX = "\U0010FFFF"
# Historical: this tool used to write one of these per overwrite, and the
# whole `agora/` folder was deleted on 2026-08-06 once it stopped. The
# filter stays because deleting a folder from CouchDB does not stop an
# Obsidian client holding stale state from pushing it back -- exactly how
# `evolve/` reappeared with twelve files a day after being renamed
# (2026-08-05). If that happens here, `recent` should stay readable.
BACKUP_PREFIX = "agora/backups/"
DEFAULT_RECENT_LIMIT = 2000
# Times shown to a human are Oslo time, not UTC -- Edvard lives there and
# asked for it directly (evolve/identity.md rule 7).
LOCAL_TZ = "Europe/Oslo"

# Nova's files live in their own CouchDB database rather than in Edvard's
# vault (his ask, 2026-08-11: "You have outgrown a poc project that is
# allowed to use my Vault as a database. Move out and get your own space").
# The document id in a LiveSync vault IS the lowercased file path, so which
# database holds a document is a pure function of its path and needs no
# lookup -- which is what makes one rule in one place possible at all.
#
# This is the second copy of a rule agora_runner/vault.py also holds, for
# the same reason the whole client is duplicated (see the module docstring):
# this image depends on neither repo. The duplication is the known cost --
# nothing detects drift between the two, and a routing rule that disagrees
# between the writer and the reader serves a file out of the wrong store.
# Keep them identical, or fix them both in the same cycle.
#
# `issues.md` and `ideas.md` deliberately stay in `obsidian`. Edvard offered
# them ("Take all of 'my' files aswell with you if you want"), but they are
# the two files Obsidian LiveSync may still write, and a second writer that
# cannot see this rule would silently re-create them in the vault Nova had
# stopped reading.
#
# Folders match by prefix; single files must match EXACTLY. Testing
# everything with startswith routed `journal-digest.md.bak` -- and any other
# file merely *beginning* with that name -- into Nova's database, which is a
# file Edvard owns being answered by the wrong store.
NOVA_DB_FOLDERS = (
    "projects/sokrates/projects/agora/nova/",
)
NOVA_DB_FILES = (
    "projects/sokrates/projects/agora/journal-digest.md",
)
NOVA_DB_TARGETS = NOVA_DB_FOLDERS + NOVA_DB_FILES

# Paths whose routing this process reports on demand. Five distinct
# behaviours of `db_for`, two of which are regressions rather than
# examples: a `.bak` beside the digest must NOT follow it into Nova's
# database, and the Nova folder Edvard asked to keep in his own vault must
# stay there. The other three are the folder rule, the exact-file rule and
# a file of his that must never move.
#
# **This tuple is a deliberate copy of agora-persona-runner's
# HEALTH_PROBE_PATHS, and copying it is the point.** The routing rule
# itself already exists twice, in two repos, with nothing detecting drift
# -- that is the risk this endpoint exists to measure, not one it
# introduces. Probing the *same* five paths from both processes is what
# makes drift a diff of two curls instead of an argument; a shorter or
# cleverer list here would only make the two answers incomparable, which
# is the failure it is built to catch.
#
# **These are real paths and must stay real.** Journal filenames are
# `<sequence>-cycle-<n>.md` where the two numbers diverge, so a plausible
# guess like `121-cycle-121.md` has never existed -- one was in the
# runner's tuple until a reviewer listed the folder. A probe pointing at a
# document nobody can open turns the one endpoint built to remove
# ambiguity into a second thing to disambiguate.
HEALTH_PROBE_PATHS = (
    "projects/sokrates/projects/agora/nova/journal/138-cycle-121.md",
    "projects/sokrates/projects/agora/journal-digest.md",
    "projects/sokrates/projects/agora/journal-digest.md.bak",
    "projects/sokrates/projects/nova/nova.md",
    # Was `agora/issues.md` until 2026-08-12, when the three files Edvard
    # writes by hand moved into the Nova folder in his own vault at his
    # ask -- *"they can be moved into the Nova folder in my Vault and not
    # be underneath the agora project folder"*. The rule it probes is
    # unchanged; the path had to move with the file, or this tuple points
    # at a document nobody can open. The runner's copy moved in the same
    # cycle: these two tuples must stay identical or the drift check above
    # is comparing two different questions.
    "projects/sokrates/projects/nova/issues.md",
)

# Stamped onto a fetched doc by file_docs() so a later chunk lookup uses the
# database the doc was really read from. Private; never written back --
# _put_raw builds its document from scratch.
_SRC_DB_KEY = "_nova_src_db"


def _local_stamp(mtime_ms):
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(LOCAL_TZ)
    except Exception:
        tz = timezone.utc
    return datetime.fromtimestamp(mtime_ms / 1000, tz).strftime("%Y-%m-%d %H:%M")


def _env(name, default=None):
    import os
    value = os.environ.get(name, default)
    if value is None:
        raise SystemExit(f"vault_tool: required env var {name} is not set")
    return value


def _req(method, base, db, auth, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        f"{base}/{db}/{path}", data=data,
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


# `database_health` only, and deliberately far below the 60s every other
# call gets. This endpoint probes each database in turn, so two unreachable
# ones would cost 2 x timeout before reporting anything -- turning the one
# instrument built to remove a slow uncertain wait into a slow uncertain
# wait. "Can I reach this database" is also the one question where a slow
# answer and no answer mean the same thing operationally, so failing fast
# loses nothing.
HEALTH_TIMEOUT_SECONDS = 5


# Content-defined chunking, in bytes. LiveSync -- the client that wrote
# every file in this vault Nova didn't -- averages ~4KB a chunk, and
# these are picked to land there.
#
# Why content-defined and not a fixed stride: `append` inserts under a
# heading near the TOP of the file, so a fixed stride would shift every
# boundary after the insertion and rewrite the whole file anyway. A
# boundary chosen by the content of the line it follows re-syncs within
# a chunk or two of the edit, so an append rewrites the tail and nothing
# else. Measured 2026-08-11 (Cycle 116, research/vault-storage-format.md):
# one-blob writes left 38.8MB of dead copies in Edvard's database against
# 1.4MB of live content -- 27.6x -- because every write stored the whole
# file again under a new content hash and deleted nothing.
CHUNK_MIN_BYTES = 2048
CHUNK_MAX_BYTES = 16384
# 1 line in 32 is a boundary candidate once past CHUNK_MIN_BYTES.
CHUNK_BOUNDARY_MASK = 0x1F


def _is_chunk_boundary(line):
    # zlib.crc32, not the builtin hash(): str hashing is salted per
    # process, so the same file would chunk differently on every run and
    # reuse nothing.
    import zlib
    return (zlib.crc32(line.encode("utf-8")) & CHUNK_BOUNDARY_MASK) == 0


def _bytes_prefix(text, limit):
    """How many characters of `text` fit in `limit` UTF-8 bytes.

    Slicing a long line by character count is wrong: CHUNK_MAX_BYTES is a
    byte budget, and 20,000 emoji measured 65,536 bytes in a single
    "chunk" -- four times the cap the chunker claims to enforce. Cutting
    on a code-point boundary is still required, so this finds the
    boundary rather than assuming one character is one byte."""
    lo, hi = 0, min(len(text), limit)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(text[:mid].encode("utf-8")) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return max(lo, 1)


def _split_chunks(content):
    """Split `content` into content-defined pieces.

    Concatenating the result reproduces `content` byte for byte --
    `assemble()` does exactly that, so this is the whole contract."""
    if not content:
        return [""]
    units = []
    for line in content.splitlines(keepends=True):
        # A single line can be longer than a chunk (a one-line JSON
        # ledger is the real case). Cut on a code-point boundary, but
        # count bytes while doing it -- see _bytes_prefix.
        while len(line.encode("utf-8")) > CHUNK_MAX_BYTES:
            cut = _bytes_prefix(line, CHUNK_MAX_BYTES)
            units.append(line[:cut])
            line = line[cut:]
        units.append(line)

    chunks, current, size = [], [], 0
    for unit in units:
        unit_bytes = len(unit.encode("utf-8"))
        # Close the chunk BEFORE the unit that would overflow it, not
        # after. Closing after lets a chunk reach almost CHUNK_MAX_BYTES
        # and then take one more whole unit -- measured live on mixed
        # text plus one long emoji line, 20,692 bytes against a 16,384
        # cap. Every unit is itself capped by the loop above, so this
        # makes the invariant hold rather than nearly hold.
        if current and size + unit_bytes > CHUNK_MAX_BYTES:
            chunks.append("".join(current))
            current, size = [], 0
        current.append(unit)
        size += unit_bytes
        if size >= CHUNK_MIN_BYTES and _is_chunk_boundary(unit):
            chunks.append("".join(current))
            current, size = [], 0
    if current:
        chunks.append("".join(current))
    return chunks


class VaultIncompleteDocument(RuntimeError):
    """A file doc references content chunks that are not in the database.

    Raised rather than returned because the text this would otherwise
    produce is *plausible*: LiveSync stores a note as an ordered list of
    content chunks, so a missing one drops a span out of the middle and
    splices the surviving neighbours together mid-word. There is no
    marker at the seam and the result parses fine.

    Measured, 2026-08-10: `projects/sokrates/projects/agora/ideas.md` was
    re-chunked by a LiveSync client into 184 chunks, 6 of which never
    reached CouchDB. `get` printed the other 178 as if nothing were
    wrong -- 1238 characters gone, including Edvard's `## Board` heading,
    its table header, rows #57 to #50, and the tail of the capture
    sentence he had just typed. A cycle read that, believed it, and had
    to reconstruct the file from an older revision. A scan of all 686
    file docs found exactly one damaged, so this is rare; it is also
    unsurvivable when it happens, because `append` and `put` callers
    read first, and writing back a silently truncated read makes the
    truncation permanent.
    """


class VaultClient:
    def __init__(self):
        self.base = _env("CDB_BASE").rstrip("/")
        self.db = _env("CDB_DB", "obsidian")
        # Unset means "one database", i.e. exactly the behaviour before
        # routing existed. The switch is the env var, in both config repos
        # at once -- flipping one client and not the other is the bug.
        self.nova_db = _env("CDB_NOVA_DB", "")
        user = _env("CDB_USER")
        pw = _env("CDB_PASS")
        self.auth = base64.b64encode(f"{user}:{pw}".encode()).decode()

    def db_for(self, path):
        """Which database holds `path`. One rule, one place, so the answer
        cannot drift between the call sites that need it.

        Note this takes a *path*. Chunk ids (`h:...`) are content hashes
        with no path at all, so they can never be routed by this function
        -- every chunk lives in the same database as the document that
        points at it, and the chunk call sites take an explicit `db`
        argument for exactly that reason. Routing a chunk id through here
        would silently resolve it to `obsidian` and turn every chunked read
        of a Nova file into a VaultIncompleteDocument.
        """
        if not self.nova_db:
            return self.db
        lowered = (path or "").lower()
        if lowered.startswith(NOVA_DB_FOLDERS) or lowered in NOVA_DB_FILES:
            return self.nova_db
        return self.db

    def dbs_for_prefix(self, prefix):
        """Every database that could hold a document under `prefix`.

        Three cases, and the middle one is the one worth naming: a prefix
        wholly inside Nova's folder needs only Nova's database; a prefix
        that is an *ancestor* of it (`""`, or `projects/`) straddles both
        and has to query both or a whole-vault listing quietly loses every
        Nova file; and anything else is Edvard's alone.
        """
        if not self.nova_db:
            return [self.db]
        lowered = (prefix or "").lower()
        if lowered.startswith(NOVA_DB_FOLDERS):
            return [self.nova_db]
        # Deliberately not `lowered in NOVA_DB_FILES -> [nova]`: as a
        # *prefix*, a single file's path also matches its own neighbours (a
        # `.bak` beside it), and those live in Edvard's database. Querying
        # both is the conservative answer and costs one extra request.
        if any(t.startswith(lowered) for t in NOVA_DB_TARGETS):
            return [self.db, self.nova_db]
        return [self.db]

    def database_health(self):
        """What this process resolved and what it can actually reach.

        Two different questions, and the gap between them is the whole risk
        during a migration: a name in `CDB_NOVA_DB` says which database
        this client *would* ask, never that an answer would come back.

        The shape is byte-for-byte the one agora-persona-runner's
        `database_health()` returns, minus the `ok` field its handler adds,
        because comparing the two is the point. Until now the bridge --
        the client every Nova cycle actually reads the vault through --
        could only be asked this by importing VaultClient and calling
        `db_for` by hand, which is a worse instrument than the write-probe
        it was supposed to replace.
        """
        names = {"main": self.db}
        if self.nova_db:
            names["nova"] = self.nova_db
        databases = {}
        for role, name in names.items():
            entry = {"name": name, "reachable": False, "doc_count": None, "error": None}
            try:
                # The database root, not a document: `_req` builds
                # `{base}/{db}/{path}`, so an empty path is the db info
                # endpoint and needs no document to exist.
                status, info = _req("GET", self.base, urllib.parse.quote(name, safe=""),
                                    self.auth, "", timeout=HEALTH_TIMEOUT_SECONDS)
                if status == 200:
                    entry["reachable"] = True
                    # Includes chunk documents, not just files -- a Nova
                    # file is one doc plus ~4KB content chunks. Named
                    # `doc_count` because that is CouchDB's own field, and
                    # renaming it here would be a second name for one
                    # number the runner already reports under the first.
                    entry["doc_count"] = info.get("doc_count")
                else:
                    entry["error"] = f"HTTP {status}"
            except Exception as e:
                entry["error"] = str(e)[:200]
            databases[role] = entry
        return {
            "routing_enabled": bool(self.nova_db),
            "databases": databases,
            "routes": [{"path": p, "database": self.db_for(p)} for p in HEALTH_PROBE_PATHS],
        }

    def _doc(self, method, doc_id, body=None, db=None):
        return _req(method, self.base, db or self.db_for(doc_id), self.auth,
                    urllib.parse.quote(doc_id, safe=""), body)

    def get_doc(self, doc_id, db=None):
        return self._doc("GET", doc_id, db=db)

    def _fetch_chunks(self, chunk_ids, db):
        """`{chunk_id: data}` for every chunk that exists, in one request.

        An id absent from the result is genuinely missing -- that is what
        `assemble` turns into VaultIncompleteDocument, so this must never
        report a chunk as absent for any reason other than absence. A
        non-200 from `_all_docs` therefore falls back to per-chunk GETs
        rather than returning an empty map, which would make every read
        of every file look like corruption."""
        keys = sorted(set(chunk_ids))
        if not keys:
            return {}
        status, body = self._doc("POST", "_all_docs",
                                 {"keys": keys, "include_docs": True}, db=db)
        if status != 200:
            out = {}
            for chunk_id in keys:
                chunk_status, chunk = self.get_doc(chunk_id, db=db)
                if chunk_status == 200:
                    out[chunk_id] = chunk.get("data", "")
            return out
        return {
            row["key"]: (row["doc"] or {}).get("data", "")
            for row in body.get("rows", [])
            if "error" not in row and row.get("doc")
        }

    def assemble(self, doc, path=None, db=None):
        kids = doc.get("children") or []
        if not kids:
            return doc.get("data", "")
        # Where the doc actually came FROM, never where db_for predicts it
        # should be. Those two agree in steady state and disagree during a
        # migration -- which is exactly when a doc's chunks would be looked
        # up in a database that does not hold them, and a chunk that is
        # merely in the other database is indistinguishable from one that
        # was never written. An intact file would report itself corrupt.
        db = db or doc.get(_SRC_DB_KEY) or self.db_for(
            path or doc.get("path") or doc.get("_id"))
        # One request for every chunk, not one per chunk. This reduces a
        # regression that chunked writes (Cycle 117) introduce; it does not
        # erase it, and the honest numbers belong here rather than in a
        # commit message. Medians of 7 against the live vault, on the same
        # 134KB file: 1 chunk 9ms either way; the same file as 16 chunks is
        # 196ms bulk against 301ms one-GET-per-chunk. So a large file does
        # get slower to read -- roughly 9ms to 196ms -- and this recovers
        # about a third of that. The trade is deliberate: the write side
        # was leaving a full dead copy of the file behind on every save.
        by_id = self._fetch_chunks(kids, db)
        parts = []
        missing = []
        for chunk_id in kids:
            if chunk_id not in by_id:
                missing.append(chunk_id)
            parts.append(by_id.get(chunk_id, ""))
        if missing:
            raise VaultIncompleteDocument(
                f"{path or doc.get('path') or doc.get('_id')}: {len(missing)} of "
                f"{len(kids)} content chunks missing from the vault "
                f"({', '.join(missing[:5])}"
                f"{', …' if len(missing) > 5 else ''}) -- refusing to serve a "
                f"partial document; recover with vault_git_revision_history"
            )
        return "".join(parts)

    def read(self, path):
        db = self.db_for(path.lower())
        status, doc = self.get_doc(path.lower(), db=db)
        if status != 200:
            return None
        # A LiveSync tombstone keeps its content chunks, so assemble()
        # happily rebuilds the text of a note that no longer exists --
        # see file_docs(). Deleted means gone; recover from the daily
        # GitHub snapshot, not from here.
        if doc.get("deleted"):
            return None
        return self.assemble(doc, path.lower(), db)

    def file_docs(self, prefix=""):
        """{doc_id: doc} for every file under `prefix` that still exists.

        Obsidian LiveSync does not remove a document when a note is
        deleted -- it sets `deleted: true` and leaves everything else in
        place, which is how other clients learn to drop their copy. So a
        deleted note stays in `_all_docs` forever with its content intact,
        and a tool that reads ids without reading the flag hands back
        files Edvard has thrown away.

        Measured on the live vault 2026-08-07: 309 of 897 documents were
        tombstones -- a third of everything this tool could see. Most were
        pre-move copies left by a vault reorganisation, sitting one prefix
        away from their live replacements; `kanban.md` was deleted with no
        replacement at all, and `prompt.md` was still sending every cycle
        to read it as the backlog.

        `_all_docs` returns ids and revs but not fields, so the ids get
        re-fetched with `include_docs=true` in batches. A Mango `_find` on
        the flag is worse -- unindexed, it scans all 10939 docs in 8.5s
        whatever the prefix.

        **Both phases are restricted to the prefix, and until 2026-08-11
        only the second one was.** Doc ids in a LiveSync vault are the
        lowercased file paths, so a folder is a contiguous key range and
        CouchDB can seek straight to it; this asked for the whole database
        and filtered the rows in Python. The docstring claimed the cost
        scaled with the prefix, which was true of the batches below and
        false of the scan above -- listing one folder or listing the vault
        paid the same 12k rows either way.

        Measured end to end on the live vault 2026-08-11, `ls` of the
        103-file journal folder: **3.3s before, 1.0s after**, byte
        identical output. The sweep itself is the part that went from
        ~2.3s to ~0.05s; the second below still re-fetches those 103
        documents to read their `deleted` flags, and that is the floor
        this cannot go under without giving up the tombstone check. `list`
        is the only caller -- `recent` goes through Mango `_find` and
        never paid this -- so the win is on `ls`, which every cycle runs
        to find the number of its own last journal entry before it can
        write one. The runner's copy of this client has had the range
        since 2026-08-09; this is that drift, in the half nobody checks.

        An empty prefix keeps the old behaviour: `""` to U+10FFFF is every
        document there is, which is what `recent()` asks for.
        """
        query = urllib.parse.urlencode({
            "startkey": json.dumps(prefix.lower()),
            "endkey": json.dumps(prefix.lower() + _ID_MAX),
        })
        prefix = prefix.lower()
        # Keyed by database, never flattened: a batch is one POST to one
        # database, so mixing ids from both into a single list of 500 would
        # send half of them to a database that has never heard of them and
        # drop them from the listing without a word.
        keys_by_db = {}
        for db in self.dbs_for_prefix(prefix):
            status, data = _req("GET", self.base, db, self.auth, f"_all_docs?{query}")
            if status != 200:
                # An empty folder and a failed sweep used to look identical
                # from here -- `ls` printed nothing either way, and "that
                # folder is empty" is a thing a cycle writes down as fact.
                # With two databases it is worse than that: one failing
                # leaves the other's rows in place, so the caller gets a
                # partial listing that looks entirely healthy. Say which
                # database, or the warning cannot be acted on.
                print(f"vault_tool: WARNING listing failed on database {db!r} "
                      f"({status}) for {prefix!r}; files under that prefix in "
                      f"that database are missing from this listing -- this is "
                      f"not an empty folder", file=sys.stderr)
                continue
            keys_by_db[db] = [
                row["id"] for row in data.get("rows", [])
                if not row["id"].startswith(INTERNAL_PREFIXES)
                and row["id"].lower().startswith(prefix)
            ]
        out = {}
        for db, keys in keys_by_db.items():
            for i in range(0, len(keys), 500):
                status, res = _req(
                    "POST", self.base, db, self.auth,
                    "_all_docs?include_docs=true", {"keys": keys[i:i + 500]},
                )
                if status != 200:
                    # Silently dropping the batch would make live files
                    # vanish from `ls` with no signal, and "that file does
                    # not exist" is a thing a cycle writes into the journal
                    # as fact.
                    print(f"vault_tool: WARNING include_docs batch failed on "
                          f"database {db!r} ({status}); up to 500 file(s) "
                          f"omitted from this listing", file=sys.stderr)
                    continue
                for row in res.get("rows", []):
                    doc = row.get("doc")
                    if doc and not doc.get("deleted"):
                        # Which database it really came from, for the chunk
                        # lookup that follows. See assemble().
                        doc[_SRC_DB_KEY] = db
                        out[row["id"]] = doc
        return out

    def list(self, prefix=""):
        return sorted(self.file_docs(prefix))

    def recent(self, hours=24, prefix="", limit=DEFAULT_RECENT_LIMIT):
        """Files modified in the last `hours`, newest first.

        Returns `(rows, truncated)` where rows is a list of
        `(mtime_ms, path)`. `truncated` matters: CouchDB applies `limit`
        before this filters anything and `_find` has no ordering
        guarantee, so a truncated result is an arbitrary subset rather
        than the newest ones. A caller that quietly believed such a list
        was complete would be worse off than having no tool at all --
        which is the exact failure this command exists to prevent -- so
        the flag is returned rather than swallowed.

        Uses Mango `_find` with a field projection, not
        `_all_docs?include_docs=true`: the vault is tens of MB of content
        across thousands of docs, and only the _id/mtime pair is needed.
        Unindexed, so CouchDB scans -- a few seconds on a vault this size,
        which is cheap enough not to warrant adding a design doc to
        someone else's database.

        `agora/backups/` is excluded unless `prefix` explicitly asks for
        it -- see BACKUP_PREFIX above for why it still exists.

        Deleted files are *kept* here and flagged, unlike in `list`/`read`
        where deleted means gone. This command answers "what changed",
        and a deletion is a change -- arguably the most important kind,
        since it is the one that invalidates a path some other file still
        points at. Hiding it is how a `kanban.md` deleted on 2026-08-06
        went a full day unnoticed while four cycles a day were told to
        read it. Rows come back as `(mtime_ms, path, deleted)`.
        """
        since_ms = int((time.time() - hours * 3600) * 1000)
        prefix = prefix.lower()
        out = []
        truncated = False
        # Once Nova's files live in their own database, "what changed
        # lately" that only asks Edvard's database answers with his edits
        # and none of Nova's -- and this is the command every cycle opens
        # with, precisely to notice what it does not already know about. A
        # failure is raised rather than warned: a silently short answer
        # here is read as "nothing changed", which is a conclusion, not a
        # gap.
        for db in self.dbs_for_prefix(prefix):
            status, data = _req(
                "POST", self.base, db, self.auth, "_find",
                {"selector": {"mtime": {"$gt": since_ms}},
                 "fields": ["_id", "mtime", "deleted"], "limit": limit},
            )
            if status != 200:
                raise SystemExit(
                    f"vault_tool: _find failed on database {db!r} ({status}): {data}")
            docs = data.get("docs", [])
            # `limit` is applied by CouchDB per query, so each database can
            # truncate independently and either one poisons the whole list.
            truncated = truncated or len(docs) >= limit
            for doc in docs:
                path = doc["_id"]
                if path.startswith(INTERNAL_PREFIXES):
                    continue
                if not path.lower().startswith(prefix):
                    continue
                if path.lower().startswith(BACKUP_PREFIX) and not prefix.startswith(BACKUP_PREFIX):
                    continue
                out.append((doc.get("mtime", 0), path, bool(doc.get("deleted"))))
        return sorted(out, reverse=True), truncated

    def _chunk_id_for(self, content_bytes):
        try:
            import xxhash
            return f"h:{xxhash.xxh64(content_bytes).hexdigest()}"
        except Exception:
            import hashlib
            return f"h:{hashlib.sha256(content_bytes).hexdigest()[:16]}"

    def _existing_chunk_ids(self, chunk_ids, db):
        """Which of `chunk_ids` are already in database `db`.

        `db` is required rather than derived: a chunk id is a content hash
        with no path, so there is nothing to derive it from, and defaulting
        would silently ask Edvard's database whether Nova's chunks exist.
        The answer would be "no" for every one of them, which is merely
        wasteful on write -- but the same mistake on read is a file that
        comes back empty.

        One `_all_docs` POST instead of a GET per chunk. A row for a
        missing id carries `error`; a row for a deleted one carries
        `value.deleted`, and both have to be rewritten."""
        keys = sorted(set(chunk_ids))
        if not keys:
            return set()
        # `_all_docs` is not a doc id, but it needs no escaping and going
        # through _doc keeps every CouchDB call in this class on one seam
        # the tests can fake -- the module-level _req is guarded against
        # in tests precisely because the bridge pod resolves the real
        # vault.
        status, body = self._doc("POST", "_all_docs", {"keys": keys}, db=db)
        if status != 200:
            return set()
        return {
            row["key"] for row in body.get("rows", [])
            if "error" not in row and not (row.get("value") or {}).get("deleted")
        }

    def _put_raw(self, path, content, existing=None):
        path = path.lower()
        # One lookup, reused for the chunks and the file doc, so a chunk
        # can never be written to a different database than the doc that
        # will point at it.
        db = self.db_for(path)
        now_ms = int(time.time() * 1000)
        content_bytes = content.encode("utf-8")
        chunk_texts = _split_chunks(content)
        chunk_ids = [self._chunk_id_for(t.encode("utf-8")) for t in chunk_texts]

        if existing is None:
            status, found = self.get_doc(path, db=db)
            existing = found if status == 200 else None

        # Chunks are content-addressed, so one that already exists holds
        # exactly this text and does not need rewriting -- that reuse is
        # the entire point of chunking, and it is what stops an append
        # from leaving a whole extra copy of the file behind.
        already = self._existing_chunk_ids(chunk_ids, db)
        written = set()
        for chunk_id, text in zip(chunk_ids, chunk_texts):
            if chunk_id in already or chunk_id in written:
                continue
            chunk = {"_id": chunk_id, "data": text, "type": "leaf", "children": []}
            chunk_status, _ = self._doc("PUT", chunk_id, chunk, db=db)
            if chunk_status == 409:
                # Content-addressed, so a conflict means this exact chunk
                # was created between the existence check above and this
                # PUT -- by the other client, or by a reply turn running
                # alongside a cycle. The id IS the hash of the content, so
                # whoever won stored exactly this text. That is success.
                # Treating it as failure aborts a perfectly good write,
                # and does so most often on the common path: a non-200
                # from the existence check reports "nothing exists", which
                # makes every unchanged chunk a blind PUT and every one of
                # them a 409.
                written.add(chunk_id)
                continue
            if chunk_status not in (200, 201):
                # Never point a file doc at a chunk that isn't there --
                # that is the VaultIncompleteDocument failure, and it is
                # silent on read. Leaving the old revision intact is the
                # safe outcome.
                return f"FAILED(chunk {chunk_id}: {chunk_status})"
            written.add(chunk_id)

        doc = {
            "_id": path, "path": path, "data": "", "children": chunk_ids,
            "size": len(content_bytes), "ctime": now_ms, "mtime": now_ms,
            "type": "plain", "eden": {},
        }
        if existing is not None:
            doc["_rev"] = existing["_rev"]
            doc["ctime"] = existing.get("ctime", now_ms)
        status, _ = self._doc("PUT", path, doc, db=db)
        return "written" if status in (200, 201) else f"FAILED({status})"

    def write(self, path, content):
        """Overwrite (or create) a file. Previous content is not copied
        anywhere first -- the daily GitHub snapshot is the recovery
        path (see module docstring)."""
        status, existing = self.get_doc(path.lower(), db=self.db_for(path.lower()))
        return self._put_raw(path, content, existing if status == 200 else None)

    def append(self, path, content, after_marker=""):
        """Add to an EXISTING file without losing what's already there.
        Fails loudly if the file doesn't exist -- 'append' implies
        something to append to, use write to create a new file."""
        existing_content = self.read(path)
        if existing_content is None:
            return f"FAILED(not found: {path} -- use write to create a new file)"
        if after_marker:
            lines = existing_content.split("\n")
            for i, line in enumerate(lines):
                if line.strip() == after_marker.strip():
                    lines[i + 1:i + 1] = ["", content.strip("\n")]
                    return self.write(path, "\n".join(lines))
            # Asking for a marker and silently getting the opposite end of the
            # file is how journal.md scrambled its own order: entries meant for
            # the top landed at the bottom, for cycles, with nothing reported.
            # An explicit marker is a positional instruction, so failing to
            # honour it is an error -- same spirit as the not-found check above.
            return (f"FAILED(after_marker not found in {path}: {after_marker!r} "
                    f"-- nothing written; omit after_marker to append at the end)")
        sep = "" if existing_content.endswith("\n\n") else ("\n" if existing_content.endswith("\n") else "\n\n")
        return self.write(path, existing_content + sep + content.strip("\n") + "\n")

    def delete(self, path):
        """Tombstone the document the way Obsidian does, instead of
        removing it from CouchDB.

        These are two different operations and only one of them syncs.
        A CouchDB `DELETE` throws the document away; LiveSync peers poll
        the changes feed for *documents*, so a phone that already holds
        the note is never told anything and keeps its local copy
        forever. Obsidian's own delete instead keeps the document and
        sets a `deleted` field inside it -- that edit is a new revision,
        peers see it, and they drop their copy.

        Every agent delete this platform has done used the first one.
        Edvard reported the symptom twice (comments.md 2026-08-12: "my
        Vault did sync multiple times and i still have the old deleted
        files on my phone... I have had this issue before, as in an
        agent deleted a file in couchdb but i still have it on my
        phone"), and Cycle 129 hand-repaired the 167 paths the
        migration had removed this way. This is that repair moved into
        the code path, so the next delete anywhere does not reproduce
        it.

        Measured on the live `obsidian` database 2026-08-12: of 283
        tombstones Obsidian itself wrote, 282 keep their `children` and
        `size` intact (the exception is an empty note that never had
        content). So the flag is the signal and nothing else is
        rewritten; `read`/`file_docs` already treat it as gone. `mtime`
        is bumped to match what Obsidian's own deletions do -- that much
        is measured. That it is specifically what settles a conflict
        against a peer's stale copy is inference from the plugin's
        behaviour, not something measured here.

        **The cost, so nobody thinks it was overlooked:** a tombstone
        keeps the file's text in CouchDB indefinitely, where a hard
        DELETE eventually became reclaimable by compaction. That is
        unavoidable -- dropping the content is what stopped the deletion
        syncing in the first place -- but it means deleted text stays
        readable to anyone with database access, and the tombstone
        fraction (309 of 897 documents when `file_docs` last measured
        it) only ever grows. Reclaiming it belongs to the orphan-chunk
        and compaction work, not here.
        """
        path = path.lower()
        db = self.db_for(path)
        status, existing = self.get_doc(path, db=db)
        if status == 404:
            return "absent"
        if status != 200:
            return f"FAILED_GET({status})"
        if existing.get("deleted"):
            # Already a tombstone. Re-flagging it would burn a revision
            # and republish a deletion peers have long since applied.
            # Distinct from "absent" on purpose: "there was nothing
            # here" and "this was already correctly deleted" are
            # different answers, and the caller only gets this string.
            return "already deleted"
        doc = dict(existing)
        doc["deleted"] = True
        doc["mtime"] = int(time.time() * 1000)
        status, _ = self._doc("PUT", path, doc, db=db)
        return "deleted" if status in (200, 201) else f"FAILED({status})"


def main(argv=None):
    # An incomplete document is a read failure, not a result. It exits
    # non-zero and prints to stderr so a caller that pipes `get` into a
    # file gets an empty file and a visible error, rather than a
    # confident-looking partial one it will go on to edit and write back.
    try:
        return _dispatch(argv)
    except VaultIncompleteDocument as e:
        print(f"[INCOMPLETE DOCUMENT] {e}", file=sys.stderr)
        return 1


def _dispatch(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 1
    client = VaultClient()
    cmd = argv[0]
    if cmd == "get":
        content = client.read(argv[1])
        print(content if content is not None else f"[not found: {argv[1]}]")
    elif cmd == "put":
        content = Path(argv[2]).read_text(encoding="utf-8")
        print(f"{client.write(argv[1], content)}: {argv[1]}")
    elif cmd == "puts":
        content = sys.stdin.read()
        print(f"{client.write(argv[1], content)}: {argv[1]}")
    elif cmd == "append":
        content = Path(argv[2]).read_text(encoding="utf-8")
        marker = argv[3] if len(argv) > 3 else ""
        print(f"{client.append(argv[1], content, marker)}: {argv[1]}")
    elif cmd == "appends":
        content = sys.stdin.read()
        # `puts` spells stdin as a literal "-", and this command's own usage
        # line did too, so a caller following it passed "-" as the marker --
        # which matched no line and silently appended at the end instead.
        rest = argv[2:]
        if rest and rest[0] == "-":
            rest = rest[1:]
        marker = rest[0] if rest else ""
        print(f"{client.append(argv[1], content, marker)}: {argv[1]}")
    elif cmd == "delete":
        print(f"{client.delete(argv[1])}: {argv[1]}")
    elif cmd == "ls":
        for p in client.list(argv[1] if len(argv) > 1 else ""):
            print(p)
    elif cmd == "recent":
        hours = float(argv[1]) if len(argv) > 1 else 24
        prefix = argv[2] if len(argv) > 2 else ""
        rows, truncated = client.recent(hours, prefix)
        if truncated:
            print(f"[INCOMPLETE: hit the {DEFAULT_RECENT_LIMIT}-doc cap, so this is an "
                  f"arbitrary subset, NOT the newest. Use a shorter window.]")
        for mtime_ms, path, deleted in rows:
            mark = "  [DELETED]" if deleted else ""
            print(f"{_local_stamp(mtime_ms)}  {path}{mark}")
        if not rows:
            print(f"[nothing modified in the last {hours:g}h]")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
