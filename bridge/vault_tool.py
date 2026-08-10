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

Env: CDB_BASE, CDB_USER, CDB_PASS, CDB_DB (default "obsidian").

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


def _req(method, base, db, auth, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        f"{base}/{db}/{path}", data=data,
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


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
        user = _env("CDB_USER")
        pw = _env("CDB_PASS")
        self.auth = base64.b64encode(f"{user}:{pw}".encode()).decode()

    def _doc(self, method, doc_id, body=None):
        return _req(method, self.base, self.db, self.auth, urllib.parse.quote(doc_id, safe=""), body)

    def get_doc(self, doc_id):
        return self._doc("GET", doc_id)

    def assemble(self, doc, path=None):
        kids = doc.get("children") or []
        if not kids:
            return doc.get("data", "")
        parts = []
        missing = []
        for chunk_id in kids:
            # One GET per chunk. This used to call get_doc twice for every
            # chunk -- once for the status, once for the data -- so a 184
            # chunk file cost 368 round trips.
            status, chunk = self.get_doc(chunk_id)
            if status != 200:
                missing.append(chunk_id)
            parts.append(chunk.get("data", "") if status == 200 else "")
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
        status, doc = self.get_doc(path.lower())
        if status != 200:
            return None
        # A LiveSync tombstone keeps its content chunks, so assemble()
        # happily rebuilds the text of a note that no longer exists --
        # see file_docs(). Deleted means gone; recover from the daily
        # GitHub snapshot, not from here.
        if doc.get("deleted"):
            return None
        return self.assemble(doc, path.lower())

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
        re-fetched with `include_docs=true` in batches. Cost scales with
        the prefix asked for: 0.46s for a project folder, 5.0s for the
        whole vault. A Mango `_find` on the flag is worse -- unindexed, it
        scans all 10939 docs in 8.5s whatever the prefix.
        """
        status, data = _req("GET", self.base, self.db, self.auth, "_all_docs")
        if status != 200:
            return {}
        prefix = prefix.lower()
        keys = [
            row["id"] for row in data.get("rows", [])
            if not row["id"].startswith(INTERNAL_PREFIXES)
            and row["id"].lower().startswith(prefix)
        ]
        out = {}
        for i in range(0, len(keys), 500):
            status, res = _req(
                "POST", self.base, self.db, self.auth,
                "_all_docs?include_docs=true", {"keys": keys[i:i + 500]},
            )
            if status != 200:
                # Silently dropping the batch would make live files vanish
                # from `ls` with no signal, and "that file does not exist"
                # is a thing a cycle writes into the journal as fact.
                print(f"vault_tool: WARNING include_docs batch failed ({status}); "
                      f"up to 500 file(s) omitted from this listing", file=sys.stderr)
                continue
            for row in res.get("rows", []):
                doc = row.get("doc")
                if doc and not doc.get("deleted"):
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
        status, data = _req(
            "POST", self.base, self.db, self.auth, "_find",
            {"selector": {"mtime": {"$gt": since_ms}},
             "fields": ["_id", "mtime", "deleted"], "limit": limit},
        )
        if status != 200:
            raise SystemExit(f"vault_tool: _find failed ({status}): {data}")
        docs = data.get("docs", [])
        prefix = prefix.lower()
        out = []
        for doc in docs:
            path = doc["_id"]
            if path.startswith(INTERNAL_PREFIXES):
                continue
            if not path.lower().startswith(prefix):
                continue
            if path.lower().startswith(BACKUP_PREFIX) and not prefix.startswith(BACKUP_PREFIX):
                continue
            out.append((doc.get("mtime", 0), path, bool(doc.get("deleted"))))
        return sorted(out, reverse=True), len(docs) >= limit

    def _chunk_id_for(self, content_bytes):
        try:
            import xxhash
            return f"h:{xxhash.xxh64(content_bytes).hexdigest()}"
        except Exception:
            import hashlib
            return f"h:{hashlib.sha256(content_bytes).hexdigest()[:16]}"

    def _put_raw(self, path, content, existing=None):
        path = path.lower()
        now_ms = int(time.time() * 1000)
        content_bytes = content.encode("utf-8")
        chunk_id = self._chunk_id_for(content_bytes)

        if existing is None:
            status, found = self.get_doc(path)
            existing = found if status == 200 else None

        chunk_status, existing_chunk = self.get_doc(chunk_id)
        chunk = {"_id": chunk_id, "data": content, "type": "leaf", "children": []}
        if chunk_status == 200:
            chunk["_rev"] = existing_chunk["_rev"]
        self._doc("PUT", chunk_id, chunk)

        doc = {
            "_id": path, "path": path, "data": "", "children": [chunk_id],
            "size": len(content_bytes), "ctime": now_ms, "mtime": now_ms,
            "type": "plain", "eden": {},
        }
        if existing is not None:
            doc["_rev"] = existing["_rev"]
            doc["ctime"] = existing.get("ctime", now_ms)
        status, _ = self._doc("PUT", path, doc)
        return "written" if status in (200, 201) else f"FAILED({status})"

    def write(self, path, content):
        """Overwrite (or create) a file. Previous content is not copied
        anywhere first -- the daily GitHub snapshot is the recovery
        path (see module docstring)."""
        status, existing = self.get_doc(path.lower())
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
        path = path.lower()
        status, existing = self.get_doc(path)
        if status == 404:
            return "absent"
        if status != 200:
            return f"FAILED_GET({status})"
        status, _ = _req(
            "DELETE", self.base, self.db, self.auth,
            f"{urllib.parse.quote(path, safe='')}?rev={existing['_rev']}",
        )
        return "deleted" if status in (200, 202) else f"FAILED({status})"


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
