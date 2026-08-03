#!/usr/bin/env python3
"""vault_tool -- CouchDB (Obsidian LiveSync) read/write, for use from
inside a claude-cli session's own Bash tool. Same wire format and same
backup-before-overwrite discipline as agora_runner/vault.py (the tool
belt agora-persona-runner's gemini/anthropic personas already use) and
~/vault-tools/vault_tool.py (the interactive-session equivalent) --
kept as a third, independent copy rather than an import because this
image has no dependency on either of those repos and shouldn't grow one
just for this.

Usage (from Bash inside the bridge pod):
  python3 -m bridge.vault_tool get    <path>
  python3 -m bridge.vault_tool put    <path> <local_file>
  python3 -m bridge.vault_tool puts   <path> -              # content from stdin
  python3 -m bridge.vault_tool append <path> <local_file> [after_marker]
  python3 -m bridge.vault_tool appends <path> - [after_marker]  # content from stdin
  python3 -m bridge.vault_tool delete <path>
  python3 -m bridge.vault_tool ls     [prefix]
  python3 -m bridge.vault_tool recent [hours] [prefix]   # what changed lately

Env: CDB_BASE, CDB_USER, CDB_PASS, CDB_DB (default "obsidian").

All paths are lowercased before use, always (standing vault-wide
convention -- see CLAUDE.md/memory `feedback-always-use-lowercase-vault-paths`).
Every overwrite/delete backs up the previous content into the vault
itself first, under `agora/backups/<timestamp> <basename>` -- mirrors
agora_runner/vault.py's vault_write_path, not the interactive tool's
local-disk backup dir, since this pod has no host filesystem a human
would ever look at.
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
# Written by this tool's own backup-before-overwrite discipline (see write/
# delete below), so they're derivative of the edits they'd otherwise drown out.
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

    def assemble(self, doc):
        kids = doc.get("children") or []
        if kids:
            return "".join(
                (self.get_doc(c)[1].get("data", "") if self.get_doc(c)[0] == 200 else "")
                for c in kids
            )
        return doc.get("data", "")

    def read(self, path):
        status, doc = self.get_doc(path.lower())
        if status != 200:
            return None
        return self.assemble(doc)

    def list(self, prefix=""):
        status, data = _req("GET", self.base, self.db, self.auth, "_all_docs")
        out = []
        for row in data.get("rows", []):
            doc_id = row["id"]
            if doc_id.startswith(INTERNAL_PREFIXES):
                continue
            if doc_id.lower().startswith(prefix.lower()):
                out.append(doc_id)
        return sorted(out)

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
        it: every overwrite writes one, so they'd bury the real edits.
        """
        since_ms = int((time.time() - hours * 3600) * 1000)
        status, data = _req(
            "POST", self.base, self.db, self.auth, "_find",
            {"selector": {"mtime": {"$gt": since_ms}},
             "fields": ["_id", "mtime"], "limit": limit},
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
            out.append((doc.get("mtime", 0), path))
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
        """Overwrite (or create) a file -- backs up any previous content
        into agora/backups/ first, same convention agora_runner/vault.py
        uses for its own writes."""
        status, existing = self.get_doc(path.lower())
        if status == 200:
            previous = self.assemble(existing)
            if previous.strip():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                base = path.rsplit("/", 1)[-1]
                self._put_raw(f"agora/backups/{stamp} {base}", previous)
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
        sep = "" if existing_content.endswith("\n\n") else ("\n" if existing_content.endswith("\n") else "\n\n")
        return self.write(path, existing_content + sep + content.strip("\n") + "\n")

    def delete(self, path):
        path = path.lower()
        status, existing = self.get_doc(path)
        if status == 404:
            return "absent"
        if status != 200:
            return f"FAILED_GET({status})"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        base = path.rsplit("/", 1)[-1]
        self._put_raw(f"agora/backups/{stamp} {base}", self.assemble(existing))
        status, _ = _req(
            "DELETE", self.base, self.db, self.auth,
            f"{urllib.parse.quote(path, safe='')}?rev={existing['_rev']}",
        )
        return "deleted" if status in (200, 202) else f"FAILED({status})"


def main(argv=None):
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
        marker = argv[2] if len(argv) > 2 else ""
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
        for mtime_ms, path in rows:
            print(f"{_local_stamp(mtime_ms)}  {path}")
        if not rows:
            print(f"[nothing modified in the last {hours:g}h]")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
