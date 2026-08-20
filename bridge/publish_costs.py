"""Publish the cost-and-cadence record into the vault, where the site can read it.

`analytics.py` turns the CLI transcripts on the PVC into a per-cycle ledger,
and `quota.py` appends a quota reading to `quota-history.jsonl` roughly once a
minute. Between them they answer every question Edvard asked on 2026-08-08 --
what a cycle costs, whether the cadence is sustainable, which lever is worth
pulling. Both files have been complete and correct for days, and nobody has
ever seen either of them, because of a fact that is easy to miss and settles
the design: **they live on the bridge's PVC, and the site does not mount it.**

`nova-site` runs from the runner image with one volume, an emptyDir on /tmp
(measured 2026-08-11). It has no route to `/data/claude-home` at all. What it
does have is the vault, which is where it reads the journal, the boards and
the comments from. So the vault is the channel, and this module is the write
end of it: rebuild the ledger, fold in the quota history, put one JSON
document where the site already knows how to look.

The stale file is the argument for doing this on a timer rather than by hand.
`/data/workspace/cycle-ledger.json` was written once, by a cycle, in the
bridge's scratch directory -- which is not the PVC and does not survive a pod
restart. On 2026-08-11 it still described 45 cycles ending 2026-08-08 while
the loop was on its 107th. A snapshot somebody has to remember to refresh is
a snapshot that is wrong, so this runs off the watcher's own end-of-cycle
reading (quota.py) and never off a human deciding it is time.

**The vault document is the record; the transcripts are a window onto the
last few hours.** This module had that backwards until 2026-08-20 and rebuilt
the whole document from `analytics.scan` every cycle. The transcripts do not
survive a bridge pod restart: the pod came up at 2026-08-19T19:27Z holding
twelve session files, the oldest under two hours old, against a vault ledger
of 265 cycles going back to 08-03. Five cycles in a row then tried to publish
5-10KB over 310KB, and the only thing that stopped them was `vault_tool`'s
collapse guard -- whose error text suggests passing `--all`, i.e. suggests
doing the destructive thing. So `build_payload` now merges into what the
vault already holds and refuses to publish at all when it cannot read it.

Everything here is best-effort by contract. It is called from a daemon thread
on the way out of a cycle, after the work is done and while the reply is
already being written -- a failure to publish costs must never cost a cycle
its reply, so every entry point returns False rather than raising.
"""
import json
import statistics
import subprocess
import sys
import time

from bridge import analytics
from bridge.log import log
from bridge.quota import history_path_for, SNAPSHOT_FILE

# Lowercase, like every vault path this system writes (vault_tool.py).
VAULT_PATH = "projects/sokrates/projects/agora/nova/resources/cost-ledger.json"

# The quota rows carry their reset timestamps and a per-model scoped window,
# neither of which a chart of the week's shape has any use for. These five
# are what "pace against the 1.0 line" and "how fast is the week burning"
# actually need.
QUOTA_FIELDS = ("at", "five_hour", "five_hour_pace", "seven_day", "seven_day_pace")


def _cycle_row(row):
    """One ledger row, cut to what a chart reads.

    `analytics.scan` returns per-session token counts in four classes plus
    the weighted total. The four classes are what `cost_share` in the summary
    is for -- they answer "which lever", once, for the whole window. Repeating
    them on all 107 rows would triple the document to say the same thing.
    """
    return {
        "session": row["session"],
        "startedAt": row["started_at"],
        "endedAt": row["ended_at"],
        "durationSeconds": row["duration_seconds"],
        "turns": row["turns"],
        "subagentTurns": row["subagent_turns"],
        "toolCalls": row["tool_calls"],
        "weightedTokens": row["weighted_tokens"],
        "models": row["models"],
    }


def read_quota_history(path=None):
    """Every quota reading, compacted. Malformed lines are skipped, not fatal.

    The file is append-only and written by a poller that can be killed
    mid-line at any moment (the pod is `Recreate` with a drain), so a
    truncated last line is a normal state of this file rather than corruption.
    """
    path = path or history_path_for(SNAPSHOT_FILE)
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                kept = {k: row[k] for k in QUOTA_FIELDS if k in row}
                if row.get("boundary"):
                    kept["boundary"] = row["boundary"]
                if kept:
                    rows.append(kept)
    except OSError as exc:
        log(f"cost publish: quota history unreadable: {type(exc).__name__}: {exc}")
    return rows


#: What `read_stored` returns when the vault could not be asked at all, as
#: distinct from `None`, which means "asked, and there is nothing there yet".
#: The two must not be confused: publishing a transcripts-only ledger over a
#: good one destroys history, and a failed read looks exactly like a first run.
UNREADABLE = object()


def read_stored(vault_path=VAULT_PATH):
    """The ledger already in the vault: a dict, None, or `UNREADABLE`.

    The transcripts this module scans do **not** survive a bridge pod restart.
    Measured 2026-08-20: the pod came up at 19:27Z with twelve session files on
    it, the oldest under two hours old, while the vault ledger held 265 cycles
    going back to 2026-08-03. So the vault document is the durable record and
    the PVC is a window onto the last few hours -- which is the opposite of
    what this module assumed when it rebuilt the whole document every cycle.

    `vault_tool get` prints `[not found: ...]` and exits 0 for a path that has
    never been written, so a missing document is a successful read of nothing.
    Anything else that goes wrong returns `UNREADABLE`, and the caller must
    treat that as a reason not to publish rather than as an empty ledger.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "bridge.vault_tool", "get", vault_path],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:
        log(f"cost publish: stored ledger unreadable: {type(exc).__name__}: {exc}")
        return UNREADABLE
    body = (proc.stdout or "").strip()
    if proc.returncode != 0:
        log(f"cost publish: stored ledger unreadable: rc={proc.returncode} "
            f"{(proc.stderr or '').strip()[:200]}")
        return UNREADABLE
    if not body or body.startswith("[not found"):
        return None
    try:
        stored = json.loads(body)
    except ValueError as exc:
        log(f"cost publish: stored ledger is not JSON: {exc}")
        return UNREADABLE
    if not isinstance(stored, dict):
        log("cost publish: stored ledger is not an object")
        return UNREADABLE
    return stored


def merge_cycles(stored_cycles, fresh_cycles):
    """Union of two cycle lists, keyed by session id, oldest first.

    A session on disk wins over the stored copy of itself: the transcript is
    the primary source and a cycle that was still running when the last
    publish happened has a longer one now. Rows the transcripts no longer
    reach are carried through untouched, which is the whole point.
    """
    by_session = {}
    for row in list(stored_cycles or []) + list(fresh_cycles or []):
        if isinstance(row, dict) and row.get("session"):
            by_session[row["session"]] = row
    return sorted(by_session.values(), key=lambda r: r.get("startedAt") or "")


def merge_quota(stored_quota, fresh_quota):
    """Union of two quota series, keyed by reading time, oldest first.

    `quota-history.jsonl` sits on the same disk as the transcripts and is lost
    in the same restart, so the series needs exactly the same treatment as the
    cycles. Measured 2026-08-20: 2,414 stored readings against 73 on disk, and
    the two do not overlap at all -- the stored series ends 08-17, the file
    begins 08-19. Merging only the cycles left the document at 76,857 bytes
    against 310,060, which is still under `vault_tool`'s 25% collapse floor,
    so the publish would have gone on failing with the history repaired.
    """
    by_at = {}
    for row in list(stored_quota or []) + list(fresh_quota or []):
        if isinstance(row, dict) and row.get("at") is not None:
            by_at[row["at"]] = row
    return sorted(by_at.values(), key=lambda r: r["at"])


def _class_totals(rows):
    """Raw and weighted per-token-class totals over `analytics.scan` rows."""
    raw = {f: sum(r.get(f, 0) for r in rows) for f in analytics.TOKEN_FIELDS}
    weighted = {
        f: sum(r.get("weighted_by_field", {}).get(f, 0) for r in rows)
        for f in analytics.TOKEN_FIELDS
    }
    return raw, weighted


def merge_summary(stored_summary, scan_rows, merged_cycles, stored_sessions=()):
    """The summary for the merged ledger.

    Everything a ledger row carries -- counts, weighted totals, durations,
    models -- is recomputed from `merged_cycles`, so it describes the whole
    history rather than whatever is on disk today.

    The per-token-class totals are the exception, because `_cycle_row` does
    not keep them and the rows they came from are gone. They are carried
    forward instead: the stored totals plus the classes of every session the
    stored summary had *not* already counted.

    **That is an accumulation, not a recomputation, and it has one limit worth
    stating plainly rather than dressing up.** A session already counted keeps
    the numbers it was counted with; if its transcript grew since, the growth
    never reaches these five counters. In this system it cannot -- `publish`
    runs after `proc.wait()`, so a transcript is closed before it is ever
    scanned (`quota.py` says so, and a reviewer checked it) -- but the ledger
    is the wrong place to assert that, so: `total_weighted` and every
    per-cycle `weightedTokens` are recomputed from the rows and are always
    right; these five are carried and can only be as right as the last write.

    An earlier version of this subtracted the re-measured session and added
    the whole scan back. That looked like exact accounting and was a
    tautology: the overlap is a subset of the scan, so the two terms cancel
    and the result is identical to the line below, with a misleading docstring
    on top. The test that was supposed to pin it could not fail, because no
    single-generation fixture can make the overlap anything but a subset.

    `other_sessions` deliberately counts only what is on disk now. It is the
    one field here that cannot be carried forward (non-cycle sessions never
    reach the document, so there is nothing to add to), and nothing reads it.
    """
    fresh_cycles = [r for r in scan_rows if r["kind"] == "cycle"]
    if not merged_cycles:
        return {"cycles": 0, "other_sessions": len(scan_rows)}

    stored_sessions = set(stored_sessions)
    stored_raw = (stored_summary or {}).get("totals") or {}
    stored_weighted = (stored_summary or {}).get("totals_weighted") or {}
    uncounted = [r for r in fresh_cycles if r["session"] not in stored_sessions]
    new_raw, new_weighted = _class_totals(uncounted)

    totals = {
        f: int(stored_raw.get(f, 0)) + new_raw[f]
        for f in analytics.TOKEN_FIELDS
    }
    totals_weighted = {
        f: round(float(stored_weighted.get(f, 0)) + new_weighted[f], 1)
        for f in analytics.TOKEN_FIELDS
    }
    class_grand = sum(totals_weighted.values())

    weighted = [r.get("weightedTokens") or 0 for r in merged_cycles]
    durations = [r["durationSeconds"] for r in merged_cycles if r.get("durationSeconds")]
    return {
        "cycles": len(merged_cycles),
        "other_sessions": len(scan_rows) - len(fresh_cycles),
        "first_cycle": merged_cycles[0].get("startedAt"),
        "last_cycle": merged_cycles[-1].get("startedAt"),
        "totals": totals,
        "totals_weighted": totals_weighted,
        "total_weighted": round(sum(weighted), 1),
        "mean_weighted": round(statistics.mean(weighted), 1),
        "median_weighted": round(statistics.median(weighted), 1),
        "max_weighted": max(weighted),
        "min_weighted": min(weighted),
        "mean_duration_seconds": round(statistics.mean(durations), 1) if durations else None,
        "median_duration_seconds": round(statistics.median(durations), 1) if durations else None,
        "cost_share": {
            f: round(totals_weighted[f] / class_grand * 100, 1) if class_grand else 0.0
            for f in analytics.TOKEN_FIELDS
        },
        "models": sorted({m for r in merged_cycles for m in (r.get("models") or ())}),
    }


def build_payload(projects_dir=None, history_path=None, vault_path=VAULT_PATH):
    """The whole document, or None if there is nothing safe to publish.

    None rather than an empty document on purpose: publishing "no cycles have
    ever run" over a good file would be worse than publishing nothing. That
    guard used to cover only the empty case, and the case it missed is the one
    that actually happened -- a transcript store that came back from a pod
    restart holding a few hours of sessions instead of three weeks of them.
    The document that would have been written was 5,281 bytes against 310,060,
    and the only thing that stopped it was a size check in `vault_tool` whose
    error text suggests passing `--all`, i.e. suggests doing the destructive
    thing. So the merge below is the fix and the refusal here is its floor.
    """
    rows = analytics.scan(projects_dir)
    if not rows:
        log("cost publish: no transcripts found, nothing to publish")
        return None
    stored = read_stored(vault_path)
    if stored is UNREADABLE:
        log("cost publish: could not read the stored ledger, refusing to "
            "replace it with a transcripts-only rebuild")
        return None

    stored_cycles = (stored or {}).get("cycles") or []
    stored_sessions = {r.get("session") for r in stored_cycles if isinstance(r, dict)}
    fresh = [_cycle_row(r) for r in rows if r["kind"] == "cycle"]
    merged = merge_cycles(stored_cycles, fresh)
    # Against the count of distinct stored sessions, not of stored rows. A
    # stored document that already holds a duplicate `session`, or a row with
    # none at all, legitimately collapses here -- and comparing against the
    # raw row count would make that a door that never reopens: every future
    # publish would refuse, forever, over one malformed row, and the only
    # sign of it would be a log line nobody is watching for.
    if len(merged) < len(stored_sessions - {None}):
        log(f"cost publish: merge lost rows ({len(stored_sessions)} stored sessions "
            f"-> {len(merged)}), refusing to publish")
        return None
    added = len(merged) - len(stored_cycles)
    log(f"cost publish: {len(stored_cycles)} stored + {len(fresh)} on disk "
        f"-> {len(merged)} cycles ({added} new)")
    return {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cycles": merged,
        "summary": merge_summary(
            (stored or {}).get("summary"), rows, merged, stored_sessions),
        "quota": merge_quota((stored or {}).get("quota"),
                             read_quota_history(history_path)),
        "weights": analytics.COST_WEIGHTS,
    }


def publish(payload, vault_path=VAULT_PATH):
    """Write the payload to the vault. True if it landed.

    Shelled out to `vault_tool` rather than imported. The reason this
    docstring gave until Cycle 108 was false and worth replacing rather than
    deleting: it claimed the module "keeps its whole write path inside
    `_dispatch`, so there is no `put` to import". There is --
    `VaultClient.write(path, content)` is public, and a reviewer went and read
    it. The real reasons are smaller and both survive checking:

    Process isolation. This runs on the watcher's daemon thread on the way
    out of a cycle. A subprocess cannot take that thread down with an
    unexpected exception, an import-time failure, or a hung socket that
    outlives the turn -- it just exits non-zero.

    And importing would not buy the clean return value it looks like it
    would: `write` returns the *string* "written" or "FAILED(<status>)", so
    the caller is checking text either way. Given that, the process boundary
    is free.

    `puts <path> -` takes the body on stdin, so the document never touches
    disk. A non-zero exit is a failure to publish, not an exception: see the
    module docstring.
    """
    if payload is None:
        return False
    body = json.dumps(payload, separators=(",", ":"))
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "bridge.vault_tool", "puts", vault_path, "-"],
            input=body, capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:
        log(f"cost publish failed: {type(exc).__name__}: {exc}")
        return False
    # The client prints FAILED and still exits 0 on some paths, which is the
    # trap prompt.md warns every cycle about when it writes a journal entry.
    # Checking both is the only way to know the document actually landed.
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 or "FAILED" in output:
        log(f"cost publish failed: rc={proc.returncode} {output.strip()[:300]}")
        return False
    return True


def refresh(projects_dir=None, history_path=None, vault_path=VAULT_PATH):
    """Build and publish in one call. Never raises -- see the module docstring."""
    try:
        # `vault_path` has to reach both halves. Merging from the default
        # document and then writing the result to a different one would read
        # one ledger and overwrite another with it.
        payload = build_payload(projects_dir, history_path, vault_path)
    except Exception as exc:
        log(f"cost publish: build failed: {type(exc).__name__}: {exc}")
        return False
    if payload is None:
        return False
    ok = publish(payload, vault_path)
    if ok:
        log(f"cost publish: {len(payload['cycles'])} cycles, "
            f"{len(payload['quota'])} quota readings -> {vault_path}")
    return ok


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    dry_run = "--dry-run" in argv
    if dry_run:
        payload = build_payload()
        if payload is None:
            return 1
        print(json.dumps(payload, indent=2)[:4000])
        print(f"\n{len(payload['cycles'])} cycles, {len(payload['quota'])} quota readings, "
              f"{len(json.dumps(payload, separators=(',', ':')))} bytes")
        return 0
    return 0 if refresh() else 1


if __name__ == "__main__":
    raise SystemExit(main())
