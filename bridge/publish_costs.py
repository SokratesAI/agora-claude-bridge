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

Everything here is best-effort by contract. It is called from a daemon thread
on the way out of a cycle, after the work is done and while the reply is
already being written -- a failure to publish costs must never cost a cycle
its reply, so every entry point returns False rather than raising.
"""
import json
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


def build_payload(projects_dir=None, history_path=None):
    """The whole document, or None if the transcripts yielded nothing.

    None rather than an empty document on purpose: publishing "no cycles have
    ever run" over a good file would be worse than publishing nothing, and the
    one condition that produces it -- a projects dir that is missing or empty
    -- is exactly what a misconfigured CLAUDE_HOME looks like.
    """
    rows = analytics.scan(projects_dir)
    if not rows:
        log("cost publish: no transcripts found, nothing to publish")
        return None
    cycles = [_cycle_row(r) for r in rows if r["kind"] == "cycle"]
    return {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cycles": cycles,
        "summary": analytics.summarize(rows),
        "quota": read_quota_history(history_path),
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
        payload = build_payload(projects_dir, history_path)
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
