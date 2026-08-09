"""Per-cycle token accounting, read back out of the Claude Code transcripts.

Edvard, 2026-08-08, on raising the cycle rate: "i want you to log how much
tokens are spent and much are left every week so we can optimize the
cycles. Maybe we can increase it or maybe we should reduce the cycles.
Only data can tell us this. ... I want you to think in Analytics, measure
and do real scientific studies to figure out the optimal way."

Before this module the only number available was `quota-snapshot.json` --
a percentage of a subscription window, sampled at whatever instant a cycle
happened to look. That is enough to answer "am I about to run out" and
nothing else. It cannot say what a cycle costs, which cycles were
expensive, or what the money actually goes on, because a percentage has no
breakdown and no history.

The transcripts do. The CLI writes one JSONL file per session under
CLAUDE_HOME/.claude/projects/<slug>/, and every assistant message in it
carries the exact `usage` block the API returned -- input, output, and the
two cache counters, per turn, with timestamps. That is a complete record
of every cycle this loop has ever run, sitting on the PVC the whole time.
This module turns it into a ledger.

THE ONE THING THAT WILL BITE WHOEVER EDITS THIS: usage is recorded per
*content block*, not per message. An assistant message with eight blocks
(some text and seven tool_use) is written as eight lines, each carrying a
byte-identical copy of the same `usage` object. Summing per line therefore
counts one API call up to eight times. Measured on cycle 5b348a4b: 163
usage-bearing lines for 85 real messages, and a naive cache-read total of
14.06M against a true 7.84M -- 1.79x too high, with no error and no
obvious tell. Every total here is deduplicated by `message.id` for that
reason, and `test_analytics.py` pins it with a fixture that repeats a
block. Do not "simplify" the dedup away.

Cost is reported in **weighted input-equivalents**, not dollars. The four
token classes are priced differently, so raw totals are not comparable to
each other -- a cycle that reads 10M cached tokens and one that writes 1M
output tokens look wildly different by volume and land in the same place
on the bill. The weights below are the published Opus price ratios
relative to a base input token. We run on a subscription, so this is a
comparability metric and a proxy for quota burn -- not an invoice.
"""
import json
import os
import statistics
import sys
from collections import Counter

from bridge.config import CLAUDE_HOME

PROJECTS_DIR = os.path.join(CLAUDE_HOME, ".claude", "projects")

# Price ratios relative to one base input token, from the published Opus
# rates ($5/MTok in, $25/MTok out, cache read ~0.1x, 5m write 1.25x, 1h
# write 2x). Ratios rather than dollars on purpose: the ratios are what
# make the four counters addable, and they have been stable across model
# generations, whereas the per-token price has not.
COST_WEIGHTS = {
    "input_tokens": 1.0,
    "output_tokens": 5.0,
    "cache_read_tokens": 0.1,
    "cache_write_5m_tokens": 1.25,
    "cache_write_1h_tokens": 2.0,
}

TOKEN_FIELDS = tuple(COST_WEIGHTS)

# A session counts as one of our cycles if its first user message opens
# with one of these. The first is the pre-heartbeat shape, where the whole
# constitution arrived as the user turn; the other two are what the runner
# sends now. Anything else in the directory (probes, one-off manual
# sessions) is still parsed but reported as kind "other", because mixing a
# 2-turn probe into a per-cycle average quietly drags it down.
CYCLE_MARKERS = (
    "You are Nova",
    "[Automatic heartbeat trigger",
    "[Manual heartbeat trigger",
)


def _first_user_text(record):
    """The text of a user turn, or "" -- content is a bare string on some
    records and a list of blocks on others."""
    message = record.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text") or ""
    return ""


def _classify(opening):
    """(kind, trigger) for a session, from its first user message."""
    for marker in CYCLE_MARKERS:
        if opening.startswith(marker):
            if opening.startswith("[Manual"):
                return "cycle", "manual"
            if opening.startswith("[Automatic"):
                return "cycle", "automatic"
            return "cycle", "embedded"
    return "other", "none"


def _usage_totals(usage):
    """Flatten one API `usage` block into our five counters.

    The two cache-write TTLs live in a nested `cache_creation` dict and are
    kept apart because they are priced 1.25x and 2x -- folding them
    together would hide a 60% swing on the most expensive input class.
    Older records predate the nested dict, so the flat
    `cache_creation_input_tokens` is the fallback and is attributed to 5m,
    which is what it meant before 1h caching existed.
    """
    creation = usage.get("cache_creation")
    if isinstance(creation, dict):
        write_5m = creation.get("ephemeral_5m_input_tokens") or 0
        write_1h = creation.get("ephemeral_1h_input_tokens") or 0
    else:
        write_5m = usage.get("cache_creation_input_tokens") or 0
        write_1h = 0
    return {
        "input_tokens": usage.get("input_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
        "cache_read_tokens": usage.get("cache_read_input_tokens") or 0,
        "cache_write_5m_tokens": write_5m,
        "cache_write_1h_tokens": write_1h,
    }


def weighted_tokens(totals):
    """Weighted input-equivalents -- the single comparable cost number."""
    return round(sum(totals.get(f, 0) * w for f, w in COST_WEIGHTS.items()), 1)


def parse_transcript(path):
    """One transcript file -> one ledger row. Never raises on a malformed
    line: these files are appended to by a live process, so the last line
    of an in-flight session is routinely half-written, and refusing to
    report on the running cycle would make this useless exactly when it is
    most interesting."""
    totals = dict.fromkeys(TOKEN_FIELDS, 0)
    seen_messages = set()
    models = set()
    opening = ""
    first_ts = last_ts = None
    tool_calls = 0
    subagent_turns = 0

    with open(path, errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue

            stamp = record.get("timestamp")
            if isinstance(stamp, str) and stamp:
                if first_ts is None or stamp < first_ts:
                    first_ts = stamp
                if last_ts is None or stamp > last_ts:
                    last_ts = stamp

            if record.get("type") == "user" and not opening:
                opening = _first_user_text(record)

            message = record.get("message")
            if not isinstance(message, dict):
                continue
            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue

            # Tool calls are counted per block, deliberately: unlike usage,
            # each tool_use block is a distinct call, so the repeated lines
            # are the correct granularity here.
            content = message.get("content")
            if isinstance(content, list):
                tool_calls += sum(
                    1 for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                )

            message_id = message.get("id")
            # A message with no id cannot be deduplicated; count it once
            # rather than dropping real spend on the floor.
            key = message_id or f"__line__{len(seen_messages)}"
            if key in seen_messages:
                continue
            seen_messages.add(key)

            if message.get("model"):
                models.add(message["model"])
            if record.get("isSidechain"):
                subagent_turns += 1
            for field, value in _usage_totals(usage).items():
                totals[field] += value

    kind, trigger = _classify(opening)
    row = {
        "session": os.path.basename(path).replace(".jsonl", ""),
        "kind": kind,
        "trigger": trigger,
        "started_at": first_ts or "",
        "ended_at": last_ts or "",
        "duration_seconds": _duration(first_ts, last_ts),
        "turns": len(seen_messages),
        "subagent_turns": subagent_turns,
        "tool_calls": tool_calls,
        "models": sorted(models),
    }
    row.update(totals)
    row["weighted_tokens"] = weighted_tokens(totals)
    return row


def _duration(first_ts, last_ts):
    """Wall-clock seconds between the first and last record, or None.

    Note this is the span of the *transcript*, which is the turn's own
    runtime -- it does not include the runner's queueing or the pod's
    startup, so it reads a little shorter than the "finished in" chip.
    """
    if not first_ts or not last_ts:
        return None
    from datetime import datetime

    try:
        start = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
        end = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return round((end - start).total_seconds(), 1)


def scan(projects_dir=None):
    """Every transcript under the projects dir, oldest first."""
    projects_dir = projects_dir or PROJECTS_DIR
    rows = []
    for root, _dirs, files in os.walk(projects_dir):
        for name in sorted(files):
            if name.endswith(".jsonl"):
                try:
                    rows.append(parse_transcript(os.path.join(root, name)))
                except OSError:
                    continue
    rows.sort(key=lambda r: r["started_at"] or "")
    return rows


def spend_events(path):
    """One transcript -> a timestamped list of individual charges.

    `parse_transcript` collapses a session into one row, which is the right
    shape for "what did that cycle cost" and the wrong shape for "what was
    spent between 14:03 and 14:57". Calibration needs the second: it joins
    spend against quota readings taken at arbitrary instants, and a session
    routinely straddles one. Spreading a session's total evenly across its
    span would be an approximation, and the intervals where it is worst are
    the short ones -- which are exactly the ones a percentage tick pins down
    most precisely. So charges are kept at the granularity the API reported
    them: one event per assistant message, at that message's timestamp.

    Same dedup rule as `parse_transcript`, and for the same reason -- one
    message written as eight content-block lines is one charge, not eight.
    """
    events = []
    seen_messages = set()
    with open(path, errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue
            key = message.get("id") or f"__line__{len(seen_messages)}"
            if key in seen_messages:
                continue
            seen_messages.add(key)
            stamp = record.get("timestamp")
            if not isinstance(stamp, str) or not stamp:
                continue
            events.append({
                "at": stamp,
                "model": message.get("model") or "",
                "weighted_tokens": weighted_tokens(_usage_totals(usage)),
            })
    events.sort(key=lambda e: e["at"])
    return events


def scan_spend_events(projects_dir=None):
    """Every charge under the projects dir, oldest first."""
    projects_dir = projects_dir or PROJECTS_DIR
    events = []
    for root, _dirs, files in os.walk(projects_dir):
        for name in sorted(files):
            if name.endswith(".jsonl"):
                try:
                    events.extend(spend_events(os.path.join(root, name)))
                except OSError:
                    continue
    events.sort(key=lambda e: e["at"])
    return events


def summarize(rows):
    """Aggregate stats over the cycle rows only.

    Median as well as mean because the distribution is not symmetric: a
    handful of research cycles run several times the length of an ordinary
    one, and a mean alone would describe a cycle that does not exist.
    """
    cycles = [r for r in rows if r["kind"] == "cycle"]
    if not cycles:
        return {"cycles": 0, "other_sessions": len(rows)}
    weighted = [r["weighted_tokens"] for r in cycles]
    totals = {f: sum(r[f] for r in cycles) for f in TOKEN_FIELDS}
    durations = [r["duration_seconds"] for r in cycles if r["duration_seconds"]]
    grand = sum(weighted)
    return {
        "cycles": len(cycles),
        "other_sessions": len(rows) - len(cycles),
        "first_cycle": cycles[0]["started_at"],
        "last_cycle": cycles[-1]["started_at"],
        "totals": totals,
        "total_weighted": round(grand, 1),
        "mean_weighted": round(statistics.mean(weighted), 1),
        "median_weighted": round(statistics.median(weighted), 1),
        "max_weighted": max(weighted),
        "min_weighted": min(weighted),
        "mean_duration_seconds": round(statistics.mean(durations), 1) if durations else None,
        "median_duration_seconds": round(statistics.median(durations), 1) if durations else None,
        # Where the weighted cost actually goes. This is the number that
        # tells you which lever is worth pulling; the raw token counts do
        # not, because they are not on the same scale.
        "cost_share": {
            field: round(totals[field] * COST_WEIGHTS[field] / grand * 100, 1) if grand else 0.0
            for field in TOKEN_FIELDS
        },
        "models": sorted(Counter(m for r in cycles for m in r["models"])),
    }


def _fmt(n):
    return f"{n:,}" if isinstance(n, int) else ("-" if n is None else f"{n:,.0f}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    projects_dir = None
    as_json = False
    cycles_only = False
    while argv:
        arg = argv.pop(0)
        if arg == "--json":
            as_json = True
        elif arg == "--cycles":
            cycles_only = True
        elif arg == "--dir":
            projects_dir = argv.pop(0) if argv else None
        else:
            print(f"usage: python3 -m bridge.analytics [--json] [--cycles] [--dir PATH]")
            return 2

    rows = scan(projects_dir)
    # Summarize over everything even when the table is filtered -- summarize()
    # already excludes non-cycles from the averages, and passing it the
    # filtered rows would make "other_sessions" report 0 every time.
    report = {
        "sessions": [r for r in rows if r["kind"] == "cycle"] if cycles_only else rows,
        "summary": summarize(rows),
        "weights": COST_WEIGHTS,
    }

    if as_json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"{'started (UTC)':20}{'kind':8}{'turns':>6}{'tools':>6}"
          f"{'in':>9}{'out':>9}{'cacheRd':>11}{'cacheWr':>10}{'weighted':>11}")
    for row in rows:
        print(f"{row['started_at'][:19]:20}{row['kind']:8}{row['turns']:>6}{row['tool_calls']:>6}"
              f"{_fmt(row['input_tokens']):>9}{_fmt(row['output_tokens']):>9}"
              f"{_fmt(row['cache_read_tokens']):>11}"
              f"{_fmt(row['cache_write_5m_tokens'] + row['cache_write_1h_tokens']):>10}"
              f"{_fmt(row['weighted_tokens']):>11}")

    summary = report["summary"]
    print()
    if not summary.get("cycles"):
        print("no cycles found")
        return 0
    print(f"cycles: {summary['cycles']}  (plus {summary['other_sessions']} other sessions)")
    print(f"weighted per cycle: mean {_fmt(summary['mean_weighted'])}  "
          f"median {_fmt(summary['median_weighted'])}  "
          f"min {_fmt(summary['min_weighted'])}  max {_fmt(summary['max_weighted'])}")
    print(f"duration per cycle: mean {summary['mean_duration_seconds']}s  "
          f"median {summary['median_duration_seconds']}s")
    print("cost share: " + "  ".join(
        f"{f.replace('_tokens', '')} {p}%" for f, p in summary["cost_share"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
