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
on the bill. One weighted unit is one Opus input token. We run on a
subscription, so this is a comparability metric and a proxy for quota
burn -- not an invoice.

A token also costs a different amount depending on which *model* spent it,
and until this was written that half was missing: everything was priced as
Opus, so every Sonnet, Haiku and Fable token was weighed wrong. It was not
a rounding error. Fable is twice Opus per token and Haiku is a fifth, so a
subagent-heavy cycle was mismeasured in both directions at once, and
`calibration.py` worked around it by discarding any quota interval that
contained non-Opus spend -- honest, and self-defeating, because subagents
are exactly the thing this loop is trying to spend more of. Cycle 61
measured the damage from the outside: the three intervals containing Fable
spend scored ~0.9M weighted per quota point against ~2.6M for the
Opus-only ones, which was the whole of a 1.68x discrepancy that had been
sitting in the journal for two cycles as an open question for Edvard.
"""
import json
import os
import statistics
import sys
from collections import Counter

from bridge.config import CLAUDE_HOME

PROJECTS_DIR = os.path.join(CLAUDE_HOME, ".claude", "projects")

# What each token class costs relative to an input token *on the same
# model*: output 5x, cache read 0.1x, 5m cache write 1.25x, 1h write 2x.
# The output multiplier is exactly 5 on every published Claude model --
# Opus $5/$25, Fable $10/$50, Sonnet $3/$15, Haiku $1/$5 per MTok -- and
# the cache multipliers are properties of the cache, not of the model. So
# the shape of the bill is model-independent and only its scale is not,
# which is why this table stays flat and MODEL_PRICE_RATIOS carries the
# rest. If a future model breaks the 5x, this splits into a real matrix.
COST_WEIGHTS = {
    "input_tokens": 1.0,
    "output_tokens": 5.0,
    "cache_read_tokens": 0.1,
    "cache_write_5m_tokens": 1.25,
    "cache_write_1h_tokens": 2.0,
}

TOKEN_FIELDS = tuple(COST_WEIGHTS)

# The scale: one input token on this model, priced in Opus input tokens.
# From published $/MTok list prices, Opus being 1.0 by definition of the
# unit. Substring match on the family name rather than an allowlist of ids,
# because ids gain date suffixes (`claude-haiku-4-5-20251001`) and admitting
# a renamed model at a known price beats silently discarding it.
#
# Two things this cannot see, both worth knowing before trusting a number
# that came out of it. Opus 5 is what we actually run and its list price is
# not published anywhere this loop can read; 1.0 is true by construction,
# but every *other* ratio here assumes Opus 5 is priced like the Opus tier
# has been ($5/MTok in). And Sonnet 5 carries an introductory $2/$10
# through 2026-08-31, which would make it 0.4 rather than 0.6 -- list price
# is used because a weight table that silently changes value on a date is
# worse than one that is consistently wrong by a knowable amount.
#
# Neither is a guess that has to stay a guess. Every interval this admits
# is now a sample, so calibration's spread across a model mix is the test:
# if a ratio is wrong, intervals containing that model drift away from the
# Opus-only ones instead of agreeing with them.
MODEL_PRICE_RATIOS = {
    "opus": 1.0,
    "fable": 2.0,
    "sonnet": 0.6,
    "haiku": 0.2,
}


def model_price_ratio(model):
    """Input-token price of `model` relative to Opus, or None if unknown.

    None rather than a 1.0 default at this boundary on purpose: an
    unrecognised model is a thing to report, not to quietly price as Opus.
    `weighted_tokens` still falls back to 1.0 so a row is never dropped,
    but `calibration.py` reads the None and excludes the interval, which is
    the behaviour that lets a newly-released model announce itself instead
    of skewing the constant for weeks.
    """
    name = (model or "").lower()
    for family, ratio in MODEL_PRICE_RATIOS.items():
        if family in name:
            return ratio
    return None


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


def weighted_tokens(totals, model=None):
    """Weighted input-equivalents -- the single comparable cost number.

    `model` scales the whole row by that model's price. Omitting it, or
    passing one no ratio is known for, prices the row as Opus: this is the
    lenient half of the boundary described on `model_price_ratio`, so that
    an unknown model still shows up in a cycle's cost instead of vanishing.
    """
    return round(sum(weighted_by_field({model: totals}).values()), 1)


def weighted_by_field(by_model):
    """{token class: weighted cost} across a {model: totals} mapping.

    The one place the two multiplications happen, so a per-class breakdown
    and a session total can never disagree about what a token cost. They did
    briefly: the total was scaled per model while `cost_share` still used the
    flat weights, which summed to 100% only because every cycle so far has
    been pure Opus. The first cycle to use a subagent would have quietly
    broken it -- and subagents are the next thing this loop plans to do.
    """
    out = dict.fromkeys(TOKEN_FIELDS, 0.0)
    for model, totals in by_model.items():
        ratio = model_price_ratio(model)
        if ratio is None:
            ratio = 1.0
        for field in TOKEN_FIELDS:
            out[field] += totals.get(field, 0) * COST_WEIGHTS[field] * ratio
    return out


def parse_transcript(path):
    """One transcript file -> one ledger row. Never raises on a malformed
    line: these files are appended to by a live process, so the last line
    of an in-flight session is routinely half-written, and refusing to
    report on the running cycle would make this useless exactly when it is
    most interesting."""
    totals = dict.fromkeys(TOKEN_FIELDS, 0)
    # The same counters again, split by model. `totals` stays whole because
    # the raw counts are model-independent and are what the ledger columns
    # report; only the weighting needs the split, and a session routinely
    # spans models now that subagents run on Sonnet and Haiku.
    by_model = {}
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

            model = message.get("model") or ""
            if model:
                models.add(model)
            if record.get("isSidechain"):
                subagent_turns += 1
            bucket = by_model.setdefault(model, dict.fromkeys(TOKEN_FIELDS, 0))
            for field, value in _usage_totals(usage).items():
                totals[field] += value
                bucket[field] += value

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
    # Kept on the row because `summarize` cannot re-derive it: the ledger
    # columns are raw counts and the model split is gone by then.
    row["weighted_by_field"] = {
        f: round(v, 1) for f, v in weighted_by_field(by_model).items()}
    row["weighted_tokens"] = round(sum(row["weighted_by_field"].values()), 1)
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
            model = message.get("model") or ""
            events.append({
                "at": stamp,
                "model": model,
                "weighted_tokens": weighted_tokens(_usage_totals(usage), model),
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
    weighted_totals = {
        f: sum(r.get("weighted_by_field", {}).get(f, 0) for r in cycles)
        for f in TOKEN_FIELDS
    }
    durations = [r["duration_seconds"] for r in cycles if r["duration_seconds"]]
    grand = sum(weighted)
    return {
        "cycles": len(cycles),
        "other_sessions": len(rows) - len(cycles),
        "first_cycle": cycles[0]["started_at"],
        "last_cycle": cycles[-1]["started_at"],
        "totals": totals,
        # The same five counters after weighting -- raw counts are not
        # comparable to each other, so this is the one that answers "where
        # does the money go" rather than "how many tokens moved".
        "totals_weighted": {f: round(v, 1) for f, v in weighted_totals.items()},
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
            field: round(weighted_totals[field] / grand * 100, 1) if grand else 0.0
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
