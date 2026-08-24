"""What a cycle costs as a share of the subscription window, not as tokens.

The owner, 2026-08-09, on the cycle-rate arithmetic a previous cycle had put
in front of him: "I think you are a bit too confident with your
calculations. You do not have 24 hours worth of data yet and a couple of
cycles today did not even go through AND we experiments with fable 5 which
uses 3.5x the tokens. ... Your calculations should be treated as hypothesis
and are tested."

He was right, and this module exists because the correction is mechanical
rather than attitudinal. `analytics.py` measures spend exactly, in weighted
tokens. `quota.py` measures the budget exactly, in integer percent. Nothing
joined them, so every cycle that wanted "can we afford this cadence" divided
a whole-window percentage by a whole-window token total and got an answer
whose error bars nobody could state. Two such estimates disagreed by 1.68x
and the standing explanation was "maybe the plan was upgraded".

It was not the plan. Most of the gap was Fable: three of the twelve observed
ticks happened during a Fable experiment, and everything was being priced as
Opus, so those intervals scored ~0.9M weighted per point against ~2.6M for
the Opus-only ones. Averaging them together is the whole error.

The first fix here was to drop any interval containing non-Opus spend. That
was right while Opus was the only model with a known price and wrong as a
resting state, because subagents run on Sonnet and Haiku -- so routine
subagent use would have disabled the very instrument meant to measure
whether subagents were worth it. `analytics.py` now knows all four models'
prices, and this module excludes only what it genuinely cannot price: a
model it has never heard of. An interval is a sample if every charge inside
it has a known ratio, whatever the mix.

THE METHOD, and why it beats dividing totals. The usage percentage is an
integer, so a single reading of "14%" means somewhere in 13.5-14.5 -- a 7%
error that no amount of token precision removes. But the *tick* from N to
N+1 is sharp. Take the instant the counter became N and the instant it
became N+1, sum the real spend between them, and that sum is one point's
worth, with the quantization cancelling instead of accumulating. Each tick
is an independent measurement, so a dozen of them give a spread rather than
a point estimate, which is what "treated as hypothesis and are tested"
actually requires.

What it still cannot do: the boundaries are known only to within one poll
gap (~2-3 minutes at each end of a ~90-minute interval, so a few percent),
and it assumes every charge against the subscription appears in a transcript
on this PVC. An interactive session elsewhere on the same account would show
up as quota spent with no tokens to explain it, which inflates the estimate
-- i.e. errs toward "cycles cost more than they do", which is the safe
direction for a budget.
"""
import json
import os
import statistics
import sys
from datetime import datetime, timezone

from bridge.analytics import (
    COST_WEIGHTS,
    model_price_ratio,
    scan,
    scan_spend_events,
    summarize,
)
from bridge.config import CLAUDE_HOME

HISTORY_FILE = os.path.join(CLAUDE_HOME, "quota-history.jsonl")


def read_history(path=None):
    """The quota history as a list of rows, oldest first. Malformed lines are
    skipped -- the file is appended to by a live process."""
    path = path or HISTORY_FILE
    rows = []
    try:
        with open(path, errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and isinstance(row.get("at"), (int, float)):
                    rows.append(row)
    except OSError:
        return []
    rows.sort(key=lambda r: r["at"])
    return rows


def ticks(history, window="seven_day"):
    """Every observed increase in a window's utilization, as
    (started_at, ticked_at, points, is_first).

    `started_at` is when the counter was last seen to *change* into its
    pre-tick value, not merely the previous poll -- that is what makes the
    interval span one whole point rather than one poll gap. A decrease means
    the window reset, which restarts the anchor: the spend either side of a
    reset belongs to different budgets.

    The earliest tick is flagged rather than dropped here, because "when did
    the counter become N" is unknowable for a value the history file opens
    on. Callers decide; `calibrate` excludes it.
    """
    out = []
    anchor = None
    first_seen = True
    for row in history:
        value = row.get(window)
        if value is None:
            continue
        if anchor is None:
            anchor = (row["at"], value)
            continue
        if value > anchor[1]:
            out.append((anchor[0], row["at"], value - anchor[1], first_seen))
            first_seen = False
            anchor = (row["at"], value)
        elif value < anchor[1]:
            # Window reset. Nothing before it is comparable to anything after.
            anchor = (row["at"], value)
            first_seen = True
        # value == anchor[1]: still inside the same point, keep the anchor.
    return out


# How much of an interval's length is allowed to be boundary uncertainty
# before it stops being a measurement. Both ends of a tick are known only to
# within one poll gap, so the error on the length is up to two of them; at
# 0.2 an interval must be at least ten poll gaps long.
#
# This is a limit with a measured danger behind it, which is the only kind
# worth having. Live on 2026-08-09 the poller ran every 120.6s (median of
# 175 gaps), making the floor ~20 minutes -- and the two intervals it
# excludes are 3 and 4 minutes long, where the uncertainty is larger than
# the interval. They scored 174k and 1.7M weighted per point against a
# ~2.3M median, and they widened the projected cost of a week from
# 82-1481%. The next shortest real interval is 61 minutes, so this is
# nowhere near a knife edge.
#
# They only became visible when per-model pricing stopped discarding them
# for containing Fable: the model filter had been doing this job by
# accident, and nothing would have noticed when it stopped.
MAX_BOUNDARY_ERROR = 0.2


def poll_gap(history):
    """Median seconds between quota readings, or None if under two rows.

    Measured rather than assumed: the poller's cadence has changed before,
    and a threshold derived from a stale constant would silently stop
    matching the data it is meant to protect.
    """
    gaps = [
        history[i + 1]["at"] - history[i]["at"]
        for i in range(len(history) - 1)
        if history[i + 1]["at"] > history[i]["at"]
    ]
    return statistics.median(gaps) if gaps else None


def _epoch(stamp):
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def spend_in(events, start, end):
    """(priced, foreign, foreign_models) weighted spend charged in [start, end).

    Split by whether the model's price is known. `analytics.py` has already
    scaled each event by its own model's ratio, so the priced half is
    directly addable across a mix of Opus, Fable, Sonnet and Haiku -- that is
    the point of per-model weights and the reason a subagent-heavy interval
    is now a usable sample instead of a discarded one.

    Foreign is what is left: a model with no known ratio. It is not
    converted, because inventing a ratio is exactly the error this module
    exists to have caught. In practice it is the `<synthetic>` model the CLI
    stamps on locally-generated turns, which costs zero and so never
    contaminates an interval, plus anything Anthropic releases that this
    table has not been taught yet. The models are named in the return rather
    than reduced to a boolean so an empty calibration can say what emptied
    it -- which, the next time a model is renamed, is the sentence that
    turns a mysterious constant into a one-line fix.
    """
    priced = foreign = 0.0
    foreign_models = set()
    for event in events:
        at = event.get("_at") if event.get("_at") is not None else _epoch(event.get("at"))
        if at is None or at < start or at >= end:
            continue
        if model_price_ratio(event.get("model")) is not None:
            priced += event["weighted_tokens"]
        elif event["weighted_tokens"]:
            foreign += event["weighted_tokens"]
            foreign_models.add(event.get("model") or "unknown")
    return priced, foreign, sorted(foreign_models)


def calibrate(events, history, window="seven_day"):
    """Weighted tokens per one percentage point of a quota window.

    Returns every interval with the reason it was kept or dropped, not just
    the answer. A calibration constant with no visible sample behind it is
    how the 1.68x disagreement survived as long as it did.
    """
    events = sorted(
        ({**e, "_at": _epoch(e.get("at"))} for e in events),
        key=lambda e: (e["_at"] is None, e["_at"] or 0),
    )
    gap = poll_gap(history)
    min_seconds = 2 * gap / MAX_BOUNDARY_ERROR if gap else 0
    intervals = []
    for start, end, points, is_first in ticks(history, window):
        priced, foreign, foreign_models = spend_in(events, start, end)
        if is_first:
            reason = "first tick in history: no known start"
        elif foreign > 0:
            reason = "no known price for %s, which also spent here" % (
                ", ".join(foreign_models),)
        elif end - start < min_seconds:
            reason = "%.0fm interval is under %.0fm: too short to time against a %.0fs poll" % (
                (end - start) / 60, min_seconds / 60, gap)
        elif priced <= 0:
            reason = "no transcript spend found in the interval"
        else:
            reason = None
        intervals.append({
            "started_at": start,
            "ticked_at": end,
            "points": points,
            "gap_seconds": round(end - start, 1),
            "priced_weighted": round(priced, 1),
            "foreign_weighted": round(foreign, 1),
            "foreign_models": foreign_models,
            "weighted_per_point": round(priced / points, 1) if points and priced > 0 else None,
            "used": reason is None,
            "excluded_because": reason,
        })

    samples = [i["weighted_per_point"] for i in intervals if i["used"]]
    result = {
        "window": window,
        "intervals": intervals,
        "samples": len(samples),
        "weighted_per_point": round(statistics.median(samples), 1) if samples else None,
        "min_weighted_per_point": round(min(samples), 1) if samples else None,
        "max_weighted_per_point": round(max(samples), 1) if samples else None,
    }
    # A single sample has no spread to report, and stdev() raises on one point.
    result["stdev_weighted_per_point"] = (
        round(statistics.stdev(samples), 1) if len(samples) > 1 else None
    )
    return result


def project(mean_weighted_per_cycle, calibration, cycles_per_day, window_days=7):
    """What a cadence costs as a share of the window, with the spread carried
    through rather than collapsed.

    The three numbers come from the median, max and min weighted-per-point
    samples. Note the inversion: the *most* tokens per point is the
    *cheapest* cycle, so `max_weighted_per_point` produces the low estimate.
    Reporting one number here would be the exact mistake this module exists
    to fix.
    """
    per_point = calibration.get("weighted_per_point")
    if not per_point or not mean_weighted_per_cycle:
        return None
    total_cycles = cycles_per_day * window_days

    def share(k):
        return round(mean_weighted_per_cycle / k * total_cycles, 2) if k else None

    return {
        "cycles_per_day": cycles_per_day,
        "cycles_per_window": total_cycles,
        "percent_of_window_per_cycle": round(mean_weighted_per_cycle / per_point, 3),
        "percent_of_window": share(per_point),
        "percent_of_window_low": share(calibration.get("max_weighted_per_point")),
        "percent_of_window_high": share(calibration.get("min_weighted_per_point")),
    }


def _fmt_time(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%m-%d %H:%M")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = False
    window = "seven_day"
    recent = 20
    while argv:
        arg = argv.pop(0)
        if arg == "--json":
            as_json = True
        elif arg == "--window":
            window = argv.pop(0) if argv else window
        elif arg == "--recent":
            recent = int(argv.pop(0)) if argv else recent
        else:
            print("usage: python3 -m bridge.calibration [--json] "
                  "[--window NAME] [--recent N]")
            return 2

    rows = scan()
    cycles = [r for r in rows if r["kind"] == "cycle"]
    calibration = calibrate(scan_spend_events(), read_history(), window)

    # Recent cycles rather than all of them: the loop's per-cycle cost roughly
    # doubled over its first four days as the constitution and journal grew,
    # so a lifetime mean describes a cheaper cycle than the one about to run.
    sample = [c["weighted_tokens"] for c in cycles[-recent:]]
    mean_recent = statistics.mean(sample) if sample else 0
    report = {
        "calibration": calibration,
        "summary": summarize(rows),
        "recent_cycles": len(sample),
        "mean_weighted_recent": round(mean_recent, 1),
        "projections": [
            p for p in (project(mean_recent, calibration, n) for n in (12, 20, 24))
            if p
        ],
        "weights": COST_WEIGHTS,
    }

    if as_json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"calibrating '{window}' against {len(calibration['intervals'])} observed ticks\n")
    print(f"{'interval (UTC)':22}{'gap':>7}{'pts':>5}{'priced w':>13}{'foreign w':>12}"
          f"{'w/point':>13}  note")
    for i in calibration["intervals"]:
        per = f"{i['weighted_per_point']:,.0f}" if i["weighted_per_point"] else "-"
        label = f"{_fmt_time(i['started_at'])} -> {_fmt_time(i['ticked_at'])[6:]}"
        print(f"{label:22}{i['gap_seconds'] / 60:>6.0f}m{i['points']:>5.0f}"
              f"{i['priced_weighted']:>13,.0f}{i['foreign_weighted']:>12,.0f}{per:>13}"
              f"  {i['excluded_because'] or ''}")

    if not calibration["samples"]:
        print("\nno usable intervals yet -- needs more quota history")
        return 0

    print(f"\n1 point of '{window}' = {calibration['weighted_per_point']:,.0f} weighted "
          f"(median of {calibration['samples']}; "
          f"{calibration['min_weighted_per_point']:,.0f}-"
          f"{calibration['max_weighted_per_point']:,.0f})")
    print(f"mean cycle over the last {report['recent_cycles']}: "
          f"{report['mean_weighted_recent']:,.0f} weighted")
    for p in report["projections"]:
        print(f"  {p['cycles_per_day']}/day: a cycle is "
              f"{p['percent_of_window_per_cycle']:.2f}% of the window, "
              f"{p['cycles_per_window']} cycles = {p['percent_of_window']:.0f}% "
              f"of it (range {p['percent_of_window_low']:.0f}-"
              f"{p['percent_of_window_high']:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
