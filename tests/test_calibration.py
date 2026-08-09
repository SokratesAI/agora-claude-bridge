import json

from bridge import analytics, calibration


def _event(at, weighted, model="claude-opus-5"):
    """A spend event already carrying its epoch, as `calibrate` normalizes to."""
    return {"at": at, "model": model, "weighted_tokens": weighted}


# 2026-08-08T22:00:00Z onwards, one hour apart, as epoch seconds.
T0 = 1786312800.0
HOUR = 3600.0


def _reading(at, seven_day):
    return {"at": at, "seven_day": seven_day}


def _iso(epoch):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def test_a_tick_interval_spans_the_whole_point_not_the_poll_gap():
    """The reason this method beats dividing totals. The counter reads 5 at
    T0, still 5 at T0+1h, and 6 at T0+2h -- so the point took two hours to
    spend, not the ten minutes between the last two polls. Anchoring on the
    poll before the tick would attribute one point to one poll gap and
    inflate the conversion factor by however often the poller runs."""
    history = [
        _reading(T0, 5.0),
        _reading(T0 + HOUR, 5.0),
        _reading(T0 + 2 * HOUR, 6.0),
    ]
    (start, end, points, is_first), = calibration.ticks(history)
    assert start == T0
    assert end == T0 + 2 * HOUR
    assert points == 1.0
    assert is_first is True


def test_a_window_reset_breaks_the_anchor():
    """Utilization dropping means the window rolled over. Spend either side
    belongs to different budgets, so the interval that straddles the reset
    must not become a sample -- it would pair post-reset spend with a
    pre-reset anchor and read as absurdly cheap."""
    history = [
        _reading(T0, 90.0),
        _reading(T0 + HOUR, 2.0),          # reset
        _reading(T0 + 2 * HOUR, 3.0),
    ]
    result = calibration.ticks(history)
    assert len(result) == 1
    start, end, points, is_first = result[0]
    assert start == T0 + HOUR, "anchor must restart at the reset, not before it"
    assert points == 1.0
    assert is_first is True, "the first tick after a reset is unanchored too"


def test_fable_intervals_are_excluded_rather_than_averaged_in():
    """The bug this module was written for. COST_WEIGHTS are Opus price
    ratios, so an interval whose spend came from another model is not
    comparable -- measured live, Fable ticks scored ~0.9M weighted per point
    against ~2.6M for Opus. Averaging the two produced the 1.68x
    disagreement that stood as an open question for two cycles."""
    history = [
        _reading(T0, 1.0),
        _reading(T0 + HOUR, 2.0),
        _reading(T0 + 2 * HOUR, 3.0),
        _reading(T0 + 3 * HOUR, 4.0),
    ]
    events = [
        _event(_iso(T0 + 0.5 * HOUR), 2_600_000),
        _event(_iso(T0 + 1.5 * HOUR), 2_400_000),
        _event(_iso(T0 + 2.5 * HOUR), 900_000, model="claude-fable-5"),
    ]
    result = calibration.calibrate(events, history)

    assert result["samples"] == 1, "first tick unanchored, third tick is Fable"
    assert result["weighted_per_point"] == 2_400_000
    fable_interval = result["intervals"][-1]
    assert fable_interval["used"] is False
    assert fable_interval["foreign_weighted"] == 900_000
    assert fable_interval["foreign_models"] == ["claude-fable-5"]
    assert "claude-fable-5" in fable_interval["excluded_because"], \
        "an empty calibration must be able to say what emptied it"


def test_a_mixed_interval_is_dropped_even_though_it_has_priced_spend():
    """A Fable call landing in the middle of an otherwise-Opus interval
    contaminates the whole interval: the point was paid for partly at
    prices these weights do not describe. Keeping the Opus half would
    understate the constant, which reads as 'cycles are cheaper than they
    are' -- the dangerous direction for a budget."""
    history = [_reading(T0, 1.0), _reading(T0 + HOUR, 2.0), _reading(T0 + 2 * HOUR, 3.0)]
    events = [
        _event(_iso(T0 + 0.5 * HOUR), 2_600_000),
        _event(_iso(T0 + 1.2 * HOUR), 2_000_000),
        _event(_iso(T0 + 1.5 * HOUR), 50_000, model="claude-fable-5"),
    ]
    result = calibration.calibrate(events, history)
    assert result["samples"] == 0
    assert result["weighted_per_point"] is None
    assert result["intervals"][-1]["priced_weighted"] == 2_000_000
    assert result["intervals"][-1]["used"] is False


def test_spend_is_attributed_by_message_timestamp_not_spread_over_a_session():
    """Charges are placed at the instant the API reported them. A long
    session straddling a tick contributes only the messages on each side --
    spreading its total evenly would smear cost across a boundary and is
    worst on short intervals, which are the sharpest samples."""
    history = [_reading(T0, 1.0), _reading(T0 + HOUR, 2.0), _reading(T0 + 2 * HOUR, 3.0)]
    events = [
        _event(_iso(T0 + 0.1 * HOUR), 1_000_000),
        _event(_iso(T0 + 0.9 * HOUR), 1_000_000),
        _event(_iso(T0 + 1.1 * HOUR), 3_000_000),
    ]
    result = calibration.calibrate(events, history)
    assert result["intervals"][0]["priced_weighted"] == 2_000_000  # excluded, but measured
    assert result["intervals"][1]["priced_weighted"] == 3_000_000
    assert result["weighted_per_point"] == 3_000_000


def test_projection_carries_the_spread_and_does_not_invert_it():
    """More tokens per point means a cheaper cycle, so the *max* sample
    yields the *low* projection. Getting this backwards would report the
    optimistic end as the worst case."""
    result = calibration.project(
        1_500_000,
        {"weighted_per_point": 2_500_000,
         "min_weighted_per_point": 2_000_000,
         "max_weighted_per_point": 3_000_000},
        cycles_per_day=24,
    )
    assert result["cycles_per_window"] == 168
    assert result["percent_of_window_per_cycle"] == 0.6
    assert result["percent_of_window"] == 100.8
    assert result["percent_of_window_low"] == 84.0
    assert result["percent_of_window_high"] == 126.0
    assert result["percent_of_window_low"] < result["percent_of_window"] \
        < result["percent_of_window_high"]


def test_an_interval_with_no_transcript_spend_is_reported_not_counted():
    """Quota moving with nothing on the PVC to explain it means spend from
    somewhere this loop cannot see. Dividing by it would produce a
    conversion factor of zero and poison the median."""
    history = [_reading(T0, 1.0), _reading(T0 + HOUR, 2.0), _reading(T0 + 2 * HOUR, 3.0)]
    events = [_event(_iso(T0 + 0.5 * HOUR), 2_600_000)]
    result = calibration.calibrate(events, history)
    assert result["samples"] == 0
    empty = result["intervals"][1]
    assert empty["priced_weighted"] == 0
    assert "no transcript spend" in empty["excluded_because"]


def test_calibration_survives_an_empty_history():
    assert calibration.calibrate([], [])["weighted_per_point"] is None
    assert calibration.project(1_000_000, {"weighted_per_point": None}, 24) is None


def test_spend_events_dedup_by_message_id_like_the_row_parser(tmp_path):
    """Same trap as parse_transcript: one message written as several
    content-block lines is one charge. A per-line sum inflated a real cycle
    by 1.79x, and calibration divides by these numbers."""
    usage = {"input_tokens": 100, "output_tokens": 20,
             "cache_read_input_tokens": 5000,
             "cache_creation": {"ephemeral_5m_input_tokens": 0,
                                "ephemeral_1h_input_tokens": 100}}
    records = [
        {"type": "assistant", "timestamp": "2026-08-08T10:00:00.000Z",
         "message": {"id": "m1", "model": "claude-opus-5", "usage": usage,
                     "content": [{"type": "text", "text": "hi"}]}}
        for _ in range(3)
    ]
    records.append(
        {"type": "assistant", "timestamp": "2026-08-08T10:01:00.000Z",
         "message": {"id": "m2", "model": "claude-opus-5", "usage": usage,
                     "content": [{"type": "text", "text": "again"}]}})
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    events = analytics.spend_events(str(path))
    assert [e["at"] for e in events] == [
        "2026-08-08T10:00:00.000Z", "2026-08-08T10:01:00.000Z"]
    expected = analytics.weighted_tokens({
        "input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 5000,
        "cache_write_5m_tokens": 0, "cache_write_1h_tokens": 100})
    assert all(e["weighted_tokens"] == expected for e in events)


def test_spend_events_totals_match_the_row_parser(tmp_path):
    """The two readers must agree, or a calibration interval covering a whole
    session would disagree with that session's own cost.

    Not exactly, though: `weighted_tokens` rounds to one decimal, and
    per-message rounding accumulates where per-session rounding does not. The
    error is bounded by 0.05 per message -- a few tokens across a whole cycle,
    against a conversion factor in the millions.
    """
    usage = {"input_tokens": 7, "output_tokens": 11,
             "cache_read_input_tokens": 1234,
             "cache_creation": {"ephemeral_5m_input_tokens": 5,
                                "ephemeral_1h_input_tokens": 9}}
    records = [
        {"type": "user", "timestamp": "2026-08-08T09:59:00.000Z",
         "message": {"role": "user", "content": "You are Nova"}},
        {"type": "assistant", "timestamp": "2026-08-08T10:00:00.000Z",
         "message": {"id": "a", "model": "claude-opus-5", "usage": usage,
                     "content": [{"type": "text", "text": "x"}]}},
        {"type": "assistant", "timestamp": "2026-08-08T10:02:00.000Z",
         "message": {"id": "b", "model": "claude-opus-5", "usage": usage,
                     "content": [{"type": "text", "text": "y"}]}},
    ]
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    row = analytics.parse_transcript(str(path))
    events = analytics.spend_events(str(path))
    assert len(events) == 2
    assert abs(sum(e["weighted_tokens"] for e in events) - row["weighted_tokens"]) \
        <= 0.05 * len(events)


def test_read_history_skips_malformed_lines(tmp_path):
    """The file is appended to by a live process, so the last line is
    routinely half-written."""
    path = tmp_path / "quota-history.jsonl"
    path.write_text(
        json.dumps({"at": T0, "seven_day": 1.0}) + "\n"
        + "{not json\n"
        + json.dumps({"seven_day": 2.0}) + "\n"          # no timestamp
        + json.dumps({"at": T0 + HOUR, "seven_day": 2.0}) + "\n"
        + '{"at": 17863'                                  # truncated tail
    )
    rows = calibration.read_history(str(path))
    assert [r["at"] for r in rows] == [T0, T0 + HOUR]
