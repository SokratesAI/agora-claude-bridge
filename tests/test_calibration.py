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


def test_a_fable_interval_is_a_sample_now_that_fable_has_a_price():
    """The inversion of the rule this module shipped with. Fable ticks used
    to be discarded because everything was priced as Opus and they scored
    ~0.9M weighted per point against ~2.6M. `analytics.py` now scales each
    event by its own model's ratio before it ever reaches here, so a Fable
    interval agrees with the Opus ones instead of reading 3x cheap -- and a
    sample that agrees is a sample worth keeping. Half the tokens at twice
    the price is the same quota point, which is the whole claim."""
    history = [
        _reading(T0, 1.0),
        _reading(T0 + HOUR, 2.0),
        _reading(T0 + 2 * HOUR, 3.0),
        _reading(T0 + 3 * HOUR, 4.0),
    ]
    events = [
        _event(_iso(T0 + 0.5 * HOUR), 2_600_000),
        _event(_iso(T0 + 1.5 * HOUR), 2_400_000),
        # 1.2M Fable tokens, already scaled by 2.0 upstream.
        _event(_iso(T0 + 2.5 * HOUR), 2_400_000, model="claude-fable-5"),
    ]
    result = calibration.calibrate(events, history)

    assert result["samples"] == 2, "only the unanchored first tick is dropped"
    fable_interval = result["intervals"][-1]
    assert fable_interval["used"] is True
    assert fable_interval["foreign_weighted"] == 0
    assert fable_interval["weighted_per_point"] == 2_400_000
    assert result["weighted_per_point"] == 2_400_000


def test_a_mixed_model_interval_is_kept_and_summed():
    """Subagents run on Sonnet and Haiku, so an interval containing more
    than one model is the normal case, not the contaminated one. Dropping
    these would mean routine subagent use disabling the instrument meant to
    measure whether subagents were worth it."""
    history = [_reading(T0, 1.0), _reading(T0 + HOUR, 2.0), _reading(T0 + 2 * HOUR, 3.0)]
    events = [
        _event(_iso(T0 + 0.5 * HOUR), 2_600_000),
        _event(_iso(T0 + 1.2 * HOUR), 2_000_000),
        _event(_iso(T0 + 1.4 * HOUR), 300_000, model="claude-sonnet-5"),
        _event(_iso(T0 + 1.5 * HOUR), 100_000, model="claude-haiku-4-5-20251001"),
    ]
    result = calibration.calibrate(events, history)

    assert result["samples"] == 1
    assert result["intervals"][-1]["used"] is True
    assert result["intervals"][-1]["priced_weighted"] == 2_400_000
    assert result["intervals"][-1]["foreign_weighted"] == 0
    assert result["weighted_per_point"] == 2_400_000


def test_an_unpriced_model_still_drops_the_interval():
    """The half of the old rule that must survive. A model with no known
    ratio cannot be converted, and pricing it as Opus by default is exactly
    the error that produced the 1.68x disagreement. So a model this table
    has not been taught yet excludes its interval and says its own name --
    the next release should announce itself, not skew the constant."""
    history = [_reading(T0, 1.0), _reading(T0 + HOUR, 2.0), _reading(T0 + 2 * HOUR, 3.0)]
    events = [
        _event(_iso(T0 + 0.5 * HOUR), 2_600_000),
        _event(_iso(T0 + 1.2 * HOUR), 2_000_000),
        _event(_iso(T0 + 1.5 * HOUR), 50_000, model="claude-something-6"),
    ]
    result = calibration.calibrate(events, history)

    assert result["samples"] == 0
    assert result["weighted_per_point"] is None
    assert result["intervals"][-1]["used"] is False
    assert result["intervals"][-1]["priced_weighted"] == 2_000_000
    assert result["intervals"][-1]["foreign_models"] == ["claude-something-6"]
    assert "claude-something-6" in result["intervals"][-1]["excluded_because"], \
        "an empty calibration must be able to say what emptied it"


def test_a_zero_cost_synthetic_turn_does_not_drop_the_interval():
    """`<synthetic>` is stamped on locally-generated turns and has no price,
    but it also has no cost. It must not be able to exclude an interval it
    contributed nothing to -- otherwise the commonest unpriced model in the
    transcripts silently eats real samples."""
    history = [_reading(T0, 1.0), _reading(T0 + HOUR, 2.0), _reading(T0 + 2 * HOUR, 3.0)]
    events = [
        _event(_iso(T0 + 0.5 * HOUR), 2_600_000),
        _event(_iso(T0 + 1.2 * HOUR), 2_400_000),
        _event(_iso(T0 + 1.5 * HOUR), 0, model="<synthetic>"),
    ]
    result = calibration.calibrate(events, history)

    assert result["samples"] == 1
    assert result["weighted_per_point"] == 2_400_000


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
