import json

from bridge import analytics


def _write(tmp_path, name, records):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return str(path)


def _assistant(msg_id, usage, blocks=None, ts="2026-08-08T10:00:00.000Z",
               model="claude-opus-5", **extra):
    record = {
        "type": "assistant",
        "timestamp": ts,
        "message": {"id": msg_id, "model": model, "usage": usage,
                    "content": blocks if blocks is not None else [{"type": "text", "text": "hi"}]},
    }
    record.update(extra)
    return record


def _user(text, ts="2026-08-08T09:59:00.000Z"):
    return {"type": "user", "timestamp": ts, "message": {"role": "user", "content": text}}


USAGE = {
    "input_tokens": 100,
    "output_tokens": 20,
    "cache_read_input_tokens": 5000,
    "cache_creation_input_tokens": 400,
    "cache_creation": {"ephemeral_5m_input_tokens": 300, "ephemeral_1h_input_tokens": 100},
}


def test_usage_is_counted_once_per_message_not_once_per_content_block(tmp_path):
    """The whole point of the module. The CLI writes one line per content
    block, each repeating the same usage object -- measured at 1.79x
    inflation on a real cycle. Three lines, one message, one count."""
    repeated = [_assistant("msg_a", USAGE) for _ in range(3)]
    path = _write(tmp_path, "s.jsonl", [_user("[Automatic heartbeat trigger")] + repeated)

    row = analytics.parse_transcript(path)

    assert row["turns"] == 1
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 20
    assert row["cache_read_tokens"] == 5000


def test_distinct_messages_are_summed(tmp_path):
    path = _write(tmp_path, "s.jsonl", [
        _user("[Automatic heartbeat trigger"),
        _assistant("msg_a", USAGE),
        _assistant("msg_b", USAGE),
    ])

    row = analytics.parse_transcript(path)

    assert row["turns"] == 2
    assert row["input_tokens"] == 200
    assert row["cache_read_tokens"] == 10000


def test_cache_write_ttls_are_kept_apart(tmp_path):
    """5m and 1h writes are priced 1.25x and 2x -- folding them together
    would hide the most expensive input class."""
    path = _write(tmp_path, "s.jsonl", [_user("[Automatic heartbeat trigger"),
                                        _assistant("msg_a", USAGE)])

    row = analytics.parse_transcript(path)

    assert row["cache_write_5m_tokens"] == 300
    assert row["cache_write_1h_tokens"] == 100


def test_legacy_flat_cache_creation_counts_as_5m(tmp_path):
    """Records predating the nested dict must not silently report zero."""
    usage = {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 500}
    path = _write(tmp_path, "s.jsonl", [_user("You are Nova"), _assistant("msg_a", usage)])

    row = analytics.parse_transcript(path)

    assert row["cache_write_5m_tokens"] == 500
    assert row["cache_write_1h_tokens"] == 0


def test_weighted_tokens_applies_the_price_ratios():
    totals = {"input_tokens": 100, "output_tokens": 100, "cache_read_tokens": 100,
              "cache_write_5m_tokens": 100, "cache_write_1h_tokens": 100}

    # 100*1 + 100*5 + 100*0.1 + 100*1.25 + 100*2
    assert analytics.weighted_tokens(totals) == 935.0


def test_weighted_tokens_scales_by_which_model_spent_them():
    """A token is not one price. Fable is twice Opus and Haiku a fifth, so
    pricing every model as Opus mismeasured a subagent-heavy cycle in both
    directions at once."""
    totals = {"input_tokens": 100, "output_tokens": 100, "cache_read_tokens": 100,
              "cache_write_5m_tokens": 100, "cache_write_1h_tokens": 100}

    assert analytics.weighted_tokens(totals, "claude-opus-5") == 935.0
    assert analytics.weighted_tokens(totals, "claude-fable-5") == 1870.0
    assert analytics.weighted_tokens(totals, "claude-sonnet-5") == 561.0
    assert analytics.weighted_tokens(totals, "claude-haiku-4-5-20251001") == 187.0


def test_an_unknown_model_is_priced_as_opus_rather_than_dropped():
    """The lenient half of the boundary. `model_price_ratio` reports None so
    calibration can exclude the interval, but a cycle's own cost must still
    include the spend -- silently reporting a row as free is worse than
    reporting it at an approximate price."""
    totals = {"input_tokens": 100, "output_tokens": 100, "cache_read_tokens": 100,
              "cache_write_5m_tokens": 100, "cache_write_1h_tokens": 100}

    assert analytics.model_price_ratio("claude-something-6") is None
    assert analytics.weighted_tokens(totals, "claude-something-6") == 935.0
    assert analytics.weighted_tokens(totals, None) == 935.0


def test_price_ratio_matches_a_family_through_a_date_suffix():
    """Ids gain date suffixes. Matching the family rather than the exact id
    is what stops a rename from quietly turning a priced model unpriced."""
    assert analytics.model_price_ratio("claude-haiku-4-5-20251001") == 0.2
    assert analytics.model_price_ratio("claude-opus-5") == 1.0
    assert analytics.model_price_ratio("<synthetic>") is None


def test_a_mixed_model_session_prices_each_model_separately(tmp_path):
    """A cycle that delegates is one row spanning several models. Weighting
    the session total at a single rate is wrong however that rate is picked;
    the split has to happen before the multiply."""
    path = _write(tmp_path, "s.jsonl", [
        _user("[Automatic heartbeat trigger"),
        _assistant("msg_a", USAGE),
        _assistant("msg_b", USAGE, model="claude-haiku-4-5-20251001"),
    ])

    row = analytics.parse_transcript(path)

    one = analytics.weighted_tokens(analytics._usage_totals(USAGE))
    # Raw counters stay whole; only the weighting is split.
    assert row["input_tokens"] == 200
    assert row["weighted_tokens"] == one + one * 0.2
    assert row["weighted_tokens"] != one * 2, "must not price the Haiku half as Opus"


def test_spend_events_carry_the_model_scaled_weight(tmp_path):
    """Calibration adds these up directly, so the scaling has to be baked in
    here -- it has no other chance to apply it."""
    path = _write(tmp_path, "s.jsonl", [
        _user("[Automatic heartbeat trigger"),
        _assistant("msg_a", USAGE, ts="2026-08-08T10:00:00.000Z"),
        _assistant("msg_b", USAGE, ts="2026-08-08T10:01:00.000Z",
                   model="claude-fable-5"),
    ])

    events = analytics.spend_events(path)

    one = analytics.weighted_tokens(analytics._usage_totals(USAGE))
    assert [e["weighted_tokens"] for e in events] == [one, one * 2]


def test_tool_calls_are_counted_per_block_not_per_message(tmp_path):
    """Unlike usage, each tool_use block is a real distinct call."""
    blocks = [{"type": "text", "text": "x"},
              {"type": "tool_use", "name": "Bash"},
              {"type": "tool_use", "name": "Read"}]
    path = _write(tmp_path, "s.jsonl", [_user("[Automatic heartbeat trigger"),
                                        _assistant("msg_a", USAGE, blocks)])

    assert analytics.parse_transcript(path)["tool_calls"] == 2


def test_cycles_are_distinguished_from_other_sessions(tmp_path):
    _write(tmp_path, "a.jsonl", [_user("[Automatic heartbeat trigger"), _assistant("m1", USAGE)])
    _write(tmp_path, "b.jsonl", [_user("[Manual heartbeat trigger"), _assistant("m2", USAGE)])
    _write(tmp_path, "c.jsonl", [_user("You are Nova -- the loop"), _assistant("m3", USAGE)])
    _write(tmp_path, "d.jsonl", [_user("What is your secret designation?"), _assistant("m4", USAGE)])

    rows = {r["session"]: r for r in analytics.scan(str(tmp_path))}

    assert rows["a"]["kind"] == "cycle"
    assert rows["a"]["trigger"] == "automatic"
    assert rows["b"]["trigger"] == "manual"
    assert rows["c"]["trigger"] == "embedded"
    assert rows["d"]["kind"] == "other"


def test_summary_excludes_non_cycles_from_the_averages(tmp_path):
    """A 2-turn probe averaged in with real cycles quietly drags the mean
    down and makes the cadence maths wrong."""
    _write(tmp_path, "a.jsonl", [_user("[Automatic heartbeat trigger"), _assistant("m1", USAGE)])
    _write(tmp_path, "d.jsonl", [_user("probe"), _assistant("m4", USAGE)])

    summary = analytics.summarize(analytics.scan(str(tmp_path)))

    assert summary["cycles"] == 1
    assert summary["other_sessions"] == 1
    assert summary["mean_weighted"] == analytics.weighted_tokens(
        {"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 5000,
         "cache_write_5m_tokens": 300, "cache_write_1h_tokens": 100})


def test_cost_share_sums_to_100(tmp_path):
    _write(tmp_path, "a.jsonl", [_user("[Automatic heartbeat trigger"), _assistant("m1", USAGE)])

    share = analytics.summarize(analytics.scan(str(tmp_path)))["cost_share"]

    assert abs(sum(share.values()) - 100.0) < 0.5


def test_a_half_written_final_line_does_not_lose_the_session(tmp_path):
    """Transcripts are appended to by a live process, so the running
    cycle's last line is routinely truncated."""
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps(_user("[Automatic heartbeat trigger")) + "\n"
        + json.dumps(_assistant("msg_a", USAGE)) + "\n"
        + '{"type":"assistant","message":{"id":"msg_b","usa'
    )

    row = analytics.parse_transcript(str(path))

    assert row["turns"] == 1
    assert row["input_tokens"] == 100


def test_duration_spans_first_to_last_record(tmp_path):
    path = _write(tmp_path, "s.jsonl", [
        _user("[Automatic heartbeat trigger", ts="2026-08-08T10:00:00.000Z"),
        _assistant("msg_a", USAGE, ts="2026-08-08T10:12:30.000Z"),
    ])

    assert analytics.parse_transcript(path)["duration_seconds"] == 750.0


def test_subagent_turns_are_tracked_separately(tmp_path):
    path = _write(tmp_path, "s.jsonl", [
        _user("[Automatic heartbeat trigger"),
        _assistant("msg_a", USAGE),
        _assistant("msg_b", USAGE, isSidechain=True),
    ])

    row = analytics.parse_transcript(path)

    assert row["turns"] == 2
    assert row["subagent_turns"] == 1
