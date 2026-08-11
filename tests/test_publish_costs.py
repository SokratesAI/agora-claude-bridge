"""The cost record that goes into the vault, where the site can actually read it."""
import json

import pytest

from bridge import publish_costs


def _history(tmp_path, lines):
    path = tmp_path / "quota-history.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_quota_history_keeps_the_fields_a_chart_reads(tmp_path):
    path = _history(tmp_path, [json.dumps({
        "at": 1786442419.698, "boundary": "start",
        "five_hour": 24.0, "five_hour_pace": 0.653,
        "five_hour_resets_at": "2026-08-11T13:09:59Z",
        "seven_day": 47.0, "seven_day_pace": 0.576,
        "weekly_scoped:Fable": 5.0,
    })])
    rows = publish_costs.read_quota_history(path)
    assert rows == [{
        "at": 1786442419.698, "five_hour": 24.0, "five_hour_pace": 0.653,
        "seven_day": 47.0, "seven_day_pace": 0.576, "boundary": "start",
    }]


def test_a_truncated_last_line_does_not_lose_the_rest_of_the_week(tmp_path):
    """The poller is killed mid-write every time the pod drains, so a half
    line is a normal state of this file rather than corruption."""
    path = _history(tmp_path, [
        json.dumps({"at": 1.0, "seven_day": 10.0}),
        json.dumps({"at": 2.0, "seven_day": 11.0}),
        '{"at": 3.0, "seven_da',
    ])
    rows = publish_costs.read_quota_history(path)
    assert [r["at"] for r in rows] == [1.0, 2.0]


def test_a_missing_history_file_is_not_fatal(tmp_path):
    assert publish_costs.read_quota_history(str(tmp_path / "nope.jsonl")) == []


def test_payload_carries_cycles_and_drops_other_sessions(tmp_path, monkeypatch):
    rows = [
        {"session": "a", "kind": "cycle", "started_at": "2026-08-11T09:00:00Z",
         "ended_at": "2026-08-11T09:20:00Z", "duration_seconds": 1200.0,
         "turns": 60, "subagent_turns": 3, "tool_calls": 70,
         "weighted_tokens": 900000.0, "models": ["claude-opus-5"],
         "input_tokens": 1, "output_tokens": 2,
         "cache_read_tokens": 3, "cache_write_5m_tokens": 0,
         "cache_write_1h_tokens": 4},
        {"session": "b", "kind": "reply", "started_at": "2026-08-11T09:30:00Z",
         "ended_at": "2026-08-11T09:31:00Z", "duration_seconds": 60.0,
         "turns": 1, "subagent_turns": 0, "tool_calls": 0,
         "weighted_tokens": 100.0, "models": ["claude-opus-5"],
         "input_tokens": 1, "output_tokens": 2,
         "cache_read_tokens": 3, "cache_write_5m_tokens": 0,
         "cache_write_1h_tokens": 4},
    ]
    monkeypatch.setattr(publish_costs.analytics, "scan", lambda d=None: rows)
    payload = publish_costs.build_payload(
        history_path=_history(tmp_path, [json.dumps({"at": 1.0, "seven_day": 47.0})]))

    assert [c["session"] for c in payload["cycles"]] == ["a"]
    assert payload["cycles"][0]["subagentTurns"] == 3
    assert payload["cycles"][0]["weightedTokens"] == 900000.0
    # The four token classes are summarised once, not repeated on every row.
    assert "cache_read_tokens" not in payload["cycles"][0]
    # ...but the summary still sees the non-cycle session, or "other_sessions"
    # would read 0 on a loop that runs a reply turn after every cycle.
    assert payload["summary"]["other_sessions"] == 1
    assert payload["quota"] == [{"at": 1.0, "seven_day": 47.0}]
    assert payload["weights"] == publish_costs.analytics.COST_WEIGHTS


def test_no_transcripts_publishes_nothing_rather_than_an_empty_ledger(monkeypatch):
    """Overwriting a good record with "no cycle has ever run" is worse than
    not writing, and an empty scan is what a misconfigured CLAUDE_HOME
    looks like."""
    monkeypatch.setattr(publish_costs.analytics, "scan", lambda d=None: [])
    assert publish_costs.build_payload() is None
    assert publish_costs.publish(None) is False


def test_publish_reports_failure_when_the_client_prints_failed(monkeypatch):
    """vault_tool can print FAILED and still exit 0 -- the trap every cycle
    is warned about when it writes a journal entry. Exit code alone is not
    enough to know the document landed."""
    class Proc:
        returncode = 0
        stdout = "FAILED: conflict\n"
        stderr = ""

    monkeypatch.setattr(publish_costs.subprocess, "run", lambda *a, **k: Proc())
    assert publish_costs.publish({"cycles": []}) is False


def test_publish_sends_the_document_on_stdin_not_via_a_temp_file(monkeypatch):
    seen = {}

    class Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["input"] = kwargs.get("input")
        return Proc()

    monkeypatch.setattr(publish_costs.subprocess, "run", fake_run)
    assert publish_costs.publish({"cycles": [], "quota": []}, "some/path.json") is True
    assert seen["cmd"][-3:] == ["puts", "some/path.json", "-"]
    assert json.loads(seen["input"]) == {"cycles": [], "quota": []}


def test_refresh_survives_a_build_that_raises(monkeypatch):
    """It runs on the way out of a cycle, while the reply is being written.
    Nothing here may cost a cycle its reply."""
    def boom(*a, **k):
        raise RuntimeError("transcripts unreadable")

    monkeypatch.setattr(publish_costs.analytics, "scan", boom)
    assert publish_costs.refresh() is False
