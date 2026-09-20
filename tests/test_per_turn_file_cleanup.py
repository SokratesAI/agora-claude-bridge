"""Do the three per-turn files in ~/.claude outlive the turn that wrote them?

They did. On 2026-09-20 the bridge pod's `~/.claude` held 566
`bridge-hooks.settings.<slot>.json`, oldest 2026-08-19, plus 5 leftover
`bridge-mcp.config.<slot>.json` each carrying that turn's (dead) bearer
token. The hook settings file had no cleanup at all; the other two are
deleted on the happy path and not when the process is killed.

Two invariants, one per hole: `_run_cli_once`'s `finally` removes all
three, and `_sweep_stale_turn_files` clears slotted leftovers older than
the cutoff while never touching the serialized lane's own fixed paths.
"""
import os
import time

from bridge import cli, quota


def _claude_dir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setattr(cli, "CLAUDE_HOME", str(home))
    monkeypatch.setattr(cli, "MCP_CONFIG_FILE",
                        str(home / ".claude" / "bridge-mcp.config.json"))
    monkeypatch.setattr(cli, "CLI_INPUT_FILE",
                        str(home / ".claude" / "bridge-input.jsonl"))
    monkeypatch.setattr(quota, "HOOK_SETTINGS_FILE",
                        str(home / ".claude" / "bridge-hooks.settings.json"))
    return home / ".claude"


def _write(d, name, age_seconds):
    p = d / name
    p.write_text("{}")
    when = time.time() - age_seconds
    os.utime(p, (when, when))
    return p


def test_sweep_removes_slotted_files_older_than_the_cutoff(tmp_path, monkeypatch):
    d = _claude_dir(tmp_path, monkeypatch)
    old = [
        _write(d, "bridge-hooks.settings.42-7.json", 10_000),
        _write(d, "bridge-mcp.config.42-7.json", 10_000),
        _write(d, "bridge-input.42-7.jsonl", 10_000),
    ]
    cli._sweep_stale_turn_files(time.time() - 3600)
    for p in old:
        assert not p.exists(), f"{p.name} survived the sweep"


def test_sweep_leaves_a_slotted_file_a_running_turn_could_still_own(tmp_path, monkeypatch):
    d = _claude_dir(tmp_path, monkeypatch)
    fresh = _write(d, "bridge-mcp.config.42-7.json", 60)
    cli._sweep_stale_turn_files(time.time() - 3600)
    assert fresh.exists()


def test_sweep_never_touches_the_serialized_lanes_own_files(tmp_path, monkeypatch):
    d = _claude_dir(tmp_path, monkeypatch)
    fixed = [
        _write(d, "bridge-mcp.config.json", 10_000),
        _write(d, "bridge-input.jsonl", 10_000),
        _write(d, "bridge-hooks.settings.json", 10_000),
    ]
    cli._sweep_stale_turn_files(time.time() - 3600)
    for p in fixed:
        assert p.exists(), f"{p.name} was swept and belongs to the locked lane"


def test_sweep_leaves_unrelated_files_alone(tmp_path, monkeypatch):
    d = _claude_dir(tmp_path, monkeypatch)
    keep = _write(d, "settings.json", 10_000)
    also = _write(d, "bridge-notes.settings.1-2.json", 10_000)
    cli._sweep_stale_turn_files(time.time() - 3600)
    assert keep.exists()
    assert also.exists()


def test_the_finally_deletes_the_hook_settings_file():
    """The cleanup that did not exist -- pinned on the source, because
    reaching this `finally` means running a real CLI subprocess."""
    src = open(os.path.join(os.path.dirname(cli.__file__), "cli.py")).read()
    finally_block = src[src.index("        cancelled = turn.cancelled"):]
    finally_block = finally_block[:finally_block.index("    elapsed = time.monotonic()")]
    for name in ("mcp_config", "input_file", "hook_settings"):
        assert f"os.remove({name})" in finally_block, f"{name} is never deleted"


def test_the_slot_sweep_calls_the_file_sweep():
    src = open(os.path.join(os.path.dirname(cli.__file__), "cli.py")).read()
    body = src[src.index("def _sweep_stale_slots():"):src.index("def _sweep_stale_turn_files(")]
    assert "_sweep_stale_turn_files(cutoff)" in body
