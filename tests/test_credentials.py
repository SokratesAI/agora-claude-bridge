"""The bootstrap's job is to leave the newest credential on disk and to be
loud when the only one it has is dead.

Cycle 266 measured both halves failing at once in the live pod: the PVC was
replaced, the Secret's credential had expired sixteen days earlier, and the
bootstrap wrote it and logged success. The loop was down 30 hours.
"""
import json
import os

from unittest.mock import patch

from bridge import credentials


def _cred(expires_at_ms, token="tok"):
    return json.dumps({
        "claudeAiOauth": {
            "accessToken": token,
            "refreshToken": "rt-" + token,
            "expiresAt": expires_at_ms,
            "scopes": ["user:inference"],
        }
    })


def _run(tmp_path, monkeypatch, secret_raw, now):
    monkeypatch.setattr(credentials, "CLAUDE_HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CREDENTIALS_JSON", secret_raw)
    logged = []
    with patch.object(credentials, "log", side_effect=logged.append):
        credentials.bootstrap_credentials(now=now)
    dest = tmp_path / ".claude" / ".credentials.json"
    return dest, logged


def _seed(tmp_path, raw):
    d = tmp_path / ".claude"
    d.mkdir(parents=True, exist_ok=True)
    (d / ".credentials.json").write_text(raw)


NOW = 1_787_000_000.0  # 2026-08-17-ish, in seconds


def test_writes_the_secret_onto_an_empty_volume(tmp_path, monkeypatch):
    fresh = _cred(int((NOW + 36000) * 1000))
    dest, logged = _run(tmp_path, monkeypatch, fresh, NOW)
    assert dest.read_text() == fresh
    assert any("bootstrapped" in m for m in logged)
    assert not any("EXPIRED" in m for m in logged)


def test_an_expired_secret_on_an_empty_volume_is_logged_as_the_outage(tmp_path, monkeypatch):
    """The 2026-08-17 case. It still writes -- a doomed credential produces a
    clearer CLI error than none at all -- but it must not read as success."""
    dead = _cred(int((NOW - 16 * 86400) * 1000))
    dest, logged = _run(tmp_path, monkeypatch, dead, NOW)
    assert dest.read_text() == dead
    alarm = [m for m in logged if "EXPIRED" in m]
    assert alarm, logged
    assert "384.0 hours ago" in alarm[0]
    assert not any(m.startswith("credentials: bootstrapped") for m in logged)


def test_a_live_refreshed_token_on_disk_is_never_clobbered(tmp_path, monkeypatch):
    """The original invariant. The CLI's own refresh always lands on disk, so
    it is always newer than the Secret's frozen snapshot -- including when the
    on-disk token has itself expired and is merely waiting to be refreshed."""
    on_disk = _cred(int((NOW - 3600) * 1000), token="live")  # expired an hour ago
    older_secret = _cred(int((NOW - 16 * 86400) * 1000), token="stale")
    _seed(tmp_path, on_disk)
    dest, logged = _run(tmp_path, monkeypatch, older_secret, NOW)
    assert dest.read_text() == on_disk
    assert any("leaving it alone" in m for m in logged)


def test_a_hand_refreshed_secret_replaces_a_stale_file(tmp_path, monkeypatch):
    """The recovery path: update the Secret, restart the pod. Before this it
    silently did nothing, because the dead file already existed."""
    on_disk = _cred(int((NOW - 16 * 86400) * 1000), token="dead")
    newer_secret = _cred(int((NOW + 36000) * 1000), token="fresh")
    _seed(tmp_path, on_disk)
    dest, logged = _run(tmp_path, monkeypatch, newer_secret, NOW)
    assert dest.read_text() == newer_secret
    assert any("a human refreshed it" in m for m in logged)
    assert not any("EXPIRED" in m or "UNREADABLE" in m for m in logged)


def test_a_newer_but_still_expired_secret_is_replaced_and_alarmed(tmp_path, monkeypatch):
    """Review finding on #60. Newer is not usable. A hurried re-auth, or a
    second stale snapshot, is newer than the dead file and still dead -- and
    this branch used to log it as 'a human refreshed it' with no alarm, which
    is the doomed-write-reported-as-success this module exists to stop."""
    on_disk = _cred(int((NOW - 16 * 86400) * 1000), token="dead")
    newer_but_dead = _cred(int((NOW - 10 * 86400) * 1000), token="alsodead")
    _seed(tmp_path, on_disk)
    dest, logged = _run(tmp_path, monkeypatch, newer_but_dead, NOW)
    assert dest.read_text() == newer_but_dead
    alarm = [m for m in logged if "EXPIRED" in m]
    assert alarm, logged
    assert "240.0 hours ago" in alarm[0]


def test_a_secret_with_no_expiry_is_alarmed_rather_than_trusted(tmp_path, monkeypatch):
    """Valid JSON the CLI will still reject -- the three-key-split shape in the
    module docstring. No opinion is a reason to leave a file alone, never a
    reason to trust one being written."""
    dest, logged = _run(tmp_path, monkeypatch, '{"access_token": "x"}', NOW)
    assert dest.exists()
    assert any("UNREADABLE" in m for m in logged)
    assert not any(m.startswith("credentials: wrote") for m in logged)


def test_equal_expiry_is_not_newer(tmp_path, monkeypatch):
    """The ordinary restart: the Secret is exactly what is already on disk.
    Rewriting it every boot is what the module has always refused to do."""
    same = _cred(int((NOW + 36000) * 1000))
    _seed(tmp_path, same)
    _, logged = _run(tmp_path, monkeypatch, same, NOW)
    assert any("leaving it alone" in m for m in logged)


def test_an_unreadable_expiry_leaves_the_file_alone(tmp_path, monkeypatch):
    """No opinion is not permission. An on-disk credential this module cannot
    parse may still be the good one."""
    _seed(tmp_path, "{}")
    newer_secret = _cred(int((NOW + 36000) * 1000))
    dest, logged = _run(tmp_path, monkeypatch, newer_secret, NOW)
    assert dest.read_text() == "{}"
    assert any("leaving it alone" in m for m in logged)


def test_the_written_file_is_600(tmp_path, monkeypatch):
    fresh = _cred(int((NOW + 36000) * 1000))
    dest, _ = _run(tmp_path, monkeypatch, fresh, NOW)
    assert oct(os.stat(dest).st_mode)[-3:] == "600"


def test_no_secret_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "CLAUDE_HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CREDENTIALS_JSON", raising=False)
    logged = []
    with patch.object(credentials, "log", side_effect=logged.append):
        credentials.bootstrap_credentials(now=NOW)
    assert not (tmp_path / ".claude" / ".credentials.json").exists()
    assert any("skipping bootstrap" in m for m in logged)


def test_garbage_secret_never_reaches_disk(tmp_path, monkeypatch):
    dest, logged = _run(tmp_path, monkeypatch, "not json", NOW)
    assert not dest.exists()
    assert any("not valid JSON" in m for m in logged)
