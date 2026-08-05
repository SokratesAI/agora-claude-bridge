"""Credentials must not reach the conversation through tool output."""
from unittest.mock import patch

from bridge import activity
from bridge.redact import redact


# The literal shape of what Cycle 20 printed: the CLI's own credentials
# file, holding a live OAuth access token and a long-lived refresh token.
CREDENTIALS_FILE = """{
  "claudeAiOauth": {
    "accessToken": "sk-ant-oat01-AbCdEf0123456789AbCdEf0123456789xyz",
    "refreshToken": "sk-ant-ort01-9876543210ZyXwVu9876543210ZyXwVuabc",
    "expiresAt": 1785200000,
    "scopes": ["user:inference", "user:profile"]
  }
}"""


def test_the_leak_that_actually_happened_is_redacted():
    out = redact(CREDENTIALS_FILE)
    assert "sk-ant-oat01-AbCdEf0123456789AbCdEf0123456789xyz" not in out
    assert "sk-ant-ort01-9876543210ZyXwVu9876543210ZyXwVuabc" not in out
    assert out.count("[redacted:") == 2
    # The structure survives: he can still see that the file has an access
    # token, a refresh token and when they expire.
    assert "accessToken" in out and "expiresAt" in out and "1785200000" in out


def test_github_tokens_are_redacted():
    out = redact("remote: https://ghp_0123456789abcdefghijklmnopqrstuvwxyz@github.com/x")
    assert "ghp_0123456789abcdefghijklmnopqrstuvwxyz" not in out
    assert "[redacted: github token]" in out
    assert "[redacted: github token]" in redact("token github_pat_11ABCDEFG0abcdefghijklmn")


def test_jwt_shaped_tokens_are_redacted():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r"
    out = redact(f"Authorization: Bearer {jwt}")
    assert jwt not in out
    assert "[redacted: jwt]" in out


def test_private_key_blocks_are_redacted_whole():
    key = ("-----BEGIN RSA PRIVATE KEY-----\n"
           "MIIEowIBAAKCAQEAx7Vn9lQ2\naGVsbG8gd29ybGQK\n"
           "-----END RSA PRIVATE KEY-----")
    out = redact(f"cat id_rsa\n{key}\ndone")
    assert "MIIEowIBAAKCAQEAx7Vn9lQ2" not in out
    assert out == "cat id_rsa\n[redacted: private key]\ndone"


def test_env_dumps_keep_the_name_and_lose_the_value():
    """`printenv` is the other realistic way this leaks. Knowing a variable
    is set is exactly what he wants to see; its value is not."""
    out = redact("ANTHROPIC_API_KEY=abc123def456ghi789\nAGORA_TOKEN: s3cr3tvalue123\nHOME=/data")
    assert "abc123def456ghi789" not in out
    assert "s3cr3tvalue123" not in out
    assert "ANTHROPIC_API_KEY=[redacted: value]" in out
    assert "AGORA_TOKEN: [redacted: value]" in out
    # Untouched: not a credential, and clipping it would be the tidying-up
    # kind of filtering this module exists to avoid.
    assert "HOME=/data" in out


def test_ordinary_output_passes_through_untouched():
    """Over-redaction is the real risk here -- output he cannot trust to be
    verbatim is worth much less than output he can."""
    diff = (
        "commit 4649ce7f0a1b2c3d4e5f60718293a4b5c6d7e8f9\n"
        "+    reporter.report(name, block.get('input'), tool_use_id)\n"
        "144 passed in 1.33s\n"
        "ghcr.io/sokratesai/agora@sha256:c62194fc96e80b42aaee3cc15199a32902d6406c\n"
    )
    assert redact(diff) == diff


def test_non_strings_pass_through():
    assert redact(None) is None
    assert redact(42) == 42
    assert redact("") == ""


def test_tool_output_is_redacted_on_the_way_out():
    posted = []
    with patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True):
        reporter = activity.ActivityReporter({"url": "http://runner/x", "token": "tok"})
        reporter.start()
        reporter.report_result("Read", "toolu_1", CREDENTIALS_FILE, is_error=False)
        reporter.close()
    assert "sk-ant-oat01-" not in posted[0]["output"]
    assert "[redacted: anthropic key]" in posted[0]["output"]


def test_chip_labels_and_narration_are_redacted_too():
    """A secret arrives just as easily in the command as in its output --
    `curl -H "Authorization: Bearer ..."` is a tool INPUT."""
    posted = []
    with patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True):
        reporter = activity.ActivityReporter({"url": "http://runner/x", "token": "tok"})
        reporter.start()
        reporter.report("Bash", {"command": "gh auth login --with-token ghp_0123456789abcdefghijklmnop"})
        reporter.report_text("I used sk-ant-oat01-AbCdEf0123456789AbCdEf0123456789xyz to authenticate.")
        reporter.close()
    assert "ghp_0123456789abcdefghijklmnop" not in posted[0]["detail"]
    assert "sk-ant-oat01-" not in posted[1]["detail"]


def test_the_reports_own_auth_token_is_not_redacted():
    """It is the runner-issued credential for this very post. Scrubbing it
    would not protect anything -- it would just fail the report."""
    posted = []
    with patch.object(activity, "_post", lambda url, payload: posted.append(payload) or True):
        reporter = activity.ActivityReporter(
            {"url": "http://runner/x", "token": "act_0123456789abcdefghij"})
        reporter.start()
        reporter.report_result("Bash", "toolu_1", "ok", is_error=False)
        reporter.close()
    assert posted[0]["token"] == "act_0123456789abcdefghij"
