"""Does the grant token for this turn's MCP server ever get written to a file?

Idea #239. For a month it did: write_mcp_config interpolated the live bearer
token into ~/.claude/bridge-mcp.config.<slot>.json on every single turn, and a
turn killed before its cleanup left the file behind -- five of them were still
on the bridge pod when Cycle 1914 counted, four stale since early September.

The fix is not better cleanup. The CLI expands ${VAR} in an --mcp-config header
out of its own process environment, so the file can carry a placeholder and the
secret can travel in the environment Popen is already given. Measured against
CLI 2.1.272 in the bridge pod on 2026-09-20 with a probe server logging the
Authorization header it received: set, all four requests carried the expanded
secret; unset, they carried the literal placeholder and `claude -p` exited 0.

These tests pin our half of that -- the file contents and the environment --
because the CLI's half is not ours to assert on in a unit test.
"""
import json
import os

from bridge import cli

TOKEN = "grant-token-that-must-not-land-on-disk"
BLOCK = {"url": "http://runner.agents.svc.cluster.local:8100/mcp", "token": TOKEN}


def test_the_written_config_does_not_contain_the_token(tmp_path):
    path = str(tmp_path / "mcp.json")
    env = {}
    assert cli.write_mcp_config(BLOCK, path=path, env=env) == path
    raw = open(path).read()
    assert TOKEN not in raw
    header = json.loads(raw)["mcpServers"][cli.MCP_SERVER_NAME]["headers"]["Authorization"]
    assert header == "Bearer ${%s}" % cli.MCP_TOKEN_ENV


def test_the_token_travels_in_the_environment_instead(tmp_path):
    env = {}
    cli.write_mcp_config(BLOCK, path=str(tmp_path / "mcp.json"), env=env)
    assert env[cli.MCP_TOKEN_ENV] == TOKEN


def test_a_failed_write_puts_no_token_in_the_environment(tmp_path):
    # "" means the CLI is started with no --mcp-config at all, so a token in
    # the environment would be a secret exported for a server nobody is
    # talking to. The env write is deliberately after the chmod for this.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    env = {}
    assert cli.write_mcp_config(BLOCK, path=str(blocker / "sub" / "mcp.json"), env=env) == ""
    assert env == {}


def test_an_unusable_block_puts_no_token_in_the_environment(tmp_path):
    env = {}
    assert cli.write_mcp_config({"url": "", "token": TOKEN},
                                path=str(tmp_path / "mcp.json"), env=env) == ""
    assert env == {}


def test_the_config_is_still_valid_json_with_no_env_given(tmp_path):
    # The asymmetry write_mcp_config's docstring is built around: an
    # unreachable MCP server costs a turn its tools, but a config file that is
    # not valid JSON stops the CLI from starting at all. A caller that passes
    # no env must still get a parseable file.
    path = str(tmp_path / "mcp.json")
    assert cli.write_mcp_config(BLOCK, path=path) == path
    json.loads(open(path).read())


def test_the_file_is_still_private(tmp_path):
    path = str(tmp_path / "mcp.json")
    cli.write_mcp_config(BLOCK, path=path, env={})
    assert oct(os.stat(path).st_mode & 0o777) == "0o600"
