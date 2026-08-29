"""The Dockerfile's version pins.

There is no other test in this repo that reads build config, and this one
exists for a specific reason rather than for coverage: the `claude` CLI was
installed unpinned, and because nothing above that line changes between
builds, Docker's layer cache never re-ran it. The version sat at 2.1.197 while
the registry was at 2.1.226, and *looked* like it tracked latest. Nothing
failed, nothing warned, and the only way to find out was to ask the pod.

So the thing worth guarding is not the version number -- that is meant to move
-- but the pin itself.
"""
import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"


def _dockerfile():
    return DOCKERFILE.read_text(encoding="utf-8")


def test_claude_cli_is_installed_at_a_pinned_version():
    """`npm install -g @anthropic-ai/claude-code` must carry an @version.

    An unpinned install is the bug this test was written for: it is frozen by
    the layer cache rather than current, and it is frozen invisibly.
    """
    text = _dockerfile()
    installs = re.findall(r"npm install -g @anthropic-ai/claude-code(\S*)", text)
    assert installs, "no claude-code install line found in the Dockerfile"
    for suffix in installs:
        assert suffix.startswith("@"), (
            "the claude CLI is installed unpinned; Docker's layer cache will "
            "freeze it at whatever version was current when the layer was "
            "first built, with nothing reporting which one that is"
        )


def test_claude_cli_version_arg_is_a_concrete_version():
    """The pin comes from an ARG, so a rebuild can override it without a diff."""
    text = _dockerfile()
    match = re.search(r"^ARG CLAUDE_CODE_VERSION=(.+)$", text, re.MULTILINE)
    assert match, "expected an ARG CLAUDE_CODE_VERSION line"
    version = match.group(1).strip()
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), (
        f"CLAUDE_CODE_VERSION should be an exact version, got {version!r} -- "
        "a range or a dist-tag reintroduces the drift this pin removes"
    )
    assert "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" in text, (
        "the install line should use ${CLAUDE_CODE_VERSION} rather than "
        "repeating the version, so the two cannot disagree"
    )


def test_openssh_client_is_installed():
    """`tools.nas_health` reaches Sonarr and Radarr over an SSH hop, or not at all.

    The NetworkPolicy on this pod opens port 22 to the NAS and nothing else, so
    the apps' own ports are unreachable from here and the only way to ask them
    anything is to run `curl` on the NAS itself. That needs an `ssh` binary in
    this image. Without it the check still reports the box as up and prints
    `CANNOT SEE FROM THIS POD` for the services -- it degrades quietly, which
    is why a build-time assertion is worth more here than a runtime one.

    Guarding the package rather than the exact apt line, because the thing that
    would take it away is a future tidy-up of an unrelated package list.
    """
    assert "openssh-client" in _dockerfile(), (
        "openssh-client is not installed; tools.nas_health can no longer judge "
        "Sonarr and Radarr from this pod, and it will say so instead of failing"
    )
