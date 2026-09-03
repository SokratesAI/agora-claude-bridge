"""The build is skipped only when the image provably cannot have changed.

`.github/image-paths.py` decides, before `build-push` starts, whether the diff
a push carries can have changed the image. Getting that wrong in one direction
costs a wasted four-minute build; getting it wrong in the other means a real
change never deploys and every check stays green. So the tests here are almost
all about the second direction.

The pinned list in `test_the_watched_paths_are_exactly_these` is a deliberate
change-detector and the only one in this file. It is what makes a new `COPY`
in the Dockerfile a red test rather than a silently unwatched path: the module
derives its answer from the Dockerfile, and this asserts that what it derives
today is still what somebody read and agreed to.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / ".github" / "image-paths.py"


def _module():
    spec = importlib.util.spec_from_file_location("image_paths", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ip = _module()
DOCKERFILE = (REPO / "Dockerfile").read_text(encoding="utf-8")


def test_the_watched_paths_are_exactly_these():
    assert ip.image_paths(DOCKERFILE) == [
        ".dockerignore",
        ".github/workflows/build.yaml",
        "Dockerfile",
        "bridge/",
        "run.py",
    ]


def test_every_copy_source_in_the_real_dockerfile_is_watched():
    """Not a restatement of the list above: this asks the Dockerfile.

    If a COPY is added and the pinned list is updated to match, both tests
    pass. If a COPY is added and the parser drops it, this one is the failure
    -- the pinned list would agree with a parser that had gone blind.
    """
    watched = ip.image_paths(DOCKERFILE)
    for source in ip.copy_sources(DOCKERFILE):
        assert source in watched


def test_every_copy_source_exists_in_the_repo():
    """A watched path that is not there watches nothing."""
    for source in ip.copy_sources(DOCKERFILE):
        assert (REPO / source.rstrip("/")).exists(), source


@pytest.mark.parametrize(
    "changed",
    [
        ["bridge/vault_tool.py"],
        ["run.py"],
        ["Dockerfile"],
        [".github/workflows/build.yaml"],
        ["tests/test_bridge.py", "bridge/reply.py"],
    ],
)
def test_a_change_the_image_ships_builds(changed):
    assert ip.affects_image(ip.image_paths(DOCKERFILE), changed) is True


@pytest.mark.parametrize(
    "changed",
    [
        ["tests/test_bridge.py"],
        ["README.md"],
        [".github/update-image-digest.py"],
        ["tools/sync_contract.py"],
    ],
)
def test_a_change_the_image_does_not_ship_skips(changed):
    assert ip.affects_image(ip.image_paths(DOCKERFILE), changed) is False


def test_a_directory_source_matches_files_under_it_not_a_prefix_of_its_name():
    paths = ["bridge/"]
    assert ip.affects_image(paths, ["bridge/nested/deep/thing.py"]) is True
    assert ip.affects_image(paths, ["bridge_notes.md"]) is False


def test_an_empty_diff_builds():
    """`git diff` printing nothing is an unknown, not a no-op."""
    assert ip.affects_image(ip.image_paths(DOCKERFILE), []) is True
    assert ip.affects_image(ip.image_paths(DOCKERFILE), ["", "  "]) is True


def test_a_copy_from_another_stage_is_not_a_repo_path():
    text = "FROM x AS b\nFROM y\nCOPY --from=b /out/bin /usr/local/bin/bin\n"
    assert ip.copy_sources(text) == []


def test_a_flag_is_not_mistaken_for_a_source():
    text = "FROM y\nCOPY --chown=1000:1000 src/ /app/src/\n"
    assert ip.copy_sources(text) == ["src/"]


def test_a_continued_copy_line_is_read_whole():
    text = "FROM y\nCOPY a.py \\\n    b.py \\\n    ./\n"
    assert ip.copy_sources(text) == ["a.py", "b.py"]


def test_a_comment_inside_the_dockerfile_is_not_a_copy():
    text = "FROM y\n# COPY secrets/ /app/\nCOPY a.py ./\n"
    assert ip.copy_sources(text) == ["a.py"]


def test_an_unparseable_copy_refuses_rather_than_dropping_a_path():
    with pytest.raises(ip.UnreadableCopy):
        ip.copy_sources('FROM y\nCOPY ["a.py", "./"]\n')
    with pytest.raises(ip.UnreadableCopy):
        ip.copy_sources("FROM y\nCOPY a.py\n")


def test_copy_dot_watches_everything():
    assert ip.affects_image(ip.copy_sources("FROM y\nCOPY . /app\n"), ["anything"]) is True


def test_the_cli_builds_when_the_dockerfile_cannot_be_read(tmp_path, capsys):
    rc = ip.main(["--dockerfile", str(tmp_path / "nope"), "--changed-files", "-"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "true"


def test_the_cli_builds_when_a_copy_is_unparseable(tmp_path, capsys):
    bad = tmp_path / "Dockerfile"
    bad.write_text('FROM y\nCOPY ["a.py", "./"]\n', encoding="utf-8")
    rc = ip.main(["--dockerfile", str(bad), "--changed-files", "-"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "true"


def test_the_cli_answers_from_a_changed_files_file(tmp_path, capsys):
    listing = tmp_path / "changed.txt"
    listing.write_text("tools/x.py\ntests/y.py\n", encoding="utf-8")
    assert ip.main(["--changed-files", str(listing)]) == 0
    assert capsys.readouterr().out.strip() == "false"

    listing.write_text("bridge/x.py\n", encoding="utf-8")
    assert ip.main(["--changed-files", str(listing)]) == 0
    assert capsys.readouterr().out.strip() == "true"
