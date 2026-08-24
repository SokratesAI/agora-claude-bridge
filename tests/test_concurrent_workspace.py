"""A concurrent turn gets a real checkout, off the shared clone, outside it.

These run against real `git`, not a mock. The whole point of the change
under test is what git actually does with two worktrees off one object
store, and a mocked `subprocess.run` would have agreed with any
implementation I wrote, including a wrong one.
"""
import os
import subprocess
import time

import pytest

from bridge import cli


def _repo(path, filename="README.md", body="hello\n"):
    os.makedirs(path, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    with open(os.path.join(path, filename), "w") as fh:
        fh.write(body)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=path, check=True)
    return path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    shared = tmp_path / "workspace"
    shared.mkdir()
    monkeypatch.setattr(cli, "CLAUDE_WORKSPACE", str(shared))
    monkeypatch.delenv("CLAUDE_CONCURRENT_ROOT", raising=False)
    return shared


def test_slot_is_a_sibling_of_the_shared_workspace_not_a_child(workspace):
    """Cycle 290's trap: a slot inside /data/workspace is indistinguishable
    from leftover work to another cycle's `for d in /data/workspace/*/`."""
    slotted = cli._workspace_for("123-456")
    assert not slotted.startswith(str(workspace) + os.sep)
    assert cli._workspace_for("") == str(workspace)


def test_provision_gives_the_slot_a_checkout_of_every_shared_repo(workspace, tmp_path):
    _repo(str(workspace / "repo-a"))
    _repo(str(workspace / "repo-b"), filename="b.txt", body="b\n")
    (workspace / "not-a-repo").mkdir()

    slot = cli._workspace_for("slot1")
    os.makedirs(slot)
    provisioned = cli._provision_workspace(slot)

    assert sorted(provisioned) == ["repo-a", "repo-b"]
    assert os.path.exists(os.path.join(slot, "repo-a", "README.md"))
    assert os.path.exists(os.path.join(slot, "repo-b", "b.txt"))
    assert not os.path.exists(os.path.join(slot, "not-a-repo"))


def test_two_slots_get_independent_working_trees_off_one_object_store(workspace):
    _repo(str(workspace / "repo-a"))
    one, two = cli._workspace_for("s1"), cli._workspace_for("s2")
    os.makedirs(one)
    os.makedirs(two)
    assert cli._provision_workspace(one) == ["repo-a"]
    assert cli._provision_workspace(two) == ["repo-a"]

    # An edit and a branch in one is invisible in the other -- the working
    # tree and the index are private, which is the isolation the lane is for.
    with open(os.path.join(one, "repo-a", "README.md"), "w") as fh:
        fh.write("changed by slot 1\n")
    subprocess.run(["git", "checkout", "-qb", "slot-1-branch"],
                   cwd=os.path.join(one, "repo-a"), check=True)
    with open(os.path.join(two, "repo-a", "README.md")) as fh:
        assert fh.read() == "hello\n"
    assert subprocess.run(["git", "status", "--porcelain"], cwd=os.path.join(two, "repo-a"),
                          capture_output=True, text=True).stdout == ""

    # Shared objects, not a second clone: `.git` is a file pointing back at
    # the shared clone's object store, not a directory of its own.
    assert not os.path.isdir(os.path.join(one, "repo-a", ".git"))

    # Detached, so provisioning leaves no branch behind in the shared clone.
    # Without --detach, `git worktree add` invents a branch named after the
    # directory, and every slot would accumulate one in the repo they share.
    branches = subprocess.run(["git", "branch", "--format=%(refname:short)"],
                              cwd=str(workspace / "repo-a"),
                              capture_output=True, text=True).stdout.split()
    assert branches == ["main", "slot-1-branch"]


def test_the_shared_checkout_is_untouched_by_provisioning(workspace):
    _repo(str(workspace / "repo-a"))
    shared_repo = str(workspace / "repo-a")
    before = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=shared_repo,
                            capture_output=True, text=True).stdout.strip()
    os.makedirs(cli._workspace_for("s1"))
    cli._provision_workspace(cli._workspace_for("s1"))
    after = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=shared_repo,
                           capture_output=True, text=True).stdout.strip()
    assert before == after == "main"
    assert subprocess.run(["git", "status", "--porcelain"], cwd=shared_repo,
                          capture_output=True, text=True).stdout == ""


def test_sweep_removes_a_dead_turns_slot_and_leaves_a_live_one(workspace, monkeypatch):
    _repo(str(workspace / "repo-a"))
    dead, live = cli._workspace_for("dead"), cli._workspace_for("live")
    os.makedirs(dead)
    os.makedirs(live)
    cli._provision_workspace(dead)
    cli._provision_workspace(live)
    old = time.time() - cli.STALE_SLOT_SECONDS - 60
    os.utime(dead, (old, old))

    cli._sweep_stale_slots()

    assert not os.path.exists(dead)
    assert os.path.exists(os.path.join(live, "repo-a", "README.md"))
    listed = subprocess.run(["git", "worktree", "list"], cwd=str(workspace / "repo-a"),
                            capture_output=True, text=True).stdout
    assert os.path.join(dead, "repo-a") not in listed
    assert os.path.join(live, "repo-a") in listed


def test_a_slot_can_be_reprovisioned_after_its_directory_was_removed(workspace):
    """A killed turn leaves a `.git/worktrees` entry claiming the path, and
    `slot` repeats -- it is pid plus a reused thread ident, so a later turn
    really does land on it. Measured while writing this: `--force` alone
    already recovers, so the prune is what keeps `.git/worktrees` from
    growing an entry per killed turn rather than what unblocks the next
    one. Both are needed and they are needed for different things."""
    _repo(str(workspace / "repo-a"))
    slot = cli._workspace_for("recycled")
    os.makedirs(slot)
    assert cli._provision_workspace(slot) == ["repo-a"]

    import shutil
    shutil.rmtree(slot)  # what a SIGKILLed turn leaves behind, minus the finally
    os.makedirs(slot)
    cli._sweep_stale_slots()
    assert cli._provision_workspace(slot) == ["repo-a"]
    assert os.path.exists(os.path.join(slot, "repo-a", "README.md"))


def _clone_with_origin(origin_path, dest_path):
    """A shared checkout that has a real `origin` to fetch from."""
    subprocess.run(["git", "clone", "-q", origin_path, dest_path], check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=dest_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=dest_path, check=True)
    return dest_path


def _read(*parts):
    with open(os.path.join(*parts)) as fh:
        return fh.read()


def test_worktree_starts_from_origin_main_not_the_branch_shared_is_parked_on(workspace, tmp_path):
    """Cycle 365 woke up on `nova/status-word-back-on-the-card`, abandoned
    two cycles earlier, because that is where a serialized turn left the
    shared checkout. The start point is origin/main, not HEAD."""
    origin = _repo(str(tmp_path / "origin"))
    shared = _clone_with_origin(origin, str(workspace / "repo-a"))

    subprocess.run(["git", "checkout", "-qb", "nova/abandoned"], cwd=shared, check=True)
    with open(os.path.join(shared, "README.md"), "w") as fh:
        fh.write("dead end\n")
    subprocess.run(["git", "commit", "-aqm", "abandoned"], cwd=shared, check=True)

    slot = cli._workspace_for("s1")
    os.makedirs(slot)
    assert cli._provision_workspace(slot) == ["repo-a"]
    assert _read(slot, "repo-a", "README.md") == "hello\n"

    # And the shared checkout is left exactly where it was -- the fetch
    # writes refs and objects, never a working tree, which is what makes
    # this safe to do under another live turn.
    assert subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=shared,
                          capture_output=True, text=True).stdout.strip() == "nova/abandoned"
    assert _read(shared, "README.md") == "dead end\n"


def test_worktree_gets_a_commit_pushed_since_the_shared_clone_last_fetched(workspace, tmp_path):
    """The other half: nothing in this loop fetches the shared checkout,
    so without the fetch here every worktree inherits code that ages by
    the hour."""
    origin = _repo(str(tmp_path / "origin"))
    _clone_with_origin(origin, str(workspace / "repo-a"))

    with open(os.path.join(origin, "README.md"), "w") as fh:
        fh.write("newer\n")
    subprocess.run(["git", "commit", "-aqm", "newer"], cwd=origin, check=True)

    slot = cli._workspace_for("s1")
    os.makedirs(slot)
    assert cli._provision_workspace(slot) == ["repo-a"]
    assert _read(slot, "repo-a", "README.md") == "newer\n"


def test_falls_back_to_head_when_the_repo_has_no_origin_main(workspace):
    """No remote at all is the old behaviour, and the old behaviour is
    still a working checkout rather than none."""
    shared = _repo(str(workspace / "repo-a"))
    assert cli._start_point(shared) == "HEAD"
    slot = cli._workspace_for("s1")
    os.makedirs(slot)
    assert cli._provision_workspace(slot) == ["repo-a"]
    assert _read(slot, "repo-a", "README.md") == "hello\n"


def test_every_repo_is_fetched_not_just_the_first(workspace, tmp_path):
    """The reviewer's mutation: hoist `_start_point` out of the loop and
    reuse one answer for every repo. The *string* is the same either way
    (`refs/remotes/origin/HEAD`), so only the fetch tells them apart --
    and with one fetch for four repos, three of them stay as stale as
    they were, which is the whole thing this change exists to stop."""
    origins = {}
    for name in ("repo-a", "repo-b"):
        origins[name] = _repo(str(tmp_path / f"origin-{name}"), body=f"old {name}\n")
        _clone_with_origin(origins[name], str(workspace / name))
    for name, origin in origins.items():
        with open(os.path.join(origin, "README.md"), "w") as fh:
            fh.write(f"new {name}\n")
        subprocess.run(["git", "commit", "-aqm", "moved"], cwd=origin, check=True)

    slot = cli._workspace_for("s1")
    os.makedirs(slot)
    assert sorted(cli._provision_workspace(slot)) == ["repo-a", "repo-b"]
    for name in ("repo-a", "repo-b"):
        assert _read(slot, name, "README.md") == f"new {name}\n"


def test_falls_back_to_head_when_the_fetch_itself_fails(workspace, tmp_path):
    """github.com was unresolvable for two cycles on 2026-08-24. A turn
    that cannot fetch still has to start, on whatever it already has --
    and what it already has is the *stale remote-tracking ref*, not the
    branch the shared checkout is parked on.

    The parked branch is what makes this test able to fail: without it,
    HEAD and origin/HEAD are the same commit and 'fell back to HEAD' and
    'used the stale ref' are indistinguishable. The reviewer caught the
    version that had no branch in it, and it passed against a mutation
    that threw the stale ref away."""
    origin = _repo(str(tmp_path / "origin"))
    shared = _clone_with_origin(origin, str(workspace / "repo-a"))
    subprocess.run(["git", "checkout", "-qb", "nova/abandoned"], cwd=shared, check=True)
    with open(os.path.join(shared, "README.md"), "w") as fh:
        fh.write("dead end\n")
    subprocess.run(["git", "commit", "-aqm", "abandoned"], cwd=shared, check=True)
    subprocess.run(["git", "remote", "set-url", "origin", str(tmp_path / "gone")],
                   cwd=shared, check=True)

    slot = cli._workspace_for("s1")
    os.makedirs(slot)
    assert cli._provision_workspace(slot) == ["repo-a"]
    assert _read(slot, "repo-a", "README.md") == "hello\n"


def test_a_git_that_raises_still_provisions_from_head(workspace, monkeypatch, tmp_path):
    """`_start_point` runs before the CLI starts, so anything it raises
    lands in `_provision_workspace`'s handler and the repo is *skipped* --
    a turn with no checkout at all, where the old literal `"HEAD"` would
    have provisioned fine. Also pins that the fetch is bounded: an
    unbounded network call here hangs the whole turn before it starts,
    and the timeout had no coverage at all until this test."""
    origin = _repo(str(tmp_path / "origin"))
    _clone_with_origin(origin, str(workspace / "repo-a"))
    real_git, seen = cli._git, []

    def exploding(args, cwd, timeout=120):
        if args[0] in ("fetch", "rev-parse"):
            seen.append((args[0], timeout))
            raise subprocess.TimeoutExpired(cmd=["git", *args], timeout=timeout)
        return real_git(args, cwd, timeout)

    monkeypatch.setattr(cli, "_git", exploding)
    slot = cli._workspace_for("s1")
    os.makedirs(slot)
    assert cli._provision_workspace(slot) == ["repo-a"]
    assert _read(slot, "repo-a", "README.md") == "hello\n"
    assert seen and seen[0][0] == "fetch"
    assert all(t is not None and t <= 60 for _, t in seen), seen


def test_git_calls_run_with_the_credential_helpers_home(workspace, monkeypatch):
    """Three of the four shared repos are public and fetch fine as
    nobody; `platform-config` is not. The HTTPS credential helper lives in
    `$CLAUDE_HOME/.gitconfig` and the bridge process's own HOME has no
    gitconfig, so without this the private repo answers `fatal: could not
    read Username for 'https://github.com'` on every concurrent turn."""
    seen = {}

    def spy(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    # The ambient HOME has to differ from CLAUDE_HOME or this asserts
    # nothing -- in the bridge pod they are the same, so the first version
    # of this test passed with the override deleted.
    monkeypatch.setenv("HOME", "/home/definitely-not-claude-home")
    monkeypatch.setattr(cli.subprocess, "run", spy)
    cli._git(["fetch", "origin"], "/nowhere")
    assert seen["env"]["HOME"] == cli.CLAUDE_HOME
    assert cli.CLAUDE_HOME != "/home/definitely-not-claude-home"


def test_provision_skips_a_broken_repo_instead_of_failing_the_turn(workspace):
    _repo(str(workspace / "good"))
    broken = workspace / "broken"
    (broken / ".git").mkdir(parents=True)  # looks like a repo, is not one
    slot = cli._workspace_for("s1")
    os.makedirs(slot)
    assert cli._provision_workspace(slot) == ["good"]
