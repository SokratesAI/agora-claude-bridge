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


def test_provision_skips_a_broken_repo_instead_of_failing_the_turn(workspace):
    _repo(str(workspace / "good"))
    broken = workspace / "broken"
    (broken / ".git").mkdir(parents=True)  # looks like a repo, is not one
    slot = cli._workspace_for("s1")
    os.makedirs(slot)
    assert cli._provision_workspace(slot) == ["good"]
