import os
import stat

from bridge import cli


def _mode(path):
    return stat.S_IMODE(os.lstat(path).st_mode)


def test_creates_a_private_directory(tmp_path):
    path = cli._messaging_runtime_dir(base=str(tmp_path))
    assert path == str(tmp_path / "claude-run-{}".format(os.getuid()))
    assert os.path.isdir(path)
    assert _mode(path) == 0o700


def test_tightens_a_directory_left_open(tmp_path):
    path = tmp_path / "claude-run-{}".format(os.getuid())
    path.mkdir()
    os.chmod(path, 0o2775)
    assert cli._messaging_runtime_dir(base=str(tmp_path)) == str(path)
    assert _mode(path) == 0o700


def test_a_file_in_the_way_leaves_messaging_off(tmp_path):
    (tmp_path / "claude-run-{}".format(os.getuid())).write_text("x")
    assert cli._messaging_runtime_dir(base=str(tmp_path)) == ""


def test_a_symlink_is_not_followed(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    os.symlink(target, tmp_path / "claude-run-{}".format(os.getuid()))
    assert cli._messaging_runtime_dir(base=str(tmp_path)) == ""


def test_an_unwritable_base_leaves_messaging_off(tmp_path):
    (tmp_path / "a-file").write_text("x")
    assert cli._messaging_runtime_dir(base=str(tmp_path / "a-file")) == ""
