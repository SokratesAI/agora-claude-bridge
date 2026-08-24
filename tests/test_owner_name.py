"""The owner's name stays out of this public repo's prose.

His ask, `issues.md` 2026-08-24: *"I do not like you using my name in public
repos. Not as comments in code, on prs or anything."* This repo is public.

This is a second, smaller copy of `tools/name_scan.py` in the runner, and the
duplication is deliberate rather than overlooked: the two repos share no
importable code, and the runner's version has to understand JS and CSS while
this one only ever sees Python and markdown. If a third copy is ever wanted,
that is the point to make it a package instead of writing this again.

Only prose is in scope. A string literal is data -- `_REAL_MANUAL_TRIGGER` in
`test_bridge.py` is a verbatim copy of the heartbeat text Agora really sends,
and rewriting it would make the test stop testing the real thing.
"""
import ast
import io
import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "Edvard"


def _docstring_starts(source):
    starts = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return starts
    for node in ast.walk(tree):
        body = getattr(node, "body", None) or []
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) or not body:
            continue
        first = body[0]
        value = getattr(first, "value", None)
        if isinstance(first, ast.Expr) and isinstance(value, ast.Constant) \
                and isinstance(value.value, str):
            starts.add((value.lineno, value.col_offset))
    return starts


def prose(path, source):
    """Comments and docstrings for Python; the whole file for markdown."""
    if path.suffix in {".md", ".txt"}:
        return [(1, source)]
    if path.suffix != ".py":
        return []
    docs = _docstring_starts(source)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    return [(t.start[0], t.string) for t in toks
            if t.type == tokenize.COMMENT
            or (t.type == tokenize.STRING and t.start in docs)]


def hits(path, source):
    return [f"{path}:{line}" for line, text in prose(Path(path), source)
            if NAME in text]


def test_no_owner_name_in_any_comment_or_docstring():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split()
    found = []
    for rel in tracked:
        path = ROOT / rel
        try:
            found += hits(rel, path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
    assert found == [], "write 'the owner' instead:\n" + "\n".join(found)


def test_a_comment_and_a_docstring_are_prose():
    assert hits("x.py", "# Edvard asked\n")
    assert hits("x.py", '"""What Edvard wanted."""\n')
    assert hits("x.md", "For Edvard.\n")


def test_a_string_literal_is_data_and_is_left_alone():
    assert hits("x.py", 'TRIGGER = "address Edvard directly"\n') == []


def test_a_broken_python_file_does_not_crash_the_scan():
    assert hits("x.py", "def (:\n") == []
