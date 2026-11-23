"""The program calls the person "the expert": no string constant of the text modules says you/your."""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "bearmind"
MODULES = ["cli.py", "inspect_server.py", "monitor.py", "actions.py"]
YOU = re.compile(r"\byou(r|rs|rself)?\b", re.I)
TOKEN = "you"  # the internal data token of "excluded by a person", never shown


def _docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def _offenders(source):
    tree = ast.parse(source)
    skip = _docstring_nodes(tree)
    return [(n.lineno, n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip
            and n.value != TOKEN and YOU.search(n.value)]


@pytest.mark.parametrize("name", MODULES)
def test_text_constants_do_not_say_you(name):
    assert _offenders((ROOT / name).read_text(encoding="utf-8")) == []


def test_the_scan_finds_a_you_text():
    assert _offenders('"""doc with you"""\nx = "open it yourself"\ny = "you"\n') == [(2, "open it yourself")]
