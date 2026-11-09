"""The student guide (docs/students) checked against the program: commands, paths, UI names, tab order, step length.

Phase 6, task 6. The guide is Russian in STE style (Simplified Technical English rules carried over to Russian):
numbered steps, one action per step, at most 20 words per sentence in a step. The program's own names (tabs,
buttons, menus, statuses) stay English and are written in **bold**; Russian bold (a warning word) is not a UI name.
File names, folders, commands and cell values are written as `code`.
"""
import ast
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GUIDE_DIR = REPO / "docs" / "students"
GUIDE = (GUIDE_DIR / "README.md", GUIDE_DIR / "series-workflow.md")
IMG_DIR = GUIDE_DIR / "img"
# The window's and the inspection page's texts. inspect_server.py holds the page's Next button and banners (ruling
# P6-3), bm_examinator.py the page's buttons (Delete sel., Save results). bearmind/monitor.py is the window's model:
# the stage columns of the Monitor and Inspection tables (STAGE_TITLE: Crop, Autoinspect) are named only there.
UI_SOURCES = (sorted((REPO / "bearmind" / "gui").glob("*.py"))
              + [REPO / "bm_examinator.py", REPO / "bearmind" / "inspect_server.py", REPO / "bearmind" / "monitor.py"])
TABS = ("Monitor", "Scan", "Thresholds", "Inspection")  # the author's order, which is the window's (checked below)
OLD_TAB_NAMES = ("Скан", "Пороги")  # the plan's Russian tab names, replaced by the program's English ones
MAX_STEP_WORDS = 20
MAX_IMAGE_BYTES = 300_000

CODE_SPAN = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"^```.*?^```", re.M | re.S)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
BOLD = re.compile(r"\*\*(.+?)\*\*")
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
STEP = re.compile(r"^\s*\d+\.\s+(.*)$")
COMMAND = re.compile(r"run_bearmind(?:\.bat)?[ \t]+([-\w]+)")
EXTENSION = re.compile(r"\.[A-Za-z]\w{0,4}$")  # a file extension; "0.86" or "1.13.2" is a number, not a file


def _text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def guide() -> dict:
    missing = [p.relative_to(REPO).as_posix() for p in GUIDE if not p.is_file()]
    assert not missing, f"guide files missing: {missing}"
    return {p: _text(p) for p in GUIDE}


def _prose(text: str) -> str:
    """The text without fenced blocks and code spans (a code span becomes one placeholder word)."""
    return CODE_SPAN.sub("CODE", FENCE.sub("", text))


# --- commands ---

def _cli_commands() -> set:
    """The subcommands `python -m bearmind --help` lists (run_bearmind.bat runs exactly that)."""
    out = subprocess.run([sys.executable, "-m", "bearmind", "--help"], cwd=REPO, capture_output=True, text=True,
                         timeout=120, check=True).stdout
    block = out.split("positional arguments:", 1)[1].split("options:", 1)[0]
    return set(re.findall(r"^ {4}(\w+)\s", block, re.M))


def test_guide_commands_exist(guide):
    commands = _cli_commands()
    assert {"scan", "run", "gui", "status"} <= commands, f"help output not parsed: {sorted(commands)}"
    used, wrong = [], []
    for p, text in guide.items():
        for token in COMMAND.findall(text):
            used.append(token)
            ok = token in ("-h", "--help") if token.startswith("-") else token in commands
            if not ok:
                wrong.append(f"{p.name}: run_bearmind.bat {token}")
    assert "--help" in used, "the guide must show how to check the install with run_bearmind.bat --help"
    assert not wrong, f"not a command of python -m bearmind --help: {wrong}"


# --- paths ---

def _series_names() -> tuple:
    """(top folders, file names) a series folder holds, from bearmind.paths (design section 5.2): paths of a series
    in the guide are not repository paths."""
    from bearmind.paths import SeriesPaths
    root = Path("SERIES")
    sp = SeriesPaths(root)
    found = []
    for name, attr in vars(SeriesPaths).items():
        if name.startswith("__"):
            continue
        if isinstance(attr, property):
            value = getattr(sp, name)
        elif inspect.isfunction(attr) and list(inspect.signature(attr).parameters)[1:] in (["sid"], ["mouse"]):
            value = getattr(sp, name)("SID")
        else:
            continue
        if isinstance(value, Path):
            found.append(value.relative_to(root))
    return {p.parts[0] for p in found}, {p.name for p in found}


def _path_tokens(text: str):
    """Words of the code spans and fenced blocks that look like a file or folder."""
    spans = CODE_SPAN.findall(FENCE.sub("", text))
    spans += [line for block in FENCE.findall(text) for line in block.splitlines()[1:-1]]
    for span in spans:
        for token in span.split():
            token = token.strip("\"'(),;")
            if "/" in token or "\\" in token or EXTENSION.search(token):
                yield token


def test_guide_paths_exist(guide):
    folders, files = _series_names()
    assert {"1_Raw", "_bearmind", "3_MC"} <= folders, f"series layout not read from bearmind.paths: {folders}"
    missing, checked = [], 0
    for p, text in guide.items():
        for target in LINK.findall(text):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            checked += 1
            if not (p.parent / target.split("#", 1)[0]).exists():
                missing.append(f"{p.name}: link {target}")
        for token in _path_tokens(text):
            norm = token.replace("\\", "/")
            # a series path, a placeholder, a URL, the user's home or a drive: not in the repository
            if ("<" in norm or "://" in norm or norm.startswith(("~", "%")) or re.match(r"[A-Za-z]:/", norm)
                    or norm.split("/", 1)[0] in folders or ("/" not in norm.rstrip("/") and norm in files)):
                continue
            checked += 1
            if not (REPO / norm).exists():
                missing.append(f"{p.name}: {token}")
    assert checked >= 5, f"only {checked} repository paths found: is the parser right?"
    assert not missing, f"named in the guide but not in the repository: {missing}"


def test_guide_images_small_and_used(guide):
    images = sorted(IMG_DIR.glob("*"))
    assert images, "the guide shows the window: docs/students/img has no pictures"
    big = [f"{f.name} {f.stat().st_size} B" for f in images if f.stat().st_size >= MAX_IMAGE_BYTES]
    assert not big, f"pictures of {MAX_IMAGE_BYTES} B or more: {big}"
    linked = {(p.parent / t).resolve() for p, text in guide.items() for t in LINK.findall(text)}
    unused = [f.name for f in images if f.resolve() not in linked]
    assert not unused, f"pictures no page shows: {unused}"


# --- STE: sentences of numbered steps ---

def _steps(text: str):
    """The text of each numbered step, its indented continuation lines included."""
    lines = FENCE.sub("", text).splitlines()
    i = 0
    while i < len(lines):
        m = STEP.match(lines[i])
        i += 1
        if not m:
            continue
        parts = [m.group(1)]
        while i < len(lines) and lines[i].startswith("   ") and lines[i].strip() \
                and not STEP.match(lines[i]) and not lines[i].lstrip().startswith(("-", "!", "|", "*")):
            parts.append(lines[i].strip())
            i += 1
        yield " ".join(parts)


def _words(sentence: str) -> int:
    return len(re.findall(r"[\w][\w\-]*", sentence))


def _sentences(step: str):
    step = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", step)  # a picture is no words
    step = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", step)  # a link counts by its text
    step = _prose(step).replace("**", "")
    return [s for s in re.split(r"(?<=[.!?])\s+", step) if s.strip()]


def test_guide_sentence_length(guide):
    n_steps, long = 0, []
    for p, text in guide.items():
        for step in _steps(text):
            n_steps += 1
            for s in _sentences(step):
                if _words(s) > MAX_STEP_WORDS:
                    long.append(f"{p.name}: {_words(s)} words: {s}")
    assert n_steps >= 20, f"only {n_steps} numbered steps found: is the parser right?"
    assert not long, "step sentences over {} words:\n{}".format(MAX_STEP_WORDS, "\n".join(long))


# --- UI names ---

def _docstrings(tree) -> set:
    nodes = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                nodes.add(id(first.value))
    return nodes


def _ui_strings() -> list:
    """String literals of the UI sources (f-string pieces included, docstrings not: they are not shown)."""
    out = []
    for f in UI_SOURCES:
        tree = ast.parse(_text(f))
        skip = _docstrings(tree)
        out += [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip]
    return out


def _bold_ui_names(text: str) -> list:
    """English bold texts: the program's names. Bold with Cyrillic is Russian emphasis."""
    return [b.strip() for b in BOLD.findall(_prose(text)) if not CYRILLIC.search(b)]


def test_guide_ui_names_exist(guide):
    strings = _ui_strings()
    names, wrong = set(), []
    for p, text in guide.items():
        for name in _bold_ui_names(text):
            names.add(name)
            word = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)")
            if not any(word.search(s) for s in strings):
                wrong.append(f"{p.name}: **{name}**")
    assert len(names) >= 20, f"only {len(names)} bold UI names found: is the parser right?"
    assert not wrong, f"bold names that are no text of the window or the inspection page: {wrong}"


# --- tabs ---

def _window_tab_names() -> tuple:
    """TAB_NAMES of bearmind/gui/app.py, read without importing Qt."""
    tree = ast.parse(_text(REPO / "bearmind" / "gui" / "app.py"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TAB_NAMES" for t in n.targets):
            return tuple(ast.literal_eval(n.value))
    raise AssertionError("TAB_NAMES not found in bearmind/gui/app.py")


def test_guide_tab_order(guide):
    assert _window_tab_names() == TABS, "the window's tab order changed: update the guide and this test"
    workflow = guide[GUIDE_DIR / "series-workflow.md"]
    for p, text in guide.items():
        prose = _prose(text)
        first = {t: prose.find(f"**{t}**") for t in TABS if f"**{t}**" in prose}
        order = sorted(first, key=first.get)
        assert order == [t for t in TABS if t in first], f"{p.name}: tabs first named in the order {order}"
        old = [w for w in OLD_TAB_NAMES if w in text]
        assert not old, f"{p.name}: the plan's Russian tab names {old}; the window says {TABS}"
    assert all(f"**{t}**" in workflow for t in TABS), "series-workflow.md must name all four tabs"


# --- the root README points to the guide (carried item П-17) ---

def test_root_readme_points_to_guide():
    text = _text(REPO / "README.md")
    targets = set(LINK.findall(text))
    assert {"docs/students/README.md", "docs/students/series-workflow.md"} <= targets, sorted(targets)
    assert all((REPO / t.split("#", 1)[0]).exists() for t in targets if "://" not in t and not t.startswith("#"))
    assert "Jupyter" not in text, "the notebook is gone: the README must not tell to launch it"
