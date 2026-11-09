"""The student guide (docs/students) checked against the program: commands, paths, quoted texts, UI names, tab order,
step length.

Phase 6, task 6. The guide is Russian in STE style (Simplified Technical English rules carried over to Russian):
numbered steps, one action per step, at most 20 words per sentence in a step. The program's own names (tabs,
buttons, menus, statuses) stay English and are written in **bold**; Russian bold (a warning word) is not a UI name.
File names, folders, commands and texts the program shows (cell values, messages) are written as `code`.
"""
import ast
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GUIDE_DIR = REPO / "docs" / "students"
GUIDE = (GUIDE_DIR / "README.md", GUIDE_DIR / "series-workflow.md")
IMG_DIR = GUIDE_DIR / "img"
BEARMIND = REPO / "bearmind"
# The window's and the inspection page's texts: where a bold UI name must be a label. inspect_server.py holds the
# page's Next button and banners (ruling P6-3), bm_examinator.py the page's buttons (Delete sel., Save results).
# bearmind/monitor.py is the window's model: the stage columns (STAGE_TITLE: Crop, Autoinspect) and the cell texts
# are named only there.
UI_SOURCES = (sorted((BEARMIND / "gui").glob("*.py"))
              + [REPO / "bm_examinator.py", BEARMIND / "inspect_server.py", BEARMIND / "monitor.py"])
# Texts quoted as `code` may also come from the modules whose strings the window or the page show as they are: the
# Scan tab's anomaly titles (scan_view) and messages (scan), check messages in Warnings (checks), the Thresholds
# tab's states and source line (threshold_model), the calcium-wave banner (wave), Start's refusals (runner,
# launcher), the install check's console lines (test_env.py). The two launchers' console lines are read as text.
TEXT_SOURCES = UI_SOURCES + [BEARMIND / f"{m}.py" for m in ("scan_view", "scan", "checks", "threshold_model", "wave",
                                                             "runner", "launcher")] + [REPO / "test_env.py"]
LAUNCHERS = (REPO / "run_bearmind.bat", REPO / "bearmind_gui.bat")
# console logs of the processes the window starts, in the series' logs folder (bearmind/launcher.py,
# bearmind/gui/inspection_tab.py); bearmind.paths does not name them
LAUNCHER_LOGS = {"runner_console.log": BEARMIND / "launcher.py",
                 "inspect_console.log": BEARMIND / "gui" / "inspection_tab.py"}
# files of the machine folder ~/.bearmind (bearmind/config.py); a message may name one bare ("details in gui.log")
HOME_FILES = {"machine.yaml": BEARMIND / "config.py", "gui.log": BEARMIND / "gui" / "app.py"}
SHELL_COMMANDS = {"run_bearmind.bat", "bearmind_gui.bat", "conda", "mamba", "pip", "git", "cd", "mkdir", "python",
                  "setx"}
PLACEHOLDER_ARGS = {"sid": "<SID>", "mouse": "<MOUSE>", "kind": "<kind>"}
TABS = ("Monitor", "Scan", "Thresholds", "Inspection")  # the author's order, which is the window's (checked below)
OLD_TAB_NAMES = ("Скан", "Пороги")  # the plan's Russian tab names, replaced by the program's English ones
MAX_STEP_WORDS = 20
MAX_IMAGE_BYTES = 300_000

CODE_SPAN = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"^```.*?^```", re.M | re.S)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
BOLD = re.compile(r"\*\*(.+?)\*\*")
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LATIN = re.compile(r"[A-Za-z]")
STEP = re.compile(r"^\s*\d+\.\s+(.*)$")
COMMAND = re.compile(r"run_bearmind(?:\.bat)?[ \t]+([-\w]+)")
EXTENSION = re.compile(r"\.[A-Za-z]\w{0,4}$")  # a file extension; "0.86" or "1.13.2" is a number, not a file
PLACEHOLDER = re.compile(r"<[^<>]+>")
MARK = "\x00"  # a value the program puts into a text ({sid}, an f-string field)
FIELD = re.compile(r"\{[^{}]*\}")
TAG = re.compile(r"<[^<>]*>")  # HTML markup of a label (<b>Details — ...</b>)


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


# --- the program's strings ---

def _docstrings(tree) -> set:
    nodes = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                nodes.add(id(first.value))
    return nodes


def _render(node, env: dict) -> str:
    """A string literal as shown: a format field ({sid}) or an f-string value becomes MARK, an f-string value that
    names a module-level text constant becomes that text."""
    if isinstance(node, ast.Constant):
        return FIELD.sub(MARK, node.value)
    out = []
    for v in node.values:
        if isinstance(v, ast.Constant):
            out.append(v.value)
        elif isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name) and v.value.id in env:
            out.append(env[v.value.id])
        else:
            out.append(MARK)
    return "".join(out)


def _strings(path: Path) -> list:
    """Every string literal of a module as shown (docstrings left out: they are not shown)."""
    tree = ast.parse(_text(path))
    env = {}
    for n in tree.body:  # module-level text constants, in order, for f-strings that use them
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.JoinedStr)) \
                and (not isinstance(n.value, ast.Constant) or isinstance(n.value.value, str)):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    env[t.id] = _render(n.value, env)
    skip = _docstrings(tree)
    return [_render(n, env) for n in ast.walk(tree)
            if (isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip)
            or isinstance(n, ast.JoinedStr)]


@pytest.fixture(scope="module")
def ui_labels() -> list:
    return [TAG.sub("", s).strip() for f in UI_SOURCES for s in _strings(f)]


@pytest.fixture(scope="module")
def program_texts() -> list:
    texts = [s for f in TEXT_SOURCES for s in _strings(f)]
    return texts + [line for f in LAUNCHERS for line in _text(f).splitlines()]


# --- commands ---

def _cli_help() -> str:
    """`python -m bearmind --help` (run_bearmind.bat runs exactly that) at a fixed width, as the guide quotes it."""
    env = dict(os.environ, COLUMNS="80")
    return subprocess.run([sys.executable, "-m", "bearmind", "--help"], cwd=REPO, capture_output=True, text=True,
                          timeout=120, check=True, env=env).stdout


def _commands(help_text: str) -> set:
    block = help_text.split("positional arguments:", 1)[1].split("options:", 1)[0]
    return set(re.findall(r"^ {4}(\w+)\s", block, re.M))


@pytest.fixture(scope="module")
def cli_help() -> str:
    return _cli_help()


def test_guide_commands_exist(guide, cli_help):
    commands = _commands(cli_help)
    assert {"scan", "run", "gui", "status"} <= commands, f"help output not parsed: {sorted(commands)}"
    used, wrong = [], []
    for p, text in guide.items():
        for token in COMMAND.findall(text):
            used.append(token)
            ok = token in ("-h", "--help") if token.startswith("-") else token in commands
            if not ok:
                wrong.append(f"{p.name}: run_bearmind.bat {token}")
        # a quoted help output: its lines are lines of the real one
        help_lines = {line.rstrip() for line in cli_help.splitlines()}
        for block in FENCE.findall(text):
            lines = block.splitlines()[1:-1]
            if lines and lines[0].startswith("usage: run_bearmind"):
                wrong += [f"{p.name}: help line {line!r}" for line in lines if line.strip()
                          and line.rstrip() not in help_lines]
    assert "--help" in used, "the guide must show how to check the install with run_bearmind.bat --help"
    assert not wrong, f"not a command or a line of python -m bearmind --help: {wrong}"


# --- paths ---

def _series_layout() -> tuple:
    """(top folders, every path a series holds with its parent folders, file names) from bearmind.paths
    (design section 5.2) with <SID> / <MOUSE> / <kind> for the arguments, plus the launchers' console logs."""
    from bearmind.paths import SeriesPaths
    root = Path("SERIES")
    sp = SeriesPaths(root)
    found = []
    for name, attr in vars(SeriesPaths).items():
        if name.startswith("__"):
            continue
        if isinstance(attr, property):
            value = getattr(sp, name)
        elif inspect.isfunction(attr):
            params = list(inspect.signature(attr).parameters)[1:]
            if not params or any(a not in PLACEHOLDER_ARGS for a in params):
                continue
            value = getattr(sp, name)(*[PLACEHOLDER_ARGS[a] for a in params])
        else:
            continue
        if isinstance(value, Path):
            found.append(value.relative_to(root).as_posix())
    for log, source in LAUNCHER_LOGS.items():
        assert log in _strings(source), f"{log} is not named in {source.name} any more"
        found.append(f"{sp.rel(sp.logs_dir)}/{log}")
    paths = {"/".join(p.split("/")[:i + 1]) for p in found for i in range(p.count("/") + 1)}
    return {p.split("/")[0] for p in found}, paths, {p.rsplit("/", 1)[-1] for p in found}


def _pathlike(token: str) -> bool:
    return "/" in token or "\\" in token or bool(EXTENSION.search(token))


def _external(norm: str) -> bool:
    """A URL, the user's home, an environment variable or a drive: not in the repository nor in a series."""
    return "://" in norm or norm.startswith(("~", "%")) or bool(re.match(r"[A-Za-z]:(/|$)", norm))


def _path_tokens(text: str):
    """Words of the code spans and fenced blocks that look like a file or folder."""
    spans = CODE_SPAN.findall(FENCE.sub("", text))
    spans += [line for block in FENCE.findall(text) for line in block.splitlines()[1:-1]]
    for span in spans:
        for token in span.split():
            token = token.strip("\"'(),;")
            if _pathlike(token):
                yield token


def _path_problem(token: str, layout, scan_text: str):
    """Why a path of the guide is wrong (None: it is right). A series output must be a path bearmind.paths builds;
    the raw-data layout (1_Raw, 0_TimeStamps, raw file names) is the scan's input: its words must be the scan's."""
    tops, series, names = layout
    norm = token.replace("\\", "/")
    bare = norm.rstrip("/")
    if _external(norm) or norm in HOME_FILES:
        return None
    top = bare.split("/", 1)[0]
    raw = top in ("1_Raw", "0_TimeStamps") or ("/" not in bare and "<" in bare and bare not in names)
    if raw:
        rest = PLACEHOLDER.sub("", bare.split("/", 1)[1] if top in ("1_Raw", "0_TimeStamps") and "/" in bare
                               else bare)
        unknown = [w for w in re.findall(r"[A-Za-z]{3,}", rest) if w not in scan_text]
        return f"raw-data layout words not used by bearmind/scan.py: {unknown}" if unknown else None
    if top in tops or ("/" not in bare and bare in names):
        return None if bare in series or bare in names else "not a path of bearmind.paths"
    return None if (REPO / norm).exists() else "not in the repository"


def test_guide_paths_exist(guide):
    layout = _series_layout()
    assert {"1_Raw", "_bearmind", "3_MC"} <= layout[0], f"series layout not read from bearmind.paths: {layout[0]}"
    assert "3_MC/<SID>_MC.avi" in layout[1] and "_bearmind/logs/runner_console.log" in layout[1]
    for name, source in HOME_FILES.items():
        assert name in _text(source), f"{name} is not named in {source.name} any more"
    scan_text = _text(BEARMIND / "scan.py")
    wrong, checked = [], 0
    for p, text in guide.items():
        for target in LINK.findall(text):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            checked += 1
            if not (p.parent / target.split("#", 1)[0]).exists():
                wrong.append(f"{p.name}: link {target}")
        for token in _path_tokens(text):
            checked += 1
            why = _path_problem(token, layout, scan_text)
            if why:
                wrong.append(f"{p.name}: {token}: {why}")
    assert checked >= 20, f"only {checked} paths found: is the parser right?"
    assert not wrong, "wrong paths in the guide:\n" + "\n".join(wrong)


def test_guide_images_small_and_used(guide):
    images = sorted(IMG_DIR.glob("*"))
    assert images, "the guide shows the window: docs/students/img has no pictures"
    big = [f"{f.name} {f.stat().st_size} B" for f in images if f.stat().st_size >= MAX_IMAGE_BYTES]
    assert not big, f"pictures of {MAX_IMAGE_BYTES} B or more: {big}"
    linked = {(p.parent / t).resolve() for p, text in guide.items() for t in LINK.findall(text)}
    unused = [f.name for f in images if f.resolve() not in linked]
    assert not unused, f"pictures no page shows: {unused}"


# --- texts quoted as code ---

def _span_regex(span: str):
    """`…` stands for any text, <X> for a value, a number for itself or for a value the program fills in."""
    out = []
    for piece in re.split(r"(<[^<>]+>|…|(?<![\w.])\d+(?:\.\d+)?(?!\w))", span):
        if not piece:
            continue
        if piece == "…":
            out.append(".*?")
        elif PLACEHOLDER.fullmatch(piece):
            out.append(".+?")
        elif re.fullmatch(r"\d+(?:\.\d+)?", piece):
            out.append(f"(?:{re.escape(piece)}|{MARK})")
        else:
            out.append(re.escape(piece))
    return re.compile("".join(out), re.S)


def _span_kind(span: str, tops: set, commands: set, env_name: str) -> str:
    from bearmind import naming
    tokens = span.split()
    if tokens[0] in SHELL_COMMANDS or span in commands:
        return "command"
    if len(tokens) == 1 and (_pathlike(span) or span.rstrip("/") in tops):
        return "path"
    if span == env_name or re.fullmatch(naming.SESSION_PATTERN, span) \
            or re.fullmatch(naming.SESSION_PATTERN, span + "_1D"):
        return "name"  # the environment, an example session or mouse
    if PLACEHOLDER.fullmatch(span) or (CYRILLIC.search(span) and not LATIN.search(span)):
        return "placeholder"  # <SID>, a Russian template
    return "text"


def test_guide_code_texts_exist(guide, program_texts, cli_help):
    """Every code span that is not a command, a path or a name is a text the program shows."""
    tops = _series_layout()[0]
    commands = _commands(cli_help)
    env_name = re.search(r"^name:\s*(\S+)", _text(REPO / "environment.yml"), re.M).group(1)
    n_texts, wrong = 0, []
    for p, text in guide.items():
        for span in CODE_SPAN.findall(FENCE.sub("", text)):
            if _span_kind(span, tops, commands, env_name) != "text":
                continue
            n_texts += 1
            rx = _span_regex(span)
            if not any(rx.search(s) for s in program_texts):
                wrong.append(f"{p.name}: `{span}`")
    assert n_texts >= 30, f"only {n_texts} quoted texts found: is the parser right?"
    assert not wrong, "quoted as code, but no text of the program:\n" + "\n".join(wrong)


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

def _bold_ui_names(text: str) -> list:
    """English bold texts: the program's names. Bold with Cyrillic is Russian emphasis."""
    return [b.strip() for b in BOLD.findall(_prose(text)) if not CYRILLIC.search(b)]


def _is_label(name: str, label: str) -> bool:
    """The whole label, or its start up to a sign ("Warnings — 3 sessions", "Next: <SID> →", "Log — <SID>"); a
    word cut short ("Stop" of "Stop after current stage") is not a label."""
    if not label.startswith(name):
        return False
    rest = label[len(name):]
    return rest == "" or re.match(r"\s*[^\w\s]", rest) is not None


def test_guide_ui_names_exist(guide, ui_labels):
    names, wrong = set(), []
    for p, text in guide.items():
        for name in _bold_ui_names(text):
            names.add(name)
            if not any(_is_label(name, label) for label in ui_labels):
                wrong.append(f"{p.name}: **{name}**")
    assert len(names) >= 20, f"only {len(names)} bold UI names found: is the parser right?"
    assert not wrong, f"bold names that are no label of the window or the inspection page: {wrong}"


# --- tabs ---

def _window_tab_names() -> tuple:
    """TAB_NAMES of bearmind/gui/app.py, read without importing Qt."""
    tree = ast.parse(_text(BEARMIND / "gui" / "app.py"))
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
