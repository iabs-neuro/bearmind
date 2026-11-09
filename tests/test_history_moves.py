"""The replaced pipeline and the MINI3 scripts live in history/; nothing outside it still reaches for them.

history/ is the record of how the paper's data were processed. bearmind does not import it; the equivalence test
does (its old side runs history.legacy_pipeline, the export comparison runs history.mini3.batch_export).
"""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# The last commit with the MINI3 scripts in the repo root and the old pipeline in bm_batch_routines.py.
BASE = "55278757e29a4e606f7a784e481fe47c050bb623"
MINI3 = ("batch_autoinspect_all.py", "batch_export_all.py", "chain_run_export_after_autoinspect.py",
         "resplit_fof.py", "batch_export.py")
# What bm_batch_routines still offers: bm_examinator imports extract_name_with_pattern.
KEPT = {"find_identifiers", "extract_name_with_pattern", "extract_session_id"}
MOVED = ("CleanMemmaps", "DrawFrameAndBox", "LoadSelectedVideos", "DrawCropper", "SaveCrops", "get_file_num_id",
         "get_bit_depth_tif", "extract_number", "DoCropAndRewrite", "DoCropAndRewriteTiff", "extract_and_copy_ts",
         "DoMotionCorrection", "DoCNMF", "ReDoCNMF", "FindMaxima", "Test_gSig_Range", "plot_gsig_range",
         "plot_min_corr_and_pnr_range", "crop_tiff", "convert_tiff_to_mp4")
GONE_MODULES = {"batch_export", "batch_autoinspect", "batch_autoinspect_all", "batch_export_all",
                "chain_run_export_after_autoinspect", "resplit_fof"}
# startup.py takes everything from bm_batch_routines with `import *` for the old notebook, which does not run any
# more (history/README.md).
STILL_STAR = {"startup.py"}
TIMEOUT_S = 600  # a cold import of CaImAn and moviepy


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, timeout=60)


def _need_git_base():
    if _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git checkout: the moved files cannot be compared with their history")
    r = _git("cat-file", "-e", f"{BASE}^{{commit}}")
    assert r.returncode == 0, f"commit {BASE} is not in this clone (a shallow clone?)"


def _py_files_outside_history():
    for p in sorted(REPO.rglob("*.py")):
        rel = p.relative_to(REPO)
        if rel.parts[0] == "history" or any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        yield rel, p


SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _own_nodes(scope):
    """The nodes of one scope, without the bodies of the functions and classes defined in it."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def _uses_of_moved_code(tree, rel) -> list[str]:
    found = []
    for scope in [tree, *(n for n in ast.walk(tree) if isinstance(n, SCOPES))]:
        aliases = set()  # names this scope binds to the module bm_batch_routines
        for node in _own_nodes(scope):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in GONE_MODULES:
                        found.append(f"{rel}:{node.lineno} import {a.name}")
                    if a.name == "bm_batch_routines":
                        aliases.add(a.asname or a.name)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module in GONE_MODULES:
                    found.append(f"{rel}:{node.lineno} from {node.module} import ...")
                if node.module == "bm_batch_routines":
                    for a in node.names:
                        if a.name not in KEPT and not (a.name == "*" and rel.as_posix() in STILL_STAR):
                            found.append(f"{rel}:{node.lineno} from bm_batch_routines import {a.name}")
        # a name like `old` means the module only where it was imported (other functions have their own `old`)
        for node in ast.walk(scope):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases
                    and node.attr not in KEPT):
                found.append(f"{rel}:{node.lineno} {node.value.id}.{node.attr}")
    return found


def test_no_imports_of_moved_names():
    found = []
    for rel, p in _py_files_outside_history():
        found += _uses_of_moved_code(ast.parse(p.read_bytes(), filename=str(rel)), rel)
    assert found == [], "code outside history/ uses what moved there:\n" + "\n".join(found)


def test_legacy_pipeline_importable(tmp_path):
    """All functions of the old bm_batch_routines but the two name helpers are in history.legacy_pipeline, with
    MotionCorrect as a module name (the equivalence test replaces it to keep the object DoMotionCorrection makes).
    The import in an empty folder leaves nothing there and nothing in BEARMIND_HOME."""
    _need_git_base()
    old = ast.parse(_git("show", f"{BASE}:bm_batch_routines.py").stdout)
    old_functions = {n.name for n in old.body if isinstance(n, ast.FunctionDef)}
    assert old_functions == set(MOVED) | (KEPT - {"extract_session_id"})
    now = ast.parse((REPO / "bm_batch_routines.py").read_bytes())
    assert {n.name for n in now.body if isinstance(n, ast.FunctionDef)} == KEPT - {"extract_session_id"}

    work, home = tmp_path / "work", tmp_path / "home"
    work.mkdir()
    code = "import json, history.legacy_pipeline as m; print(json.dumps(sorted(dir(m))))"
    env = {**os.environ, "PYTHONPATH": str(REPO), "BEARMIND_HOME": str(home)}
    r = subprocess.run([sys.executable, "-c", code], cwd=work, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=TIMEOUT_S)
    assert r.returncode == 0, r.stderr[-2000:]
    names = set(json.loads(r.stdout.strip().splitlines()[-1]))
    assert {"DoMotionCorrection", "DoCNMF", "ReDoCNMF", "MotionCorrect", *MOVED} <= names
    assert sorted(p.name for p in work.iterdir()) == []
    assert not home.exists() or sorted(p.name for p in home.iterdir()) == []


def test_mini3_scripts_kept_verbatim():
    """Byte for byte the scripts that processed the paper's data (git's own line-ending rules on both sides)."""
    _need_git_base()
    for name in MINI3:
        moved = REPO / "history" / "mini3" / name
        assert moved.is_file(), moved
        assert not (REPO / name).exists(), f"{name} is still in the repo root"
        r = _git("hash-object", str(moved))
        assert r.returncode == 0, r.stderr
        assert r.stdout.decode().strip() == _git("rev-parse", f"{BASE}:{name}").stdout.decode().strip(), name


def _imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def test_export_library_stays():
    """The batch's export library stays in the root: the export stage, the runner and ae_launch import it."""
    import export_estimates_data

    assert Path(export_estimates_data.__file__).resolve() == REPO / "export_estimates_data.py"
    for rel in ("bearmind/stages/export.py", "bearmind/runner.py", "ae_launch.py"):
        assert "export_estimates_data" in _imported_modules(REPO / rel), rel


def test_deleted_scripts_gone():
    assert not (REPO / "batch_autoinspect.py").exists()
    assert sorted(REPO.glob("history/**/batch_autoinspect.py")) == []
