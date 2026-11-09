"""The replaced pipeline and the MINI3 scripts live in history/; nothing outside it still reaches for them.

history/ is the record of how the paper's data were processed. bearmind does not import it; the equivalence test
does (its old side runs history.legacy_pipeline, the export comparison runs history.mini3.batch_export).
"""
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# git blob ids of the MINI3 scripts as they last were in the repo root, before the move to history/mini3. The
# content is pinned, not a commit: a squashed, rebased or shallow history must not break the check.
MINI3 = {"batch_autoinspect_all.py": "7c1e449b89c6689b17999e6d539fccb147a35e70",
         "batch_export_all.py": "161be541df4a0140b7bb7c37640496db45e33c25",
         "chain_run_export_after_autoinspect.py": "cef73f50ccbc0ffc93c926d466fcbc6df84f71f0",
         "resplit_fof.py": "a52630932e24fe5d3b230cff175be4f56d2a457f",
         "batch_export.py": "4ab30d284a40c49a19702ce0fff6a6872579fda2"}
# What bm_batch_routines still offers: bm_examinator imports extract_name_with_pattern.
KEPT = {"find_identifiers", "extract_name_with_pattern", "extract_session_id"}
MOVED = ("CleanMemmaps", "DrawFrameAndBox", "LoadSelectedVideos", "DrawCropper", "SaveCrops", "get_file_num_id",
         "get_bit_depth_tif", "extract_number", "DoCropAndRewrite", "DoCropAndRewriteTiff", "extract_and_copy_ts",
         "DoMotionCorrection", "DoCNMF", "ReDoCNMF", "FindMaxima", "Test_gSig_Range", "plot_gsig_range",
         "plot_min_corr_and_pnr_range", "crop_tiff", "convert_tiff_to_mp4")
GONE_MODULES = {"batch_export", "batch_autoinspect", "batch_autoinspect_all", "batch_export_all",
                "chain_run_export_after_autoinspect", "resplit_fof"}
TIMEOUT_S = 600  # a cold import of CaImAn and moviepy


def _blob_id(path: Path) -> str:
    """git's blob id of the file as git stores it, computed without git. With core.autocrlf=true a Windows checkout
    has CRLF while the blob has LF, so CRLF is folded back to LF before hashing."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


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
                        if a.name not in KEPT:  # `import *` too: code written that way expects the moved names
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


def _functions(path: Path) -> set[str]:
    return {n.name for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef)}


def test_legacy_pipeline_importable(tmp_path):
    """All functions of the old bm_batch_routines but the two name helpers are in history.legacy_pipeline, with
    MotionCorrect as a module name (the equivalence test replaces it to keep the object DoMotionCorrection makes).
    The import in an empty folder leaves nothing there and nothing in BEARMIND_HOME."""
    assert _functions(REPO / "history" / "legacy_pipeline.py") == set(MOVED)
    assert _functions(REPO / "bm_batch_routines.py") == KEPT - {"extract_session_id"}

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
    """Byte for byte the MINI3 scripts as they were in the repo root (line endings as git stores them)."""
    for name, blob in MINI3.items():
        moved = REPO / "history" / "mini3" / name
        assert moved.is_file(), moved
        assert not (REPO / name).exists(), f"{name} is still in the repo root"
        assert _blob_id(moved) == blob, name


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
