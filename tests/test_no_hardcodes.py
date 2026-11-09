"""No machine paths, file-name slicing or second copies of what has one source (design appendix A, phase 6 task 4).

Production code is the root modules and bearmind/. The absolute-path check also reads tests/. Not read: history/ (the
record of how the paper's data were processed, kept verbatim), paper/, and the research and check records
experiments/, ml/ and docs/ (scripts kept as they were run; they are not edited).
"""
import ast
import inspect
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bearmind import config as bm_config  # noqa: E402

# As test_import_side_effects: conftest.py and the root test_*.py script (test_env.py) belong to pytest.
NOT_MODULES = {"conftest.py"}
ROOT_MODULES = sorted(p for p in REPO.glob("*.py") if p.name not in NOT_MODULES and not p.name.startswith("test_"))
PRODUCTION = ROOT_MODULES + sorted((REPO / "bearmind").rglob("*.py"))


def _rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


def _parsed(files):
    for p in files:
        src = p.read_text(encoding="utf-8")
        yield _rel(p), src, ast.parse(src)


def _docstrings(tree) -> set:
    nodes = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                nodes.add(id(first.value))
    return nodes


def _code_strings(tree):
    """String constants of the code; docstrings are prose and may name a file."""
    skip = _docstrings(tree)
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip]


# --- absolute paths ---

# A drive letter, slashes or backslashes (doubled inside a string literal), then a folder of a person's account or of
# a lab machine's data tree.
ABSOLUTE = re.compile(r"[A-Za-z]:[\\/]+(?:Users|DATASET|_Projects)[\\/]", re.IGNORECASE)


def test_no_absolute_user_paths():
    files = PRODUCTION + sorted((REPO / "tests").rglob("*.py"))
    hits = [f"{_rel(p)}:{i}: {line.strip()[:100]}"
            for p in files for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if ABSOLUTE.search(line)]
    assert hits == []


def test_absolute_path_pattern_sees_both_slashes():
    assert ABSOLUTE.search("ROOT = 'C:" + "\\\\Users\\\\1\\\\data'")
    assert ABSOLUTE.search("default=r'D:" + "/_Projects/NOF'")
    assert ABSOLUTE.search("W:" + "\\DATASET\\NOF")
    assert not ABSOLUTE.search("H:/Projects/LT1")


# --- file-name slicing ---

NAMEY = re.compile(r"name|session|stem", re.IGNORECASE)
# capcan_validation keys the validation corpus by the first 10 characters of its file names (build_session_mapping,
# load_fps_data) and names its output folders capcan_artifacts_<key>. The corpus (data/final_compressed) is not on
# this machine, so it cannot be shown that extract_session_id gives the same keys; for an 11-character ID such as
# LNOF_J01_1D it would not. Left as is (ruling P-10 of the phase 6 audit).
ALLOWED_SLICES = {"capcan_validation.py": 2}


def _int_const(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, int):
        return n.value
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant):
        return -n.operand.value
    return None


def _is_path_split(call) -> bool:
    """x.split(os.sep), x.split('/') or x.split('\\')."""
    if not (isinstance(call, ast.Call) and getattr(call.func, "attr", None) in ("split", "rsplit") and call.args):
        return False
    sep = call.args[0]
    if isinstance(sep, ast.Attribute) and sep.attr in ("sep", "altsep"):
        return True
    return isinstance(sep, ast.Constant) and sep.value in ("/", "\\")


def _name_slices(src, tree):
    """Character slices of a name by fixed counts (name[:10], name[-20:-10]) and path parts picked by position
    (split(os.sep)[-2], split('/')[6:]); the last part (the file name, [-1]) is not a position."""
    split_vars = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) and _is_path_split(n.value)
                  for t in n.targets if isinstance(t, ast.Name)}
    hits = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Subscript):
            continue
        s = n.slice
        text = ast.get_source_segment(src, n)
        if isinstance(s, ast.Slice):
            bounds = [_int_const(b) for b in (s.lower, s.upper) if b is not None]
            if any(b is not None and abs(b) >= 3 for b in bounds) and NAMEY.search(ast.get_source_segment(src, n.value)):
                hits.append(text)
                continue
        on_path = _is_path_split(n.value) or (isinstance(n.value, ast.Name) and n.value.id in split_vars)
        if on_path and not (_int_const(s) == -1):
            hits.append(text)
    return hits


def test_name_slice_detector():
    src = ("a = name[:10]\nb = fname[-20:-10]\nparts = p.split(os.sep)\nc = parts[-2]\nd = p.split('/')[6:]\n"
           "e = p.split('/')[-1]\nf = names[:-1]\ng = arr[:10]\n")
    assert _name_slices(src, ast.parse(src)) == ["name[:10]", "fname[-20:-10]", "parts[-2]", "p.split('/')[6:]"]


def test_no_name_slicing():
    found = {rel: hits for rel, src, tree in _parsed(PRODUCTION) if (hits := _name_slices(src, tree))}
    assert {rel: len(h) for rel, h in found.items()} == ALLOWED_SLICES, found


def test_session_name_from_path_layouts(tmp_path, monkeypatch):
    import config

    bonsai = tmp_path / "HM_NOF_1D" / "NOF_H01_1D" / "0.avi"
    monkeypatch.setitem(config.CONFIG, "DATA_PATHWAY", "bonsai")
    assert config.get_session_name_from_path(str(bonsai)) == "NOF_H01_1D"
    legacy = tmp_path / "NOF_H02_1D" / "2021_03_04" / "12_00_00" / "Miniscope" / "0.avi"
    monkeypatch.setitem(config.CONFIG, "DATA_PATHWAY", "legacy")
    assert config.get_session_name_from_path(str(legacy)) == "NOF_H02_1D"


# --- autoinspection parameters: bearmind/defaults.yaml ---

PARAMS = {"ml_threshold", "n_iter", "pxlthr_distance_boundary", "brain", "ml_model_path", "model_path", "ml_model",
          "detect_corner_artifacts"}
# Library and view defaults that the pipeline does not use (ruling P-7): auto_inspector's own function defaults,
# the manual-inspection view of bm_examinator, and capcan_validation's validation set (its own rules).
NOT_PIPELINE = {"auto_inspector.py", "bm_examinator.py", "capcan_validation.py"}


def _literal(n) -> bool:
    return isinstance(n, ast.Constant) and n.value is not None


def _param_literals(tree):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = n.args
            pos = a.posonlyargs + a.args
            pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + list(zip(a.kwonlyargs, a.kw_defaults))
            out += [f"{arg.arg}={d.value!r} (line {d.lineno})" for arg, d in pairs if arg.arg in PARAMS and _literal(d)]
        elif isinstance(n, ast.Call):
            out += [f"{k.arg}={k.value.value!r} (line {k.value.lineno})" for k in n.keywords
                    if k.arg in PARAMS and _literal(k.value)]
            if getattr(n.func, "attr", None) == "add_argument" and n.args and isinstance(n.args[0], ast.Constant):
                flag = str(n.args[0].value).lstrip("-").replace("-", "_")
                out += [f"--{flag} default={k.value.value!r} (line {k.value.lineno})" for k in n.keywords
                        if k.arg == "default" and flag in PARAMS and _literal(k.value)]
        elif isinstance(n, (ast.Assign, ast.AnnAssign)) and _literal(n.value):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            out += [f"{t.id} = {n.value.value!r} (line {n.lineno})" for t in targets
                    if isinstance(t, ast.Name) and any(p in t.id.lower() for p in PARAMS)]
        elif isinstance(n, ast.Dict):
            out += [f"{k.value!r}: {v.value!r} (line {k.lineno})" for k, v in zip(n.keys, n.values)
                    if isinstance(k, ast.Constant) and k.value in PARAMS and _literal(v)]
    return out


def test_param_literal_detector():
    src = ("def f(x, n_iter=2, *, ml_threshold=0.71, brain=None): pass\n"
           "g(pxlthr_distance_boundary=15)\nDEFAULT_ML_THRESHOLD = 0.72\nd = {'brain': 'ml'}\n"
           "p.add_argument('--n-iter', type=int, default=2)\nh(n_iter=n)\n")
    assert len(_param_literals(ast.parse(src))) == 6


def test_autoinspect_params_single_source():
    scope = [p for p in PRODUCTION if _rel(p) not in NOT_PIPELINE]
    found = {rel: hits for rel, _, tree in _parsed(scope) if (hits := _param_literals(tree))}
    assert found == {}


def test_ae_launch_defaults_are_defaults_yaml():
    """A direct call of run_auto_inspection runs as the autoinspect stage does (the stage passes every one of these)."""
    import ae_launch

    section = bm_config.load_defaults()["autoinspect"]
    params = inspect.signature(ae_launch.run_auto_inspection).parameters
    for key, value in section.items():
        if key == "model_path":
            assert Path(params["ml_model_path"].default) == REPO / value
            assert (REPO / value).is_file()
        else:
            assert params[key].default == value, key


# Parameters of run_auto_inspection that the stage does not pass and its hash does not hold (audit row N22): the
# stage runs with these defaults, so changing one would change autoinspection silently.
HIDDEN_DEFAULTS = {"comps_to_select": None, "num_sessions": 1, "match_threshold": 3, "sf": None, "ef": None, "ds": 1,
                   "corner_artifact_params": None, "wavelet_backend": "auto", "deletion_rules": None,
                   "track_criteria_failures": True, "save_artifacts": True, "save_estimates": True,
                   "save_matrices": True, "save_corner_detection": True, "compress_estimates": True}


def test_ae_launch_hidden_defaults_unchanged():
    import ae_launch

    params = inspect.signature(ae_launch.run_auto_inspection).parameters
    assert {k: params[k].default for k in HIDDEN_DEFAULTS} == HIDDEN_DEFAULTS


def test_export_fallback_threshold_is_defaults_yaml():
    """A processed file without autoinspection_config is exported at the threshold the stage hash assumes for it
    (runner.stage_params: autoinspect.ml_threshold)."""
    import export_estimates_data

    assert export_estimates_data.DEFAULT_ML_THRESHOLD == bm_config.load_defaults()["autoinspect"]["ml_threshold"]


# --- fps_data.csv: one reader ---

def test_one_fps_table_reader():
    """Only the reader, bearmind/timestamps.py, builds a path to fps_data.csv (a string that is exactly the file
    name); messages and help texts may name it."""
    hits = sorted({rel for rel, _, tree in _parsed(PRODUCTION) for s in _code_strings(tree) if s == "fps_data.csv"})
    assert hits == ["bearmind/timestamps.py"]


def test_fps_lookup_rounds_only_for_autoinspection():
    import ae_launch
    import export_estimates_data
    from bearmind.timestamps import fps_from_table, read_fps_table

    exact = read_fps_table()["NOF_H01_1D"]
    assert exact == pytest.approx(19.76)
    assert fps_from_table("x/NOF_H01_1D_estimates.pickle") == exact
    assert fps_from_table("NOF_H01_1D_2T") == exact  # a trial without its own row: the session's rate
    assert fps_from_table("3DM_D17_1D_1T") == read_fps_table()["3DM_D17_1D_1T"]
    assert fps_from_table("XYZ_Q99_9D") is None
    # autoinspection: integer fps, as the ML features and the paper data were built
    assert ae_launch.get_fps_from_table("NOF_H01_1D") == 20
    assert isinstance(ae_launch.get_fps_from_table("NOF_H01_1D"), int)
    # export: the stored rate, the time axis of the exported traces
    assert export_estimates_data.get_fps_from_table("NOF_H01_1D") == exact
    assert export_estimates_data.get_fps("NOF_H01_1D") == exact


def test_fps_missing_from_table():
    import ae_launch
    import export_estimates_data

    with pytest.raises(ValueError, match="XYZ_Q99_9D"):
        ae_launch.get_fps_from_table("XYZ_Q99_9D")  # was a silent 30
    assert ae_launch.get_fps_from_table("XYZ_Q99_9D", default_fps=30) == 30
    assert export_estimates_data.get_fps_from_table("XYZ_Q99_9D") is None
    with pytest.raises(ValueError, match="XYZ_Q99_9D"):
        export_estimates_data.get_fps("XYZ_Q99_9D")


def test_fps_table_refuses_a_name_twice(tmp_path):
    from bearmind.timestamps import read_fps_table

    f = tmp_path / "fps.csv"
    f.write_text("#;Filename;FPS\n1;NOF_H01_1D;19.76\n2;NOF_H01_1D;29.78\n", encoding="utf-8")
    with pytest.raises(ValueError, match="NOF_H01_1D"):
        read_fps_table(f)


def test_capcan_fps_map_unchanged():
    """The validation's FPS map, built from the shared reader, equals the map its own reader built."""
    import capcan_validation

    old = {}
    for _, row in pd.read_csv(REPO / "fps_data.csv", sep=";").iterrows():
        old[row["Filename"][:10]] = int(round(row["FPS"]))
    assert capcan_validation.load_fps_data() == old
    assert capcan_validation.fps_map == old


def test_get_fps_has_one_meaning():
    """table_routines.get_fps (rate from a timestamps table) and export_estimates_data.get_fps (rate from
    fps_data.csv) share a name: no module takes table_routines' names wholesale."""
    star = sorted(rel for rel, _, tree in _parsed(PRODUCTION) for n in ast.walk(tree)
                  if isinstance(n, ast.ImportFrom) and n.module == "table_routines" and any(a.name == "*" for a in n.names))
    assert star == []


# --- PNR -> imax: bm_qc.pnr_to_imax ---

def test_pnr_to_imax_single_source():
    hits = sorted(rel for rel, src, _ in _parsed(PRODUCTION) if re.search(r"pnr\w*\s*\*\s*255", src))
    assert hits == ["bm_qc.py"]


def test_pnr_to_imax_as_the_cnmf_stage_made_it():
    import bm_qc

    rng = np.random.default_rng(0)
    pnr = rng.uniform(-5, 120, (20, 30)).astype(np.float32)
    pnr[0, :3] = [np.nan, np.inf, -np.inf]
    old = np.clip(np.nan_to_num(pnr, nan=0.0), 0, 70)
    old = (old * 255 / float(old.max())).astype(np.uint8)
    got = bm_qc.pnr_to_imax(pnr)
    assert got.dtype == np.uint8 and np.array_equal(got, old)
    assert np.array_equal(bm_qc.pnr_to_imax(np.zeros((3, 4), np.float32)), np.zeros((3, 4), np.uint8))


def test_manual_seeds_image_unchanged(tmp_path, monkeypatch):
    """build_average_image (ManualSeeds) keeps its notebook look: a pixel with infinite PNR is black."""
    import bm_examinator
    import tifffile

    f = tmp_path / "NOF_H01_1D_MC.tif"
    tifffile.imwrite(f, np.zeros((10, 5, 6), np.uint8))
    pnr = np.linspace(0, 100, 30, dtype=np.float32).reshape(5, 6)
    pnr[0, :2] = [np.inf, np.nan]
    monkeypatch.setattr(bm_examinator.cm.summary_images, "correlation_pnr",
                        lambda data, gSig, swap_dim: (None, pnr.copy()))
    old = pnr.copy()
    old[np.where(old == np.inf)] = 0
    old[np.where(old > 70)] = 70
    old[np.isnan(old)] = 0
    old = (old * 255 / np.max(old)).astype("uint8")
    assert np.array_equal(bm_examinator.build_average_image(str(f), 4), old)


# --- export library: no command line ---

def test_export_library_has_no_command_line():
    import export_estimates_data as lib

    tree = ast.parse(Path(lib.__file__).read_text(encoding="utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "argparse" not in imported and not hasattr(lib, "main")
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name)
                and n.left.id == "__name__"]
    stage = (REPO / "bearmind" / "stages" / "export.py").read_text(encoding="utf-8")
    used = set(re.findall(r"\bold\.(\w+)", stage)) | {"MERGE_GROUPING", "get_fps", "get_fps_from_table"}
    assert used <= set(dir(lib)), used - set(dir(lib))


PROCESS_CALLS = {"run", "call", "check_call", "check_output", "Popen", "system", "spawnv", "spawnl"}


def _runs_export_script(tree) -> list:
    """Lines of process calls whose arguments name export_estimates_data (as a file or as -m)."""
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and (getattr(n.func, "attr", None) or getattr(n.func, "id", None)) in PROCESS_CALLS
            and any(isinstance(c, ast.Constant) and isinstance(c.value, str) and "export_estimates_data" in c.value
                    for a in [*n.args, *(k.value for k in n.keywords)] for c in ast.walk(a))]


def test_export_script_detector():
    src = ("subprocess.run([sys.executable, str(REPO / 'export_estimates_data.py'), p], check=True)\n"
           "subprocess.check_call([sys.executable, '-m', 'export_estimates_data'])\n"
           "import export_estimates_data\nAUDITED = ['export_estimates_data.py']\n")
    assert _runs_export_script(ast.parse(src)) == [1, 2]


def test_no_test_runs_the_export_script():
    """export_estimates_data has no command line: a test that runs it as a script gets exit 0 and no files."""
    found = {rel: lines for rel, _, tree in _parsed(sorted((REPO / "tests").rglob("*.py")))
             if (lines := _runs_export_script(tree))}
    assert found == {}


# --- broad excepts ---

# The modules of audit row A28. In auto_inspector the handlers only gained a log line: narrowing them would change
# decisions on edge cases (ruling P-11).
AUDITED_EXCEPTS = ["auto_inspector.py", "bm_spike_detection.py", "estimates_compression.py", "table_routines.py",
                   "ae_launch.py", "export_estimates_data.py"]


def _broad(h) -> bool:
    if h.type is None:
        return True
    names = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
    return any(getattr(n, "id", None) in ("Exception", "BaseException") for n in names)


def _leaves_a_trace(h) -> bool:
    """The handler raises, logs, prints or warns, or passes the exception on."""
    for n in ast.walk(ast.Module(body=h.body, type_ignores=[])):
        if isinstance(n, ast.Raise) or (h.name and isinstance(n, ast.Name) and n.id == h.name):
            return True
        if isinstance(n, ast.Call):
            f = n.func
            if getattr(f, "id", None) == "print" or getattr(f, "attr", None) == "warn" \
                    or getattr(getattr(f, "value", None), "id", None) in ("log", "logging", "warnings"):
                return True
    return False


def test_no_silent_broad_except():
    silent = [f"{rel}:{h.lineno}" for rel, _, tree in _parsed(REPO / f for f in AUDITED_EXCEPTS)
              for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler) and _broad(h) and not _leaves_a_trace(h)]
    assert silent == []


def test_no_bare_except():
    bare = [f"{rel}:{h.lineno}" for rel, _, tree in _parsed(PRODUCTION)
            for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler) and h.type is None]
    assert bare == []
