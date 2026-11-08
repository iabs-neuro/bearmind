"""Importing a root module leaves no trace: no file in the start folder, no window, no global switch.

The notebook-era config.py wrote config.json into whatever folder Python started in, so the command line and the
runner changed to the repo root before work; research scripts chose the wavelet backend, blocked torch and switched
matplotlib's backend while being imported. Each root module is imported here in its own fresh interpreter, started
in an empty folder, with tkinter.Tk replaced by a failure.
"""
import ast
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import config  # noqa: E402

# These leave the root later in this phase, so this test does not hold them: the MINI3 scripts and batch_export.py
# move verbatim to history/mini3 and batch_autoinspect.py is deleted (task 3); startup.py is deleted with the
# notebook and the root test_*.py scripts move to experiments/testing (task 5). conftest.py belongs to pytest.
LEAVING = {"batch_autoinspect.py", "batch_autoinspect_all.py", "batch_export.py", "batch_export_all.py",
           "chain_run_export_after_autoinspect.py", "resplit_fof.py", "startup.py", "conftest.py"}
ROOT_MODULES = sorted(p.stem for p in REPO.glob("*.py")
                      if p.name not in LEAVING and not p.name.startswith("test_"))

# Loaded by every interpreter the tests start (sitecustomize on PYTHONPATH): a Tk window opened on import, as
# startup.py does, fails the import instead of appearing on the screen.
NO_TK = ("import tkinter\n"
         "class _NoTk:\n"
         "    def __init__(self, *args, **kwargs):\n"
         "        raise RuntimeError('tkinter.Tk created')\n"
         "tkinter.Tk = _NoTk\n")

TIMEOUT_S = 600  # one cold import of CaImAn, driada and Bokeh takes 15-30 s; several run at once


def _env(aux: Path) -> dict:
    site = aux / "site"
    site.mkdir(parents=True, exist_ok=True)
    (site / "sitecustomize.py").write_text(NO_TK, encoding="utf-8")
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), str(REPO)]), "BEARMIND_HOME": str(aux / "home")}


def _python(args, cwd, env):
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=TIMEOUT_S)


def _repo_state():
    """Names at the repo's top level and the bytes of its config.json (None when there is none)."""
    cfg = REPO / "config.json"
    return ({p.name for p in REPO.iterdir() if p.name != "__pycache__"},
            cfg.read_bytes() if cfg.is_file() else None)


def test_window_guard_works(tmp_path):
    """The guard below is real: a module that opens a Tk window fails under it."""
    work = tmp_path / "work"
    work.mkdir()
    r = _python(["-c", "from tkinter.filedialog import Tk\nTk()"], work, _env(tmp_path / "aux"))
    assert r.returncode != 0 and "tkinter.Tk created" in r.stderr, r.stderr[-2000:]


def test_root_modules_import_cleanly(tmp_path):
    assert {"config", "bm_examinator", "ae_launch", "collect_nof_aligned_events"} <= set(ROOT_MODULES)
    env = _env(tmp_path / "aux")
    before = _repo_state()

    def one(module):
        cwd = tmp_path / "cwd" / module  # one empty start folder per module: they import in parallel
        cwd.mkdir(parents=True)
        r = _python(["-c", f"import {module}"], cwd, env)
        return module, r, sorted(p.name for p in cwd.iterdir())

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        results = list(pool.map(one, ROOT_MODULES))
    failed = {m: r.stderr[-1500:] for m, r, _ in results if r.returncode != 0}
    left = {m: files for m, _, files in results if files}
    assert not failed, failed
    assert not left, f"files left in the start folder: {left}"
    assert _repo_state() == before, "an import changed the repo's top level or its config.json"


def test_research_scripts_set_nothing_global_on_import(tmp_path):
    """The wavelet backend (SSQ_* variables and wavelet_backend's one-shot choice), the torch block and matplotlib's
    backend belong to the script's main(); importing the module leaves them as they were."""
    code = ("import os, sys, matplotlib\n"
            "mpl = matplotlib.get_backend()\n"
            "import collect_nof_aligned_events, collect_wavelet_event_stats, summarize_nof_events, wavelet_backend\n"
            "assert 'SSQ_GPU' not in os.environ and 'SSQ_PARALLEL' not in os.environ, 'SSQ_* set'\n"
            "assert wavelet_backend.get_current_backend() == 'not_set', wavelet_backend.get_current_backend()\n"
            "assert sys.modules.get('torch', 'absent') is not None, 'torch blocked'\n"
            "assert matplotlib.get_backend() == mpl, (mpl, matplotlib.get_backend())\n")
    env = {k: v for k, v in _env(tmp_path / "aux").items() if not k.startswith("SSQ_") and k != "MPLBACKEND"}
    work = tmp_path / "work"
    work.mkdir()
    r = _python(["-c", code], work, env)
    assert r.returncode == 0, r.stderr[-2000:]


@pytest.fixture
def main_config(tmp_path, monkeypatch):
    """config.py's main file moved to tmp_path; CONFIG's contents come back afterwards (the session shares it)."""
    saved = dict(config.CONFIG)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    yield tmp_path / "config.json"
    config.CONFIG.clear()
    config.CONFIG.update(saved)


def test_update_config_visible_to_importers(main_config, tmp_path):
    from config import CONFIG as imported  # what bm_examinator and bm_batch_routines hold

    assert not main_config.exists()
    assert config.read_config() == config.DEFAULT_CONFIG  # no file yet: the defaults, and nothing is written
    assert not main_config.exists()
    config.update_config({"ROOT": str(tmp_path / "data") + os.sep, "DATA_PATHWAY": "legacy"})
    assert imported is config.CONFIG
    assert imported["ROOT"] == os.path.normpath(str(tmp_path / "data"))
    assert imported["DATA_PATHWAY"] == "legacy"
    saved = json.loads(main_config.read_text(encoding="utf-8"))
    assert saved["ROOT"] == imported["ROOT"] and saved["DATA_PATHWAY"] == "legacy"
    assert saved["TEMP_PATHWAY"] == config.DEFAULT_CONFIG["TEMP_PATHWAY"]  # the rest from the defaults
    assert config.DEFAULT_CONFIG["DATA_PATHWAY"] == "bonsai"  # the defaults themselves stay as they were


def test_unknown_data_pathway(tmp_path, monkeypatch):
    monkeypatch.setitem(config.CONFIG, "DATA_PATHWAY", "miniscope_v4")
    with pytest.raises(ValueError) as e:
        config.get_session_name_from_path(str(tmp_path / "HM_NOF_1D" / "NOF_H01_1D" / "0.avi"))
    assert str(e.value) == "unknown DATA_PATHWAY: 'miniscope_v4'; expected one of 'bonsai', 'legacy'"


def test_backend_does_not_depend_on_start_folder(tmp_path):
    """Autoinspection takes its joblib backend from config.json (config.get_parallelization_backend). A config.json
    in the start folder is not the machine's: the choice from any folder equals the one from the repo root."""
    env = _env(tmp_path / "aux")
    code = ["-c", "import config; print(config.get_parallelization_backend())"]
    r = _python(code, REPO, env)
    assert r.returncode == 0, r.stderr[-2000:]
    from_root = r.stdout.split()[-1]
    stray = "multiprocessing" if from_root != "multiprocessing" else "loky"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "config.json").write_text(json.dumps({**config.DEFAULT_CONFIG, "PARALLELIZATION_BACKEND": stray}),
                                           encoding="utf-8")
    r = _python(code, elsewhere, env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split()[-1] == from_root


def test_cli_from_other_cwd(tmp_path):
    from bearmind.paths import SeriesPaths
    from bearmind.registry import save_record
    from tests.bearmind.test_runner import SIDS, _record

    paths = SeriesPaths(tmp_path / "series")
    save_record(paths, _record(SIDS[0]))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    before = _repo_state()
    r = _python(["-m", "bearmind", "status", str(paths.root)], elsewhere, _env(tmp_path / "aux"))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert SIDS[0] in r.stdout
    assert sorted(tmp_path.rglob("config.json")) == []
    assert _repo_state() == before


@pytest.mark.parametrize("rel", ["bearmind/cli.py", "bearmind/runner.py"])
def test_cli_has_no_chdir(rel):
    """Paths are made absolute where they come in; nothing depends on the current folder."""
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    lines = [n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Call) and "chdir" in (getattr(n.func, "attr", None), getattr(n.func, "id", None))]
    assert lines == [], f"{rel}: chdir at lines {lines}"
