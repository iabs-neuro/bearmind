"""Helpers of test_lt1_snapshot: copy a series' service folder (registry and configs) into a temporary series and
find the finished stages whose stored hash no longer matches the code. Not a test module."""
import os
import shutil
from pathlib import Path

from bearmind import config, registry, runner
from bearmind.paths import SeriesPaths
from bearmind.registry import STAGES, State

SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "lt1_snapshot"
SNAPSHOT_SIDS = ("LT1_J01_1D", "LT1_J03_2D", "LT1_J05_1D", "LT1_J06_1D", "LT1_J10_1D", "LT1_J10_2D", "LT1_J17_2D",
                 "LT1_J24_3D", "LT1_J52_1D")
# The machine of the LT1 batch (~/.bearmind/machine.yaml, 2026-11-16). The RAM budget picks ssub_B, an input of
# the CNMF-E hash, so another machine would show every finished CNMF-E stage as changed.
MACHINE = {"temp_dir": "C:/bearmind_test_temp", "n_cpus": 32, "ram_gb": 191.2, "cnmf_ram_budget_gb": 152.96}


def copy_series(bearmind_dir: Path, root: Path) -> SeriesPaths:
    """Copy `<bearmind_dir>/registry/*.json` and the config tree into `root/_bearmind`, write MACHINE to the
    test's machine file and make an empty file for every required output of every DONE main stage (StageView.current
    then judges by hashes, as in a real series). The source is only read."""
    bearmind_dir, root = Path(bearmind_dir), Path(root)
    paths = SeriesPaths(root)
    (root / "1_Raw").mkdir(parents=True, exist_ok=True)
    dst = root / "_bearmind"
    (dst / "registry").mkdir(parents=True, exist_ok=True)
    for f in sorted((bearmind_dir / "registry").glob("*.json")):
        shutil.copyfile(f, dst / "registry" / f.name)
    if (bearmind_dir / "config").is_dir():
        shutil.copytree(bearmind_dir / "config", dst / "config", dirs_exist_ok=True)
    config.update_yaml(config.machine_file(), MACHINE)
    recs, _ = registry.read_all(paths)
    for rec in recs:
        for stage in STAGES:
            st = rec.stages.get(stage)
            if st is None or st.state is not State.DONE:
                continue
            for key in runner.REQUIRED[stage]:
                if key in st.outputs:
                    f = root / st.outputs[key]
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.touch()
    return paths


def stale_finished(paths) -> list[tuple[str, str]]:
    """(sid, stage) of each main stage in state DONE or RUNNING whose stored params_hash differs from the hash of
    the current code and configs; ascending."""
    recs, _ = registry.read_all(paths)
    view = runner.StageView(paths)
    out = []
    for rec in recs:
        for stage in STAGES:
            st = rec.stages.get(stage)
            if st is not None and st.state in (State.DONE, State.RUNNING) and st.params_hash != view.hash(rec, stage):
                out.append((rec.sid, stage))
    return sorted(out)


def live_bearmind_dir() -> Path | None:
    src = os.environ.get("BEARMIND_LT1_SERIES")
    return Path(src) / "_bearmind" if src else None
