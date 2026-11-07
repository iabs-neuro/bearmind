"""End-to-end check of a finished real-data series (Task 17).

BEARMIND_TEST_SERIES names one or more series folders (separated by os.pathsep) that went through the whole
pipeline: scan, run, thresholds, gate, run. Without it every test here is skipped. The tests only read what
the run left behind; the last one starts `run --once` and expects it to do nothing (it refuses to start it
when a stage is ready, so a stale series never launches hours of processing from a test run).
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bearmind import runner
from bearmind.paths import SeriesPaths
from bearmind.registry import STAGES, State, all_records

_ENV = os.environ.get("BEARMIND_TEST_SERIES", "")
SERIES = [Path(p) for p in _ENV.split(os.pathsep) if p.strip()]
REPO = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.realdata,
              pytest.mark.skipif(not SERIES, reason="BEARMIND_TEST_SERIES is not set")]

EXPORT_KEYS = ("npz", "json", "timestamps", "filters")
# outputs of the side stages, made only when preview.enabled was set for the run
PREVIEW_KEYS = {"mc_preview": ("preview_raw", "preview_raw_dff", "preview_mc", "preview_mc_dff"),
                "cnmf_preview": ("preview_cnmf_qc",)}


@pytest.fixture(params=SERIES, ids=[p.name for p in SERIES])
def series(request):
    paths = SeriesPaths(request.param.resolve())
    if not paths.registry_file("x").parent.is_dir():
        pytest.fail(f"{paths.root}: no registry; scan and run the series first")
    return paths


def _active(paths):
    recs = [r for r in all_records(paths) if not r.excluded]
    assert recs, f"{paths.root}: no non-excluded session"
    return recs


def test_every_stage_done(series):
    bad = {r.sid: {s: (r.stages.get(s).state.value if r.stages.get(s) else None)
                   for s in STAGES if r.stages.get(s) is None or r.stages[s].state is not State.DONE}
           for r in _active(series)}
    assert not {k: v for k, v in bad.items() if v}


def test_export_outputs(series):
    for r in _active(series):
        outputs = r.stages["export"].outputs
        keys = [k for k in EXPORT_KEYS
                if k != "filters" or not any(a.code == "no_filters" for a in r.anomalies)]
        for k in keys:
            assert k in outputs, f"{r.sid}: export output {k} not recorded"
            assert (series.root / outputs[k]).is_file(), f"{r.sid}: {outputs[k]} missing"
        meta = json.loads((series.root / outputs["json"]).read_text(encoding="utf-8"))
        assert meta["fps"] == pytest.approx(r.fps)
        assert "provenance" in meta
        assert meta["component_rule"] in ("ml_or_merged_not_deleted", "idx_components")
        if meta["component_rule"] == "ml_or_merged_not_deleted":
            assert meta["n_snr_duplicates_exported"] == 0 and meta["n_merge_products_dropped"] == 0
        with np.load(series.root / outputs["npz"], allow_pickle=False) as d:
            assert d.files, f"{r.sid}: empty npz"
        with open(series.root / outputs["timestamps"], encoding="utf-8") as fh:
            # bonsai files have no header, UCLA files one: count the data rows only
            n_rows = sum(1 for line in fh if line.split(",")[0].strip().lstrip("-").isdigit())
        mismatch = any(a.code == "export_frames_mismatch" for a in r.anomalies)
        assert n_rows == r.n_frames or mismatch, f"{r.sid}: {n_rows} timestamp rows, {r.n_frames} frames"


def test_previews_and_summary(series):
    recs = _active(series)
    for r in recs:
        out = r.stages["cnmf"].outputs.get("residual_png")
        assert out and (series.root / out).is_file(), f"{r.sid}: residual maps missing"
        for stage, keys in PREVIEW_KEYS.items():
            st = r.stages.get(stage)
            if st is None or st.state is not State.DONE:
                continue
            for k in keys:
                out = st.outputs.get(k)
                assert out and (series.root / out).is_file(), f"{r.sid}: preview {k} missing"
    with open(series.summary_csv, encoding="utf-8", newline="") as fh:
        rows = {row["sid"]: row for row in csv.DictReader(fh)}
    for r in recs:
        assert r.sid in rows, f"{r.sid}: not in summary.csv"
        assert rows[r.sid]["export"] == "done"
        assert int(float(rows[r.sid]["export_n_neurons"])) == r.stages["export"].metrics["n_neurons"]


def test_second_run_does_nothing(series):
    records = all_records(series)
    ready = runner.next_ready(records, series)
    assert ready is None, f"stage ready, refusing to start a real run from a test: {ready}"
    before = {p.name: p.read_bytes() for p in series.registry_file("x").parent.glob("*.json")}
    log_size = series.series_log.stat().st_size if series.series_log.exists() else 0
    r = subprocess.run([sys.executable, "-m", "bearmind", "run", str(series.root), "--once"], cwd=REPO,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "done 0, failed 0" in r.stdout
    after = {p.name: p.read_bytes() for p in series.registry_file("x").parent.glob("*.json")}
    assert after == before, "a registry record changed during a run that had nothing to do"
    with open(series.series_log, encoding="utf-8") as fh:
        fh.seek(log_size)
        new = fh.read()
    assert " started" not in new.replace("run started", ""), new
