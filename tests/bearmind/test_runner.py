"""Scheduler tests: stage functions are stubs injected through runner.STAGE_FUNCS (gate is the real one)."""
import copy
import errno
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import psutil
import pytest
import yaml

from bearmind import config, fileio, provenance, registry, runner
from bearmind.paths import SeriesPaths
from bearmind.registry import (STAGES, Anomaly, Decision, RawPart, SessionRecord, StageRecord, State,
                               load_record, save_record)
from bearmind.stages import StageResult, WaitingForInput, gate
from bearmind.stages import mc as mc_stage

SIDS = ["NOF_H01_1D", "NOF_H01_2D", "NOF_H02_1D"]
S1, S2, S3 = SIDS
THRESH = {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}}
ALL = [(sid, st) for st in STAGES for sid in SIDS]
ZERO = {"done": 0, "failed": 0, "waiting": 0, "skipped": 0}


def _tail(sid, first):
    return [(sid, s) for s in STAGES[STAGES.index(first):]]


def _record(sid, decided=True, excluded=False):
    part = RawPart(folder=f"1_Raw/{sid}", chunks=[f"{sid}_0.avi"], n_frames=100, shape=(16, 16),
                   timestamps_file=f"0_TimeStamps/{sid}_Mini_TS.csv", ts_format="bonsai_ticks",
                   start_ms=None, metadata_file=None)
    # R28: a manual decision counts only if made after the current CNMF output; far future = always
    decisions = [Decision("manual_inspection", "not_needed", "viktor", "2099-01-01T00:00:00")] if decided else []
    return SessionRecord(sid=sid, mouse=sid.rsplit("_", 1)[0], mode="bonsai", parts=[part], n_frames=100,
                         fps=19.76, fps_info={"source": "timestamps"},
                         anomalies=[Anomaly("scan_note", "warning", "from the scan")], decisions=decisions,
                         excluded=excluded, stages={s: StageRecord() for s in STAGES})


def _touch(path, data=b"x"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class Stubs:
    """Stage stubs writing their real output paths; per-(sid, stage) failures, hooks and extras."""

    def __init__(self):
        self.calls = []
        self.cwd = []
        self.fail = {}
        self.hooks = {}
        self.metrics = {}
        self.anomalies = {}
        self.drop = {}  # (sid, stage) -> output key the stub "forgets" to write
        self.fov = {}  # sid -> fov returned by crop (default a 10x9 box)

    def __call__(self, stage):
        def fn(paths, rec, cfg):
            key = (rec.sid, stage)
            self.calls.append(key)
            self.cwd.append(os.getcwd())
            if key in self.hooks:
                self.hooks[key](paths, rec, cfg)
            if key in self.fail:
                raise self.fail[key]
            res = getattr(self, "_" + stage)(paths, rec, cfg)
            res.metrics.update(self.metrics.get(key, {}))
            res.anomalies.extend(copy.deepcopy(self.anomalies.get(key, [])))
            if key in self.drop:
                res.outputs.pop(self.drop[key])
            return res
        return fn

    @staticmethod
    def _out(paths, **files):
        return {k: paths.rel(_touch(v)) for k, v in files.items()}

    def _crop(self, paths, rec, cfg):
        return StageResult(metrics={"fov": self.fov.get(rec.sid, [2, 3, 10, 9]), "frame_shape": [16, 16]})

    def _mc(self, paths, rec, cfg):
        crop = mc_stage._crop(cfg)  # the real stage's crop rule: waits for the box, uncropped only without FOV
        s = rec.sid
        return StageResult(
            outputs=self._out(paths, mc_video=paths.mc_video(s), mc_shifts=paths.mc_shifts(s),
                              preview_mc=paths.preview(s, "mc")),
            metrics={"max_shift_px": 2.0, "mean_template_corr": 0.9,
                     "crop": None if crop is None else [crop.top, crop.left, crop.height, crop.width]})

    def _thresholds(self, paths, rec, cfg):
        return StageResult(outputs=self._out(paths, threshold_maps=paths.threshold_maps(rec.sid)),
                           metrics={"suggested": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})

    def _cnmf(self, paths, rec, cfg):
        if any(cfg["cnmf"].get(k) is None for k in ("gSig", "min_corr", "min_pnr")):
            raise WaitingForInput("thresholds")
        s = rec.sid
        return StageResult(
            outputs=self._out(paths, estimates=paths.estimates(s),
                              preview_cnmf_qc=paths.preview(s, "cnmf_qc"), residual_png=paths.residual_png(s)),
            metrics={"n_components": 100, "missed_cells": 1, "gSig": cfg["cnmf"]["gSig"]})

    def _gate(self, paths, rec, cfg):
        return gate.run(paths, rec, cfg)

    def _autoinspect(self, paths, rec, cfg):
        f = paths.inspection_dir(rec.sid) / f"{rec.sid}_processed.pickle"
        return StageResult(outputs=self._out(paths, processed=f),
                           metrics={"n_initial": 100, "n_after_merge": 90, "n_deleted": 20, "n_merge_groups": 2,
                                    "n_kept": 70, "deleted_frac": 0.2})

    def _export(self, paths, rec, cfg):
        s = rec.sid
        return StageResult(outputs=self._out(paths, npz=paths.export_npz(s), json=paths.export_json(s),
                                             timestamps=paths.export_timestamps(s)),
                           metrics={"n_neurons": 80, "median_snr": 5.0})


@pytest.fixture
def series(tmp_path, monkeypatch):
    home = Path(os.environ["BEARMIND_HOME"])
    temp = tmp_path / "temp"
    config.update_yaml(home / "machine.yaml", {"temp_dir": str(temp), "n_cpus": 2, "ram_gb": 16.0,
                                               "cnmf_ram_budget_gb": 12.8})
    paths = SeriesPaths(tmp_path / "series")
    model = _touch(tmp_path / "model.pkl", b"model")
    # the stubs are closures: they run in this process (the isolation is tested below and in test_isolate.py)
    config.update_yaml(paths.series_config, {"autoinspect": {"model_path": str(model)},
                                             "runner": {"isolate_stages": False}})
    for sid in SIDS:
        save_record(paths, _record(sid))
        config.update_yaml(paths.session_config(sid), THRESH)
    stubs = Stubs()
    for st in STAGES:
        monkeypatch.setitem(runner.STAGE_FUNCS, st, stubs(st))
    return SimpleNamespace(paths=paths, stubs=stubs, temp=temp, tmp=tmp_path)


def _run(s, **kw):
    kw.setdefault("once", True)
    return runner.run(s.paths.root, **kw)


def _states(paths, sid):
    rec = load_record(paths, sid)
    return [rec.stages[s].state for s in STAGES]


def _heartbeat(paths):
    try:
        return json.loads(paths.heartbeat.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


# --- the tests required by the brief ---

def test_order_and_resume(series):
    s = series
    res = _run(s)
    assert s.stubs.calls == ALL  # lowest stage first across sessions, then sid order
    assert res == {"done": 21, "failed": 0, "waiting": 0, "skipped": 0}
    for sid in SIDS:
        rec = load_record(s.paths, sid)
        assert _states(s.paths, sid) == [State.DONE] * 7
        assert all(st.params_hash and st.params for st in rec.stages.values())
        assert rec.stages["cnmf"].params["fr"] == 19.76 and rec.stages["cnmf"].params["gSig"] == [4, 4]
        assert rec.stages["mc"].params["box_pos"] == _yaml(s.paths.session_config(sid))["crop"]["box_pos"]
    # finalize_mouse: one box size per mouse, a position per session
    assert _yaml(s.paths.mouse_config("NOF_H01"))["crop"]["box"] == [16, 16]
    assert load_record(s.paths, S1).stages["crop"].metrics["box"] == [0, 0, 16, 16]

    s.stubs.calls.clear()
    assert _run(s) == ZERO
    assert s.stubs.calls == []

    config.update_yaml(s.paths.series_config, {"cnmf": {"merge_thr": 0.7}})
    res = _run(s)
    assert s.stubs.calls == [(sid, st) for st in ("cnmf", "gate", "autoinspect", "export") for sid in SIDS]
    assert res["done"] == 12


def test_failure_isolated(series):
    s = series
    s.stubs.fail[(S2, "mc")] = RuntimeError("boom")
    res = _run(s)
    rec = load_record(s.paths, S2)
    assert rec.stages["mc"].state is State.FAILED
    assert rec.stages["mc"].error == "RuntimeError: boom"
    assert [rec.stages[x].state for x in STAGES[2:]] == [State.SKIPPED] * 5
    for sid in (S1, S3):
        assert _states(s.paths, sid) == [State.DONE] * 7
    assert res == {"done": 15, "failed": 1, "waiting": 0, "skipped": 5}
    log = s.paths.session_log(S2).read_text(encoding="utf-8")
    assert "Traceback" in log and "boom" in log
    assert "Traceback" not in s.paths.session_log(S1).read_text(encoding="utf-8")
    # a failed stage is not retried while its parameters are unchanged
    del s.stubs.fail[(S2, "mc")]
    s.stubs.calls.clear()
    assert _run(s) == ZERO and s.stubs.calls == []
    # changing a parameter of the failed stage retries it
    config.update_yaml(s.paths.session_config(S2), {"mc": {"max_deviation_rigid": 10}})
    _run(s)
    assert s.stubs.calls == _tail(S2, "mc")


def test_waiting_does_not_block_others(series):
    s = series
    save_record(s.paths, _record(S1, decided=False))
    s.paths.session_config(S3).unlink()  # S3 has no thresholds yet
    res = _run(s)
    s1 = load_record(s.paths, S1)
    assert s1.stages["gate"].state is State.WAITING
    assert s1.stages["gate"].error == "waiting for manual_inspection"
    assert [s1.stages[x].state for x in ("autoinspect", "export")] == [State.PENDING] * 2
    s3 = load_record(s.paths, S3)
    assert s3.stages["cnmf"].state is State.WAITING and s3.stages["cnmf"].error == "waiting for thresholds"
    assert _states(s.paths, S2) == [State.DONE] * 7
    assert res["waiting"] == 2 and res["failed"] == 0
    assert s.stubs.calls.count((S1, "gate")) == 1  # once=True: each waiting stage is tried once
    hb = _heartbeat(s.paths)
    assert hb["counts"] == {"done": 1, "running": 0, "waiting_thresholds": 1, "waiting_manual": 1,
                            "waiting_crop": 0, "failed": 0, "skipped": 0, "pending": 0}  # per session
    assert hb["by_stage"] == {"crop": 3, "mc": 3, "thresholds": 3, "cnmf": 2, "gate": 1, "autoinspect": 1,
                              "export": 1}
    assert hb["status"] == "idle" and hb["current"] is None and hb["pid"] == os.getpid()


def test_stale_running_reset(series):
    s = series
    _run(s)
    s.stubs.calls.clear()
    f = s.paths.registry_file(S1)
    d = json.loads(f.read_text(encoding="utf-8"))
    d["stages"]["cnmf"]["state"] = "running"  # the previous process was killed during CNMF
    f.write_text(json.dumps(d), encoding="utf-8")
    est = s.paths.estimates(S1)
    _touch(est.with_name(est.name + ".partial"))
    _touch(s.temp / S1 / "Yr_d1_16_d2_16_d3_1_order_C_frames_100.mmap")
    _run(s)
    assert s.stubs.calls == _tail(S1, "cnmf")
    assert _states(s.paths, S1) == [State.DONE] * 7
    assert not (s.temp / S1).exists()  # temp of the killed run removed
    assert "running at start" in s.paths.series_log.read_text(encoding="utf-8")


def test_disk_full_cleans_temp(series):
    s = series

    def fill(paths, rec, cfg):
        d = Path(cfg["machine"]["temp_dir"]) / rec.sid
        d.mkdir(parents=True)
        (d / "mc_u8.dat").write_bytes(b"0" * 1000)

    s.stubs.hooks[(S2, "cnmf")] = fill
    s.stubs.fail[(S2, "cnmf")] = OSError(errno.ENOSPC, "No space left on device")
    _run(s)
    st = load_record(s.paths, S2).stages["cnmf"]
    assert st.state is State.FAILED and "No space left on device" in st.error
    assert not s.temp.exists() or list(s.temp.iterdir()) == []
    assert _states(s.paths, S1) == [State.DONE] * 7 and _states(s.paths, S3) == [State.DONE] * 7


def _hold_run_lock(s, sid=S1, stage="crop"):
    """Start a runner in a thread that blocks inside one stage; returns (thread, release event)."""
    entered, release = threading.Event(), threading.Event()

    def block(paths, rec, cfg):
        entered.set()
        assert release.wait(30)

    s.stubs.hooks[(sid, stage)] = block
    t = threading.Thread(target=lambda: _run(s), daemon=True)
    t.start()
    assert entered.wait(60)
    return t, release


def test_second_runner_refused(series):
    """M4: the run lock is an OS lock held for the whole run; the PID in it is for the message."""
    s = series
    t, release = _hold_run_lock(s)
    with pytest.raises(RuntimeError, match=f"series is already being processed by PID {os.getpid()}"):
        _run(s)
    release.set()
    t.join(60)
    assert not t.is_alive()
    del s.stubs.hooks[(S1, "crop")]
    assert _run(s) == ZERO  # released: a new run starts


def test_live_runner_kept_when_lock_file_content_is_unreadable(series):
    s = series
    t, release = _hold_run_lock(s)
    with open(s.paths.lock, "r+b") as fh:  # the PID text is for display; the OS lock decides
        fh.write(b"garbage")
    try:
        with pytest.raises(RuntimeError, match="already being processed"):
            _run(s)
    finally:
        release.set()
        t.join(60)


def test_summary_csv(series):
    s = series
    s.stubs.fail[(S2, "cnmf")] = RuntimeError("boom")
    s.stubs.metrics[(S3, "mc")] = {"max_shift_px": 31.0}
    _run(s)
    df = runner.summary(s.paths)
    assert list(df["sid"]) == SIDS
    assert set(STAGES) <= set(df.columns)
    assert {"mouse", "excluded", "crop_box", "mc_max_shift_px", "cnmf_n_components", "gate_decision",
            "autoinspect_n_kept", "autoinspect_deleted_frac", "export_n_neurons", "export_median_snr",
            "gSig", "min_corr", "min_pnr", "worst_check", "waiting_for", "error"} <= set(df.columns)
    row = df.set_index("sid")
    assert row.loc[S1, "export"] == "done" and row.loc[S2, "cnmf"] == "failed"
    assert row.loc[S2, "export"] == "skipped"
    assert row.loc[S3, "mc_max_shift_px"] == 31.0 and row.loc[S1, "cnmf_n_components"] == 100
    assert row.loc[S1, "crop_box"] == "16x16@0,0"
    assert row.loc[S1, "gSig"] == 4 and row.loc[S1, "min_corr"] == 0.8 and row.loc[S1, "min_pnr"] == 8.0
    assert row.loc[S1, "gate_decision"] == "not_needed" and row.loc[S1, "export_n_neurons"] == 80
    assert row.loc[S1, "autoinspect_n_kept"] == 70 and row.loc[S1, "autoinspect_deleted_frac"] == 0.2
    assert row.loc[S1, "worst_check"] == "ok"
    assert row.loc[S3, "worst_check"] == "error"  # max shift 31 px > error threshold 30
    assert row.loc[S2, "worst_check"] == "error"  # a failed stage
    assert "boom" in row.loc[S2, "error"]
    csv = pd.read_csv(s.paths.summary_csv)
    assert list(csv["sid"]) == SIDS and list(csv.columns) == list(df.columns)


def test_reset_from_reruns_only_downstream(series):
    s = series
    _run(s)
    s.stubs.calls.clear()
    est = s.paths.estimates(S1)
    runner.reset_from(s.paths, S1, "cnmf", "viktor")
    rec = load_record(s.paths, S1)
    assert _states(s.paths, S1) == [State.DONE] * 3 + [State.PENDING] * 4
    d = rec.decisions[-1]
    assert (d.what, d.value, d.who) == ("rerun_from", "cnmf", "viktor") and d.when
    assert est.exists()  # old outputs stay until the stage replaces them
    _run(s)
    assert s.stubs.calls == _tail(S1, "cnmf")
    with pytest.raises(ValueError):
        runner.reset_from(s.paths, S1, "nonsense", "viktor")


def test_config_writes_do_not_invalidate_upstream(series):
    s = series
    s.paths.session_config(S1).unlink()  # no thresholds yet: CNMF waits after the threshold maps
    _run(s)
    assert load_record(s.paths, S1).stages["cnmf"].state is State.WAITING
    s.stubs.calls.clear()
    # what finalize and the thresholds window write after crop / thresholds are done
    box = _yaml(s.paths.mouse_config("NOF_H01"))["crop"]["box"]
    config.update_yaml(s.paths.mouse_config("NOF_H01"), {"crop": {"box": box}})
    config.update_yaml(s.paths.session_config(S1), THRESH)
    _run(s)
    assert s.stubs.calls == _tail(S1, "cnmf")
    # a new crop position re-runs MC and everything after it, never crop itself
    s.stubs.calls.clear()
    config.update_yaml(s.paths.session_config(S2), {"crop": {"box_pos": [1, 2]}})
    _run(s)
    assert s.stubs.calls == _tail(S2, "mc")


def test_missing_required_output_reruns(series):
    s = series
    _run(s)
    s.stubs.calls.clear()
    s.paths.estimates(S1).unlink()
    _run(s)
    assert s.stubs.calls == _tail(S1, "cnmf")
    s.stubs.calls.clear()
    s.paths.preview(S2, "cnmf_qc").unlink()
    s.paths.preview(S2, "mc").unlink()
    s.paths.residual_png(S2).unlink()
    assert _run(s) == ZERO and s.stubs.calls == []


def test_concurrent_gate_decision_not_lost(series):
    s = series
    save_record(s.paths, _record(S1, decided=False))
    # a person decides while the runner holds S1's record (loaded before the gate started); a decision made
    # during CNMF would not count (R28), so it lands while the gate itself runs
    def decide_once(paths, rec, cfg):
        del s.stubs.hooks[(S1, "gate")]
        gate.decide(paths, S1, "not_needed", "viktor")

    s.stubs.hooks[(S1, "gate")] = decide_once
    _run(s)
    rec = load_record(s.paths, S1)
    assert [d.what for d in rec.decisions] == ["manual_inspection"]
    assert _states(s.paths, S1) == [State.DONE] * 7
    assert rec.stages["gate"].metrics["gate"] == "not_needed"


def test_waits_and_picks_up_decision(series):
    s = series
    save_record(s.paths, _record(S1, decided=False))
    out = {}
    t = threading.Thread(target=lambda: out.update(res=runner.run(s.paths.root, poll_s=0.05)), daemon=True)
    t.start()
    hb = None
    deadline = time.time() + 60
    while time.time() < deadline:
        hb = _heartbeat(s.paths)
        if hb and hb["status"] == "waiting":
            break
        time.sleep(0.05)
    assert hb and hb["status"] == "waiting" and hb["counts"]["waiting_manual"] == 1
    assert t.is_alive()
    assert _states(s.paths, S2) == [State.DONE] * 7
    gate.decide(s.paths, S1, "not_needed", "viktor")
    t.join(60)
    assert not t.is_alive()
    assert _states(s.paths, S1) == [State.DONE] * 7
    assert out["res"]["waiting"] == 0
    assert "waiting: thresholds 0, manual inspection 1" in s.paths.series_log.read_text(encoding="utf-8")


def test_corrupt_registry_file_skipped(series):
    s = series
    s.paths.registry_file("NOF_H09_1D").write_text("{not json", encoding="utf-8")
    res = _run(s)
    assert res["done"] == 21
    log = s.paths.series_log.read_text(encoding="utf-8")
    assert "NOF_H09_1D.json" in log


# --- further behaviour ---

def test_stage_hash_inputs(series):
    s = series
    rec = load_record(s.paths, S1)
    cfg = config.resolve(s.paths, sid=S1)
    base = {st: runner.stage_hash(rec, st, cfg, s.paths) for st in STAGES}

    def hashes(change):
        c = copy.deepcopy(cfg)
        change(c)
        return {st: runner.stage_hash(rec, st, c, s.paths) for st in STAGES}

    def machine(c):
        c["machine"].update(n_cpus=64, temp_dir="X:/tmp", ram_gb=1.0, cnmf_ram_budget_gb=0.5)

    assert hashes(machine) == base  # machine settings are not inputs
    h = hashes(lambda c: c["crop"].update(box=[10, 10], box_pos=[1, 2]))
    assert h["crop"] == base["crop"] and h["mc"] != base["mc"] and h["export"] != base["export"]
    h = hashes(lambda c: c["cnmf"].update(gSig=6))
    assert [h[x] == base[x] for x in STAGES] == [True, True, True, False, False, False, False]
    h = hashes(lambda c: c["cnmf"].update(tsub=2))
    assert h["mc"] == base["mc"] and h["thresholds"] != base["thresholds"]
    h = hashes(lambda c: c["autoinspect"].update(ml_threshold=0.5))
    assert h["gate"] == base["gate"] and h["autoinspect"] != base["autoinspect"]
    h = hashes(lambda c: c["preview"].update(crf=30))
    assert h == base  # previews are not results
    h = hashes(lambda c: c["crop"].update(fov_threshold_frac=0.3))  # R30: detection parameters
    assert h["crop"] != base["crop"]
    rec2 = copy.deepcopy(rec)
    rec2.parts[0].start_ms = 5.0  # R31: timestamp inputs belong to export only
    assert [runner.stage_hash(rec2, x, cfg, s.paths) == base[x] for x in STAGES] == [True] * 6 + [False]
    # box_pos null (no field of view: uncropped) differs from box_pos not written yet
    h1 = hashes(lambda c: c["crop"].update(box_pos=None))
    h2 = hashes(lambda c: c["crop"].pop("box_pos", None))
    assert h1["mc"] != h2["mc"]
    # parts and fps of the session
    rec2 = copy.deepcopy(rec)
    rec2.parts[0].n_frames = 99
    assert runner.stage_hash(rec2, "crop", cfg, s.paths) != base["crop"]
    rec2 = copy.deepcopy(rec)
    rec2.fps = 30.0
    assert runner.stage_hash(rec2, "crop", cfg, s.paths) == base["crop"]
    assert runner.stage_hash(rec2, "mc", cfg, s.paths) != base["mc"]
    # gate: decision and the manual result file
    rec2 = copy.deepcopy(rec)
    rec2.decisions.append(Decision("manual_inspection", "cleaned", "v", "t"))
    _touch(s.paths.manual_result(S1), b"a")
    h_clean = runner.stage_hash(rec2, "gate", cfg, s.paths)
    assert h_clean != base["gate"]
    _touch(s.paths.manual_result(S1), b"ab")
    assert runner.stage_hash(rec2, "gate", cfg, s.paths) != h_clean
    # autoinspect: the model file content
    model = Path(cfg["autoinspect"]["model_path"])
    model.write_bytes(b"model v2")
    assert runner.stage_hash(rec, "autoinspect", cfg, s.paths) != base["autoinspect"]


def test_stage_params_export_threshold_from_autoinspect(series):
    s = series
    _run(s)
    rec = load_record(s.paths, S1)
    cfg = config.resolve(s.paths, sid=S1)
    assert rec.stages["export"].params["ml_threshold"] == 0.72
    assert rec.stages["export"].params["timestamps"] == [
        {"timestamps_file": f"0_TimeStamps/{S1}_Mini_TS.csv", "ts_format": "bonsai_ticks", "start_ms": None,
         "n_frames": 100, "file": None}]
    assert rec.stages["autoinspect"].params["fps"] == 20
    assert len(rec.stages["autoinspect"].params["model_sha256"]) == 64
    rec.stages["autoinspect"].params["autoinspect"]["ml_threshold"] = 0.6  # what the pickle was made with
    assert runner.stage_params("export", rec, cfg, s.paths)["ml_threshold"] == 0.6
    assert provenance.collect(s.paths, rec, cfg)["stage_params"]["cnmf"]["fr"] == 19.76


def test_heartbeat_logs_and_cwd(series, monkeypatch):
    s = series
    monkeypatch.setattr(runner, "HEARTBEAT_S", 0.05)
    seen = []

    def slow(paths, rec, cfg):
        seen.append(_heartbeat(paths))
        time.sleep(0.4)
        seen.append(_heartbeat(paths))

    s.stubs.hooks[(S1, "crop")] = slow
    cwd = os.getcwd()
    _run(s)
    assert os.getcwd() == cwd
    assert set(s.stubs.cwd) == {str(provenance.REPO_ROOT)}
    a, b = seen
    assert a["current"] == {"sid": S1, "stage": "crop"} and a["status"] == "running"
    assert b["time"] != a["time"]  # the background thread keeps the pulse while a stage runs
    hb = _heartbeat(s.paths)
    assert hb["status"] == "idle" and hb["current"] is None
    assert hb["counts"] == {"done": 3, "running": 0, "waiting_thresholds": 0, "waiting_manual": 0,
                            "waiting_crop": 0, "failed": 0, "skipped": 0, "pending": 0}
    assert hb["by_stage"] == {st: 3 for st in STAGES}
    assert not any(t.name == "bearmind-heartbeat" for t in threading.enumerate())
    for sid in SIDS:
        text = s.paths.session_log(sid).read_text(encoding="utf-8")
        assert f"{sid} export: done" in text and SIDS[(SIDS.index(sid) + 1) % 3] not in text
    series_log = s.paths.series_log.read_text(encoding="utf-8")
    assert f"{S3} export: done" in series_log
    assert not s.paths.lock.exists()
    assert s.paths.summary_csv.exists()
    assert not logging.getLogger("bearmind").handlers


def test_interrupt_returns_stage_to_pending(series):
    s = series
    s.stubs.fail[(S2, "mc")] = KeyboardInterrupt()
    cwd = os.getcwd()
    with pytest.raises(KeyboardInterrupt):
        _run(s)
    rec = load_record(s.paths, S2)
    assert rec.stages["mc"].state is State.PENDING
    assert not s.paths.lock.exists()
    assert os.getcwd() == cwd
    assert not any(t.name == "bearmind-heartbeat" for t in threading.enumerate())
    assert _heartbeat(s.paths)["status"] == "idle"
    del s.stubs.fail[(S2, "mc")]
    s.stubs.calls.clear()
    _run(s)
    assert (S2, "mc") in s.stubs.calls and (S1, "crop") not in s.stubs.calls


def test_dead_or_reused_pid_lock_taken_over(series):
    s = series
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    s.paths.lock.parent.mkdir(parents=True, exist_ok=True)
    s.paths.lock.write_text(json.dumps({"pid": p.pid, "create_time": 1.0}), encoding="utf-8")
    assert _run(s)["done"] == 21
    assert not s.paths.lock.exists()
    me = psutil.Process()  # same PID number, other process start: the PID was reused
    s.paths.lock.write_text(json.dumps({"pid": me.pid, "create_time": me.create_time() - 1000}),
                            encoding="utf-8")
    assert _run(s) == ZERO
    assert not s.paths.lock.exists()


def test_sessions_and_until(series):
    s = series
    _run(s, until="mc", sids=[S1])
    # crop runs for every session of the mouse (one box size per mouse), the rest only for S1
    assert s.stubs.calls == [(S1, "crop"), (S2, "crop"), (S1, "mc")]
    assert _states(s.paths, S2)[1] is State.PENDING
    assert _states(s.paths, S3) == [State.PENDING] * 7
    with pytest.raises(ValueError):
        _run(s, until="nonsense")


def test_excluded_and_failed_crop_do_not_block_mouse(series):
    s = series
    save_record(s.paths, _record(S2, excluded=True))
    s.stubs.fail[(S3, "crop")] = RuntimeError("bad video")
    _run(s)
    assert all(sid != S2 for sid, _ in s.stubs.calls)
    assert _states(s.paths, S1) == [State.DONE] * 7
    assert "crop" not in _yaml(s.paths.session_config(S2))
    s3 = load_record(s.paths, S3)
    assert s3.stages["crop"].state is State.FAILED and s3.stages["mc"].state is State.SKIPPED


def test_crop_failure_of_one_session_does_not_block_its_mouse(series):
    s = series
    s.stubs.fail[(S2, "crop")] = RuntimeError("bad video")
    _run(s)
    assert _states(s.paths, S1) == [State.DONE] * 7
    assert "box_pos" in _yaml(s.paths.session_config(S1))["crop"]
    # the session fixed later gets the mouse box without changing the others
    del s.stubs.fail[(S2, "crop")]
    s.stubs.calls.clear()
    runner.reset_from(s.paths, S2, "crop", "viktor")
    _run(s)
    assert s.stubs.calls == _tail(S2, "crop")
    assert load_record(s.paths, S2).stages["crop"].metrics["box"] == [0, 0, 16, 16]


def test_no_fov_anomaly_tagged_with_crop(series):
    s = series
    s.stubs.fov[S3] = None
    _run(s)
    rec = load_record(s.paths, S3)
    assert rec.stages["crop"].metrics["box"] is None
    assert _yaml(s.paths.session_config(S3))["crop"]["box_pos"] is None
    found = [a for a in rec.anomalies if a.code == "fov_not_found"]
    assert len(found) == 1 and found[0].data["stage"] == "crop"
    assert _states(s.paths, S3) == [State.DONE] * 7


def test_stage_anomalies_tagged_and_replaced(series):
    s = series
    s.stubs.anomalies[(S1, "thresholds")] = [Anomaly("no_threshold_suggestion", "warning", "x")]
    _run(s)
    rec = load_record(s.paths, S1)
    codes = {a.code: a.data.get("stage") for a in rec.anomalies}
    assert codes == {"scan_note": None, "no_threshold_suggestion": "thresholds"}
    del s.stubs.anomalies[(S1, "thresholds")]
    runner.reset_from(s.paths, S1, "thresholds", "viktor")
    _run(s)
    assert [a.code for a in load_record(s.paths, S1).anomalies] == ["scan_note"]


def test_missing_output_from_stage_fails(series):
    s = series
    s.stubs.drop[(S1, "mc")] = "mc_shifts"
    _run(s)
    st = load_record(s.paths, S1).stages["mc"]
    assert st.state is State.FAILED and "mc_shifts" in st.error


def test_checks_use_series_metrics(series):
    s = series
    s.stubs.metrics[(S3, "cnmf")] = {"n_components": 1000}
    _run(s)
    checks = {c.name: c.level for c in load_record(s.paths, S3).stages["cnmf"].checks}
    assert checks["cnmf_n_components_vs_median"] == "error"  # 1000 vs median 100
    assert {c.name for c in load_record(s.paths, S1).stages["mc"].checks} == {"mc_max_shift_px",
                                                                              "mc_template_corr"}


def test_summary_write_failure_is_warning(series, monkeypatch, caplog):
    """M3: one attempt per write (no retry loop blocking the batch), one warning until a write succeeds."""
    s = series
    _run(s)
    monkeypatch.setattr(fileio, "REPLACE_STEP_S", 0.0)
    real = os.replace
    calls = {"n": 0, "deny": True}

    def replace(src, dst):
        if str(dst).endswith("summary.csv"):
            calls["n"] += 1
            if calls["deny"]:
                raise PermissionError(13, "summary.csv is open in Excel")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    s.paths.summary_csv.unlink()  # unchanged content is not written again: each call below has to write
    with caplog.at_level(logging.WARNING, logger="bearmind"):
        assert len(runner.summary(s.paths)) == 3
        runner.summary(s.paths)
        assert calls["n"] == 2
        assert caplog.text.count("summary.csv not written") == 1
        calls["deny"] = False
        runner.summary(s.paths)
        calls["deny"] = True
        registry.update_record(s.paths, S1, lambda r: r.anomalies.append(Anomaly("note", "warning", "new row")))
        runner.summary(s.paths)
        assert caplog.text.count("summary.csv not written") == 2
    assert not list(s.paths.summary_csv.parent.glob("*.tmp"))


def test_unstorable_metrics_fail_only_that_stage(series):
    s = series
    s.stubs.metrics[(S1, "mc")] = {"when": object()}  # a stage bug: JSON cannot store it
    res = _run(s)
    st = load_record(s.paths, S1).stages["mc"]
    assert st.state is State.FAILED and "not JSON serializable" in st.error
    assert _states(s.paths, S2) == [State.DONE] * 7 and _states(s.paths, S3) == [State.DONE] * 7
    assert res["failed"] == 1


def test_broken_session_config_holds_its_mouse_finalize(series):
    s = series
    good = s.paths.session_config(S2).read_text(encoding="utf-8")
    s.paths.session_config(S2).write_text("cnmf: [unclosed" + chr(10), encoding="utf-8")
    _run(s)
    assert _states(s.paths, S2) == [State.PENDING] * 7  # held, not failed: nothing to reset later
    s1 = load_record(s.paths, S1)  # same mouse: the box waits for S2's crop (M1)
    assert s1.stages["crop"].state is State.DONE and s1.stages["mc"].state is State.WAITING
    assert s1.stages["mc"].error == "waiting for crop_finalize"
    assert _states(s.paths, S3) == [State.DONE] * 7
    assert f"{S2}: cannot be scheduled: config" in s.paths.series_log.read_text(encoding="utf-8")
    # I2: visible where the operator looks, not only in the log
    row = pd.read_csv(s.paths.summary_csv).set_index("sid")
    assert row.loc[S2, "error"].startswith("cannot be scheduled: config: ")
    assert f"{S2}.yaml" in row.loc[S2, "error"]
    assert _heartbeat(s.paths)["unschedulable"][S2].startswith("config: ")
    state = runner.series_state(s.paths)
    assert state.waiting == [] and state.blocked == [(S1, "mc", "crop_finalize", [S2])]
    assert set(state.problems) == {S2}
    s.paths.session_config(S2).write_text(good, encoding="utf-8")
    s.stubs.calls.clear()
    _run(s)
    assert s.stubs.calls == [(S2, "crop")] + [(sid, st) for st in STAGES[1:] for sid in (S1, S2)]


def test_waiting_session_that_cannot_be_scheduled_does_not_spin(series, monkeypatch):
    s = series
    save_record(s.paths, _record(S1, decided=False))
    save_record(s.paths, _record(S3, decided=False))  # a wait a person can resolve keeps the runner polling
    clock = [1000.0]
    sleeps = []

    def fake_sleep(x):
        sleeps.append(x)
        if len(sleeps) == 1:  # after the first try of S1's gate its session config breaks
            s.paths.session_config(S1).write_text("cnmf: [unclosed" + chr(10), encoding="utf-8")
        if len(sleeps) >= 4:
            raise KeyboardInterrupt  # stop the waiting runner
        clock[0] += x

    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=fake_sleep))
    with pytest.raises(KeyboardInterrupt):
        runner.run(s.paths.root, poll_s=5)
    assert len(sleeps) == 4 and all(x >= 5 for x in sleeps), sleeps


# --- fix round 1 ---

def test_decision_inside_poll_window_is_picked_up(series, monkeypatch):
    """C1: the decision lands while the runner sleeps and it wakes before the stage is due again."""
    s = series
    save_record(s.paths, _record(S1, decided=False))
    clock = [1000.0]
    sleeps = []

    def fake_sleep(x):
        sleeps.append(x)
        if len(sleeps) == 1:
            gate.decide(s.paths, S1, "not_needed", "viktor")
        if len(sleeps) > 20:
            raise KeyboardInterrupt  # the runner never got back to S1
        clock[0] += x / 2  # coarse clock: woke inside the poll window

    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=fake_sleep))
    res = runner.run(s.paths.root, poll_s=5)
    assert _states(s.paths, S1) == [State.DONE] * 7
    assert res["waiting"] == 0


def test_cleaned_decision_must_follow_current_cnmf(series):
    """R28: a manual decision and the manual result must be newer than the CNMF output they clean."""
    s = series
    save_record(s.paths, _record(S1, decided=False))
    _run(s)
    assert load_record(s.paths, S1).stages["gate"].state is State.WAITING
    _touch(s.paths.manual_result(S1), b"manual result of cnmf v1")
    gate.decide(s.paths, S1, "cleaned", "viktor")
    _run(s)
    assert _states(s.paths, S1) == [State.DONE] * 7

    runner.reset_from(s.paths, S1, "cnmf", "viktor")

    def decide_during_cnmf(paths, rec, cfg):  # the person confirms while the new CNMF still runs
        gate.decide(paths, S1, "cleaned", "viktor")
        time.sleep(0.05)

    s.stubs.hooks[(S1, "cnmf")] = decide_during_cnmf
    s.stubs.calls.clear()
    _run(s)
    st = load_record(s.paths, S1).stages["gate"]
    assert st.state is State.WAITING and "older than the current CNMF output" in st.error
    assert (S1, "gate") in s.stubs.calls and (S1, "autoinspect") not in s.stubs.calls
    assert _heartbeat(s.paths)["counts"]["waiting_manual"] == 1

    del s.stubs.hooks[(S1, "cnmf")]
    _touch(s.paths.manual_result(S1), b"manual result of cnmf v2")
    gate.decide(s.paths, S1, "cleaned", "viktor")
    s.stubs.calls.clear()
    _run(s)
    assert s.stubs.calls == [(S1, "gate"), (S1, "autoinspect"), (S1, "export")]
    assert load_record(s.paths, S1).stages["gate"].params["decision"]["value"] == "cleaned"


def test_crop_detection_params_rerun_crop(series):
    """R30: the crop section is an input of crop, except the keys the pipeline writes."""
    s = series
    _run(s)
    s.stubs.calls.clear()
    config.update_yaml(s.paths.mouse_config("NOF_H02"), {"crop": {"fov_threshold_frac": 0.3}})
    _run(s)
    assert s.stubs.calls == _tail(S3, "crop")


def test_export_hashes_timestamp_inputs(series):
    """R31: a rescan that changes the timestamps of a part re-runs export only."""
    s = series
    _run(s)
    s.stubs.calls.clear()

    def shift(d):
        d.parts[0].start_ms = 1728989165178.0

    registry.update_record(s.paths, S1, shift)
    _run(s)
    assert s.stubs.calls == [(S1, "export")]
    s.stubs.calls.clear()
    rec = load_record(s.paths, S2)
    _touch(s.paths.root / rec.parts[0].timestamps_file, b"0" + bytes([10]) + b"500000" + bytes([10]))
    _run(s)
    assert s.stubs.calls == [(S2, "export")]


def test_lock_timeout_on_registry_write_does_not_abort(series, monkeypatch):
    """I3: a registry lock timeout skips the stage for this loop; the batch goes on."""
    s = series
    real = registry.update_record
    hits = {"running": 0, "done": 0}

    def flaky(paths, sid, change):
        name = getattr(change, "__name__", "")
        if sid == S2 and name in hits and hits[name] == 0:
            hits[name] += 1
            raise fileio.LockTimeout(f"lock is held by another writer: {sid}")
        return real(paths, sid, change)

    monkeypatch.setattr(registry, "update_record", flaky)
    res = _run(s)
    assert hits == {"running": 1, "done": 1}
    assert all(_states(s.paths, sid) == [State.DONE] * 7 for sid in SIDS)
    assert res["failed"] == 0
    assert "lock" in s.paths.series_log.read_text(encoding="utf-8")


def test_finalize_error_is_retried_not_failed(series, monkeypatch):
    """M2: a finalize error keeps crop DONE; MC waits and finalize is retried by the next run."""
    s = series
    from bearmind.stages import crop as crop_stage

    calls = {"n": 0}

    def flaky_finalize(paths, mouse, records, cfg):
        calls["n"] += 1
        if mouse == "NOF_H01" and calls["n"] == 1:
            raise fileio.LockTimeout("lock is held by another writer: mouse config")
        return crop_stage.finalize_mouse(paths, mouse, records, cfg)

    monkeypatch.setattr(runner, "FINALIZE_FUNC", flaky_finalize)
    res = _run(s)
    for sid in (S1, S2):
        rec = load_record(s.paths, sid)
        assert rec.stages["crop"].state is State.DONE
        assert rec.stages["mc"].state is State.WAITING and rec.stages["mc"].error == "waiting for crop_finalize"
    assert _states(s.paths, S3) == [State.DONE] * 7
    assert res["failed"] == 0
    _run(s)
    assert all(_states(s.paths, sid) == [State.DONE] * 7 for sid in SIDS)


# --- stages in their own processes (runner.isolate_stages: true, the default) ---

FAKES = "tests.bearmind.isolate_fakes:{}"


@pytest.mark.parametrize("how, retries, attempts", [("crash", 1, 2), ("crash", 0, 1), ("fail", 1, 1)])
def test_crashed_stage_retried_once_then_failed(series, monkeypatch, how, retries, attempts):
    """Two sessions of two mice; S1's MC process crashes (or raises): FAILED after `attempts` runs, an
    automatic retry only for a crash; the other session is done and run() returns."""
    s = series
    save_record(s.paths, _record(S2, excluded=True))
    notes = s.tmp / "notes"
    config.update_yaml(s.paths.series_config, {"runner": {"isolate_stages": True, "max_auto_retries": retries},
                                               "fake": {"dir": str(notes), how: [S1]}})
    monkeypatch.setitem(runner.STAGE_FUNCS, "crop", FAKES.format("crop"))
    monkeypatch.setitem(runner.STAGE_FUNCS, "mc", FAKES.format("mc"))
    t0 = time.monotonic()
    res = _run(s, until="mc")
    assert time.monotonic() - t0 < 60
    calls = (notes / "calls.txt").read_text(encoding="utf-8").splitlines()
    assert calls.count(f"{S1} mc") == attempts and calls.count(f"{S3} mc") == 1
    rec = load_record(s.paths, S1)
    assert rec.stages["mc"].state is State.FAILED
    assert all(rec.stages[st].state is State.SKIPPED for st in STAGES[2:])
    assert _states(s.paths, S3)[:2] == [State.DONE, State.DONE]
    assert res["failed"] == 1 and res["done"] == 3
    text = s.paths.session_log(S1).read_text(encoding="utf-8")
    crashed = [a for a in rec.anomalies if a.code == "stage_crashed"]
    if how == "crash":
        assert "stage process exited with code 0xc0000005" in rec.stages["mc"].error
        assert [(a.level, a.data) for a in crashed] == [("error", {"exit_code": 0xC0000005, "stage": "mc"})]
        assert "Windows fatal exception: access violation" in text  # faulthandler of the stage process
    else:
        assert "ValueError: boom" in rec.stages["mc"].error and crashed == []
        assert "isolate_fakes.py" in text  # the child's traceback, logged by the runner
    assert text.count("automatic retry") == (attempts - 1)
    if attempts == 2:
        assert "automatic retry 1 of 1" in text
    assert "automatic retry" not in s.paths.session_log(S3).read_text(encoding="utf-8")


@pytest.mark.parametrize("stage, cfg_patch, what", [
    ("cnmf", {"cnmf": {"gSig": None}}, "waiting for thresholds"),
    ("mc", {}, "waiting for crop_finalize"),
    ("gate", {}, "waiting for manual_inspection"),
])
def test_waiting_stage_does_not_start_a_stage_process(series, monkeypatch, stage, cfg_patch, what):
    """A stage polled while it waits for a person must not cost a process start (0.5-12 s) per poll."""
    s = series
    rec = _record(S1, decided=False)
    cfg = config.deep_merge(config.resolve(s.paths, sid=S1), {"runner": {"isolate_stages": True}, **cfg_patch})
    cfg["crop"].pop("box_pos", None)
    monkeypatch.setitem(runner.STAGE_FUNCS, stage, f"bearmind.stages.{stage}:run")
    started = []
    monkeypatch.setattr(runner.isolate, "run_isolated", lambda *a, **k: started.append(a))
    with pytest.raises(WaitingForInput, match=f"^{what}$"):
        runner._call_stage(s.paths, S1, stage, rec, cfg, None, runner._Run(s.paths, None, None, True, 1))
    assert started == []
    # inputs present: the stage process is started
    rec.decisions = _record(S1).decisions
    cfg = config.deep_merge(cfg, {"cnmf": THRESH["cnmf"], "crop": {"box_pos": [0, 0], "box": [16, 16]}})
    runner._call_stage(s.paths, S1, stage, rec, cfg, None, runner._Run(s.paths, None, None, True, 1))
    assert len(started) == 1 and started[0][0] == f"bearmind.stages.{stage}:run"


# --- final fix wave 2 ---

def _isolated_in_process(s, monkeypatch):
    """runner.isolate_stages: true, as in production (the wait checks run in the runner), but the stage
    functions run in this process instead of a stage process."""
    config.update_yaml(s.paths.series_config, {"runner": {"isolate_stages": True}})
    monkeypatch.setattr(runner.isolate, "run_isolated",
                        lambda spec, paths, rec, cfg, log, **kw: spec(paths, rec, cfg))


def _snapshot(paths):
    files = sorted(paths.registry_file("x").parent.glob("*.json")) + [paths.summary_csv]
    return {f.name: (f.stat().st_mtime_ns, f.read_bytes()) for f in files}


def test_waiting_recheck_writes_nothing(series, monkeypatch):
    """I1: a stage still waiting for the same thing is checked again without any registry or summary write
    (no RUNNING on disk while it is checked, no rewrite of every record and of summary.csv per poll)."""
    s = series
    save_record(s.paths, _record(S1, decided=False))
    _isolated_in_process(s, monkeypatch)
    _run(s)
    st = load_record(s.paths, S1).stages["gate"]
    assert st.state is State.WAITING and st.error == "waiting for manual_inspection"
    before = _snapshot(s.paths)
    writes = []
    real = registry.update_record
    monkeypatch.setattr(registry, "update_record", lambda p, sid, change: (writes.append((sid, change)),
                                                                          real(p, sid, change))[1])
    s.stubs.calls.clear()
    _run(s)
    assert writes == [] and s.stubs.calls == []
    assert _snapshot(s.paths) == before
    # a decision makes the same check pass: the gate starts and the session finishes
    gate.decide(s.paths, S1, "not_needed", "viktor")
    _run(s)
    assert _states(s.paths, S1) == [State.DONE] * 7
    assert f"{S1} gate: started" in s.paths.session_log(S1).read_text(encoding="utf-8")


def _run_bounded(s, **kw):
    """runner.run(once=False) in a daemon thread; (thread, result holder): a runner that never returns
    fails the test instead of hanging it."""
    out = {}
    t = threading.Thread(target=lambda: out.update(res=runner.run(s.paths.root, **kw)), daemon=True)
    t.start()
    return t, out


def test_wait_blocked_by_unschedulable_session_ends_the_run(series):
    """I2: once=False returns when every remaining wait needs a config fix, naming the blocking session."""
    s = series
    s.paths.session_config(S2).write_text("cnmf: [unclosed" + chr(10), encoding="utf-8")
    t, out = _run_bounded(s, poll_s=0.05)
    t.join(60)
    assert not t.is_alive(), "the runner waits for a session that only a config fix can unblock"
    assert out["res"]["waiting"] == 1  # S1 mc, blocked
    log_text = s.paths.series_log.read_text(encoding="utf-8")
    assert "nothing can progress" in log_text
    assert f"{S1} mc waits for crop_finalize, blocked by {S2}" in log_text
    assert "waiting: thresholds 0, manual inspection 0" not in log_text  # never counted as a person's wait


def test_run_keeps_waiting_while_a_person_can_act(series):
    """I2: a blocked wait next to a wait a person can resolve keeps the runner polling."""
    s = series
    s.paths.session_config(S2).write_text("cnmf: [unclosed" + chr(10), encoding="utf-8")
    save_record(s.paths, _record(S3, decided=False))
    t, out = _run_bounded(s, poll_s=0.05)
    deadline = time.time() + 60
    while time.time() < deadline and load_record(s.paths, S3).stages["gate"].state is not State.WAITING:
        time.sleep(0.05)
    time.sleep(0.5)
    assert t.is_alive()
    msg = "waiting: thresholds 0, manual inspection 1; blocked by a session that cannot be scheduled: "
    assert msg + f"{S1} mc <- {S2}" in s.paths.series_log.read_text(encoding="utf-8")
    gate.decide(s.paths, S3, "not_needed", "viktor")
    t.join(60)
    assert not t.is_alive() and _states(s.paths, S3) == [State.DONE] * 7


def test_session_without_fps_is_visible(series):
    """I2: a record scanned before fps-less sessions were excluded is shown as not schedulable."""
    s = series
    rec = _record(S3)
    rec.fps = None
    save_record(s.paths, rec)
    _run(s)
    assert _states(s.paths, S3) == [State.PENDING] * 7
    err = pd.read_csv(s.paths.summary_csv).set_index("sid").loc[S3, "error"]
    assert "fps_unavailable" in err and "scan" in err
    assert all(_states(s.paths, sid) == [State.DONE] * 7 for sid in (S1, S2))


def test_corrupt_registry_file_is_a_visible_row(series):
    s = series
    bad = s.paths.registry_file("NOF_H09_1D")
    bad.write_text("{not json", encoding="utf-8")
    _run(s)
    row = pd.read_csv(s.paths.summary_csv).set_index("sid").loc["NOF_H09_1D"]
    assert str(bad) in row["error"] and "JSONDecodeError" in row["error"] and "scan" in row["error"]
    assert "NOF_H09_1D" in _heartbeat(s.paths)["unschedulable"]


# --- I4: the mouse crop box ---

def _set_box(paths, mouse, **crop):
    """Edit the mouse config as a person does (keys set to None are removed)."""
    f = paths.mouse_config(mouse)
    y = _yaml(f)
    c = y.setdefault("crop", {})
    for k, v in crop.items():
        if v is None:
            c.pop(k, None)
        else:
            c[k] = v
    f.write_text(yaml.safe_dump(y), encoding="utf-8")


def test_removed_mouse_box_is_placed_again_never_uncropped(series):
    s = series
    _run(s)
    s.stubs.calls.clear()
    _set_box(s.paths, "NOF_H01", box=None)
    _run(s)
    assert _yaml(s.paths.mouse_config("NOF_H01"))["crop"]["box"] == [16, 16]
    assert s.stubs.calls == []  # the same box again: MC is up to date, it never ran uncropped
    assert load_record(s.paths, S1).stages["mc"].metrics["crop"] == [0, 0, 16, 16]


def test_removed_mouse_box_waits_while_finalize_cannot_run(series):
    """box_pos without box: MC waits for crop_finalize (here finalize waits for an unschedulable mate)."""
    s = series
    _run(s)
    s.stubs.calls.clear()
    _set_box(s.paths, "NOF_H01", box=None)
    s.paths.session_config(S2).write_text("cnmf: [unclosed" + chr(10), encoding="utf-8")
    _run(s)
    assert (S1, "mc") in s.stubs.calls  # tried, and it waits
    st = load_record(s.paths, S1).stages["mc"]
    assert st.state is State.WAITING and st.error.startswith("waiting for crop_finalize")
    assert not st.outputs and "crop" not in st.metrics


def test_resized_mouse_box_places_the_sessions_again(series):
    """A resized box: positions re-centred inside the frame, MC re-runs cropped; the box is kept (manual)."""
    s = series
    config.update_yaml(s.paths.series_config, {"crop": {"margin_px": 1}})
    _run(s)
    assert _yaml(s.paths.mouse_config("NOF_H01"))["crop"] == {"box": [12, 11], "box_source": "auto"}
    assert _yaml(s.paths.session_config(S1))["crop"]["box_pos"] == [1, 2]  # +15 would pass the 16 px edge
    s.stubs.calls.clear()
    _set_box(s.paths, "NOF_H01", box=[15, 15])
    _run(s)
    assert s.stubs.calls == _tail(S1, "mc")[:1] + _tail(S2, "mc")[:1] + [
        (sid, st) for st in STAGES[2:] for sid in (S1, S2)]
    for sid in (S1, S2):
        top, left = _yaml(s.paths.session_config(sid))["crop"]["box_pos"]
        assert 0 <= top and top + 15 <= 16 and 0 <= left and left + 15 <= 16
        assert load_record(s.paths, sid).stages["mc"].metrics["crop"] == [top, left, 15, 15]
    assert _yaml(s.paths.mouse_config("NOF_H01"))["crop"] == {"box": [15, 15], "box_source": "manual"}
    assert runner.summary(s.paths).set_index("sid").loc[S1, "crop_box"] == "15x15@0,0"


@pytest.mark.parametrize("source, box", [("auto", [8, 8]), ("manual", [12, 11])])
def test_detection_parameter_change_recomputes_an_automatic_box(series, source, box):
    s = series
    config.update_yaml(s.paths.series_config, {"crop": {"margin_px": 1}})
    _run(s)
    if source == "manual":
        _set_box(s.paths, "NOF_H01", box_source="manual")
    s.stubs.fov.update({S1: [4, 4, 6, 6], S2: [5, 5, 6, 6]})  # what the new threshold finds
    config.update_yaml(s.paths.mouse_config("NOF_H01"), {"crop": {"fov_threshold_frac": 0.3}})
    s.stubs.calls.clear()
    _run(s)
    assert (S1, "crop") in s.stubs.calls and (S2, "crop") in s.stubs.calls
    assert _yaml(s.paths.mouse_config("NOF_H01"))["crop"]["box"] == box
    for sid in (S1, S2):
        assert load_record(s.paths, sid).stages["mc"].metrics["crop"][2:] == box
