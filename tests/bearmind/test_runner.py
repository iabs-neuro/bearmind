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

from bearmind import config, fileio, provenance, runner
from bearmind.paths import SeriesPaths
from bearmind.registry import (STAGES, Anomaly, Decision, RawPart, SessionRecord, StageRecord, State,
                               load_record, save_record)
from bearmind.stages import StageResult, WaitingForInput, gate

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
    decisions = [Decision("manual_inspection", "not_needed", "viktor", "2026-10-31T10:00:00")] if decided else []
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
        if "box_pos" not in (cfg.get("crop") or {}):  # as the real stage
            raise WaitingForInput("crop_finalize")
        s = rec.sid
        return StageResult(
            outputs=self._out(paths, mc_video=paths.mc_video(s), mc_shifts=paths.mc_shifts(s),
                              preview_mc=paths.preview(s, "mc")),
            metrics={"max_shift_px": 2.0, "mean_template_corr": 0.9})

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
                           metrics={"n_initial": 100, "n_final": 80, "deleted_frac": 0.2})

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
    config.update_yaml(paths.series_config, {"autoinspect": {"model_path": str(model)}})
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
    assert hb["counts"]["waiting_manual"] == 1 and hb["counts"]["waiting_thresholds"] == 1
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


def test_second_runner_refused(series):
    s = series
    me = psutil.Process()
    s.paths.lock.parent.mkdir(parents=True, exist_ok=True)
    s.paths.lock.write_text(json.dumps({"pid": me.pid, "create_time": me.create_time()}), encoding="utf-8")
    with pytest.raises(RuntimeError, match=f"series is already being processed by PID {me.pid}"):
        _run(s)
    assert s.stubs.calls == []
    assert s.paths.lock.exists()  # the other runner's lock stays


def test_summary_csv(series):
    s = series
    s.stubs.fail[(S2, "cnmf")] = RuntimeError("boom")
    s.stubs.metrics[(S3, "mc")] = {"max_shift_px": 31.0}
    _run(s)
    df = runner.summary(s.paths)
    assert list(df["sid"]) == SIDS
    assert set(STAGES) <= set(df.columns)
    assert {"mouse", "excluded", "crop_box", "mc_max_shift_px", "cnmf_n_components", "gate_decision",
            "autoinspect_n_final", "autoinspect_deleted_frac", "export_n_neurons", "export_median_snr",
            "gSig", "min_corr", "min_pnr", "worst_check", "waiting_for", "error"} <= set(df.columns)
    row = df.set_index("sid")
    assert row.loc[S1, "export"] == "done" and row.loc[S2, "cnmf"] == "failed"
    assert row.loc[S2, "export"] == "skipped"
    assert row.loc[S3, "mc_max_shift_px"] == 31.0 and row.loc[S1, "cnmf_n_components"] == 100
    assert row.loc[S1, "crop_box"] == "16x16@0,0"
    assert row.loc[S1, "gSig"] == 4 and row.loc[S1, "min_corr"] == 0.8 and row.loc[S1, "min_pnr"] == 8.0
    assert row.loc[S1, "gate_decision"] == "not_needed" and row.loc[S1, "export_n_neurons"] == 80
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
    # a person decides while the runner holds S1's record (loaded before CNMF started)
    s.stubs.hooks[(S1, "cnmf")] = lambda paths, rec, cfg: gate.decide(paths, S1, "not_needed", "viktor")
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
    assert rec.stages["export"].params == {"ml_threshold": 0.72}
    assert rec.stages["autoinspect"].params["fps"] == 20
    assert len(rec.stages["autoinspect"].params["model_sha256"]) == 64
    rec.stages["autoinspect"].params["autoinspect"]["ml_threshold"] = 0.6  # what the pickle was made with
    assert runner.stage_params("export", rec, cfg, s.paths) == {"ml_threshold": 0.6}
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
    assert hb["counts"] == {"done": 21, "running": 0, "waiting_thresholds": 0, "waiting_manual": 0,
                            "waiting_crop": 0, "failed": 0, "skipped": 0, "pending": 0}
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
    s = series
    _run(s)

    def locked(src, dst):
        raise PermissionError(13, "summary.csv is open in Excel")

    monkeypatch.setattr(fileio, "replace", locked)
    with caplog.at_level(logging.WARNING, logger="bearmind"):
        df = runner.summary(s.paths)
    assert len(df) == 3
    assert "summary.csv" in caplog.text


def test_unstorable_metrics_fail_only_that_stage(series):
    s = series
    s.stubs.metrics[(S1, "mc")] = {"when": object()}  # a stage bug: JSON cannot store it
    res = _run(s)
    st = load_record(s.paths, S1).stages["mc"]
    assert st.state is State.FAILED and "not JSON serializable" in st.error
    assert _states(s.paths, S2) == [State.DONE] * 7 and _states(s.paths, S3) == [State.DONE] * 7
    assert res["failed"] == 1


def test_broken_session_config_holds_only_that_session(series):
    s = series
    good = s.paths.session_config(S2).read_text(encoding="utf-8")
    s.paths.session_config(S2).write_text("cnmf: [unclosed\n", encoding="utf-8")
    _run(s)
    assert _states(s.paths, S2) == [State.PENDING] * 7  # held, not failed: nothing to reset later
    assert _states(s.paths, S1) == [State.DONE] * 7
    assert f"{S2}: cannot be scheduled: config" in s.paths.series_log.read_text(encoding="utf-8")
    s.paths.session_config(S2).write_text(good, encoding="utf-8")
    s.stubs.calls.clear()
    _run(s)
    assert s.stubs.calls == _tail(S2, "crop")


def test_waiting_session_that_cannot_be_scheduled_does_not_spin(series, monkeypatch):
    s = series
    save_record(s.paths, _record(S1, decided=False))
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
