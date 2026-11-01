"""Command line tests: tiny series, stage stubs through runner.STAGE_FUNCS, exit codes and registry/config effects."""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bearmind import cli, config, runner
from bearmind.paths import SeriesPaths
from bearmind.registry import STAGES, Anomaly, State, load_record, save_record, update_record
from bearmind.stages import gate
from tests.bearmind.test_runner import SIDS, THRESH, Stubs, _record

S1, S2, S3 = SIDS
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def series(tmp_path, monkeypatch):
    home = Path(os.environ["BEARMIND_HOME"])
    config.update_yaml(home / "machine.yaml", {"temp_dir": str(tmp_path / "temp"), "n_cpus": 2, "ram_gb": 16.0,
                                               "cnmf_ram_budget_gb": 12.8})
    paths = SeriesPaths(tmp_path / "series")
    model = tmp_path / "model.pkl"
    model.write_bytes(b"model")
    # the stubs are closures: they run in this process, not in stage processes
    config.update_yaml(paths.series_config, {"autoinspect": {"model_path": str(model)},
                                             "runner": {"isolate_stages": False}})
    for sid in SIDS:
        save_record(paths, _record(sid, decided=False))
    stubs = Stubs()
    for st in STAGES:
        monkeypatch.setitem(runner.STAGE_FUNCS, st, stubs(st))
    monkeypatch.chdir(tmp_path)  # main() changes the folder; monkeypatch restores it
    return SimpleNamespace(paths=paths, stubs=stubs, root=paths.root, tmp=tmp_path)


def _main(*args):
    return cli.main([str(a) for a in args])


def _set(paths, sid, stage, state, error=None, **metrics):
    def change(rec):
        st = rec.stages[stage]
        st.state, st.error = state, error
        st.metrics.update(metrics)
    update_record(paths, sid, change)


def _yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def test_unknown_series(tmp_path, capsys):
    code = _main("status", tmp_path / "nope")
    out = capsys.readouterr()
    assert code == 2
    assert "[FAILED] series folder not found:" in out.out + out.err
    assert "nope" in out.out + out.err


def test_help_subprocess():
    r = subprocess.run([sys.executable, "-m", "bearmind", "--help"], cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0
    assert "status" in r.stdout and "thresholds" in r.stdout and "retry" in r.stdout


def test_scan(tmp_path, make_series, capsys):
    root = make_series(tmp_path / "scanseries")
    assert _main("scan", root) == 0
    assert "[OK]" in capsys.readouterr().out
    assert load_record(SeriesPaths(root), "NOF_H01_1D").mouse == "NOF_H01"


def test_run_and_until(series, capsys):
    assert _main("run", series.root, "--once", "--until", "mc", "--sessions", S1) == 0
    assert load_record(series.paths, S1).stages["mc"].state is State.DONE
    assert load_record(series.paths, S3).stages["crop"].state is State.PENDING  # other session untouched
    out = capsys.readouterr().out
    assert "done" in out and out.isascii()


def test_run_failed_exit_code(series):
    series.stubs.fail[(S1, "crop")] = RuntimeError("boom")
    assert _main("run", series.root, "--once") == 1
    assert load_record(series.paths, S1).stages["crop"].state is State.FAILED


def test_run_busy(series, capsys, monkeypatch):
    def busy(*a, **k):
        raise RuntimeError("series is already being processed by PID 4242")
    monkeypatch.setattr(runner, "run", busy)
    assert _main("run", series.root, "--once") == 3
    assert "[FAILED] series is already being processed by PID 4242" in capsys.readouterr().out


def test_run_ctrl_c(series, capsys, monkeypatch):
    def stop(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(runner, "run", stop)
    assert _main("run", series.root) == 130
    assert "same command" in capsys.readouterr().out


def test_run_bad_input(series, capsys):
    assert _main("run", series.root, "--sessions", "NOPE_X01_1D") == 2
    assert "NOPE_X01_1D" in capsys.readouterr().out
    assert _main("run", series.root, "--until", "bogus") == 2
    assert "bogus" in capsys.readouterr().err


def test_run_unexpected_error_one_line(series, capsys, monkeypatch):
    def broken(*a, **k):
        raise ValueError("kaput")
    monkeypatch.setattr(runner, "run", broken)
    assert _main("run", series.root) == 1
    out = capsys.readouterr()
    text = out.out + out.err
    assert "Traceback" not in text and "series.log" in text
    assert "kaput" in series.paths.series_log.read_text(encoding="utf-8")


def test_thresholds_set(series):
    code = _main("thresholds", series.root, "set", S1, "--gsig", 5, "--min-corr", 0.7, "--min-pnr", 9)
    assert code == 0
    assert _yaml(series.paths.session_config(S1))["cnmf"] == {"gSig": 5, "min_corr": 0.7, "min_pnr": 9.0}
    assert _main("thresholds", series.root, "set", "NOPE_X01_1D", "--gsig", 5, "--min-corr", 0.7,
                 "--min-pnr", 9) == 2


def test_thresholds_accept_suggested(series, capsys):
    sug = {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0, "method": "auto"}
    for sid in (S1, S2):
        _set(series.paths, sid, "thresholds", State.DONE, suggested=sug)
        _set(series.paths, sid, "cnmf", State.WAITING, error="waiting for thresholds")
    # S3 waits too but has no suggestion
    _set(series.paths, S3, "thresholds", State.DONE, suggested=None)
    _set(series.paths, S3, "cnmf", State.WAITING, error="waiting for thresholds")
    assert _main("thresholds", series.root, "accept-suggested") == 1
    out = capsys.readouterr().out
    assert f"[FAILED] {S3}: no threshold suggestion" in out
    for sid in (S1, S2):
        assert _yaml(series.paths.session_config(sid))["cnmf"] == {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}
    assert not series.paths.session_config(S3).exists()


def test_thresholds_accept_suggested_sessions(series):
    sug = {"gSig": 3, "min_corr": 0.6, "min_pnr": 7.0, "method": "auto"}
    _set(series.paths, S2, "thresholds", State.DONE, suggested=sug)
    assert _main("thresholds", series.root, "accept-suggested", "--sessions", S2) == 0
    assert _yaml(series.paths.session_config(S2))["cnmf"]["gSig"] == 3
    assert _main("thresholds", series.root, "accept-suggested", "--sessions", S1) == 1  # never ran


def test_gate_not_needed_and_all_waiting(series):
    _set(series.paths, S1, "gate", State.WAITING, error="waiting for manual_inspection")
    _set(series.paths, S2, "gate", State.WAITING, error="waiting for manual_inspection")
    assert _main("gate", series.root, "--all-waiting", "not-needed", "--who", "tester") == 0
    for sid in (S1, S2):
        rec = load_record(series.paths, sid)
        d = rec.decisions[-1]
        assert (d.what, d.value, d.who) == ("manual_inspection", "not_needed", "tester")
        assert rec.stages["gate"].state is State.PENDING
    assert not [d for d in load_record(series.paths, S3).decisions if d.what == "manual_inspection"]
    assert _main("gate", series.root, S3, "not-needed") == 0
    assert load_record(series.paths, S3).decisions[-1].who  # default: the Windows user name


def test_gate_all_waiting_takes_a_gate_being_checked(series, capsys):
    """I1: --all-waiting also takes a session whose CNMF is done and whose gate is being checked right now
    (not WAITING on disk at that moment); a passed gate and a decision that already counts are left alone."""
    for sid in SIDS:
        _set(series.paths, sid, "cnmf", State.DONE)
    _set(series.paths, S1, "gate", State.RUNNING)  # the runner's check of the gate in progress
    _set(series.paths, S2, "gate", State.DONE)  # passed
    gate.decide(series.paths, S3, "not_needed", "earlier")  # counts already
    assert _main("gate", series.root, "--all-waiting", "not-needed", "--who", "tester") == 0
    assert [d.who for d in load_record(series.paths, S1).decisions] == ["tester"]
    assert load_record(series.paths, S2).decisions == []
    assert [d.who for d in load_record(series.paths, S3).decisions] == ["earlier"]


def test_gate_cleaned(series, capsys):
    assert _main("gate", series.root, S1, "cleaned") == 1  # manual result missing
    assert "[FAILED]" in capsys.readouterr().out
    f = series.paths.manual_result(S1)
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    assert _main("gate", series.root, S1, "cleaned") == 0
    assert load_record(series.paths, S1).decisions[-1].value == "cleaned"


def test_gate_bad_input(series):
    assert _main("gate", series.root, S1, "maybe") == 2
    assert _main("gate", series.root, "NOPE_X01_1D", "not-needed") == 2


def test_decide(series):
    assert _main("decide", series.root, S1, "exclude", "--who", "me") == 0
    rec = load_record(series.paths, S1)
    assert rec.excluded is True and (rec.decisions[-1].what, rec.decisions[-1].value) == ("exclude", "true")
    assert _main("decide", series.root, S1, "include") == 0
    rec = load_record(series.paths, S1)
    assert rec.excluded is False and rec.decisions[-1].value == "false"


def test_decide_accept_mismatch(series):
    def add(rec):
        rec.anomalies.append(Anomaly("frames_ts_mismatch", "error", "frames and timestamps differ"))
        rec.excluded = True
    update_record(series.paths, S2, add)
    assert _main("decide", series.root, S2, "accept-mismatch") == 0
    rec = load_record(series.paths, S2)
    assert (rec.decisions[-1].what, rec.decisions[-1].value) == ("accept_mismatch", "true")
    assert rec.excluded is False
    assert rec.anomalies[-1].level == "warning"


def test_rerun(series):
    for st in STAGES:
        _set(series.paths, S1, st, State.DONE)
    assert _main("rerun", series.root, S1, "--from", "cnmf", "--who", "me") == 0
    rec = load_record(series.paths, S1)
    assert [rec.stages[s].state for s in STAGES] == [State.DONE] * 3 + [State.PENDING] * 4
    assert (rec.decisions[-1].what, rec.decisions[-1].value) == ("rerun_from", "cnmf")
    assert _main("rerun", series.root, S1, "--from", "bogus") == 2
    assert _main("rerun", series.root, "NOPE_X01_1D", "--from", "mc") == 2


def test_retry(series, capsys):
    _set(series.paths, S1, "crop", State.DONE)
    _set(series.paths, S1, "mc", State.FAILED, error="disk full")
    _set(series.paths, S1, "thresholds", State.SKIPPED, error="skipped: mc failed")
    _set(series.paths, S2, "crop", State.DONE)
    assert _main("retry", series.root, S1) == 0
    rec = load_record(series.paths, S1)
    assert rec.stages["crop"].state is State.DONE
    assert [rec.stages[s].state for s in STAGES[1:]] == [State.PENDING] * 6
    assert _main("retry", series.root, S2) == 1  # nothing failed there
    assert "no failed stage" in capsys.readouterr().out
    _set(series.paths, S3, "cnmf", State.FAILED, error="x")
    assert _main("retry", series.root, "--all-failed") == 0
    assert load_record(series.paths, S3).stages["cnmf"].state is State.PENDING
    assert _main("retry", series.root, "--all-failed") == 0  # none left
    assert _main("retry", series.root) == 2


def _heartbeat(paths, age_s, pid=4321, current=None, status="running"):
    when = (datetime.now() - timedelta(seconds=age_s)).isoformat(timespec="milliseconds")
    paths.heartbeat.parent.mkdir(parents=True, exist_ok=True)
    paths.heartbeat.write_text(json.dumps({"pid": pid, "time": when, "current": current, "status": status,
                                           "counts": {}, "by_stage": {}}), encoding="utf-8")


def _status(series, capsys):
    capsys.readouterr()
    assert _main("status", series.root) == 0
    out = capsys.readouterr().out
    assert out.isascii()
    return out


def test_status(series, capsys):
    _set(series.paths, S1, "crop", State.DONE)
    _set(series.paths, S1, "mc", State.FAILED, error="disk full")
    _set(series.paths, S2, "cnmf", State.WAITING, error="waiting for thresholds")
    _set(series.paths, S3, "gate", State.WAITING, error="waiting for manual_inspection")
    update_record(series.paths, S3, lambda r: setattr(r, "excluded", True))
    out = _status(series, capsys)
    assert "runner: not running" in out
    assert series.paths.summary_csv.is_file()
    assert "thresholds" in out and "disk full" in out
    assert all(s in out for s in SIDS)

    _heartbeat(series.paths, 5, current={"sid": S1, "stage": "mc"})
    assert f"runner: running PID 4321, current {S1}/mc" in _status(series, capsys)
    _heartbeat(series.paths, 600)
    assert "runner: not running" in _status(series, capsys)
    _heartbeat(series.paths, 5, status="idle")
    assert "runner: not running" in _status(series, capsys)


def test_bat_file_mentions_conda_and_module():
    text = (REPO / "run_bearmind.bat").read_text(encoding="ascii")
    assert "-m bearmind %*" in text and "--no-capture-output" in text and "%~dp0" in text


@pytest.mark.parametrize("flag,value", [("--min-corr", "92"), ("--min-corr", "nan"), ("--min-corr", "0"),
                                        ("--min-pnr", "inf"), ("--min-pnr", "0"), ("--min-pnr", "nan"),
                                        ("--gsig", "2.5"), ("--gsig", "0")])
def test_thresholds_set_rejects_bad_values(series, capsys, flag, value):
    args = {"--gsig": "5", "--min-corr": "0.7", "--min-pnr": "9", flag: value}
    flat = [x for kv in args.items() for x in kv]
    assert _main("thresholds", series.root, "set", S1, *flat) == 2
    assert not series.paths.session_config(S1).exists()
    assert value in capsys.readouterr().err


def test_gsig_message(series, capsys):
    _main("thresholds", series.root, "set", S1, "--gsig", "2.5", "--min-corr", "0.7", "--min-pnr", "9")
    assert "--gsig must be a whole number >= 1, got 2.5" in capsys.readouterr().err


def test_accept_suggested_invalid_suggestion(series, capsys):
    _set(series.paths, S1, "thresholds", State.DONE,
         suggested={"gSig": 4, "min_corr": 92, "min_pnr": 8.0, "method": "auto"})
    _set(series.paths, S2, "thresholds", State.DONE,
         suggested={"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0, "method": "auto"})
    assert _main("thresholds", series.root, "accept-suggested", "--sessions", S1, S2) == 1
    out = capsys.readouterr().out
    assert f"[FAILED] {S1}: suggested thresholds are invalid:" in out and "92" in out
    assert not series.paths.session_config(S1).exists()
    assert _yaml(series.paths.session_config(S2))["cnmf"]["gSig"] == 4


def test_warn_lines(series, capsys):
    assert _main("decide", series.root, S1, "accept-mismatch") == 0
    assert "[WARN]" in capsys.readouterr().out
    _main("decide", series.root, S2, "exclude")
    capsys.readouterr()
    assert _main("gate", series.root, S2, "not-needed") == 0
    assert "[WARN]" in capsys.readouterr().out
    _main("thresholds", series.root, "set", S2, "--gsig", "4", "--min-corr", "0.8", "--min-pnr", "8")
    assert "[WARN]" in capsys.readouterr().out


def test_cyrillic_path_not_escaped(tmp_path, capsys):
    folder = tmp_path / "серия"
    folder.mkdir()
    save_record(SeriesPaths(folder), _record(S1, decided=False))
    assert _main("status", folder) == 0
    assert "серия" in capsys.readouterr().out


def test_bat_setlocal():
    text = (REPO / "run_bearmind.bat").read_text(encoding="ascii")
    assert text.lower().splitlines()[1].strip() == "setlocal"


def test_warn_accept_mismatch_on_warning_level_mismatch(series, capsys):
    # a warning-level mismatch is not changed by the decision, so the student is told
    update_record(series.paths, S1, lambda r: r.anomalies.append(
        Anomaly("frames_ts_mismatch", "warning", "small mismatch")))
    assert _main("decide", series.root, S1, "accept-mismatch") == 0
    assert "[WARN]" in capsys.readouterr().out


def test_status_shows_sessions_that_cannot_be_scheduled(series, capsys):
    """I2: a broken session config, the mouse-mate it blocks and a corrupt registry file are on screen."""
    series.paths.session_config(S2).parent.mkdir(parents=True, exist_ok=True)
    series.paths.session_config(S2).write_text("cnmf: [unclosed" + chr(10), encoding="utf-8")
    _set(series.paths, S1, "crop", State.DONE)
    _set(series.paths, S1, "mc", State.WAITING, error="waiting for crop_finalize")
    bad = series.paths.registry_file("NOF_H09_1D")
    bad.write_text("{not json", encoding="utf-8")
    out = _status(series, capsys)
    lines = {ln.split()[0]: ln for ln in out.splitlines() if ln.split() and ln.split()[0] in SIDS + ["NOF_H09_1D"]}
    assert "cannot be scheduled: config: " in lines[S2]
    assert f"blocked by {S2}" in lines[S1]
    assert "broken registry file" in lines["NOF_H09_1D"] and "run scan" in lines["NOF_H09_1D"]
    head = out.splitlines()[1]
    assert "cannot be scheduled: 2" in head and "blocked: 1" in head


def test_decide_include_without_fps_warns(tmp_path, make_series, capsys):
    root = make_series(tmp_path / "noTS", with_ts=False)
    assert _main("scan", root) == 0
    capsys.readouterr()
    assert _main("decide", root, "NOF_H01_1D", "include") == 0
    out = capsys.readouterr().out
    assert "[WARN]" in out and "frame rate" in out and "stays excluded" in out
    assert load_record(SeriesPaths(root), "NOF_H01_1D").excluded is True
    status = _status(SimpleNamespace(root=root), capsys)
    assert "excluded: no_timestamps, fps_unavailable" in status


def test_scan_reports_a_recreated_registry_file(tmp_path, make_series, capsys):
    root = make_series(tmp_path / "re")
    assert _main("scan", root) == 0
    SeriesPaths(root).registry_file("NOF_H01_1D").write_text("{not json", encoding="utf-8")
    capsys.readouterr()
    assert _main("scan", root) == 0
    out = capsys.readouterr().out
    assert "[WARN] NOF_H01_1D: registry file was unreadable" in out and ".broken" in out


def test_resumed_run_reports_the_failed_sessions(series, capsys):
    """I3: a run with nothing left to do still reports the series state, and exits 1 while sessions fail."""
    series.stubs.fail[(S1, "crop")] = RuntimeError("boom")
    assert _main("run", series.root, "--once") == 1
    capsys.readouterr()
    assert _main("run", series.root, "--once") == 1
    out = capsys.readouterr().out
    assert out.isascii()
    # S2 and S3 wait for thresholds
    assert "[FAILED] this run: done 0, failed 0, skipped 0 | series now: failed 1, waiting for a person 2" in out
    assert f"{S1} failed at crop: RuntimeError: boom" in out
    assert f"after fixing the cause: retry {series.root} --all-failed" in out
    assert "[WARN] 2 sessions wait for a person" in out


def test_run_ends_when_only_a_config_fix_helps(series, capsys):
    """I2/I3: once=False returns (exit 1, the blocking session named) when every wait needs a config fix."""
    for sid in (S1, S3):
        save_record(series.paths, _record(sid))  # decided and thresholds set: no wait for a person
        config.update_yaml(series.paths.session_config(sid), THRESH)
    series.paths.session_config(S2).parent.mkdir(parents=True, exist_ok=True)
    series.paths.session_config(S2).write_text("cnmf: [unclosed" + chr(10), encoding="utf-8")
    out = {}
    t = threading.Thread(target=lambda: out.update(code=_main("run", series.root)), daemon=True)
    t.start()
    t.join(60)
    assert not t.is_alive(), "run waits for a session only a config fix can unblock"
    text = capsys.readouterr().out
    assert out["code"] == 1
    assert "series now: failed 0, waiting for a person 0, cannot be scheduled 1" in text
    assert f"[FAILED] {S2} cannot be scheduled: config: ConfigError: " in text
    assert f"{S1} mc waits for crop box: blocked by {S2}" in text


def test_finished_series_is_ok(series, capsys):
    for sid in SIDS:
        save_record(series.paths, _record(sid))
        config.update_yaml(series.paths.session_config(sid), THRESH)
    assert _main("run", series.root, "--once") == 0
    out = capsys.readouterr().out
    assert ("[OK] this run: done 21, failed 0, skipped 0 | series now: failed 0, waiting for a person 0, "
            "cannot be scheduled 0, excluded 0") in out


def test_accept_suggested_after_run_until_thresholds(series, capsys):
    """I5: the documented workflow run --until thresholds -> thresholds accept-suggested sets the thresholds of
    every session that has a suggestion and none in its config (CNMF is PENDING then, not WAITING)."""
    _write = config.update_yaml
    _write(series.paths.session_config(S3), {"cnmf": {"gSig": 5, "min_corr": 0.7, "min_pnr": 9.0}})  # set before
    assert _main("run", series.root, "--once", "--until", "thresholds") == 0
    assert load_record(series.paths, S1).stages["cnmf"].state is State.PENDING
    capsys.readouterr()
    assert _main("thresholds", series.root, "accept-suggested") == 0
    out = capsys.readouterr().out
    for sid in (S1, S2):
        assert _yaml(series.paths.session_config(sid))["cnmf"] == {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}
        assert f"[OK] {sid}: gSig=4" in out
    assert _yaml(series.paths.session_config(S3))["cnmf"]["gSig"] == 5 and S3 not in out


def test_scan_and_status_report_unrecognized_folders(tmp_path, make_series, capsys):
    """I8: a folder the scan cannot read as a session does not vanish from the operator's view."""
    root = make_series(tmp_path / "un")
    odd = root / "1_Raw" / "HM_NOF_1D" / "NOF_H01_1D_part2"
    odd.mkdir()
    (odd / "NOF_H01_1D_part2_0.avi").write_bytes(b"x")
    for i in range(11):
        junk = root / "1_Raw" / f"junk_{i:02d}"
        junk.mkdir()
        (junk / "0.avi").write_bytes(b"x")
    assert _main("scan", root) == 0
    out = capsys.readouterr().out
    csv_path = root / "_bearmind" / "scan_unrecognized.csv"
    assert "[WARN] 12 folders not recognized" in out
    assert "1_Raw/HM_NOF_1D/NOF_H01_1D_part2 - AVI names do not match <SID>_<n>.avi" in out
    assert sum(" - folder name is not a session id" in ln for ln in out.splitlines()) == 9  # 10 lines at most
    assert str(csv_path) in out
    status = _status(SimpleNamespace(root=root), capsys)
    assert f"unrecognized folders: 12; list: {csv_path}" in status


def test_status_without_unrecognized_folders_says_nothing(series, capsys):
    assert "unrecognized" not in _status(series, capsys)


def test_run_prints_progress_lines(series, capsys):
    """10e: one ASCII console line per stage start / done / failure; tracebacks stay in the logs."""
    series.stubs.fail[(S1, "crop")] = RuntimeError("доступ запрещён")  # a localized Windows error text
    assert _main("run", series.root, "--once", "--until", "mc") == 1
    out = capsys.readouterr().out
    assert out.isascii()
    lines = out.splitlines()
    assert any(ln.endswith(f"{S2} crop: started") for ln in lines)
    assert any(ln.endswith(f"{S2} crop: done (checks: ok)") for ln in lines)
    assert any(f"{S1} crop: failed: RuntimeError: " in ln and r"\u0434" in ln for ln in lines)
    assert "Traceback" not in out
    assert len(lines) < 20  # events only


def test_waiting_runner_prints_what_it_waits_for(series, capsys):
    s = series
    for sid in SIDS:
        config.update_yaml(s.paths.session_config(sid), THRESH)
    out = {}
    # poll 1 s: three in-process re-checks of a waiting gate take longer than a 0.05 s poll
    t = threading.Thread(target=lambda: out.update(res=runner.run(s.root, poll_s=1, console=True)), daemon=True)
    t.start()
    deadline = time.time() + 60
    text = ""
    while time.time() < deadline and "waiting: thresholds 0, manual inspection 3" not in text:
        text += capsys.readouterr().out
        time.sleep(0.05)
    assert "waiting: thresholds 0, manual inspection 3" in text and text.isascii()
    for sid in SIDS:
        gate.decide(s.paths, sid, "not_needed", "viktor")
    t.join(60)
    assert not t.is_alive()
