import os
import threading
import time

import numpy as np
import pytest

from bearmind import scan
from bearmind.paths import SeriesPaths
from bearmind.registry import (Anomaly, Decision, StageRecord, State, load_record, save_record,
                               update_record)
from bearmind.video import SessionReader


def codes(rec):
    return {a.code: a for a in rec.anomalies}


def _numbered_chunks(make_avi, folder, names, first, n):
    """Chunks of constant 16x16 frames whose pixel value is the running frame number from `first`."""
    for j, nm in enumerate(names):
        vals = np.arange(first + j * n, first + (j + 1) * n).astype(np.uint8)
        make_avi(folder / nm, np.broadcast_to(vals[:, None, None], (n, 16, 16)).copy())


def test_bonsai_nested_and_ts_dir(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai")
    recs = scan.scan_series(root)
    assert [r.sid for r in recs] == ["NOF_H01_1D"]
    r = recs[0]
    assert r.mode == "bonsai" and r.n_frames == 30 and not r.excluded
    assert r.parts[0].chunks == ["NOF_H01_1D_0.avi", "NOF_H01_1D_1.avi", "NOF_H01_1D_2.avi"]
    assert r.parts[0].timestamps_file == "0_TimeStamps/NOF_H01_1D_Mini_TS.csv"
    assert r.fps == pytest.approx(20.0)
    assert set(r.stages) >= {"crop", "export"}
    assert load_record(SeriesPaths(root), "NOF_H01_1D").n_frames == 30
    assert SeriesPaths(root).series_config.exists()


def test_bonsai_ts_in_session_folder(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai", with_ts_in_session=True)
    r = scan.scan_series(root)[0]
    assert r.parts[0].timestamps_file.endswith("NOF_H01_1D/NOF_H01_1D_Mini_TS.csv")


def test_legacy_split_is_stitched(make_series, tmp_path):
    root = make_series(tmp_path / "s", "legacy_split")
    recs = scan.scan_series(root)
    assert len(recs) == 1
    r = recs[0]
    assert r.mode == "legacy" and len(r.parts) == 2
    a = codes(r)
    assert a["split_recording"].level == "warning"
    assert len(a["split_recording"].data["gaps_s"]) == 1
    assert r.n_frames == sum(p.n_frames for p in r.parts) == 60
    assert not r.excluded
    assert r.parts[0].start_ms < r.parts[1].start_ms


def test_split_part_surplus_frames_do_not_shift_next_part(make_series, make_avi, tmp_path):
    from bearmind.timestamps import session_timestamps
    root = make_series(tmp_path / "s", "legacy_split", n_chunks=2, chunk_frames=60)
    minis = sorted((root / "1_Raw").rglob("Miniscope"))  # 10_00_00 (part A), 11_00_00 (part B)
    for k, mini in enumerate(minis):
        _numbered_chunks(make_avi, mini, ["0.avi", "1.avi"], k * 120, 60)
    # part A: 120 frames but 118 timestamp rows; part B: 120 / 120
    (minis[0] / "timeStamps.csv").write_text("Frame Number,Time Stamp (ms),Buffer Index\n"
                                             + "".join(f"{i},{i * 50},0\n" for i in range(118)),
                                             encoding="utf-8")
    r = scan.scan_series(root)[0]
    assert [p.n_frames for p in r.parts] == [118, 120] and r.n_frames == 238
    a = codes(r)["frames_ts_mismatch"]
    assert a.level == "warning" and a.auto_fixed
    assert a.data == {"diff": 2, "where": "part 0 end", "part": 0}
    assert a.message.startswith(f"part 1 ({r.parts[0].folder}): ")  # counted from 1 in texts (M4)
    assert not r.excluded
    frames = np.concatenate(list(SessionReader(root, r).iter_chunks(50)))
    assert frames[:, 0, 0].tolist() == list(range(118)) + list(range(120, 240))
    ts = session_timestamps(SeriesPaths(root), r)
    assert len(ts.raw) == r.n_frames
    # frame k of part B gets timestamp row k of part B (part B starts 1 h after part A)
    assert ts.raw[118:].tolist() == [3600_000 + k * 50 for k in range(120)]


def test_legacy_single_part(make_series, tmp_path):
    root = make_series(tmp_path / "s", "legacy")
    r = scan.scan_series(root)[0]
    assert r.n_frames == 30 and len(r.parts) == 1 and "fps_outlier" not in codes(r)


def test_legacy_without_metadata(make_series, tmp_path):
    r = scan.scan_series(make_series(tmp_path / "s", "legacy", with_meta=False))[0]
    assert codes(r)["no_metadata"].level == "warning" and not r.excluded
    r2 = scan.scan_series(make_series(tmp_path / "s2", "legacy_split", with_meta=False))[0]
    assert codes(r2)["split_recording"].level == "error" and r2.excluded


def test_missing_timestamps_excluded(make_series, tmp_path):
    r = scan.scan_series(make_series(tmp_path / "s", "bonsai", with_ts=False))[0]
    a = codes(r)["no_timestamps"]
    assert a.level == "error" and r.excluded and r.fps is None and r.fps_info == {}
    assert r.n_frames == 30


def test_chunk_tail_rule(make_series, tmp_path):
    r = scan.scan_series(make_series(tmp_path / "s", "bonsai", ts_rows=28))[0]
    assert r.parts[0].drop_chunk_tail is True
    a = codes(r)["chunk_tail_frames"]
    assert a.level == "warning" and a.auto_fixed
    assert r.n_frames == 28 and r.parts[0].n_frames == 28 and not r.excluded


def test_broken_last_frame(make_series, tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "last_frame_readable", lambda p: False)
    r = scan.scan_series(make_series(tmp_path / "s", "bonsai", ts_rows=29))[0]
    assert r.parts[0].drop_last_frame is True
    assert codes(r)["broken_last_frame"].auto_fixed
    assert r.n_frames == 29 and not r.excluded


def test_truncated_last_chunk_on_real_file(make_series, make_avi, tmp_path):
    import av
    root = make_series(tmp_path / "s", "bonsai", ts_rows=29)
    folder = root / "1_Raw" / "HM_NOF_1D" / "NOF_H01_1D"
    names = [f"NOF_H01_1D_{i}.avi" for i in range(3)]
    _numbered_chunks(make_avi, folder, names, 0, 10)
    last = folder / names[-1]
    with av.open(str(last)) as c:
        pk = [p for p in c.demux(c.streams.video[0]) if p.size][-1]
        cut = pk.pos + pk.size // 2  # recording stopped in the middle of writing the last frame
    last.write_bytes(last.read_bytes()[:cut])
    r = scan.scan_series(root)[0]
    assert r.parts[0].drop_last_frame and codes(r)["broken_last_frame"].auto_fixed
    assert r.parts[0].chunk_frames == [10, 10, 10]
    assert r.n_frames == 29 and "frames_ts_mismatch" not in codes(r) and not r.excluded
    frames = np.concatenate(list(SessionReader(root, r).iter_chunks(8)))
    assert frames[:, 0, 0].tolist() == list(range(29))


def test_small_mismatch_trimmed(make_series, tmp_path):
    r = scan.scan_series(make_series(tmp_path / "s", "bonsai", chunk_frames=40, ts_rows=119))[0]
    a = codes(r)["frames_ts_mismatch"]
    assert a.level == "warning" and a.auto_fixed and a.data == {"diff": 1, "where": "end"}
    assert r.n_frames == 119 and not r.excluded


def test_large_mismatch_excluded(make_series, tmp_path):
    r = scan.scan_series(make_series(tmp_path / "s", "bonsai", ts_rows=25))[0]
    assert codes(r)["frames_ts_mismatch"].level == "error" and r.excluded
    assert r.n_frames == 25


def test_unrecognized_folder_reported(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai")
    make_series(tmp_path / "s", "bonsai", sid="NOF_H02_1D")
    bad = root / "1_Raw" / "nof_h01_1d_part2"
    bad.mkdir()
    (bad / "nof_h01_1d_part2_0.avi").write_bytes(b"x")
    recs = scan.scan_series(root)
    assert [r.sid for r in recs] == ["NOF_H01_1D", "NOF_H02_1D"]
    lines = (root / "_bearmind" / "scan_unrecognized.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "path,reason"
    assert any(ln.startswith("1_Raw/nof_h01_1d_part2,") for ln in lines[1:])


def test_unrecognized_csv_header_only(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai")
    scan.scan_series(root)
    assert (root / "_bearmind" / "scan_unrecognized.csv").read_text(encoding="utf-8").strip() == "path,reason"


def test_rescan_keeps_decisions(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai")
    paths = SeriesPaths(root)
    r = scan.scan_series(root)[0]
    r.decisions.append(Decision("accept", "yes", "me", "2026-01-01"))
    r.decisions.append(Decision("exclude", "true", "me", "2026-01-01"))
    save_record(paths, r)
    r2 = scan.scan_series(root)[0]
    assert r2.excluded and [d.what for d in r2.decisions] == ["accept", "exclude"]


def test_rescan_reevaluates_exclusion(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai", with_ts=False)
    assert scan.scan_series(root)[0].excluded
    ts = root / "0_TimeStamps" / "NOF_H01_1D_Mini_TS.csv"
    ts.parent.mkdir(parents=True)
    ts.write_text("".join(f"{i * 500000}\n" for i in range(30)), encoding="utf-8")
    r = scan.scan_series(root)[0]
    assert "no_timestamps" not in codes(r) and not r.excluded


def test_latest_user_exclude_decision_wins(make_series, tmp_path):
    # an error the stages can live with (a large frames/timestamps mismatch, the session still has its
    # frame rate); a session without one stays excluded (test_include_decision_cannot_include_...)
    root = make_series(tmp_path / "s", "bonsai", ts_rows=25)
    paths = SeriesPaths(root)
    r = scan.scan_series(root)[0]
    assert r.excluded
    r.decisions.append(Decision("exclude", "true", "me", "2026-01-01"))
    r.decisions.append(Decision("exclude", "false", "me", "2026-01-02"))
    save_record(paths, r)
    r = scan.scan_series(root)[0]
    assert codes(r)["frames_ts_mismatch"].level == "error" and not r.excluded


def test_accept_mismatch_keeps_large_mismatch_included(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai", ts_rows=25)
    paths = SeriesPaths(root)
    r = scan.scan_series(root)[0]
    assert r.excluded
    r.decisions.append(Decision("accept_mismatch", "true", "me", "2026-01-01"))
    save_record(paths, r)
    r = scan.scan_series(root)[0]
    a = codes(r)["frames_ts_mismatch"]
    assert a.level == "warning" and a.auto_fixed and a.data["diff"] == 5
    assert r.n_frames == 25 and not r.excluded


def _series_at(make_series, root, rates):
    """Bonsai sessions at the given frame rates (sid -> fps)."""
    for sid, fps in rates.items():
        make_series(root, "bonsai", sid=sid)
        tick = round(1e7 / fps)  # bonsai timestamps count 100 ns ticks
        (root / "0_TimeStamps" / f"{sid}_Mini_TS.csv").write_text("".join(f"{i * tick}\n" for i in range(30)),
                                                                  encoding="utf-8")
    return {r.sid: r for r in scan.scan_series(root)}


def test_fps_outlier_vs_mouse_median(make_series, tmp_path):
    """R1: mice of one series may be recorded at different rates; a session is compared with its own mouse."""
    recs = _series_at(make_series, tmp_path / "s", {
        "NOF_H01_1D": 20, "NOF_H01_2D": 20, "NOF_H01_3D": 20, "NOF_H01_4D": 10,  # one odd session
        "NOF_H02_1D": 30, "NOF_H02_2D": 30, "NOF_H02_3D": 30})  # a whole mouse at another rate
    a = codes(recs["NOF_H01_4D"])["fps_outlier"]
    assert a.message == "fps 10.0; other sessions of the mouse: 20.0" and a.level == "warning"
    assert a.data["nearest"] == pytest.approx(20.0) and a.data["mouse"] == "NOF_H01"
    assert [sid for sid, r in recs.items() if "fps_outlier" in codes(r)] == ["NOF_H01_4D"]


def test_no_fps_outlier_for_a_mouse_switching_rates(make_series, tmp_path):
    """R1: NOF mice switch between 19.8 and 29.8 fps from day to day; the mouse median (24.8) lies between
    them, but every session has another session of the mouse at its own rate, so none is an outlier."""
    recs = _series_at(make_series, tmp_path / "s", {
        "NOF_H01_1D": 19.8, "NOF_H01_2D": 19.8, "NOF_H01_3D": 29.8, "NOF_H01_4D": 29.8})
    assert not [sid for sid, r in recs.items() if "fps_outlier" in codes(r)]


def test_fps_outlier_names_the_other_rates(make_series, tmp_path):
    recs = _series_at(make_series, tmp_path / "s", {
        "NOF_H01_1D": 30.0, "NOF_H01_2D": 30.0, "NOF_H01_3D": 30.0, "NOF_H01_4D": 24.9})
    assert [sid for sid, r in recs.items() if "fps_outlier" in codes(r)] == ["NOF_H01_4D"]
    a = codes(recs["NOF_H01_4D"])["fps_outlier"]
    assert a.message == "fps 24.9; other sessions of the mouse: 30.0"
    recs = _series_at(make_series, tmp_path / "t", {
        "NOF_H01_1D": 19.8, "NOF_H01_2D": 19.8, "NOF_H01_3D": 29.8, "NOF_H01_4D": 24.9})
    a = codes(recs["NOF_H01_4D"])["fps_outlier"]
    assert a.message == "fps 24.9; other sessions of the mouse: 19.8, 29.8"
    assert a.data["nearest"] == pytest.approx(29.8, abs=0.01)


def test_no_fps_outlier_for_a_mouse_with_few_sessions(make_series, tmp_path):
    """R1: fewer than checks.min_sessions_for_relative sessions with a rate give no reference."""
    recs = _series_at(make_series, tmp_path / "s", {
        "NOF_H01_1D": 20, "NOF_H01_2D": 20, "NOF_H01_3D": 20,
        "NOF_H02_1D": 10, "NOF_H02_2D": 20,  # two sessions, different rates
        "NOF_H03_1D": 10})
    assert not [sid for sid, r in recs.items() if "fps_outlier" in codes(r)]


def _write_bonsai_ts(root, values_ms):
    ts = root / "0_TimeStamps" / "NOF_H01_1D_Mini_TS.csv"
    ts.write_text("".join(f"{int(round(v * 1e4))}\n" for v in values_ms), encoding="utf-8")


def _gappy_series(make_series, tmp_path, shifts):
    """400 frames at 50 ms; a shift of s periods from frame j on is a gap of 1 + s periods at j (2 m is not above 2 m: use s above 1)."""
    root = make_series(tmp_path / "s", "bonsai", n_chunks=4, chunk_frames=100)
    t = np.arange(400) * 50.0
    for j, s in shifts:
        t[j:] += s * 50.0
    _write_bonsai_ts(root, t)
    return root


def test_lost_frames_flag_lists_each_gap(make_series, tmp_path):
    root = _gappy_series(make_series, tmp_path, [(200, 1.05)])
    a = codes(scan.scan_series(root)[0])["dropped_frames"]
    assert a.level == "warning" and a.data["rule"] == "window"
    assert a.data["n_gaps"] == 1 and a.data["n_dropped"] == 1 and a.data["max_gap_ms"] == 102.5
    assert a.data["frames_probable"] == 1 and a.data["frames_certain"] == 0
    first, second = a.message.splitlines()
    assert first == "1 frame lost (50 ms) in 1 gap: probable 1 frame lost (50 ms)"
    assert second.startswith("1 frame lost (") and second.endswith("at 0:10.0, probable")


def test_gap_without_loss_is_one_info_line_and_no_dropped_frames(make_series, tmp_path):
    """Frame 200 is late by 1.05 periods and the next two catch up: the interval is above 2 m, the clock goes on."""
    root = make_series(tmp_path / "s", "bonsai", n_chunks=4, chunk_frames=100)
    t = np.arange(400) * 50.0
    t[200:203] += np.array([1.05, 1.0, 0.5]) * 50.0
    _write_bonsai_ts(root, t)
    c = codes(scan.scan_series(root)[0])
    assert "dropped_frames" not in c
    assert c["timing_jitter"].level == "info" and c["timing_jitter"].message == "timing jitter: 1 gap, no frames lost"


def test_gaps_are_judged_per_part_not_across_the_pause(make_series, tmp_path):
    """The pause between two parts is not a lost-frame gap (split_recording reports it)."""
    root = make_series(tmp_path / "s", "legacy_split", n_chunks=2, chunk_frames=100)
    recs = scan.scan_series(root)
    c = codes(recs[0])
    assert "split_recording" in c and "dropped_frames" not in c and "timing_jitter" not in c


def test_damaged_timestamp_row_skips_the_test_and_says_so(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai", n_chunks=4, chunk_frames=100)
    t = np.arange(400) * 50.0
    t[300] = t[299] - 10
    _write_bonsai_ts(root, t)
    rec = scan.scan_series(root)[0]
    c = codes(rec)
    assert c["gap_test_skipped"].level == "info" and "damaged row" in c["gap_test_skipped"].message
    assert "dropped_frames" not in c and rec.fps is not None
    assert not rec.excluded


def test_cyrillic_space_root(make_series, tmp_path):
    root = make_series(tmp_path / "серия 1", "legacy")
    r = scan.scan_series(root)[0]
    assert r.n_frames == 30 and r.mode == "legacy"


def test_detect_mode(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai")
    assert scan.detect_mode(root / "1_Raw" / "HM_NOF_1D" / "NOF_H01_1D") == "bonsai"
    assert scan.detect_mode(root / "1_Raw") is None
    root2 = make_series(tmp_path / "t", "legacy")
    assert scan.detect_mode(root2 / "1_Raw" / "LNOF_J01_1D" / "2024_10_15" / "10_00_00" / "Miniscope") == "legacy"


def test_legacy_unattributable_folder_reported(make_series, tmp_path):
    root = make_series(tmp_path / "s", "legacy")
    bad = root / "1_Raw" / "badname" / "2024_10_15" / "10_00_00" / "Miniscope"
    bad.mkdir(parents=True)
    (bad / "0.avi").write_bytes(b"x")
    odd = root / "1_Raw" / "LNOF_J02_1D" / "2024_10_15" / "10_00_00" / "Miniscope"
    odd.mkdir(parents=True)
    (odd / "video.avi").write_bytes(b"x")
    cam = root / "1_Raw" / "LNOF_J01_1D" / "2024_10_15" / "10_00_00" / "BehavCam_0"
    cam.mkdir(parents=True)
    (cam / "0.avi").write_bytes(b"x")
    scan.scan_series(root)
    text = (root / "_bearmind" / "scan_unrecognized.csv").read_text(encoding="utf-8")
    assert "badname/2024_10_15/10_00_00/Miniscope,legacy folder name is not a session id" in text
    assert "LNOF_J02_1D/2024_10_15/10_00_00/Miniscope,Miniscope AVI names are not <n>.avi" in text
    assert "BehavCam" not in text


def test_corrupt_chunk_does_not_abort_series(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai")
    make_series(tmp_path / "s", "bonsai", sid="NOF_H02_1D")
    bad = next((root / "1_Raw").rglob("NOF_H02_1D_1.avi"))
    bad.write_bytes(bytes(range(256)) * 20)
    recs = {r.sid: r for r in scan.scan_series(root)}
    assert recs["NOF_H01_1D"].n_frames == 30 and not recs["NOF_H01_1D"].excluded
    a = codes(recs["NOF_H02_1D"])["unreadable_video"]
    assert a.level == "error" and "NOF_H02_1D_1.avi" in a.message
    assert recs["NOF_H02_1D"].excluded


def test_fps_unavailable_excludes(make_series, tmp_path):
    """I2: the stages need the frame rate (MC video rate, CNMF-E fr): a session without one is excluded."""
    root = make_series(tmp_path / "s", "bonsai", chunk_frames=1, n_chunks=2, ts_rows=2)
    r = scan.scan_series(root)[0]
    assert r.fps is None and r.excluded and codes(r)["fps_unavailable"].level == "error"


def test_include_decision_cannot_include_a_session_without_fps(make_series, tmp_path):
    r = scan.scan_series(make_series(tmp_path / "s", "bonsai", with_ts=False))[0]
    r.decisions.append(Decision("exclude", "false", "viktor", "2026-11-01T00:00:00"))
    assert scan.evaluate_excluded(r) is True
    a = codes(r)["fps_unavailable"]
    assert a.level == "error" and not a.data.get("stage")
    assert [x.code for x in r.anomalies].count("fps_unavailable") == 1
    assert scan.evaluate_excluded(r) is True and [x.code for x in r.anomalies].count("fps_unavailable") == 1


def test_scan_recreates_an_unreadable_registry_file(make_series, tmp_path):
    """I2: a corrupt registry json does not crash the scan; the record is made again, the old file kept."""
    root = make_series(tmp_path / "s", "bonsai")
    paths = SeriesPaths(root)
    scan.scan_series(root)
    f = paths.registry_file("NOF_H01_1D")
    f.write_text("{not json", encoding="utf-8")
    r = scan.scan_series(root)[0]
    assert load_record(paths, "NOF_H01_1D").stages["crop"].state is State.PENDING
    kept = list(f.parent.glob("NOF_H01_1D.json.broken*"))
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8") == "{not json"
    a = codes(r)["registry_recreated"]
    assert a.level == "warning" and str(kept[0]) in a.message and "JSONDecodeError" in a.message
    assert not r.excluded


@pytest.mark.parametrize("rate,expect_outlier", [("30FPS", True), ("20.0 FPS", False), ("fast", False)])
def test_nominal_frame_rate_lenient(make_series, tmp_path, rate, expect_outlier):
    import json
    root = make_series(tmp_path / "s", "legacy")
    meta = next((root / "1_Raw").rglob("Miniscope/metaData.json"))
    meta.write_text(json.dumps({"frameRate": rate}), encoding="utf-8")
    r = scan.scan_series(root)[0]
    assert ("fps_outlier" in codes(r)) == expect_outlier



def test_rescan_keeps_stage_anomalies(make_series, tmp_path):
    """R29: a rescan replaces the scan's own anomalies only; stage-tagged ones stay."""
    root = make_series(tmp_path / "s", "bonsai", ts_rows=28)
    paths = SeriesPaths(root)
    r = scan.scan_series(root)[0]
    scan_codes = sorted(a.code for a in r.anomalies)

    def tag(d):
        d.anomalies.append(Anomaly("fov_not_found", "warning", "x", data={"stage": "crop"}))
        d.anomalies.append(Anomaly("stage_trouble", "error", "y", data={"stage": "mc"}))

    update_record(paths, r.sid, tag)
    r2 = scan.scan_series(root)[0]
    assert sorted(a.code for a in r2.anomalies if not a.data.get("stage")) == scan_codes
    assert {a.code for a in r2.anomalies if a.data.get("stage")} == {"fov_not_found", "stage_trouble"}
    assert not r2.excluded  # exclusion is decided by scan anomalies and user decisions only
    assert load_record(paths, r.sid) == r2


def test_rescan_keeps_a_concurrent_update(make_series, tmp_path):
    """R29: the rescan rereads under the lock, so a runner write made meanwhile survives."""
    root = make_series(tmp_path / "s", "bonsai")
    paths = SeriesPaths(root)
    sid = scan.scan_series(root)[0].sid
    entered, release = threading.Event(), threading.Event()

    def slow(d):
        entered.set()
        release.wait(20)
        d.stages["mc"] = StageRecord(state=State.DONE, params_hash="h")

    t = threading.Thread(target=update_record, args=(paths, sid, slow))
    t.start()
    assert entered.wait(10)
    sc = threading.Thread(target=scan.scan_series, args=(root,))
    sc.start()
    time.sleep(1.5)  # the scan has read the videos and wants the record
    release.set()
    t.join(20)
    sc.join(20)
    assert load_record(paths, sid).stages["mc"].params_hash == "h"


def test_progress_called_per_session(make_series, tmp_path):
    """I2: one call before each session is read (the slow part), one after the writes."""
    root = tmp_path / "s"
    sids = ["NOF_H01_1D", "NOF_H01_2D", "NOF_H02_1D"]
    for sid in sids:
        make_series(root, "bonsai", sid=sid)
    calls = []
    scan.scan_series(root, progress=lambda *a: calls.append(a))
    assert calls == [(0, 3, ""), (0, 3, sids[0]), (1, 3, sids[1]), (2, 3, sids[2]), (3, 3, sids[2])]


def test_progress_names_the_session_being_read(make_series, tmp_path, monkeypatch):
    """I2: while a session's files are read (patched to be slow), the last progress call names it."""
    root = tmp_path / "s"
    sids = ["NOF_H01_1D", "NOF_H01_2D", "NOF_H02_1D"]
    for sid in sids:
        make_series(root, "bonsai", sid=sid)
    calls, seen = [], []
    real = scan._build_record

    def slow(paths, sid, items, cfg_scan):
        seen.append(calls[-1])
        time.sleep(0.05)
        return real(paths, sid, items, cfg_scan)

    monkeypatch.setattr(scan, "_build_record", slow)
    scan.scan_series(root, progress=lambda *a: calls.append(a))
    assert seen == [(0, 3, sids[0]), (1, 3, sids[1]), (2, 3, sids[2])]


def _registry_files(paths):
    return {f.name: f.read_bytes() for f in sorted(paths.registry_file("x").parent.glob("*.json"))}


def test_progress_does_not_change_registry_files(make_series, tmp_path):
    root = make_series(tmp_path / "a", "legacy_split")
    scan.scan_series(root)
    plain = _registry_files(SeriesPaths(root))
    calls = []
    scan.scan_series(root, lambda i, n, sid: calls.append((i, n, sid)))
    assert calls and plain
    assert _registry_files(SeriesPaths(root)) == plain


def test_progress_raising_leaves_registry_readable(make_series, tmp_path):
    from bearmind.registry import read_all
    root = make_series(tmp_path / "s", "legacy_split")
    paths = SeriesPaths(root)

    def boom(i, n, sid):
        if i >= 1:
            raise RuntimeError("stop")
    with pytest.raises(RuntimeError):
        scan.scan_series(root, boom)
    records, broken = read_all(paths)
    assert records and not broken


def test_missing_raw_raises_before_any_write(tmp_path):
    """I1: a series without 1_Raw (renamed, a disconnected share) fails the scan; nothing is written."""
    root = tmp_path / "s"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="1_Raw not found"):
        scan.scan_series(root)
    assert list(root.iterdir()) == []


def test_raw_gone_on_rescan_keeps_the_registry(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai")
    paths = SeriesPaths(root)
    scan.scan_series(root)
    before = {f.name: f.read_bytes() for f in (root / "_bearmind").rglob("*") if f.is_file()}
    (root / "1_Raw").rename(tmp_path / "moved")
    with pytest.raises(FileNotFoundError, match="1_Raw not found"):
        scan.scan_series(root)
    assert {f.name: f.read_bytes() for f in (root / "_bearmind").rglob("*") if f.is_file()} == before
    assert not load_record(paths, "NOF_H01_1D").excluded


@pytest.mark.skipif(os.name != "nt", reason="directory junctions are a Windows feature")
def test_dangling_raw_junction_raises(tmp_path):
    import _winapi
    root = tmp_path / "s"
    root.mkdir()
    target = tmp_path / "share"
    target.mkdir()
    _winapi.CreateJunction(str(target), str(root / "1_Raw"))
    target.rmdir()  # the share is gone; the junction stays
    try:
        with pytest.raises(FileNotFoundError, match="1_Raw not found"):
            scan.scan_series(root)
        assert [p.name for p in root.iterdir()] == ["1_Raw"]
    finally:
        os.rmdir(root / "1_Raw")


def test_session_not_found_is_excluded_and_comes_back(make_series, tmp_path):
    """I1: a session folder removed in Explorer: the record stays (stages, decisions) with the error
    not_found_in_scan, so the batch skips it; a scan that finds the folder again includes it again."""
    root = make_series(tmp_path / "s", "bonsai")
    make_series(root, "bonsai", sid="NOF_H02_1D")
    paths = SeriesPaths(root)
    scan.scan_series(root)

    def done_crop(r):
        r.stages["crop"] = StageRecord(state=State.DONE, params_hash="h")
        r.decisions.append(Decision("exclude", "false", "me", "2026-01-01"))  # an include cannot keep it in

    update_record(paths, "NOF_H02_1D", done_crop)
    folder = next((root / "1_Raw").rglob("NOF_H02_1D"))
    folder.rename(tmp_path / "away")
    calls = []
    recs = scan.scan_series(root, lambda *a: calls.append(a))
    assert [r.sid for r in recs] == ["NOF_H01_1D"] and calls[0] == (0, 1, "")
    r = load_record(paths, "NOF_H02_1D")
    a = codes(r)["not_found_in_scan"]
    assert a.level == "error" and a.message == "session folder not found by the last scan"
    assert r.excluded and r.stages["crop"].params_hash == "h" and r.decisions[-1].what == "exclude"
    assert not load_record(paths, "NOF_H01_1D").excluded
    scan.scan_series(root)  # a second scan does not add a second anomaly
    assert [a.code for a in load_record(paths, "NOF_H02_1D").anomalies].count("not_found_in_scan") == 1
    (tmp_path / "away").rename(folder)
    scan.scan_series(root)
    r = load_record(paths, "NOF_H02_1D")
    assert "not_found_in_scan" not in codes(r) and not r.excluded
    assert r.stages["crop"].params_hash == "h"


def test_unreadable_registry_file_of_a_missing_session_is_left_alone(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai")
    paths = SeriesPaths(root)
    scan.scan_series(root)
    f = paths.registry_file("NOF_X99_1D")
    f.write_text("{garbage", encoding="utf-8")
    scan.scan_series(root)
    assert f.read_text(encoding="utf-8") == "{garbage"


def test_unrecognized_csv_written_atomically_with_retries(make_series, tmp_path, monkeypatch):
    """I5: scan_unrecognized.csv goes through fileio.write_atomic, which retries a replace that Windows
    refuses while another program holds the file."""
    from bearmind import fileio
    root = make_series(tmp_path / "s", "bonsai")
    real = os.replace
    refused = []

    def busy(src, dst):
        if str(dst).endswith("scan_unrecognized.csv") and len(refused) < 2:
            refused.append(dst)
            raise PermissionError("file is open in another program")
        return real(src, dst)

    monkeypatch.setattr(fileio, "REPLACE_STEP_S", 0.0)
    monkeypatch.setattr(fileio.os, "replace", busy)
    scan.scan_series(root)
    assert len(refused) == 2
    assert (root / "_bearmind" / "scan_unrecognized.csv").read_text(encoding="utf-8") == "path,reason\n"


def test_unreadable_video_keeps_the_session_folder(make_series, tmp_path):
    """M6: Open folder needs the session folder of a session whose videos cannot be read (it has no parts)."""
    root = make_series(tmp_path / "s", "bonsai")
    next((root / "1_Raw").rglob("NOF_H01_1D_1.avi")).write_bytes(bytes(range(256)) * 20)
    r = scan.scan_series(root)[0]
    assert r.parts == [] and codes(r)["unreadable_video"].data == {"folder": "1_Raw/HM_NOF_1D/NOF_H01_1D"}


def test_unrecognized_list_read_with_retries(make_series, tmp_path, monkeypatch):
    """I5: on Windows a reader can meet the list while a scan replaces it (PermissionError); it retries."""
    import pathlib
    root = make_series(tmp_path / "s", "bonsai")
    bad = root / "1_Raw" / "junk"
    bad.mkdir()
    (bad / "a.avi").write_bytes(b"x")
    scan.scan_series(root)
    real = pathlib.Path.read_text
    refused = []

    def busy(self, *a, **k):
        if self.name == "scan_unrecognized.csv" and len(refused) < 2:
            refused.append(self)
            raise PermissionError("being replaced")
        return real(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", busy)
    assert scan.read_unrecognized(SeriesPaths(root)) == [("1_Raw/junk", "folder name is not a session id")]
    assert len(refused) == 2


def test_fps_outlier_with_min_sessions_one(make_series, tmp_path):
    """checks.min_sessions_for_relative 1: a mouse with one session has no other session to compare with."""
    from bearmind import config
    root = tmp_path / "s"
    config.update_yaml(SeriesPaths(root).series_config, {"checks": {"min_sessions_for_relative": 1}})
    recs = _series_at(make_series, root, {"NOF_H01_1D": 20})
    assert "fps_outlier" not in codes(recs["NOF_H01_1D"])
