import pytest

from bearmind import scan
from bearmind.paths import SeriesPaths
from bearmind.registry import Decision, load_record, save_record


def codes(rec):
    return {a.code: a for a in rec.anomalies}


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
    r.excluded = True
    r.decisions.append(Decision("accept", "yes", "me", "2026-01-01"))
    save_record(paths, r)
    r2 = scan.scan_series(root)[0]
    assert r2.excluded and r2.decisions[0].what == "accept"


def test_fps_outlier_vs_series_median(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai", sid="NOF_H01_1D")
    for sid in ("NOF_H02_1D", "NOF_H03_1D"):
        make_series(tmp_path / "s", "bonsai", sid=sid)
    ts = root / "0_TimeStamps" / "NOF_H03_1D_Mini_TS.csv"
    ts.write_text("".join(f"{i * 1000000}\n" for i in range(30)), encoding="utf-8")  # 10 fps
    recs = {r.sid: r for r in scan.scan_series(root)}
    assert "fps_outlier" in codes(recs["NOF_H03_1D"])
    assert "fps_outlier" not in codes(recs["NOF_H01_1D"])


def test_dropped_frames_flag(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai")
    ts = root / "0_TimeStamps" / "NOF_H01_1D_Mini_TS.csv"
    vals = [i * 500000 for i in range(30)]
    vals[15:] = [v + 1500000 for v in vals[15:]]
    ts.write_text("".join(f"{v}\n" for v in vals), encoding="utf-8")
    assert "dropped_frames" in codes(scan.scan_series(root)[0])


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


def test_fps_unavailable_reported(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai", chunk_frames=1, n_chunks=2, ts_rows=2)
    r = scan.scan_series(root)[0]
    assert r.fps is None and codes(r)["fps_unavailable"].level == "warning"


@pytest.mark.parametrize("rate,expect_outlier", [("30FPS", True), ("20.0 FPS", False), ("fast", False)])
def test_nominal_frame_rate_lenient(make_series, tmp_path, rate, expect_outlier):
    import json
    root = make_series(tmp_path / "s", "legacy")
    meta = next((root / "1_Raw").rglob("Miniscope/metaData.json"))
    meta.write_text(json.dumps({"frameRate": rate}), encoding="utf-8")
    r = scan.scan_series(root)[0]
    assert ("fps_outlier" in codes(r)) == expect_outlier
