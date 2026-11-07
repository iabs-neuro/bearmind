import re

from bearmind import monitor, scan, scan_view
from bearmind.paths import SeriesPaths
from bearmind.registry import (STAGES, Anomaly, Decision, RawPart, SessionRecord, StageRecord, save_record,
                               update_record)

CYRILLIC = re.compile("[Ѐ-ӿ]")


def _rec(sid="NOF_H01_1D", anomalies=(), decisions=(), excluded=False, fps=30.0, n_frames=900):
    return SessionRecord(
        sid=sid, mouse="_".join(sid.split("_")[:2]), mode="bonsai",
        parts=[RawPart("HM/" + sid, ["a.avi"], n_frames, (16, 16), None, None, None, None, chunk_frames=[n_frames])],
        n_frames=n_frames, fps=fps,
        fps_info={"fps": fps, "n_outliers": 0, "n_gaps": 0, "max_gap_ms": 70.0, "median_interval_ms": 33.0,
                  "source": "timestamps"},
        anomalies=list(anomalies), decisions=list(decisions), excluded=excluded,
        stages={s: StageRecord() for s in STAGES})


def _save(tmp_path, *recs):
    paths = SeriesPaths(tmp_path / "s")
    for r in recs:
        save_record(paths, r)
    return paths


def _decide(paths, sid, what, value, excluded=None):
    def change(r):
        r.decisions.append(Decision(what, value, "me", "2026-11-07 14:02"))
        if excluded is not None:
            r.excluded = excluded
    update_record(paths, sid, change)


def test_row_fields(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai")
    rec = scan.scan_series(root)[0]
    v = scan_view.scan_view(SeriesPaths(root))
    row = v.rows[0]
    assert v.scanned and row.sid == rec.sid and row.mode == "bonsai"
    assert row.n_video_frames == sum(sum(p.chunk_frames) for p in rec.parts) == 30
    assert row.n_chunks == sum(len(p.chunks) for p in rec.parts) == 3
    assert row.n_parts == 1 and row.n_frames == rec.n_frames
    assert row.duration_min == rec.n_frames / rec.fps / 60
    assert row.folders == [p.folder for p in rec.parts]
    assert row.level == "ok" and row.summary == "–" and not row.excluded and row.excluded_by == ""
    assert row.fps_text.startswith(f"{rec.fps:.1f} from timestamps · ")


def test_mismatch_line(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai", ts_rows=25)
    rec = scan.scan_series(root)[0]
    row = scan_view.scan_view(SeriesPaths(root)).rows[0]
    an = next(a for a in rec.anomalies if a.code == "frames_ts_mismatch")
    line = next(x for x in row.anomalies if x.code == "frames_ts_mismatch")
    assert line.level == "error" and line.text == an.message
    assert line.short == f"frames ≠ timestamps: {an.data['diff']:+d} at the end"
    assert row.level == "error" and row.excluded and row.excluded_by == "anomaly"


def test_auto_fixed_marked_not_counted(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai", ts_rows=28)
    scan.scan_series(root)
    v = scan_view.scan_view(SeriesPaths(root))
    line = next(x for x in v.rows[0].anomalies if x.code == "chunk_tail_frames")
    assert line.auto_fixed and v.rows[0].level == "ok" and v.n_problems == 0


def test_split_recording_reviewed(make_series, tmp_path):
    root = make_series(tmp_path / "s", "legacy_split")
    scan.scan_series(root)
    paths = SeriesPaths(root)
    row = scan_view.scan_view(paths).rows[0]
    line = next(x for x in row.anomalies if x.code == "split_recording")
    assert row.level == "warning" and line.short.startswith("split recording: 2 parts, pause")
    assert not line.reviewed
    _decide(paths, row.sid, "stitch_confirmed", "true")
    v = scan_view.scan_view(paths)
    assert next(x for x in v.rows[0].anomalies if x.code == "split_recording").reviewed
    assert v.rows[0].level == "ok" and v.n_problems == 0


def test_fps_outlier_short_names_mouse_median(tmp_path):
    an = Anomaly("fps_outlier", "warning", "m", data={"fps": 24.9, "median": 30.0})
    line = scan_view.scan_view(_save(tmp_path, _rec(anomalies=[an]))).rows[0].anomalies[0]
    assert line.short == "FPS outlier: 24.9 (mouse median 30.0)"


def test_excluded_by(make_series, tmp_path):
    bad = SeriesPaths(make_series(tmp_path / "s", "bonsai", with_ts=False))
    scan.scan_series(bad.root)
    row = scan_view.scan_view(bad).rows[0]
    assert row.excluded and row.excluded_by == "anomaly"

    ok = SeriesPaths(make_series(tmp_path / "t", "bonsai"))
    sid = scan.scan_series(ok.root)[0].sid
    _decide(ok, sid, "exclude", "true", excluded=True)
    row = scan_view.scan_view(ok).rows[0]
    assert row.excluded and row.excluded_by == "you"
    assert row.decisions == ["2026-11-07 14:02 me: exclude = true"]
    _decide(ok, sid, "exclude", "false", excluded=False)
    row = scan_view.scan_view(ok).rows[0]
    assert not row.excluded and row.excluded_by == ""


def test_stage_anomalies_not_shown(tmp_path):
    stage = Anomaly("stage_crashed", "error", "boom", data={"stage": "cnmf"})
    row = scan_view.scan_view(_save(tmp_path, _rec(anomalies=[stage]))).rows[0]
    assert row.anomalies == [] and row.level == "ok"


def test_only_unrecognized(tmp_path):
    paths = SeriesPaths(tmp_path / "s")
    paths.scan_unrecognized_csv.parent.mkdir(parents=True)
    paths.scan_unrecognized_csv.write_text("path,reason\n1_Raw/a,no video\n1_Raw/b,no id\n", encoding="utf-8")
    v = scan_view.scan_view(paths)
    assert v.scanned and v.rows == [] and len(v.unrecognized) == 2


def test_many_anomalies_summarized(tmp_path):
    many = [Anomaly("dropped_frames", "warning", "g", data={"n_gaps": i + 1, "max_gap_ms": 100.0})
            for i in range(300)]
    row = scan_view.scan_view(_save(tmp_path, _rec(anomalies=many))).rows[0]
    assert row.summary.endswith("and 297 more") and len(row.anomalies) == 300
    assert row.summary.count("dropped frames") == 3


def test_scan_tab_shows_all_monitor_keeps_one_kind(tmp_path):
    an = [Anomaly("frames_ts_mismatch", "error", "e", data={"diff": 5, "where": "end"}),
          Anomaly("fps_outlier", "warning", "o", data={"fps": 20.0, "median": 30.0}),
          Anomaly("dropped_frames", "warning", "d", data={"n_gaps": 2, "max_gap_ms": 90.0}),
          Anomaly("no_metadata", "warning", "n")]
    paths = _save(tmp_path, _rec(anomalies=an, excluded=True))
    assert len(scan_view.scan_view(paths).rows[0].anomalies) == 4
    ws = [w for w in monitor.series_view(paths).warnings if w.sid == "NOF_H01_1D"]
    assert len(ws) == 1 and ws[0].kind == "scan_data"


def test_summary_text(tmp_path):
    warn = Anomaly("no_metadata", "warning", "n")
    paths = _save(tmp_path,
                  _rec("NOF_H01_1D", excluded=True, decisions=[Decision("exclude", "true", "me", "d")]),
                  _rec("NOF_H01_2D"),
                  _rec("NOF_H02_1D", anomalies=[warn]))
    v = scan_view.scan_view(paths)
    assert v.n_mice == 2 and v.n_excluded == 1 and v.n_problems == 1
    assert v.modes == {"bonsai": 3, "legacy": 0}
    assert v.summary_text() == ("Found 3 sessions of 2 mice · bonsai 3 · legacy 0 · excluded 1 · "
                                "with problems 1 · unrecognized folders 0")


def test_texts_are_english(make_series, tmp_path):
    root = make_series(tmp_path / "s", "legacy_split")
    scan.scan_series(root)
    paths = SeriesPaths(root)
    _decide(paths, scan_view.scan_view(paths).rows[0].sid, "exclude", "true")
    v = scan_view.scan_view(paths)
    texts = list(scan_view.ANOMALY_TITLE.values()) + [v.summary_text()]
    for row in v.rows:
        texts += [row.summary, row.fps_text, *row.decisions]
        for x in row.anomalies:
            texts += [x.short, x.text]
    assert texts and not [t for t in texts if CYRILLIC.search(t)]
    assert set(scan_view.ANOMALY_TITLE) >= {"frames_ts_mismatch", "registry_recreated", "fps_unavailable"}


def test_broken_registry_file_listed(make_series, tmp_path):
    root = make_series(tmp_path / "s", "bonsai")
    scan.scan_series(root)
    paths = SeriesPaths(root)
    paths.registry_file("NOF_X99_1D").write_text("{garbage", encoding="utf-8")
    v = scan_view.scan_view(paths)
    assert [p.name for p, _ in v.broken] == ["NOF_X99_1D.json"] and v.broken[0][1]
    assert len(v.rows) == 1


def test_not_scanned(tmp_path):
    v = scan_view.scan_view(SeriesPaths(tmp_path / "s"))
    assert not v.scanned and v.rows == []
