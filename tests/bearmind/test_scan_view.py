import re

from bearmind import config, monitor, scan, scan_view
from bearmind.paths import SeriesPaths
from bearmind.registry import (STAGES, Anomaly, Decision, RawPart, SessionRecord, StageRecord, load_record,
                               save_record, update_record)

CYRILLIC = re.compile("[Ѐ-ӿ]")
RULE = {"max_gap_s": 1.0, "share": 0.01}  # defaults.yaml scan.dropped_frames_warn


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
    assert line.note == "fixed automatically"
    # R2: the Anomalies column marks it and greys it, so it does not look like an error
    assert v.rows[0].summary_parts == [("extra chunk frames: 2 (fixed)", True)]


def test_split_recording_reviewed(make_series, tmp_path):
    root = make_series(tmp_path / "s", "legacy_split")
    scan.scan_series(root)
    paths = SeriesPaths(root)
    row = scan_view.scan_view(paths).rows[0]
    line = next(x for x in row.anomalies if x.code == "split_recording")
    assert row.level == "warning" and line.short.startswith("split recording: 2 parts, pause")
    assert not line.reviewed
    _decide(paths, row.sid, "stitch_confirmed", scan.stitch_fingerprint(load_record(paths, row.sid).parts))
    v = scan_view.scan_view(paths)
    assert next(x for x in v.rows[0].anomalies if x.code == "split_recording").reviewed
    assert v.rows[0].level == "ok" and v.n_problems == 0


def test_fps_outlier_short_names_mouse_median(tmp_path):
    an = Anomaly("fps_outlier", "warning", "m", data={"fps": 24.9, "median": 30.0})
    line = scan_view.scan_view(_save(tmp_path, _rec(anomalies=[an]))).rows[0].anomalies[0]
    assert line.short == "FPS outlier: 24.9 (mouse median 30.0)"  # a record scanned before 2026-11-08


def test_fps_outlier_short_names_the_other_rates(tmp_path):
    an = Anomaly("fps_outlier", "warning", "m", data={"fps": 24.9, "nearest": 29.8, "others": [19.8, 19.81, 29.8]})
    line = scan_view.scan_view(_save(tmp_path, _rec(anomalies=[an]))).rows[0].anomalies[0]
    assert line.short == "FPS outlier: 24.9 (other sessions 19.8, 29.8)"


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
    assert set(scan_view.ANOMALY_TITLE) >= {"frames_ts_mismatch", "registry_recreated", "fps_unavailable",
                                         "not_found_in_scan"}


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


def _three_part_line(n_gaps, max_gap_ms):
    split = Anomaly("split_recording", "warning", "split", data={"gaps_s": [60.0, 30.0]})
    drops = Anomaly("dropped_frames", "warning", "drops", data={"n_gaps": n_gaps, "max_gap_ms": max_gap_ms})
    rec = _rec(anomalies=[split, drops])
    rec.parts = rec.parts * 3
    rec.decisions = [Decision("stitch_confirmed", scan.stitch_fingerprint(rec.parts), "me", "t")]
    return next(x for x in scan_view._row(rec, RULE).anomalies if x.code == "dropped_frames")


def test_three_parts_two_pauses_no_drop_reviewed():
    assert _three_part_line(2, 60000.0).reviewed


def test_three_parts_two_pauses_plus_real_drop_stays_unreviewed():
    assert not _three_part_line(3, 60000.0).reviewed


def test_session_not_found_row(make_series, tmp_path):
    """I1: a session whose folder the last scan did not find is a red row excluded by anomaly."""
    root = make_series(tmp_path / "s", "bonsai")
    make_series(root, "bonsai", sid="NOF_H02_1D")
    scan.scan_series(root)
    next((root / "1_Raw").rglob("NOF_H02_1D")).rename(tmp_path / "away")
    scan.scan_series(root)
    row = next(r for r in scan_view.scan_view(SeriesPaths(root)).rows if r.sid == "NOF_H02_1D")
    assert row.excluded and row.excluded_by == "anomaly" and row.level == "error"
    assert row.summary == "not found by the last scan"
    assert scan_view.ANOMALY_TITLE["not_found_in_scan"] == "not found by the last scan"


def _dropped(data, n_frames=18000, fps=30.0):
    an = Anomaly("dropped_frames", "warning", "d", data=data)
    return scan_view._row(_rec(anomalies=[an], fps=fps, n_frames=n_frames), RULE)


def test_single_frame_drops_are_information():
    """I7: LT1 has 1-18 single-frame gaps of ~70 ms per session; they are shown, not counted as problems."""
    row = _dropped({"n_gaps": 18, "max_gap_ms": 70.0, "n_dropped": 18})
    line = row.anomalies[0]
    assert line.info and row.level == "ok"
    assert row.summary_parts == [("dropped frames: 18 gaps, longest 70 ms", False)]
    assert line.note == "information: no gap longer than 1 s, at most 1% of frames lost"
    assert _dropped({"n_gaps": 1, "max_gap_ms": 72.0, "n_dropped": 1}).level == "ok"


def test_dropped_frames_count_above_the_limits():
    assert _dropped({"n_gaps": 1, "max_gap_ms": 1500.0, "n_dropped": 44}).level == "warning"  # longest gap
    assert _dropped({"n_gaps": 50, "max_gap_ms": 300.0, "n_dropped": 400}).level == "warning"  # 2.2 % lost
    line = _dropped({"n_gaps": 50, "max_gap_ms": 300.0, "n_dropped": 400}).anomalies[0]
    assert not line.info and line.note == ""


def test_dropped_frames_scanned_before_the_estimate():
    """A record without n_dropped: the gap rule alone, and the line says the share is not known yet."""
    line = _dropped({"n_gaps": 1, "max_gap_ms": 72.0}).anomalies[0]
    assert line.info and "known after a rescan" in line.note
    # each gap loses at least one frame: a gap count above the share is enough to count it
    assert _dropped({"n_gaps": 300, "max_gap_ms": 72.0}).level == "warning"


def test_dropped_frames_rule_from_the_series_config(tmp_path):
    an = Anomaly("dropped_frames", "warning", "d", data={"n_gaps": 2, "max_gap_ms": 70.0, "n_dropped": 2})
    paths = _save(tmp_path, _rec(anomalies=[an], n_frames=18000))
    assert scan_view.scan_view(paths).n_problems == 0
    paths.series_config.parent.mkdir(parents=True, exist_ok=True)
    paths.series_config.write_text("scan:\n  dropped_frames_warn: {max_gap_s: 0.05}\n", encoding="utf-8")
    assert scan_view.scan_view(paths).n_problems == 1
    assert config.load_defaults()["scan"]["dropped_frames_warn"] == {"max_gap_s": 1.0, "share": 0.01}


def test_odd_record_becomes_a_row_that_says_so(tmp_path):
    """I6: one record the view cannot read (hand-edited, an older version) does not hide the table."""
    odd = Anomaly("frames_ts_mismatch", "error", "e", data={"diff": "abc", "where": "end"})
    paths = _save(tmp_path, _rec("NOF_H01_1D", anomalies=[odd], excluded=True), _rec("NOF_H01_2D"))
    rows = {r.sid: r for r in scan_view.scan_view(paths).rows}
    assert rows["NOF_H01_1D"].summary.startswith("cannot show: ValueError")
    assert rows["NOF_H01_1D"].level == "error" and rows["NOF_H01_1D"].n_video_frames is None
    assert rows["NOF_H01_1D"].excluded and rows["NOF_H01_1D"].excluded_by == "anomaly"
    assert rows["NOF_H01_2D"].summary == "–"
    null_gap = Anomaly("dropped_frames", "warning", "2 gaps", data={"n_gaps": 2, "max_gap_ms": None})
    assert scan_view._row(_rec(anomalies=[null_gap]), RULE).sid == "NOF_H01_1D"


def test_short_texts():
    def short(code, data, level="warning"):
        return scan_view._row(_rec(anomalies=[Anomaly(code, level, "m", data=data)]),
                              RULE).anomalies[0].short
    assert short("dropped_frames", {"n_gaps": 1, "max_gap_ms": 72.0}) == "dropped frames: 1 gap, longest 72 ms"
    assert short("dropped_frames", {"n_gaps": 2, "max_gap_ms": 3598000.0}) == \
        "dropped frames: 2 gaps, longest 59 min 58 s"
    # M4: parts counted from 1 for a person
    assert short("frames_ts_mismatch", {"diff": 2, "where": "part 0 end", "part": 0}) == \
        "frames ≠ timestamps: +2 at the end of part 1"
    assert [scan.duration_text(ms) for ms in (72.0, 1500.0, 38000.0, 3598000.0, 7380000.0)] == \
        ["72 ms", "1.5 s", "38 s", "59 min 58 s", "2 h 3 min"]


def test_split_pause_line_is_labelled():
    """M5: a dropped_frames line that only reports the pause between the parts says so."""
    split = Anomaly("split_recording", "warning", "split", data={"gaps_s": [3598.0]})
    drops = Anomaly("dropped_frames", "warning", "drops", data={"n_gaps": 1, "max_gap_ms": 3598000.0})
    rec = _rec(anomalies=[split, drops])
    rec.parts = rec.parts * 2
    lines = scan_view._row(rec, RULE).anomalies
    assert lines[0].short == "split recording: 2 parts, pause 59 min 58 s"
    assert lines[1].short == "dropped frames: 1 gap, longest 59 min 58 s (the pause between parts)"


def test_accepted_mismatch_says_accepted_by_you(make_series, tmp_path):
    """M1: an accepted mismatch is not "fixed automatically"."""
    root = make_series(tmp_path / "s", "bonsai", ts_rows=25)
    scan.scan_series(root)
    paths = SeriesPaths(root)
    sid = scan_view.scan_view(paths).rows[0].sid
    from bearmind import actions
    actions.accept_mismatch(paths, [sid], "me")
    row = scan_view.scan_view(paths).rows[0]
    line = next(x for x in row.anomalies if x.code == "frames_ts_mismatch")
    assert line.auto_fixed and line.note == "accepted by you"
    assert row.summary_parts == [("frames ≠ timestamps: +5 at the end (accepted)", True)]


def test_counted_lines_lead_the_summary():
    fixed = Anomaly("broken_last_frame", "warning", "b", auto_fixed=True)
    err = Anomaly("no_timestamps", "error", "n")
    row = scan_view._row(_rec(anomalies=[fixed, err]), RULE)
    assert row.summary_parts == [("no timestamps found", False), ("broken last frame (fixed)", True)]
    assert row.summary == "no timestamps found; broken last frame (fixed)"


def test_open_folder_of_a_session_without_parts():
    """M6: a session whose only part cannot be read has no parts; Open folder uses the anomaly's folder."""
    rec = _rec(anomalies=[Anomaly("unreadable_video", "error", "x", data={"folder": "1_Raw/HM/NOF_H01_1D"})])
    assert scan_view._row(rec, RULE).folder == "HM/NOF_H01_1D"
    rec.parts = []
    row = scan_view._row(rec, RULE)
    assert row.folder == "1_Raw/HM/NOF_H01_1D" and row.folders == []


def test_fps_text():
    """M11: the intervals left out of the estimate are not "dropped" frames. Only the gaps are left out; the
    uneven intervals (n_outliers, outside 0.5-1.5 of the median) are counted, so the text does not call them
    ignored."""
    rec = _rec(fps=20.0)
    assert scan_view._row(rec, RULE).fps_text == \
        ("20.0 from timestamps · 0 gaps left out of the FPS estimate"
         " · 0 uneven intervals (outside 0.5-1.5 of the median) · longest pause 70 ms")
    rec.fps_info["n_outliers"], rec.fps_info["n_gaps"] = 1, 1
    text = scan_view._row(rec, RULE).fps_text
    assert "· 1 gap left out of the FPS estimate ·" in text and "· 1 uneven interval (" in text
    assert "ignored" not in text


def test_summary_counts_not_found_and_unreadable(make_series, tmp_path):
    """I1, M8: sessions the last scan did not find and unreadable registry files are named in the summary."""
    root = make_series(tmp_path / "s", "bonsai")
    make_series(root, "bonsai", sid="NOF_H02_1D")
    scan.scan_series(root)
    next((root / "1_Raw").rglob("NOF_H02_1D")).rename(tmp_path / "away")
    scan.scan_series(root)
    paths = SeriesPaths(root)
    paths.registry_file("NOF_X99_1D").write_text("{garbage", encoding="utf-8")
    assert scan_view.scan_view(paths).summary_text() == (
        "Found 1 session of 1 mouse · bonsai 1 · legacy 0 · excluded 1 · with problems 1 · unrecognized folders 0"
        " · not found by the last scan 1 · unreadable registry files 1")


def test_included_by_you_over_an_error():
    """I4: a person's include that overrides an error is shown as such."""
    err = Anomaly("frames_ts_mismatch", "error", "e", data={"diff": 50, "where": "end"})
    rec = _rec(anomalies=[err], decisions=[Decision("exclude", "false", "me", "t")])
    row = scan_view._row(rec, RULE)
    assert not row.excluded and row.excluded_by == "included" and row.level == "error"
    plain = scan_view._row(_rec(decisions=[Decision("exclude", "false", "me", "t")]), RULE)
    assert plain.excluded_by == ""
