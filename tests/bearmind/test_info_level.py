"""The scan writes anomalies of level "info" (timing_jitter, gap_test_skipped). Every reader of anomaly levels must
treat them as no problem: no warning, no exclusion, no wait, no crash."""
from bearmind import monitor, provenance, registry, runner, scan, scan_view
from bearmind.paths import SeriesPaths
from bearmind.registry import Anomaly
from tests.bearmind.gui.test_monitor_tab import done, put

INFO = [Anomaly("timing_jitter", "info", "timing jitter: 5 gaps, no frames lost", data={"n_gaps": 5}),
        Anomaly("gap_test_skipped", "info", "frame loss test skipped: the timestamps do not increase",
                data={"parts": [0]})]


def _series(tmp_path):
    paths = SeriesPaths(tmp_path / "s")
    paths.registry_file("x").parent.mkdir(parents=True)
    rec = put(paths, "LT1_J01_1D", {s: done() for s in ("crop", "mc", "thresholds")})
    registry.update_record(paths, rec.sid, lambda r: r.anomalies.extend(INFO))
    return paths, rec.sid


def test_info_anomalies_survive_the_registry_round_trip(tmp_path):
    paths, sid = _series(tmp_path)
    assert [(a.code, a.level) for a in registry.load_record(paths, sid).anomalies] == \
        [("timing_jitter", "info"), ("gap_test_skipped", "info")]


def test_info_anomalies_do_not_exclude_or_warn_in_monitor(tmp_path):
    paths, sid = _series(tmp_path)
    rec = registry.load_record(paths, sid)
    assert scan.evaluate_excluded(rec) is False
    view = monitor.series_view(paths)
    assert view.warnings == [] and view.n_warning_sessions == 0
    assert all(c.level != "error" for c in view.rows[0].cells.values())


def test_info_anomalies_are_no_problem_in_the_scan_view(tmp_path):
    paths, sid = _series(tmp_path)
    v = scan_view.scan_view(paths)
    assert v.n_problems == 0 and v.rows[0].level == "ok" and not v.rows[0].excluded
    assert all(x.info and not x.counts for x in v.rows[0].anomalies)


def test_info_anomalies_do_not_block_the_batch_state_or_the_summary(tmp_path):
    paths, sid = _series(tmp_path)
    state = runner.series_state(paths)
    assert state.problems == {}
    runner.summary(paths, state.records, problems=state.problems, broken=state.broken, view=state.view)
    assert paths.summary_csv.is_file()


def test_info_anomalies_export_into_the_provenance_as_they_are(tmp_path):
    paths, sid = _series(tmp_path)
    out = provenance.anomaly_list(registry.load_record(paths, sid).anomalies)
    assert [(a["code"], a["level"]) for a in out] == [("timing_jitter", "info"), ("gap_test_skipped", "info")]
