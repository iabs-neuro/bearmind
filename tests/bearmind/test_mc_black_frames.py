"""MC max shift check without black frames, and recheck of finished MC stages."""
import numpy as np
import pytest

from bearmind import actions, registry, runner
from bearmind.registry import Check, State
from bearmind.stages import mc
from tests.bearmind.test_cli import S1, S2, make_current, series  # noqa: F401  (the series fixture)


def _frames(means, shape=(4, 4)):
    return [np.full(shape, m, np.uint8) for m in means]


def test_black_frame_with_the_largest_shift_is_left_out():
    shifts = np.array([[35.7, 0], [2.0, 1.0], [0.5, 6.7], [0, 0]], np.float32)
    v, left = mc.max_shift_without_black(shifts, _frames([0, 40, 40, 40]).__getitem__, 0.5)
    assert v == pytest.approx(6.7) and left == [0]


def test_dark_frame_counts():
    shifts = np.array([[35.7, 0], [2.0, 1.0]], np.float32)
    assert mc.max_shift_without_black(shifts, _frames([2, 40]).__getitem__, 0.5)[1] == []


def test_several_black_frames():
    shifts = np.array([[30, 0], [0, 33], [5, 0], [1, 0]], np.float32)
    v, left = mc.max_shift_without_black(shifts, _frames([0, 0, 40, 0]).__getitem__, 0.5)
    assert v == pytest.approx(5.0) and left == [0, 1]


def test_all_frames_black():
    shifts = np.array([[3, 0], [1, 0]], np.float32)
    assert mc.max_shift_without_black(shifts, _frames([0, 0]).__getitem__, 0.5) == (None, [0, 1])


def test_only_frames_down_to_the_answer_are_read():
    shifts = np.array([[35, 0], [9, 0], [1, 0], [2, 0]], np.float32)
    frames, read = _frames([0, 40, 40, 40]), []
    mc.max_shift_without_black(shifts, lambda i: read.append(i) or frames[i], 0.5)
    assert read == [0, 1]


def test_a_shift_that_is_not_finite_is_reported_not_hidden():
    from bearmind import checks
    shifts = np.array([[1.0, 0], [np.nan, 0], [2.0, 0]], np.float32)
    v, left = mc.max_shift_without_black(shifts, _frames([40, 40, 40]).__getitem__, 0.5)
    assert np.isnan(v) and left == []
    cfg = {"checks": {"mc_max_shift_px": {"warn": 10, "error": 30}}}
    got = checks.evaluate("mc", {"max_shift_px": v}, [], cfg)
    assert any("max_shift_px is not a number" in c.message for c in got)


def test_shift_metrics_keys():
    shifts = np.array([[35.7, 0], [2.0, 6.7]], np.float32)
    m = mc.shift_metrics(shifts, _frames([0, 40]), 0.5)
    assert m["max_shift_px"] == pytest.approx(6.7) and m["black_frames_left_out"] == [0]
    assert m["max_shift_all_px"] == pytest.approx(35.7)


def _mc_session(series, make_avi, sid=S1, first_black=True):
    """MC of `sid` done with a stored error-level check, real shifts file and MC video (frame 0 black)."""
    paths = series.paths
    make_current(paths, sid, "crop", "mc", "thresholds", "cnmf")
    shifts = np.array([[35.7, 0], [2.0, 1.0], [0.5, 6.7], [0, 0]], np.float32)
    paths.mc_video(sid).parent.mkdir(parents=True, exist_ok=True)
    np.save(paths.mc_shifts(sid), shifts)
    movie = np.full((4, 8, 8), 40, np.uint8)
    if first_black:
        movie[0] = 0
    make_avi(paths.mc_video(sid), movie)

    def change(rec):
        st = rec.stages["mc"]
        st.metrics = {"max_shift_px": 35.7, "mean_template_corr": 0.9, "n_frames": 4, "crop": [0, 0, 8, 8]}
        st.checks = [Check("mc", "mc_max_shift_px", "error", 35.7, "max shift 35.7 px (warn > 10, error > 30)"),
                     Check("mc", "mc_template_corr", "ok", 0.9, "template corr 0.9 (warn < 0.7)")]
    registry.update_record(paths, sid, change)
    return paths


def _snapshot(paths, sid):
    rec = registry.load_record(paths, sid)
    return rec, runner.StageView(paths).hashes(rec)


def test_recheck_rewrites_only_metrics_and_checks(series, make_avi):
    """Frame 0 black with a 35.7 px shift; the rest 6.7 px at most: error -> ok; hashes, other metric keys and
    the other stages untouched."""
    paths = _mc_session(series, make_avi)
    before, hashes = _snapshot(paths, S1)
    rows = actions.recheck_mc(paths, [S1])
    assert [(r.sid, r.before, r.after, r.note) for r in rows] == [(S1, "error", "ok", "")]
    after, hashes_after = _snapshot(paths, S1)
    st, old = after.stages["mc"], before.stages["mc"]
    assert st.metrics["max_shift_px"] == pytest.approx(6.7)
    assert st.metrics["black_frames_left_out"] == [0]
    assert st.metrics["max_shift_all_px"] == pytest.approx(35.7)
    for key in ("mean_template_corr", "n_frames", "crop"):
        assert st.metrics[key] == old.metrics[key]
    assert {c.name: c.level for c in st.checks} == {"mc_max_shift_px": "ok", "mc_template_corr": "ok"}
    assert st.checks[0].message.startswith("max shift 6.7 px, 1 black frame left out (")
    assert (st.params_hash, st.params, st.finished, st.outputs) == (old.params_hash, old.params, old.finished,
                                                                   old.outputs)
    assert hashes_after == hashes  # MC, thresholds and CNMF-E hashes: nothing goes out of date
    view = runner.StageView(paths)
    assert all(view.current(after, s) for s in ("mc", "thresholds", "cnmf"))
    for s in ("crop", "thresholds", "cnmf"):
        assert after.stages[s] == before.stages[s]


def test_recheck_skips_missing_outputs(series, make_avi):
    """No shifts file: the row says so; the record is unchanged."""
    paths = _mc_session(series, make_avi)
    paths.mc_shifts(S1).unlink()
    before = registry.load_record(paths, S1)
    rows = actions.recheck_mc(paths, [S1])
    assert [(r.sid, r.note) for r in rows] == [(S1, "MC outputs missing")]
    assert registry.load_record(paths, S1) == before


def test_recheck_ignores_sessions_without_finished_mc(series, make_avi):
    paths = _mc_session(series, make_avi)
    assert [r.sid for r in actions.recheck_mc(paths)] == [S1]  # S2, S3 have no MC done
    assert registry.load_record(paths, S2).stages["mc"].state is not State.DONE
