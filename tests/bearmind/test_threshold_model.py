"""Threshold model without Qt: passing mask, blob counters, maps file, session states, start values."""
import re
import statistics
import time

import numpy as np

from bearmind import actions, config, registry, runner, threshold_model as tm
from bearmind.registry import State, update_record
from tests.bearmind.test_cli import S1, S2, S3, _set, make_current, series  # noqa: F401  (the series fixture)

CYR = re.compile("[Ѐ-ӿ]")


def _blob(shape, cy, cx, s=2.0):
    y, x = np.mgrid[:shape[0], :shape[1]]
    return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * s ** 2))


def _maps(mirror=False, single=True, g=4, gsigs=None):
    """Maps of gSig g; `gsigs` lists more gSigs, each with the same maps."""
    shape = (60, 60)
    cn = 0.1 * np.random.default_rng(0).random(shape)
    cn = cn + _blob(shape, 15, 15) + _blob(shape, 40, 40)
    pnr = 20 * cn
    if single:
        for yx in ((5, 50), (30, 5), (55, 20)):
            cn[yx], pnr[yx] = 0.95, 20.0
    cn_m, pnr_m = {}, {}
    if mirror:
        cnm = 0.1 * np.ones(shape)
        pnrm = 5 * np.ones(shape)
        cnm[10:13, 10:13], pnrm[10:13, 10:13] = 0.9, 15.0
        cn_m, pnr_m = {g: cnm}, {g: pnrm}
    gs = list(gsigs or [g])
    return tm.ThresholdMaps(gs, cn, pnr, {k: cn for k in gs}, {k: pnr for k in gs}, cn_m, pnr_m)


def test_passing_mask():
    cn, pnr = np.array([[0.9, 0.9]]), np.array([[10.0, 5.0]])
    assert tm.passing_mask(cn, pnr, 0.8, 8.0).tolist() == [[True, False]]


def test_counters_blobs_and_singles():
    c = tm.counters(_maps(), 4, 0.8, 8.0, 2)
    assert c.n_blobs == 2 and abs(c.single_frac - 3 / 5) < 1e-12
    assert c.n_mirror_blobs is None


def test_counters_mirror():
    c = tm.counters(_maps(mirror=True), 4, 0.8, 8.0, 2)
    assert c.n_mirror_blobs == 1


def test_nan_in_maps():
    m = _maps()
    m.cn[4][0, :] = np.nan
    m.cn[4][:, 0] = np.inf
    m.pnr[4][:, 0] = np.inf
    mask = tm.passing_mask(m.cn[4], m.pnr[4], 0.8, 8.0)
    assert not mask[0].any() and not mask[:, 0].any()
    assert tm.counters(m, 4, 0.8, 8.0, 2).n_blobs == 2
    img = tm.display_u8(m.cn[4])
    assert img.dtype == np.uint8 and (img[0] == 0).all()
    assert tm.gsig_thumbnail(m, 4, 30).shape == (30, 30)


def test_thumbnail_keeps_aspect():
    m = _maps()
    m.cn[4], m.pnr[4] = m.cn[4][:, :30], m.pnr[4][:, :30]
    assert tm.gsig_thumbnail(m, 4, 20).shape == (20, 10)


def _save(path, m, mirror):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"gsig": np.array(m.gsigs), "mean": m.mean, "max": m.max}
    for g in m.gsigs:
        arrays[f"cn_{g}"], arrays[f"pnr_{g}"] = m.cn[g], m.pnr[g]
        if mirror:
            arrays[f"cn_mirror_{g}"], arrays[f"pnr_mirror_{g}"] = m.cn_mirror[g], m.pnr_mirror[g]
    np.savez_compressed(path, **arrays)


def test_load_maps_roundtrip(tmp_path):
    for mirror in (False, True):
        m = _maps(mirror=mirror)
        _save(tmp_path / "m.npz", m, mirror)
        back = tm.load_maps(tmp_path / "m.npz")
        assert back.gsigs == [4]
        assert np.array_equal(back.cn[4], m.cn[4]) and np.array_equal(back.pnr[4], m.pnr[4])
        assert np.array_equal(back.mean, m.mean)
        assert bool(back.cn_mirror) is mirror and bool(back.pnr_mirror) is mirror


def _items(series):
    return {i.sid: i for i in tm.threshold_items(series.paths)}


def test_threshold_items_states(series):
    p = series.paths
    _set(p, S1, "thresholds", State.PENDING)
    assert _items(series)[S1].state == "no_maps"
    _save(p.threshold_maps(S2), _maps(), False)
    make_current(p, S2, "crop", "mc", "thresholds")
    _set(p, S2, "thresholds", State.DONE, suggested=None)
    _set(p, S2, "cnmf", State.WAITING, error="waiting for thresholds")
    it = _items(series)[S2]
    assert it.state == "waiting" and it.chosen is None and it.waiting and it.first_stale == "cnmf"
    # a suggestion and no thresholds: CNMF-E is PENDING, not WAITING
    _set(p, S2, "cnmf", State.PENDING, error=None)
    _set(p, S2, "thresholds", State.DONE,
         suggested={"gSig": 4, "min_corr": 0.85, "min_pnr": 8.0, "method": "robust_bg"})
    it = _items(series)[S2]
    assert it.state == "waiting" and it.suggested["gSig"] == 4
    config.update_yaml(p.session_config(S2), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    it = _items(series)[S2]
    assert it.state == "chosen" and it.chosen == {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}
    _set(p, S2, "cnmf", State.DONE)  # done with other thresholds (no hash of these): the batch runs it again
    assert _items(series)[S2].state == "chosen"
    make_current(p, S2, "cnmf")
    assert _items(series)[S2].state == "cnmf_done"
    update_record(p, S2, lambda r: setattr(r, "excluded", True))
    assert _items(series)[S2].state == "excluded"
    assert [i.sid for i in tm.threshold_items(p)] == sorted([S1, S2, S3])


def test_chosen_from_mouse_config(series):
    p = series.paths
    _save(p.threshold_maps(S1), _maps(), False)
    config.update_yaml(p.mouse_config("NOF_H01"), {"cnmf": {"gSig": 5, "min_corr": 0.7, "min_pnr": 6.0}})
    assert _items(series)[S1].chosen["gSig"] == 5


def test_no_maps_reason(series):
    p = series.paths
    _set(p, S1, "thresholds", State.FAILED,
         error="Traceback (most recent call last):\n  File x\nMemoryError: cannot allocate\n")
    it = _items(series)[S1]
    assert it.state == "no_maps" and it.reason == "MemoryError: cannot allocate"
    _set(p, S1, "thresholds", State.PENDING, error=None)
    assert _items(series)[S1].reason == "not computed yet"


def _item(suggested=None, chosen=None):
    return tm.ThresholdItem("S", "M", "waiting", "", suggested, chosen)


CFG = {"thresholds": {"fallback": {"min_corr": 0.85, "min_pnr": 8}}}


def test_initial_values_order():
    m = _maps(gsigs=[3, 4, 5])
    sug = {"gSig": 4, "min_corr": 0.85, "min_pnr": 8.0, "method": "robust_bg"}
    ch = {"gSig": 5, "min_corr": 0.7, "min_pnr": 6.0}
    assert tm.initial_values(_item(sug, ch), m, CFG) == (5, 0.7, 6.0, "chosen")
    assert tm.initial_values(_item(sug), m, CFG) == (4, 0.85, 8.0, "suggested")
    assert tm.initial_values(_item(), m, CFG)[3] == "fallback"


def test_initial_values_without_suggestion():
    assert tm.initial_values(_item(), _maps(gsigs=[3, 4, 5, 6, 7]), CFG) == (4, 0.85, 8.0, "fallback")
    assert tm.initial_values(_item(), _maps(gsigs=[5, 6, 7]), CFG)[0] == 6


def test_source_texts_english():
    sug = {"gSig": 4, "min_corr": 0.85, "min_pnr": 8.0, "method": "robust_bg"}
    assert tm.source_text("chosen", _item()) == "Your earlier choice"
    assert tm.source_text("suggested", _item(sug)) == \
        "Autotune suggestion (robust_bg): gSig 4, min_corr 0.85, min_pnr 8.0"
    assert tm.source_text("fallback", _item()) == "No suggestion — start values from the config"
    for t in (*tm.STATE_TEXT.values(), tm.source_text("fallback", _item()), tm.source_text("chosen", _item())):
        assert not CYR.search(t)
    assert tm.STATE_TEXT["cnmf_done"] == "CNMF-E done" and tm.STATE_TEXT["no_maps"] == "no maps"


def test_defaults_and_slider_range():
    cfg = config.load_defaults()
    assert cfg["thresholds"]["min_blob_px"] == 2
    assert cfg["thresholds"]["fallback"] == {"min_corr": 0.85, "min_pnr": 8}
    m = _maps()
    assert tm.slider_range_pnr(m, 4) >= 30.0
    m.pnr[4] = m.pnr[4] * 10
    assert tm.slider_range_pnr(m, 4) > 30.0


def test_mask_speed():
    rng = np.random.default_rng(1)
    cn, pnr = rng.random((600, 600)), 20 * rng.random((600, 600))
    m = tm.ThresholdMaps([4], cn, pnr, {4: cn}, {4: pnr}, {4: cn}, {4: pnr})
    times = []
    for _ in range(20):
        t = time.perf_counter()
        tm.passing_mask(cn, pnr, 0.8, 8.0)
        tm.counters(m, 4, 0.8, 8.0, 2)
        times.append(time.perf_counter() - t)
    assert statistics.median(times) < 0.2  # the window's budget is 50 ms; the bound leaves room for a loaded machine


def test_maps_without_a_suggestion_wait(series):
    """I1: maps up to date and no thresholds: the batch waits whether or not autotune suggested any, and whatever
    the registry says of CNMF-E (a FAILED one is offered again: its parameters changed with the thresholds)."""
    p = series.paths
    _save(p.threshold_maps(S1), _maps(), False)
    make_current(p, S1, "crop", "mc", "thresholds")
    _set(p, S1, "thresholds", State.DONE, suggested=None)
    for st in (State.PENDING, State.FAILED, State.WAITING):
        _set(p, S1, "cnmf", st)
        it = _items(series)[S1]
        assert it.state == "waiting" and it.waiting and it.suggested is None
        assert actions.waiting_threshold_sids(p) == [S1]


def test_cnmf_done_maps_deleted_is_no_maps(series):
    p = series.paths
    _set(p, S1, "thresholds", State.DONE)
    _set(p, S1, "cnmf", State.DONE)
    it = _items(series)[S1]
    assert it.state == "no_maps" and it.reason == "maps file not found (CNMF-E done)"


def test_waiting_count_equals_actions(series):
    p = series.paths
    # the registry says CNMF-E waits, but the maps failed: the batch redoes the thresholds stage after a retry
    _set(p, S1, "thresholds", State.FAILED, error="MemoryError: x")
    _set(p, S1, "cnmf", State.WAITING, error="waiting for thresholds")
    # suggestion, no thresholds, maps
    _save(p.threshold_maps(S2), _maps(), False)
    make_current(p, S2, "crop", "mc", "thresholds")
    _set(p, S2, "thresholds", State.DONE,
         suggested={"gSig": 4, "min_corr": 0.85, "min_pnr": 8.0, "method": "robust_bg"})
    _set(p, S2, "cnmf", State.PENDING)
    # no suggestion, CNMF-E pending: waits too (I1)
    _save(p.threshold_maps(S3), _maps(), False)
    make_current(p, S3, "crop", "mc", "thresholds")
    _set(p, S3, "thresholds", State.DONE, suggested=None)
    _set(p, S3, "cnmf", State.PENDING)
    items = _items(series)
    assert items[S1].state == "no_maps" and not items[S1].waiting
    assert items[S2].state == "waiting" and items[S3].state == "waiting"
    assert sorted(i.sid for i in items.values() if i.waiting) == actions.waiting_threshold_sids(p) == [S2, S3]
    update_record(p, S1, lambda r: setattr(r, "excluded", True))
    assert sorted(i.sid for i in _items(series).values() if i.waiting) == actions.waiting_threshold_sids(p)


def test_gsig_without_map_falls_back():
    m = _maps()
    m.gsigs = [4]
    ch = {"gSig": 9, "min_corr": 0.7, "min_pnr": 6.0}
    it = _item(None, ch)
    assert tm.initial_values(it, m, CFG) == (4, 0.7, 6.0, "chosen")
    assert tm.missing_gsig_note(it, m, "chosen") == "gSig 9 has no map; showing 4"
    assert tm.missing_gsig_note(_item(), m, "fallback") == ""
    assert tm.counters(m, 9, 0.8, 8.0, 2).n_blobs == 2
    assert tm.slider_range_pnr(m, 9) >= 30.0
    assert tm.gsig_thumbnail(m, 9, 30).shape == (30, 30)


def test_fallback_gsig_rule_shared():
    assert config.fallback_gsig([3, 4, 5, 6, 7]) == 4 and config.fallback_gsig([5, 6, 7]) == 6


def test_usable_gsigs_and_start_values():
    """A gSig listed in the file without both maps cannot be shown or chosen: the shared fallback among the usable
    gSigs takes its place, and the note says so."""
    m = _maps(gsigs=[3, 4, 5])
    del m.cn[4], m.pnr[4]
    assert m.gsigs == [3, 4, 5] and m.usable_gsigs == [3, 5]
    sug = {"gSig": 4, "min_corr": 0.85, "min_pnr": 8.0, "method": "robust_bg"}
    assert tm.initial_values(_item(sug), m, CFG) == (5, 0.85, 8.0, "suggested")
    assert tm.start_values(_item(sug), m, CFG) == (5, 0.85, 8.0, "suggested", "gSig 4 has no map; showing 5")
    assert tm.start_values(_item(), m, CFG) == (5, 0.85, 8.0, "fallback", "")
    assert tm.counters(m, 4, 0.8, 8.0, 2).n_blobs == 2  # counted on the fallback, not a KeyError
    assert tm.slider_range_pnr(m, 4) >= 30.0 and tm.gsig_thumbnail(m, 4, 30).shape == (30, 30)
    ch = {"gSig": 9, "min_corr": 0.7, "min_pnr": 6.0}
    assert tm.start_values(_item(sug, ch), m, CFG) == (5, 0.7, 6.0, "chosen", "gSig 9 has no map; showing 5")


def test_last_line_public():
    assert tm.last_line("Traceback\n  File x\nMemoryError: y\n\n") == "MemoryError: y"
    assert tm.last_line("") == "" and tm.last_line(None) == ""


# --- phase 4 final fix wave: states by the runner's view ---

def test_out_of_date_maps_state(series):
    """A3: maps DONE but not up to date (here a crop input changed): out_of_date, named after the stage the batch
    starts from; the maps are not offered (the note replaces them); a choice made earlier says the maps will be
    recomputed."""
    p = series.paths
    _save(p.threshold_maps(S1), _maps(), False)
    make_current(p, S1, "crop", "mc", "thresholds")
    config.update_yaml(p.mouse_config("NOF_H01"), {"crop": {"fov_threshold_frac": 0.3}})
    it = _items(series)[S1]
    assert (it.state, it.first_stale, it.waiting) == ("out_of_date", "crop", False)
    assert tm.state_text(it) == "maps out of date: the batch redoes crop first"
    assert it.note == "The maps are out of date: the batch redoes crop first."
    config.update_yaml(p.session_config(S1), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    assert tm.state_text(_items(series)[S1]) == "chosen · maps will be recomputed"
    _set(p, S1, "thresholds", State.FAILED, error="MemoryError: x")
    assert tm.state_text(_items(series)[S1]) == "chosen · maps out of date: the thresholds stage failed"
    for t in (tm.state_text(i) for i in _items(series).values()):
        assert not CYR.search(t)


def test_cnmf_done_only_when_current(series):
    """A3: CNMF-E DONE in the registry counts as done only while the batch would not redo it."""
    p = series.paths
    _save(p.threshold_maps(S1), _maps(), False)
    config.update_yaml(p.session_config(S1), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    make_current(p, S1, "crop", "mc", "thresholds", "cnmf")
    assert _items(series)[S1].state == "cnmf_done"
    config.update_yaml(p.session_config(S1), {"cnmf": {"min_pnr": 9.0}})  # new thresholds: CNMF-E runs again
    assert _items(series)[S1].state == "chosen"


def test_config_error_state_from_the_view(series):
    """B1 / A3: a config that cannot be read is "cannot be scheduled" with the reason of the runner's view, the one
    Monitor shows; the session never counts as one without thresholds."""
    p = series.paths
    _save(p.threshold_maps(S1), _maps(), False)
    make_current(p, S1, "crop", "mc", "thresholds")
    p.session_config(S1).parent.mkdir(parents=True, exist_ok=True)
    p.session_config(S1).write_text("cnmf: [unclosed\n", encoding="utf-8")
    it = _items(series)[S1]
    view = runner.StageView(p)
    rec = registry.load_record(p, S1)
    assert it.state == "config_error" and it.reason == view.problem(rec) and it.reason.startswith("config: ")
    assert tm.state_text(it) == f"cannot be scheduled: {it.reason}" and not it.waiting and it.chosen is None
    assert it.note == f"The session cannot be scheduled: {it.reason}"


def test_threshold_items_take_the_callers_view_and_records(series):
    """One view per refresh: the view and records of the caller are used; nothing is resolved twice."""
    p = series.paths
    records = registry.all_records(p)[:2]
    calls = []
    view = runner.StageView(p, resolve=lambda rec: calls.append(rec.sid) or config.resolve(p, sid=rec.sid))
    items = tm.threshold_items(p, view, records)
    assert [i.sid for i in items] == [S1, S2] and sorted(calls) == [S1, S2]


def test_choice_source_names_the_configs(series):
    """D5: the config layers the session's thresholds come from, the closest first."""
    p = series.paths
    assert tm.choice_source(p, S1, "NOF_H01") == ""
    config.update_yaml(p.series_config, {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    assert tm.choice_source(p, S1, "NOF_H01") == "series config"
    config.update_yaml(p.mouse_config("NOF_H01"), {"cnmf": {"min_corr": 0.7}})
    assert tm.choice_source(p, S1, "NOF_H01") == "mouse and series config"
    config.update_yaml(p.session_config(S1), {"cnmf": {"gSig": 5, "min_corr": 0.6, "min_pnr": 7.0}})
    assert tm.choice_source(p, S1, "NOF_H01") == "session config"
    sug = {"gSig": 4, "min_corr": 0.85, "min_pnr": 8.0, "method": "robust_bg"}
    assert tm.source_text("chosen", _item(sug), "session config") == "Your earlier choice (session config)"
    p.session_config(S1).write_text("cnmf: [unclosed\n", encoding="utf-8")
    assert tm.choice_source(p, S1, "NOF_H01") == ""  # cannot be read: nothing is claimed
