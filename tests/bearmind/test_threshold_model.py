"""Threshold model without Qt: passing mask, blob counters, maps file, session states, start values."""
import re
import statistics
import time

import numpy as np

from bearmind import config, threshold_model as tm
from bearmind.registry import State, update_record
from tests.bearmind.test_cli import S1, S2, S3, _set, series  # noqa: F401  (the series fixture)

CYR = re.compile("[Ѐ-ӿ]")


def _blob(shape, cy, cx, s=2.0):
    y, x = np.mgrid[:shape[0], :shape[1]]
    return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * s ** 2))


def _maps(mirror=False, single=True, g=4):
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
    return tm.ThresholdMaps([g], cn, pnr, {g: cn}, {g: pnr}, cn_m, pnr_m)


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
    _set(p, S2, "thresholds", State.DONE, suggested=None)
    _set(p, S2, "cnmf", State.WAITING, error="waiting for thresholds")
    it = _items(series)[S2]
    assert it.state == "waiting" and it.chosen is None
    # a suggestion and no thresholds: CNMF-E is PENDING, not WAITING
    _set(p, S2, "cnmf", State.PENDING, error=None)
    _set(p, S2, "thresholds", State.DONE,
         suggested={"gSig": 4, "min_corr": 0.85, "min_pnr": 8.0, "method": "robust_bg"})
    it = _items(series)[S2]
    assert it.state == "waiting" and it.suggested["gSig"] == 4
    config.update_yaml(p.session_config(S2), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    it = _items(series)[S2]
    assert it.state == "chosen" and it.chosen == {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}
    _set(p, S2, "cnmf", State.DONE)
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
    m = _maps()
    sug = {"gSig": 4, "min_corr": 0.85, "min_pnr": 8.0, "method": "robust_bg"}
    ch = {"gSig": 5, "min_corr": 0.7, "min_pnr": 6.0}
    assert tm.initial_values(_item(sug, ch), m, CFG) == (5, 0.7, 6.0, "chosen")
    assert tm.initial_values(_item(sug), m, CFG) == (4, 0.85, 8.0, "suggested")
    assert tm.initial_values(_item(), m, CFG)[3] == "fallback"


def test_initial_values_without_suggestion():
    m = _maps()
    m.gsigs = [3, 4, 5, 6, 7]
    assert tm.initial_values(_item(), m, CFG) == (4, 0.85, 8.0, "fallback")
    m.gsigs = [5, 6, 7]
    assert tm.initial_values(_item(), m, CFG)[0] == 6


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
    assert statistics.median(times) < 0.05
