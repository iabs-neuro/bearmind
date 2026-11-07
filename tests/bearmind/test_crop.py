from types import SimpleNamespace

import numpy as np
import pytest

from bearmind import config
from bearmind.crop import (ACTIVITY_SIGMA_PX, MIN_POS_NEG, activity_field, activity_fov, detect_fov, lit_fov,
                           mouse_box_size, place_box)
from bearmind.paths import SeriesPaths
from bearmind.registry import RawPart, SessionRecord
from bearmind.stages import crop as crop_stage
from bearmind.stages.crop import finalize_mouse

# Synthetic session: a 200 x 240 frame, a lit disk of radius 90 centred at (100, 120), noise sigma 4; cells are
# gaussians (sigma 3) with rare transients, as tests/test_bm_qc.synthetic_movie.
H, W, CY, CX, R = 200, 240, 100, 120, 90
T = 600
KERNEL = np.exp(-np.arange(60) / 20.)


def _disk_background(inside=100.0, outside=10.0):
    yy, xx = np.mgrid[:H, :W]
    return np.where((yy - CY) ** 2 + (xx - CX) ** 2 < R ** 2, inside, outside).astype(np.float32)


def _movie(rng, background, cells, spot=None):
    """uint8 (T, H, W): background + cells (y, x, amplitude) + an optional 3 x 3 spot blinking in 1 % of frames."""
    yy, xx = np.mgrid[:H, :W]
    mov = np.repeat(background[None], T, 0)
    for cy, cx, amp in cells:
        trace = amp * np.convolve((rng.random(T) < 0.01).astype(float), KERNEL)[:T]
        fp = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3. ** 2))
        mov += trace[:, None, None].astype(np.float32) * fp[None].astype(np.float32)
    if spot is not None:
        on = rng.random(T) < 0.01
        mov[on, spot[0] - 1:spot[0] + 2, spot[1] - 1:spot[1] + 2] += 80
    mov += rng.normal(0, 4, mov.shape).astype(np.float32)
    return np.clip(mov, 0, 255).astype(np.uint8)


def _left_cells(rng, n=100, x_max=110):
    """n cell centres in the disk (15 px from its edge), the whole cell (3 sigma) at x < x_max."""
    out = []
    while len(out) < n:
        cy, cx = rng.uniform(CY - R, CY + R), rng.uniform(CX - R, x_max - 9)
        if (cy - CY) ** 2 + (cx - CX) ** 2 < (R - 15) ** 2:
            out.append((cy, cx, 40.))
    return out


@pytest.fixture(scope="module")
def left_scene():
    rng = np.random.default_rng(0)
    cells = _left_cells(rng)
    return _movie(rng, _disk_background(), cells), cells


def _field(mov):
    """(lit box, ActivityField) as the crop stage computes them (gSig 3 for these cells)."""
    import bm_qc  # CaImAn: only the tests that compute seed maps pay for it

    lit_box, lit_mask = lit_fov(mov.mean(axis=0, dtype=np.float64))
    maps = bm_qc.compute_init_maps(mov, gSig=3, tsub=1, mirror=True, verbose=False)
    return lit_box, activity_field(maps, lit_box, lit_mask, 3, 0.005, 20)


def _inside(box, outer):
    return (box[0] >= outer[0] and box[1] >= outer[1] and box[0] + box[2] <= outer[0] + outer[2]
            and box[1] + box[3] <= outer[1] + outer[3])


def _covers(box, y, x, spare):
    return box[0] + spare <= y <= box[0] + box[2] - 1 - spare and box[1] + spare <= x <= box[1] + box[3] - 1 - spare


def test_activity_fov_covers_the_active_part(left_scene):
    mov, cells = left_scene
    lit_box, found = _field(mov)
    box = found.field
    assert found.reason == "activity" and box is not None and _inside(box, lit_box)
    assert all(_covers(box, cy, cx, 5) for cy, cx, _ in cells)
    # Right of the cells only noise: smoothed noise carries 0.3-0.7 % of the positive mass there, at the 0.5 %
    # cut, so that edge lands up to 12 px beyond the cells' tails (3 sigma) plus the margin (8 seeds measured).
    assert box[1] + box[3] <= 110 + 20 + 3 * ACTIVITY_SIGMA_PX + 15
    assert box[3] <= 0.8 * lit_box[3]


def test_strong_cell_beyond_the_active_area_is_kept():
    """A very active cell 40 px right of the active area: its excess and the ring of its sign-inverted movie
    cancel after smoothing (E ~ 0 at its centre), so the activity map alone cuts it; the strong-seed guard
    keeps it."""
    rng = np.random.default_rng(0)
    cells = _left_cells(rng)
    lit_box, found = _field(_movie(rng, _disk_background(), cells + [(100., 150., 120.)]))
    assert found.reason == "activity"
    assert _covers(found.field, 100, 150, 5) and _inside(found.field, lit_box)
    assert all(_covers(found.field, cy, cx, 5) for cy, cx, _ in cells)
    plain = activity_fov(found.activity, lit_box, 0.005, 20)  # the same field without the guard
    assert not _covers(plain, 100, 150, 0)
    assert any(abs(y - 100) <= 3 and abs(x - 150) <= 3 for y, x in found.seeds)


def test_activity_field_reasons():
    """Hand-made seed maps: why the field is what it is."""
    cn, pnr = np.full((H, W), 0.5, np.float32), np.full((H, W), 2.0, np.float32)
    maps = {"cn": cn.copy(), "pnr": pnr.copy(), "cn_neg": cn.copy(), "pnr_neg": pnr.copy()}
    lit_mask = np.zeros((H, W), bool)
    lit_mask[10:190, 30:210] = True
    lit_box = (10, 30, 180, 180)
    maps["cn"][40:160, 40:120], maps["pnr"][40:160, 40:120] = 1.0, 20.0  # strong cells all over a block
    found = activity_field(maps, lit_box, lit_mask, 4, 0.005, 20)
    assert found.reason == "activity" and found.field is not None and found.pos_neg >= MIN_POS_NEG
    assert len(found.seeds) > 0 and all(_covers(found.field, y, x, 0) for y, x in found.seeds)
    small = {k: v.copy() for k, v in maps.items()}
    small["cn"][:], small["pnr"][:] = 0.5, 2.0
    small["cn"][95:105, 115:125], small["pnr"][95:105, 115:125] = 1.0, 20.0
    found = activity_field(small, lit_box, lit_mask, 4, 0.005, 0)  # no margin: the smoothed spot stays small
    assert found.reason == "fallback_small" and found.field is None
    noise = {k: v.copy() for k, v in maps.items()}
    noise["cn_neg"][40:160, 40:120], noise["pnr_neg"][40:160, 40:120] = 1.0, 20.0  # the mirror is as strong
    found = activity_field(noise, lit_box, lit_mask, 4, 0.005, 20)
    assert found.reason == "fallback_pos_neg" and found.field is None and found.pos_neg < MIN_POS_NEG


def test_activity_fov_arithmetic():
    """Noise-free map: the cut, the margin and the clip to the lit box exactly."""
    a = np.zeros((H, W))
    a[40:160, 40:120] = 1.0  # uniform activity: 0.5 % of 80 columns is 0.4 of a column
    assert activity_fov(a, (10, 30, 180, 180), 0.005, 20) == (20, 30, 160, 110)  # left edge clipped at the lit box
    assert activity_fov(a, (10, 30, 180, 180), 0.005, 0) == (40, 40, 120, 80)
    # protected points widen the field by the margin around them, clipped to the lit box; never narrow it
    assert activity_fov(a, (10, 30, 180, 180), 0.005, 20, protect=[(100, 200)]) == (20, 30, 160, 180)
    assert activity_fov(a, (10, 30, 180, 180), 0.005, 20, protect=[(15, 80)]) == (10, 30, 170, 110)
    assert activity_fov(a, (10, 30, 180, 180), 0.005, 20, protect=[(100, 80)]) == (20, 30, 160, 110)
    total = a.sum()
    a[60, 200] = -total / (MIN_POS_NEG + 1)  # positive mass MIN_POS_NEG + 1 times the negative: kept
    assert activity_fov(a, (10, 30, 180, 180), 0.005, 20) == (20, 30, 160, 110)
    a[60, 200] = -total / (MIN_POS_NEG - 1)  # under MIN_POS_NEG times: noise, whatever is protected
    assert activity_fov(a, (10, 30, 180, 180), 0.005, 20, protect=[(100, 200)]) is None
    small = np.zeros((H, W))
    small[95:105, 115:125] = 1.0
    assert activity_fov(small, (10, 30, 180, 180), 0.005, 20) is None  # 50 x 50 < 25 % of the lit box
    # a protected point does not rescue a small field: the lit area is used, which covers it anyway
    assert activity_fov(small, (10, 30, 180, 180), 0.005, 20, protect=[(20, 40), (180, 200)]) is None
    assert activity_fov(np.zeros((H, W)), (10, 30, 180, 180), 0.005, 20) is None


def test_activity_fov_keeps_dim_active_tissue():
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:H, :W]
    r2 = (yy - CY) ** 2 + (xx - CX) ** 2
    bg = np.zeros((H, W), np.float32)
    bg[r2 < R ** 2] = 40.0  # dim tissue, 25 % of the bright core
    bg[r2 < 35 ** 2] = 160.0
    core = []
    while len(core) < 40:
        a, rr = rng.uniform(0, 2 * np.pi), 30 * np.sqrt(rng.uniform())
        core.append((CY + rr * np.sin(a), CX + rr * np.cos(a), 40.))
    # the lit edge sits ~5 px inside the disk (the mean is smoothed before the threshold): 20 px inside it
    dim = [(CY + 65 * np.sin(a), CX + 65 * np.cos(a), 10.) for a in np.deg2rad([150, 165, 180, 195, 210])]
    lit_box, found = _field(_movie(rng, bg, core + dim))
    box = found.field
    assert found.reason == "activity" and box != lit_box  # activity placed it
    assert all(_covers(box, cy, cx, 3) for cy, cx, _ in dim)


@pytest.mark.parametrize("scene", ["noise", "hot_spot"])
def test_activity_fov_does_not_collapse_without_cells(scene):
    rng = np.random.default_rng(2)
    mov = _movie(rng, _disk_background(), [], spot=(100, 150) if scene == "hot_spot" else None)
    lit_box, found = _field(mov)
    box = found.field
    assert box is None or box[2] * box[3] >= 0.9 * lit_box[2] * lit_box[3]


def _session(tmp_path, make_avi, mov):
    (tmp_path / "1_Raw" / "s").mkdir(parents=True)
    make_avi(tmp_path / "1_Raw" / "s" / "0.avi", mov)
    part = RawPart(folder="1_Raw/s", chunks=["0.avi"], n_frames=len(mov), shape=mov.shape[1:],
                   timestamps_file=None, ts_format=None, start_ms=None, metadata_file=None,
                   chunk_frames=[len(mov)])
    rec = SessionRecord(sid="M1_d1", mouse="M1", mode="bonsai", parts=[part], n_frames=len(mov), fps=20.0,
                        fps_info={}, anomalies=[], decisions=[], excluded=False, stages={})
    cfg = config.load_defaults()
    cfg["machine"] = {"ram_gb": 16, "n_cpus": 2}
    return SeriesPaths(tmp_path), rec, cfg


def test_detect_metrics_and_projection(tmp_path, make_avi, left_scene):
    paths, rec, cfg = _session(tmp_path, make_avi, left_scene[0])
    res = crop_stage.detect(paths, rec, cfg)
    m = res.metrics
    assert set(m) == {"fov", "fov_lit", "fov_activity", "fov_method", "activity_pos_neg", "frame_shape"}
    assert m["frame_shape"] == [H, W] and m["fov_method"] == "activity"
    assert type(m["activity_pos_neg"]) is float and m["activity_pos_neg"] >= MIN_POS_NEG
    assert m["fov"] == m["fov_activity"] and _inside(m["fov"], m["fov_lit"]) and m["fov"] != m["fov_lit"]
    assert all(type(v) is int for v in m["fov"] + m["fov_lit"])
    assert res.outputs == {"projection": paths.rel(paths.crop_projection("M1_d1"))}
    with np.load(paths.crop_projection("M1_d1")) as z:
        assert set(z.files) == {"mean", "activity", "seeds"}
        assert z["mean"].shape == z["activity"].shape == (H, W) and z["mean"].dtype == np.float16
        assert z["seeds"].ndim == 2 and z["seeds"].shape[1] == 2 and len(z["seeds"]) > 0
        assert abs(float(z["mean"][CY, CX]) - 100) < 3
    assert not list(paths.crop_projection("M1_d1").parent.glob("*.partial*"))


def test_detect_brightness_method(tmp_path, make_avi, monkeypatch, left_scene):
    import bm_qc

    def boom(*a, **k):
        raise AssertionError("seed maps computed with fov_method brightness")

    monkeypatch.setattr(bm_qc, "compute_init_maps", boom)
    paths, rec, cfg = _session(tmp_path, make_avi, left_scene[0])
    cfg["crop"]["fov_method"] = "brightness"
    m = crop_stage.detect(paths, rec, cfg).metrics
    assert m["fov"] == m["fov_lit"] == list(detect_fov(left_scene[0]))
    assert m["fov_activity"] is None and m["fov_method"] == "brightness" and m["activity_pos_neg"] is None
    with np.load(paths.crop_projection("M1_d1")) as z:
        assert set(z.files) == {"mean"}
    cfg["crop"]["fov_method"] = "brightest"
    with pytest.raises(ValueError, match="fov_method"):
        crop_stage.detect(paths, rec, cfg)


def test_detect_fov_bright_disk():
    yy, xx = np.mgrid[:200, :240]
    img = np.full((200, 240), 10.0)
    img[(yy - 90) ** 2 + (xx - 120) ** 2 < 50**2] = 200.0
    rng = np.random.default_rng(0)
    frames = np.clip(img + rng.normal(0, 5, (20, 200, 240)), 0, 255).astype(np.uint8)
    top, left, h, w = detect_fov(frames)
    assert abs(top - 40) <= 6 and abs(left - 70) <= 6 and abs(h - 101) <= 8 and abs(w - 101) <= 8  # low threshold grows the box a little: cropping too little is safe
    assert all(type(v) is int for v in (top, left, h, w))


def test_no_fov_on_uniform_noise():
    rng = np.random.default_rng(1)
    frames = rng.integers(100, 110, (20, 100, 100)).astype(np.uint8)
    assert detect_fov(frames) is None
    assert detect_fov(np.full((5, 50, 50), 7, np.uint8)) is None


def test_box_same_size_moves_with_fov():
    s = mouse_box_size([(100, 120, 300, 310), (110, 90, 290, 300)], (608, 608), margin=10)
    a = place_box((100, 120, 300, 310), s, (608, 608))
    b = place_box((110, 90, 290, 300), s, (608, 608))
    assert (a.height, a.width) == (b.height, b.width) == s and (a.top, a.left) != (b.top, b.left)


def test_box_clipped_inside_frame():
    s = mouse_box_size([(0, 0, 300, 300)], (400, 400), margin=10)
    box = place_box((0, 0, 300, 300), s, (400, 400))
    assert (box.height, box.width) == s == (320, 320)
    assert box.top >= 0 and box.left >= 0
    assert box.top + box.height <= 400 and box.left + box.width <= 400
    big = mouse_box_size([(0, 0, 390, 390)], (400, 400), margin=10)
    assert big == (400, 400)
    assert place_box((0, 0, 390, 390), big, (400, 400)) == place_box((0, 0, 390, 390), (400, 400), (400, 400))


def _rec(sid, fov, shape=(608, 608)):
    return SimpleNamespace(
        sid=sid, anomalies=[],
        stages={"crop": SimpleNamespace(metrics={"fov": fov, "frame_shape": list(shape)})},
    )


def test_finalize_keeps_existing_mouse_size(tmp_path):
    paths = SeriesPaths(tmp_path)
    m = "M1"
    r1 = _rec("M1_d1", [100, 120, 300, 310])
    out = finalize_mouse(paths, m, [r1], config.resolve(paths, mouse=m))
    cfg = config.resolve(paths, mouse=m)
    assert cfg["crop"]["box"] == [320, 330] and cfg["crop"]["box_source"] == "auto"
    assert (out["M1_d1"].height, out["M1_d1"].width) == (320, 330)
    b = out["M1_d1"]
    r1.stages["crop"].metrics["box"] = [b.top, b.left, b.height, b.width]  # what the runner records
    r2 = _rec("M1_d2", [10, 10, 500, 500])  # a session added later gets the same box size
    r3 = _rec("M1_d3", None)
    out = finalize_mouse(paths, m, [r1, r2, r3], config.resolve(paths, mouse=m))
    assert config.resolve(paths, mouse=m)["crop"]["box"] == [320, 330]
    assert (out["M1_d2"].height, out["M1_d2"].width) == (320, 330)
    assert out["M1_d3"] is None
    assert config.resolve(paths, sid="M1_d3")["crop"]["box_pos"] is None
    assert config.resolve(paths, sid="M1_d1")["crop"]["box_pos"] == [out["M1_d1"].top, out["M1_d1"].left]
    assert [a.code for a in r3.anomalies] == ["fov_not_found"] and r3.anomalies[0].level == "warning"


def test_finalize_no_fov_anywhere(tmp_path):
    paths = SeriesPaths(tmp_path)
    r = _rec("M2_d1", None)
    out = finalize_mouse(paths, "M2", [r], config.resolve(paths, mouse="M2"))
    assert out == {"M2_d1": None}
    assert "box" not in (config.resolve(paths, mouse="M2")["crop"] or {})
    # finalized as uncropped (explicit null), so MC does not wait for crop_finalize
    crop = config.resolve(paths, sid="M2_d1")["crop"]
    assert "box_pos" in crop and crop["box_pos"] is None
    assert [a.code for a in r.anomalies] == ["fov_not_found"]


def test_finalize_skips_records_without_crop_metrics(tmp_path):
    paths = SeriesPaths(tmp_path)
    ok = _rec("M3_d1", [100, 120, 300, 310])
    no_shape = SimpleNamespace(sid="M3_d2", anomalies=[], stages={  # big FOV, must not count for the size
        "crop": SimpleNamespace(metrics={"fov": [10, 10, 500, 500]})})
    failed = SimpleNamespace(sid="M3_d3", anomalies=[], stages={"crop": SimpleNamespace(metrics={})})
    no_stage = SimpleNamespace(sid="M3_d4", anomalies=[], stages={})
    recs = [ok, no_shape, failed, no_stage]
    out = finalize_mouse(paths, "M3", recs, config.resolve(paths, mouse="M3"))
    assert set(out) == {"M3_d1"} and (out["M3_d1"].height, out["M3_d1"].width) == (320, 330)
    assert config.resolve(paths, mouse="M3")["crop"]["box"] == [320, 330]
    for sid in ("M3_d2", "M3_d3", "M3_d4"):
        assert "box_pos" not in config.resolve(paths, sid=sid)["crop"]
    assert all(r.anomalies == [] for r in recs)


def test_finalize_twice_adds_fov_not_found_once(tmp_path):
    paths = SeriesPaths(tmp_path)
    recs = [_rec("M4_d1", [100, 120, 300, 310]), _rec("M4_d2", None)]
    for _ in range(2):
        finalize_mouse(paths, "M4", recs, config.resolve(paths, mouse="M4"))
    assert [a.code for a in recs[1].anomalies] == ["fov_not_found"]


def test_detect_fov_covers_dim_tissue():
    yy, xx = np.mgrid[:300, :300]
    r2 = (yy - 150) ** 2 + (xx - 150) ** 2
    img = np.zeros((300, 300))
    img[r2 < 120**2] = 50.0   # dim tissue, ~25% of the bright level
    img[r2 < 40**2] = 200.0   # bright core
    rng = np.random.default_rng(2)
    frames = np.clip(img + rng.normal(0, 3, (10, 300, 300)), 0, 255).astype(np.uint8)
    top, left, h, w = detect_fov(frames)
    assert top <= 40 and left <= 40 and h >= 220 and w >= 220  # bright core alone would give ~80
