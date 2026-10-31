from types import SimpleNamespace

import numpy as np

from bearmind import config
from bearmind.crop import detect_fov, mouse_box_size, place_box
from bearmind.paths import SeriesPaths
from bearmind.stages.crop import finalize_mouse


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
    assert cfg["crop"]["box"] == [320, 330]
    assert (out["M1_d1"].height, out["M1_d1"].width) == (320, 330)
    r2 = _rec("M1_d2", [10, 10, 500, 500])
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
