import builtins
from types import SimpleNamespace

import cv2
import numpy as np

from bearmind.crop_overview import UNREADABLE, _projection, write_overview
from bearmind.paths import SeriesPaths

SHAPE = (60, 80)
BOX = (8, 10, 34, 44)


def _rec(sid, method="activity", box=BOX):
    m = {"fov": [10, 12, 30, 40], "fov_lit": [5, 5, 50, 70], "fov_activity": [10, 12, 30, 40] if method == "activity"
         else None, "fov_method": method, "activity_pos_neg": 42.0 if method == "activity" else None,
         "frame_shape": list(SHAPE), "box": None if box is None else list(box)}
    return SimpleNamespace(sid=sid, stages={"crop": SimpleNamespace(metrics=m)})


def _projection_file(paths, sid, activity=True, seeds=((20, 30),)):
    rng = np.random.default_rng(0)
    f = paths.crop_projection(sid)
    f.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"mean": rng.uniform(20, 120, SHAPE).astype(np.float16)}
    if activity:
        arrays["activity"] = rng.normal(0, 1, SHAPE).astype(np.float16)
        arrays["seeds"] = np.array(seeds, np.int32).reshape(-1, 2)
    np.savez_compressed(f, **arrays)


def _pixels(img, bgr_test) -> int:
    b, g, r = (img[..., k].astype(int) for k in range(3))
    return int(np.sum(bgr_test(b, g, r)))


def _cyan(b, g, r):
    return (b > 200) & (g > 170) & (r < 60)


def _red(b, g, r):  # pure red: the magma map never gets there (its reds carry green and blue)
    return (r > 230) & (g < 40) & (b < 40)


def test_overview_draws_every_session(tmp_path):
    paths = SeriesPaths(tmp_path)
    recs = [_rec("M1_d1"), _rec("M1_d2", method="brightness"), _rec("M2_d1", box=None)]
    _projection_file(paths, "M1_d1")
    _projection_file(paths, "M1_d2", activity=False)  # brightness: no activity map was computed
    out = write_overview(paths, recs)  # M2_d1 has no projection: drawn with its boxes on an empty panel
    assert out == paths.crop_overview
    one_row = cv2.imread(str(out))
    assert one_row is not None and _pixels(one_row, _cyan) > 0  # the mouse boxes are drawn
    recs += [_rec("M2_d2"), SimpleNamespace(sid="M3_d1", stages={})]  # no crop metrics: not drawn
    two_rows = cv2.imread(str(write_overview(paths, recs)))
    assert two_rows.shape[0] > one_row.shape[0] and two_rows.shape[1] == one_row.shape[1]
    assert not list(out.parent.glob("*.partial*"))


def test_overview_marks_strong_seeds_outside_the_box(tmp_path):
    paths = SeriesPaths(tmp_path)
    _projection_file(paths, "M1_d1", seeds=[(20, 30)])
    inside = _pixels(cv2.imread(str(write_overview(paths, [_rec("M1_d1")]))), _red)  # the legend's red dot
    _projection_file(paths, "M1_d1", seeds=[(20, 30), (50, 70)])  # (50, 70) is right of and below the box
    assert _pixels(cv2.imread(str(write_overview(paths, [_rec("M1_d1")]))), _red) > inside


def test_overview_survives_a_damaged_projection(tmp_path):
    paths = SeriesPaths(tmp_path)
    _projection_file(paths, "M1_d1")
    bad = paths.crop_projection("M1_d2")
    bad.write_bytes(b"PK\x03\x04 not a zip archive")
    assert _projection(paths, "M1_d2", SHAPE) is UNREADABLE
    out = write_overview(paths, [_rec("M1_d1"), _rec("M1_d2")])
    assert out == paths.crop_overview and cv2.imread(str(out)) is not None
    # only damaged projections: the placeholders are still drawn
    paths.crop_projection("M1_d1").unlink()
    assert write_overview(paths, [_rec("M1_d1"), _rec("M1_d2")]) == paths.crop_overview


def test_overview_none_without_projections(tmp_path, monkeypatch):
    paths = SeriesPaths(tmp_path)
    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name.split(".")[0] == "matplotlib":
            raise AssertionError("matplotlib imported with nothing to draw")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    recs = [_rec("M1_d1"), SimpleNamespace(sid="M1_d2", stages={})]
    assert write_overview(paths, recs) is None
    assert write_overview(paths, []) is None
    assert not paths.crop_overview.exists()
