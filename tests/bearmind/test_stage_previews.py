import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_bm_qc import synthetic_movie  # noqa: E402

from bearmind import config  # noqa: E402
from bearmind.paths import SeriesPaths  # noqa: E402
from bearmind.registry import RawPart, SessionRecord, StageRecord, State  # noqa: E402
from bearmind.stages import previews  # noqa: E402
from bearmind.video import write_ffv1  # noqa: E402

from tests.bearmind.test_stage_mc import _raw_frames  # noqa: E402

SID = "NOF_H01_1D"
T, FULL = 300, 72
CROP = [4, 4, 64, 64]


def frame_shape(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return s.codec_context.height, s.codec_context.width


def _cfg(tmp_path, **pv):
    cfg = config.load_defaults()
    cfg["machine"] = dict(cfg.get("machine") or {}, temp_dir=str(tmp_path / "tmp"))
    cfg["preview"].update(enabled=True, **pv)
    return cfg


def _mc_setup(tmp_path, make_avi, **pv):
    root = tmp_path / "series"
    folder = root / "1_Raw" / "HM_NOF_1D" / SID
    folder.mkdir(parents=True)
    raw = _raw_frames()
    make_avi(folder / f"{SID}_0.avi", raw)
    part = RawPart(folder=f"1_Raw/HM_NOF_1D/{SID}", chunks=[f"{SID}_0.avi"], n_frames=T,
                   shape=(FULL, FULL), timestamps_file=None, ts_format=None, start_ms=None,
                   metadata_file=None)
    rec = SessionRecord(sid=SID, mouse="NOF_H01", mode="bonsai", parts=[part], n_frames=T, fps=20.0,
                        fps_info={}, anomalies=[], decisions=[], excluded=False,
                        stages={"mc": StageRecord(state=State.DONE, metrics={"crop": CROP, "n_frames": T})})
    paths = SeriesPaths(root)
    paths.mc_video(SID).parent.mkdir(parents=True)
    write_ffv1([raw[:, 4:68, 4:68]], paths.mc_video(SID), 20.0)
    return paths, rec, _cfg(tmp_path, **pv)


def _temp_gone(paths, tmp_path):
    t = paths.temp_dir(tmp_path / "tmp", SID)
    return not t.exists() or not any(t.iterdir())


def test_preview_params_off_and_on():
    cfg = config.load_defaults()
    assert previews.preview_params(cfg) == {"enabled": False}
    cfg["preview"]["tbin"] = 4
    assert previews.preview_params(cfg) == {"enabled": False}  # settings of a disabled preview do not matter
    cfg["preview"]["enabled"] = True
    assert previews.preview_params(cfg) == {"enabled": True, "tbin": 4, "ssub": 1, "crf": 23, "speed": 4}


@pytest.mark.parametrize("key,bad", [("tbin", 0), ("tbin", True), ("ssub", 1.5), ("ssub", None),
                                     ("ssub", "2"), ("crf", False), ("speed", None)])
def test_preview_params_rejects_bad_values(key, bad):
    cfg = config.load_defaults()
    cfg["preview"].update(enabled=True, **{key: bad})
    with pytest.raises(ValueError, match=rf"preview\.{key}.*series\.yaml"):
        previews.preview_params(cfg)


def test_mc_previews_writes_four_videos(tmp_path, make_avi):
    paths, rec, cfg = _mc_setup(tmp_path, make_avi)
    res = previews.mc_previews(paths, rec, cfg)
    assert set(res.outputs) == {"preview_raw", "preview_raw_dff", "preview_mc", "preview_mc_dff"}
    got = {k: paths.root / v for k, v in res.outputs.items()}
    assert frame_shape(got["preview_raw"]) == (72, 72) and frame_shape(got["preview_mc"]) == (64, 64)
    # both dF/F videos are of the 64 x 64 crop
    assert frame_shape(got["preview_raw_dff"]) == frame_shape(got["preview_mc_dff"])
    assert _temp_gone(paths, tmp_path)
    assert res.metrics["tbin"] == 2 and res.metrics["ssub"] == 1
    assert res.metrics["bytes"] == sum(p.stat().st_size for p in got.values()) > 0


def test_mc_previews_ssub_shrinks_frames(tmp_path, make_avi):
    paths, rec, cfg = _mc_setup(tmp_path, make_avi, ssub=2)
    res = previews.mc_previews(paths, rec, cfg)
    assert frame_shape(paths.root / res.outputs["preview_raw"]) == (36, 36)
    assert frame_shape(paths.root / res.outputs["preview_mc"]) == (32, 32)
    assert res.metrics["ssub"] == 2


def test_mc_previews_without_crop_in_record(tmp_path, make_avi):
    paths, rec, cfg = _mc_setup(tmp_path, make_avi)
    rec.stages["mc"].metrics = {}
    with pytest.raises(ValueError, match="crop"):
        previews.mc_previews(paths, rec, cfg)


def test_mc_previews_failure_cleans_temp(tmp_path, make_avi, monkeypatch):
    import bm_qc
    paths, rec, cfg = _mc_setup(tmp_path, make_avi)

    def bad_dff(movie, out_path=None, **k):
        held = movie[0:1]  # a frame of the traceback keeps the memmap, as in a real writer failure  # noqa: F841
        raise RuntimeError("dff failed")

    monkeypatch.setattr(bm_qc, "make_dff_video", bad_dff)
    with pytest.raises(RuntimeError, match="dff failed"):
        previews.mc_previews(paths, rec, cfg)
    assert not paths.temp_dir(tmp_path / "tmp", SID).exists()


def test_cnmf_preview_writes_qc_video(tmp_path):
    mov, centers, fps, traces, bg = synthetic_movie(T=200, n_cells=5, amp=60., seed=3, return_model=True)
    paths = SeriesPaths(tmp_path / "series")
    paths.mc_video(SID).parent.mkdir(parents=True)
    write_ffv1([mov], paths.mc_video(SID), 20.0)
    A = sp.csc_matrix(np.stack([f.ravel(order="F") for f in fps], 1))
    est = SimpleNamespace(A=A, C=traces[:, :200].astype(np.float32), W=None, b0=None,
                          b=bg.ravel(order="F")[:, None], f=np.ones((1, 200)), idx_components=np.arange(4),
                          cnmf_dict={"gSig": [3, 3], "gSiz": [13, 13], "min_corr": 0.8, "min_pnr": 6})
    paths.estimates(SID).parent.mkdir(parents=True)
    with open(paths.estimates(SID), "wb") as f:
        pickle.dump(est, f)
    rec = SessionRecord(sid=SID, mouse="NOF_H01", mode="bonsai", parts=[], n_frames=200, fps=20.0,
                        fps_info={}, anomalies=[], decisions=[], excluded=False, stages={})
    res = previews.cnmf_preview(paths, rec, _cfg(tmp_path))
    assert res.outputs == {"preview_cnmf_qc": paths.rel(paths.preview(SID, "cnmf_qc"))}
    assert paths.preview(SID, "cnmf_qc").stat().st_size > 0
    assert not paths.residual_png(SID).exists() and _temp_gone(paths, tmp_path)
    assert res.metrics["bytes"] == paths.preview(SID, "cnmf_qc").stat().st_size
