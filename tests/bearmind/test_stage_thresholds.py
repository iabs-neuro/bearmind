import pytest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_bm_qc import synthetic_movie  # noqa: E402

from bearmind import config  # noqa: E402
from bearmind.paths import SeriesPaths  # noqa: E402
from bearmind.registry import SessionRecord  # noqa: E402
from bearmind.stages import thresholds  # noqa: E402
from bearmind.video import VideoArray, write_ffv1  # noqa: E402

SID = "NOF_H01_1D"


def _no_videoarray_reads(monkeypatch):
    def boom(self, key):
        raise AssertionError("the stage must read the movie from its decoded uint8 memmap, not VideoArray")

    monkeypatch.setattr(VideoArray, "__getitem__", boom)


def test_thresholds_stage(tmp_path, monkeypatch):
    mov, _ = synthetic_movie(T=300)
    paths = SeriesPaths(tmp_path / "series")
    mc = paths.mc_video(SID)
    mc.parent.mkdir(parents=True)
    write_ffv1([np.clip(np.round(mov), 0, 255).astype(np.uint8)], mc, 20.0)
    rec = SessionRecord(sid=SID, mouse="NOF_H01", mode="bonsai", parts=[], n_frames=300, fps=20.0,
                        fps_info={}, anomalies=[], decisions=[], excluded=False, stages={})
    cfg = config.load_defaults()
    cfg["machine"] = {"n_cpus": 2, "temp_dir": str(tmp_path / "tmp"), "ram_gb": 16}
    cfg["thresholds"]["gsig_range"] = [3, 4]
    _no_videoarray_reads(monkeypatch)
    calls = []
    real = thresholds.bm_qc.compute_init_maps

    def spy(movie, **kw):
        calls.append((isinstance(movie, np.memmap), kw["max_mem_gb"]))
        return real(movie, **kw)

    monkeypatch.setattr(thresholds.bm_qc, "compute_init_maps", spy)

    res = thresholds.run(paths, rec, cfg)

    assert calls == [(True, 2), (True, 2)]  # decoded once; budget clip(0.1 * 16 GB, 2, 32)
    assert not paths.temp_dir(tmp_path / "tmp", SID).exists()

    out = paths.threshold_maps(SID)
    assert res.outputs == {"threshold_maps": paths.rel(out)}
    assert not list(out.parent.glob("*.partial*"))
    d = np.load(out)
    shape = mov.shape[1:]
    assert list(d["gsig"]) == [3, 4]
    for g in (3, 4):
        for k in ("cn", "pnr", "cn_mirror", "pnr_mirror"):
            a = d[f"{k}_{g}"]
            assert a.shape == shape and a.dtype == np.float32
    for k in ("mean", "max"):
        assert d[k].shape == shape and d[k].dtype == np.float32
    s = res.metrics["suggested"]
    assert s["gSig"] == 4
    assert 0 < s["min_corr"] <= 1
    assert s["min_pnr"] > 0
    assert s["method"] in ("mirror_fdr", "robust_bg")


def _setup(tmp_path):
    mov, _ = synthetic_movie(T=200)
    paths = SeriesPaths(tmp_path / "series")
    mc = paths.mc_video(SID)
    mc.parent.mkdir(parents=True)
    write_ffv1([mov], mc, 20.0)
    rec = SessionRecord(sid=SID, mouse="NOF_H01", mode="bonsai", parts=[], n_frames=200, fps=20.0,
                        fps_info={}, anomalies=[], decisions=[], excluded=False, stages={})
    cfg = config.load_defaults()
    cfg["machine"] = {"n_cpus": 2, "temp_dir": str(tmp_path / "tmp"), "ram_gb": 16}
    cfg["thresholds"]["gsig_range"] = [4]
    return paths, rec, cfg


@pytest.mark.parametrize("mode", ["empty", "nan", "raise"])
def test_no_suggestion_keeps_maps(tmp_path, monkeypatch, mode):
    paths, rec, cfg = _setup(tmp_path)
    none_r = {"min_corr": None, "min_pnr": None}
    nan_r = {"min_corr": float("nan"), "min_pnr": 3.0}

    def fake(maps, gSiz=None, **kw):
        if mode == "raise":
            raise RuntimeError("boom")
        r = none_r if mode == "empty" else nan_r
        return {"mirror_fdr": r, "robust_bg": r}, None

    monkeypatch.setattr(thresholds.bm_qc, "suggest_init_thresholds", fake)
    res = thresholds.run(paths, rec, cfg)
    assert paths.threshold_maps(SID).exists()
    assert res.metrics["suggested"] is None
    assert [a.code for a in res.anomalies] == ["no_threshold_suggestion"]
    assert res.anomalies[0].level == "warning"


def test_thresholds_failure_removes_temp(tmp_path, monkeypatch):
    paths, rec, cfg = _setup(tmp_path)

    def boom(movie, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(thresholds.bm_qc, "compute_init_maps", boom)
    with pytest.raises(RuntimeError, match="boom"):
        thresholds.run(paths, rec, cfg)
    assert not paths.temp_dir(tmp_path / "tmp", SID).exists()
    assert not paths.threshold_maps(SID).exists()


def test_thresholds_reports_temp_left_behind(tmp_path, monkeypatch):
    paths, rec, cfg = _setup(tmp_path)
    monkeypatch.setattr(thresholds.shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(thresholds.time, "sleep", lambda s: None)
    res = thresholds.run(paths, rec, cfg)
    assert "temp_not_removed" in [a.code for a in res.anomalies]
    assert not (paths.temp_dir(tmp_path / "tmp", SID) / "mc_u8.dat").exists()  # the decoded copy itself is deleted
