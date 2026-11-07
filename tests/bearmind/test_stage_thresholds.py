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


def test_thresholds_fail_at_once_without_temp_space(tmp_path, monkeypatch):
    """10d: the decoded uint8 copy of the MC movie, 1 byte per pixel-frame."""
    from tests.bearmind.test_stage_mc import _check_space_error, _little_free_space

    from bearmind.stages import NotEnoughTempSpace

    paths, rec, cfg = _setup(tmp_path)
    _little_free_space(monkeypatch)
    monkeypatch.setattr(thresholds, "decode_to_memmap", None)  # any decoding would fail differently
    with pytest.raises(NotEnoughTempSpace) as e:
        thresholds.run(paths, rec, cfg)
    T, h, w = 200, *VideoArray(paths.mc_video(SID)).shape[1:]
    _check_space_error(e.value, paths.temp_dir(tmp_path / "tmp", SID), round(T * h * w / 1e9, 1))
    assert not (tmp_path / "tmp").exists()


def test_map_peak_follows_compute_init_maps_strips():
    assert thresholds.map_peak_gb(17861, 461, 608, 7, 1, 19.12) == pytest.approx(19.12, rel=0.02)
    assert thresholds.map_peak_gb(107107, 567, 520, 7, 1, 19.12) == pytest.approx(50.8, abs=0.2)
    assert thresholds.map_peak_gb(100, 20, 30, 7, 1, 19.12) == pytest.approx(24 * 100 * 30 * 20 / 1e9)


def test_map_workers():
    cfg = {"machine": {"n_cpus": 32, "ram_gb": 191.2, "cnmf_ram_budget_gb": 152.96}}
    g = [3, 4, 5, 6, 7]
    assert thresholds.map_workers(17861, 461, 608, g, 1, cfg) == 5
    assert thresholds.map_workers(107107, 567, 520, g, 1, cfg) == 3
    cfg["machine"]["n_cpus"] = 2
    assert thresholds.map_workers(17861, 461, 608, g, 1, cfg) == 1
    cfg["machine"].update(n_cpus=32, cnmf_ram_budget_gb=30)
    assert thresholds.map_workers(107107, 567, 520, g, 1, cfg) == 1


@pytest.mark.slow
def test_parallel_maps_equal_serial(tmp_path):
    mov, _ = synthetic_movie(T=300)
    frames = np.clip(np.round(mov), 0, 255).astype(np.uint8)
    rec = SessionRecord(sid=SID, mouse="NOF_H01", mode="bonsai", parts=[], n_frames=300, fps=20.0,
                        fps_info={}, anomalies=[], decisions=[], excluded=False, stages={})
    res, outs = [], []
    for name, machine in (("serial", {"n_cpus": 2}), ("parallel", {"n_cpus": 8, "cnmf_ram_budget_gb": 64})):
        paths = SeriesPaths(tmp_path / name)
        mc = paths.mc_video(SID)
        mc.parent.mkdir(parents=True)
        write_ffv1([frames], mc, 20.0)
        cfg = config.load_defaults()
        cfg["machine"] = {"temp_dir": str(tmp_path / f"tmp_{name}"), "ram_gb": 16, **machine}
        cfg["thresholds"]["gsig_range"] = [3, 4, 5]
        res.append(thresholds.run(paths, rec, cfg))
        outs.append(paths.threshold_maps(SID))
    res1, res2 = res
    assert res2.metrics["map_workers"] == 3 and res1.metrics["map_workers"] == 1
    d1, d2 = np.load(outs[0]), np.load(outs[1])
    assert sorted(d1.files) == sorted(d2.files)
    for k in d1.files:
        assert np.array_equal(d1[k], d2[k]), k
    assert res1.metrics["suggested"] == res2.metrics["suggested"]
