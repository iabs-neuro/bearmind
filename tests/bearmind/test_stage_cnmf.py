import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_bm_qc import synthetic_movie  # noqa: E402

import bm_qc  # noqa: E402  (the stage imports it inside its functions)

from bearmind import config  # noqa: E402
from bearmind.config import load_defaults  # noqa: E402
from bearmind.paths import SeriesPaths  # noqa: E402
from bearmind.registry import RawPart, SessionRecord, StageRecord, State  # noqa: E402
from bearmind.stages import WaitingForInput  # noqa: E402
from bearmind.stages import cnmf  # noqa: E402
from bearmind.stages.cnmf import (BackgroundDownsample, background_downsample, cnmf_movie_shape,  # noqa: E402
                                  cnmf_need_gib, cnmf_params, downsample_anomaly, patch_workers)
from bearmind.video import VideoArray, decode_to_memmap, write_ffv1  # noqa: E402

SID = "NOF_H01_1D"
FPS = 19.76


def _rec(n_frames):
    return SessionRecord(sid=SID, mouse="NOF_H01", mode="bonsai", parts=[], n_frames=n_frames, fps=FPS,
                         fps_info={}, anomalies=[], decisions=[], excluded=False, stages={})


def _setup(tmp_path, mov=None, thresholds=True):
    paths = SeriesPaths(tmp_path / "series")
    T = 0
    if mov is not None:
        T = mov.shape[0]
        mc = paths.mc_video(SID)
        mc.parent.mkdir(parents=True)
        write_ffv1([mov], mc, FPS)
    cfg = config.load_defaults()
    cfg["machine"] = {"n_cpus": 3, "temp_dir": str(tmp_path / "tmp"), "ram_gb": 16,
                      "cnmf_ram_budget_gb": 10}
    if thresholds:
        cfg["cnmf"].update(gSig=3, min_corr=0.8, min_pnr=10)
    return paths, _rec(T), cfg


def _temp_empty(paths, tmp_path):
    t = paths.temp_dir(tmp_path / "tmp", SID)
    return not t.exists() or not any(t.iterdir())


def test_fr_equals_fps_regardless_of_tsub():
    cfg = load_defaults(); cfg["cnmf"].update(gSig=4, min_corr=0.85, min_pnr=8, tsub=2)
    p = cnmf_params(cfg, fps=19.76)
    assert p["fr"] == 19.76 and p["tsub"] == 2 and p["gSiz"] == (17, 17)
    assert p["gSig"] == (4, 4) and p["min_corr"] == 0.85 and p["min_pnr"] == 8
    assert p["rf"] == 40 and p["method_init"] == "corr_pnr"  # notebook base dict is carried over
    assert cfg["cnmf"]["gSig"] == 4 and "fr" not in cfg["cnmf"]  # the caller's cfg is not mutated


@pytest.mark.parametrize("missing", ["all", "gSig", "min_corr", "min_pnr"])
def test_cnmf_params_waits_for_thresholds(missing):
    cfg = load_defaults()
    if missing != "all":
        cfg["cnmf"].update(gSig=4, min_corr=0.85, min_pnr=8)
        cfg["cnmf"][missing] = None
    with pytest.raises(WaitingForInput) as e:
        cnmf_params(cfg, fps=20.0)
    assert e.value.what == "thresholds"
    assert str(e.value) == "waiting for thresholds"


def test_waiting_without_thresholds(tmp_path):
    # no mc video either: the stage must stop before touching any input or the temp folder
    paths, rec, cfg = _setup(tmp_path, thresholds=False)
    env = {k: os.environ.get(k) for k in ("CAIMAN_TEMP", "OPENBLAS_NUM_THREADS")}
    with pytest.raises(WaitingForInput) as e:
        cnmf.run(paths, rec, cfg)
    assert e.value.what == "thresholds"
    assert not (tmp_path / "tmp").exists()
    assert {k: os.environ.get(k) for k in env} == env


def test_patch_workers_budget():
    cfg = {"machine": {"n_cpus": 32, "cnmf_ram_budget_gb": 153}, "memory": {"patch_gb_per_mframe": {1: 245, 2: 163}}}
    assert patch_workers(108000, 1, cfg) == 5 and patch_workers(108000, 2, cfg) == 8 and patch_workers(2000, 1, cfg) == 31


def test_patch_workers_bounds():
    cfg = {"machine": {"n_cpus": 32, "cnmf_ram_budget_gb": 10}, "memory": {"patch_gb_per_mframe": {1: 245, 2: 163}}}
    assert patch_workers(108000, 1, cfg) == 1  # budget below one worker still runs one
    cfg["machine"]["n_cpus"] = 1
    assert patch_workers(2000, 1, cfg) == 1


def test_memmap_strips_identical_to_save_memmap(tmp_path, monkeypatch):
    import caiman as cm

    rng = np.random.default_rng(3)
    T, H, W = 37, 20, 13
    mov = rng.integers(0, 256, (T, H, W), dtype=np.uint8)
    avi = tmp_path / "m.avi"
    write_ffv1([mov], avi, 20.0)
    tif = tmp_path / "m.tif"
    tifffile.imwrite(str(tif), mov)

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    monkeypatch.setenv("CAIMAN_TEMP", str(ref_dir))  # save_memmap writes into CAIMAN_TEMP, not next to base_name
    ref = Path(cm.save_memmap([str(tif)], base_name="Yr", order="C"))  # default add_to_movie

    ours_dir = tmp_path / "ours"
    ours_dir.mkdir()
    u8 = decode_to_memmap(avi, tmp_path / "mc_u8.dat")
    monkeypatch.setattr(cnmf, "TILE", 8)  # 37 frames = 8 + 8 + 8 + 8 + 5
    mem_gb = 4 * 4 * T * H / 1e9  # float32 strip of 4 columns: 13 = 4 + 4 + 4 + 1, the last strip partial
    ours = cnmf.write_memmap(u8, ours_dir, mem_gb=mem_gb)

    assert ours.parent == ours_dir
    assert ours.name == ref.name
    assert ours.read_bytes() == ref.read_bytes()
    Yr, dims, TT = cm.load_memmap(str(ours))
    assert dims == (H, W) and TT == T
    images = Yr.T.reshape((T,) + dims, order="F")
    np.testing.assert_array_equal(images, mov.astype(np.float32) + np.float32(1e-4))
    del Yr, images, u8


def test_mem_budget_from_machine_ram():
    assert cnmf.mem_budget_gb({"machine": {"ram_gb": 16}}) == 2  # 1.6 raised to the floor
    assert cnmf.mem_budget_gb({"machine": {"ram_gb": 192}}) == pytest.approx(19.2)
    assert cnmf.mem_budget_gb({"machine": {"ram_gb": 512}}) == 32


def test_imax_from_threshold_maps(tmp_path, monkeypatch):
    paths = SeriesPaths(tmp_path / "series")
    pnr = np.linspace(-5, 100, 6 * 7, dtype=np.float32).reshape(6, 7)
    out = paths.threshold_maps(SID)
    out.parent.mkdir(parents=True)
    np.savez_compressed(out, pnr_4=pnr, pnr_3=np.zeros((6, 7), np.float32))

    def boom(*a, **k):
        raise AssertionError("must not recompute maps when the threshold maps have pnr_<gSig>")

    monkeypatch.setattr(bm_qc, "compute_init_maps", boom)
    imax = cnmf._imax(paths, SID, gSig=4, tsub=1, dims=(6, 7), movie=None, mem_gb=2)
    c = np.clip(pnr, 0, 70)
    np.testing.assert_array_equal(imax, (c * 255 / c.max()).astype(np.uint8))
    assert imax.dtype == np.uint8 and imax.max() == 255 and imax.min() == 0


def test_imax_fallback_uses_decoded_movie(tmp_path, monkeypatch):
    paths = SeriesPaths(tmp_path / "series")
    mov = np.random.default_rng(5).integers(0, 256, (10, 6, 7), dtype=np.uint8)
    # maps for another gSig only: the fallback must run for gSig 4
    out = paths.threshold_maps(SID)
    out.parent.mkdir(parents=True)
    np.savez_compressed(out, pnr_3=np.ones((6, 7), np.float32))
    calls = []
    pnr = np.full((6, 7), 35.0, np.float32)
    pnr[0, 0] = np.inf

    def fake(movie, gSig, tsub=2, max_mem_gb=4, **kw):
        calls.append((movie is mov, gSig, tsub, max_mem_gb))
        return {"pnr": pnr, "cn": np.zeros_like(pnr)}

    monkeypatch.setattr(bm_qc, "compute_init_maps", fake)
    imax = cnmf._imax(paths, SID, gSig=4, tsub=2, dims=(6, 7), movie=mov, mem_gb=3.5)
    assert calls == [(True, 4, 2, 3.5)]
    assert imax[0, 0] == 255 and imax[1, 1] == 127


def _no_videoarray_reads(monkeypatch):
    def boom(self, key):
        raise AssertionError("the stage must read the movie from its decoded uint8 memmap, not VideoArray")

    monkeypatch.setattr(VideoArray, "__getitem__", boom)


@pytest.mark.slow
def test_cnmf_on_synthetic(tmp_path, monkeypatch):
    mov, centers = synthetic_movie(T=1000, n_cells=6)
    paths, rec, cfg = _setup(tmp_path, mov)
    cfg["cnmf"].update(rf=20, stride=10)  # several patches, so the pool and the memmap name are exercised
    env = {k: os.environ.get(k) for k in ("CAIMAN_TEMP", "OPENBLAS_NUM_THREADS")}
    _no_videoarray_reads(monkeypatch)  # no threshold maps here, so imax takes the fallback path too
    qc_calls = []
    real_qc = bm_qc.cnmf_qc

    def spy(*a, **k):
        qc_calls.append(k.get("make_video"))  # not the kwargs: they hold the movie memmap and block its removal
        return real_qc(*a, **k)

    monkeypatch.setattr(bm_qc, "cnmf_qc", spy)

    res = cnmf.run(paths, rec, cfg)

    assert qc_calls == [False]
    assert set(res.outputs) == {"estimates", "residual_png"}
    assert res.outputs["estimates"] == paths.rel(paths.estimates(SID))
    assert not paths.preview(SID, "cnmf_qc").exists()
    assert res.outputs["residual_png"] == paths.rel(paths.residual_png(SID))
    for rel in res.outputs.values():
        assert (paths.root / rel).stat().st_size > 0, rel
    assert not list(paths.estimates(SID).parent.glob("*.partial"))

    m = res.metrics
    assert m["n_components"] >= 4
    assert 0 < m["n_accepted"] <= m["n_components"]
    assert m["median_snr"] is not None and m["median_snr"] > 0
    assert m["gSig"] == 3 and m["min_corr"] == 0.8 and m["min_pnr"] == 10
    assert m["patch_workers"] == patch_workers(1000, 1, cfg) == 2  # n_cpus - 1
    assert "missed_cells" in m
    assert m["ssub_B"] == 2 and m["need_gib"] > 0
    assert not [a for a in res.anomalies if a.code == "cnmf_downsampled"]

    with open(paths.estimates(SID), "rb") as f:
        est = pickle.load(f)
    assert est.mc_video == "3_MC/NOF_H01_1D_MC.avi"
    assert getattr(est, "tif_name", None) is None
    assert est.cnmf_dict["fr"] == FPS and est.cnmf_dict["gSig"] == (3, 3) and est.cnmf_dict["gSiz"] == (13, 13)
    assert est.patch_workers == 2
    assert est.imax.dtype == np.uint8 and est.imax.shape == mov.shape[1:] and est.imax.max() == 255
    assert est.A.shape == (mov.shape[1] * mov.shape[2], m["n_components"])

    # every simulated cell is matched by a component centre
    A = est.A.toarray().reshape(mov.shape[1], mov.shape[2], -1, order="F")
    com = np.array([np.unravel_index(np.argmax(A[:, :, k]), A.shape[:2]) for k in range(A.shape[2])])
    dist = np.sqrt(((centers[:, None, :] - com[None]) ** 2).sum(-1))
    assert np.sum(dist.min(1) < 4) >= 4

    assert _temp_empty(paths, tmp_path)
    assert {k: os.environ.get(k) for k in env} == env


@pytest.mark.slow
@pytest.mark.parametrize("where", ["fit", "qc"])
def test_cnmf_failure_cleans_up(tmp_path, monkeypatch, where):
    from caiman.source_extraction.cnmf.cnmf import CNMF
    from caiman.source_extraction.cnmf.estimates import Estimates

    mov, _ = synthetic_movie(T=200, n_cells=2)
    paths, rec, cfg = _setup(tmp_path, mov)
    env = {k: os.environ.get(k) for k in ("CAIMAN_TEMP", "OPENBLAS_NUM_THREADS")}
    seen = {}

    def boom(*a, **k):
        raise RuntimeError("boom")

    def fit(self, images):
        seen["mmap"] = Path(images.filename)
        seen["temp"] = sorted(p.name for p in paths.temp_dir(tmp_path / "tmp", SID).iterdir())
        seen["env"] = (os.environ.get("CAIMAN_TEMP"), os.environ.get("OPENBLAS_NUM_THREADS"))
        if where == "fit":
            raise RuntimeError("boom")

    monkeypatch.setattr(CNMF, "fit", fit)
    if where == "qc":  # CNMF "succeeds", the QC step fails: no estimates file may be left
        monkeypatch.setattr(Estimates, "evaluate_components", lambda self, *a, **k: None)
        monkeypatch.setattr(bm_qc, "cnmf_qc", boom)
    with pytest.raises(RuntimeError, match="boom"):
        cnmf.run(paths, rec, cfg)
    assert seen["mmap"].parent == paths.temp_dir(tmp_path / "tmp", SID)
    assert seen["temp"] == [seen["mmap"].name]  # the uint8 copy is gone before the fit
    assert seen["env"] == (str(paths.temp_dir(tmp_path / "tmp", SID)), "1")
    assert not seen["mmap"].exists()
    assert _temp_empty(paths, tmp_path)
    est_dir = paths.estimates(SID).parent
    assert not est_dir.exists() or not any(est_dir.iterdir())  # neither the pickle nor a .partial
    assert {k: os.environ.get(k) for k in env} == env


def test_cnmf_fails_at_once_without_temp_space(tmp_path, monkeypatch):
    """10d: CNMF needs about 5 bytes per pixel-frame (uint8 copy + float32 memmap); 159 GB for an hour
    of 567 x 520 at 30 fps."""
    from tests.bearmind.test_stage_mc import _check_space_error, _little_free_space

    from bearmind.stages import NotEnoughTempSpace

    mov, _ = synthetic_movie(T=200, n_cells=2)
    paths, rec, cfg = _setup(tmp_path, mov)
    _little_free_space(monkeypatch)
    monkeypatch.setattr(cnmf, "decode_to_memmap", None)  # any decoding would fail differently
    monkeypatch.setattr(cnmf, "video_info", lambda p: (108_000, 567, 520, 30.0))
    with pytest.raises(NotEnoughTempSpace) as e:
        cnmf.run(paths, rec, cfg)
    _check_space_error(e.value, paths.temp_dir(tmp_path / "tmp", SID), 159.2)
    assert not (tmp_path / "tmp").exists()


# --- automatic background downsampling (ssub_B) ---

HOUR = 107107


def _cfg(budget=152.96, ssub_B=2):
    cfg = load_defaults(); cfg["cnmf"]["ssub_B"] = ssub_B
    cfg["machine"] = {"n_cpus": 32, "ram_gb": 191.2, "cnmf_ram_budget_gb": budget}
    return cfg


def test_downsample_lnof_auto_box():
    d = background_downsample(HOUR, 567, 520, _cfg())
    assert (d.ssub_B, d.configured, d.fits) == (3, 2, True)
    assert d.need_configured_gib == pytest.approx(156.5, abs=0.3) and d.need_gib == pytest.approx(74.0, abs=0.3)


def test_downsample_keeps_the_config_when_it_fits():
    for T, h, w in ((HOUR, 472, 436), (HOUR, 458, 458), (17861, 461, 608), (2000, 608, 608)):
        assert background_downsample(T, h, w, _cfg()).ssub_B == 2


def test_downsample_takes_the_weakest_that_fits():
    assert background_downsample(HOUR, 567, 520, _cfg(budget=60)).ssub_B == 4     # 74.0 > 60 >= 45.1
    assert background_downsample(HOUR, 608, 608, _cfg()).ssub_B == 3              # 194.2, 90.8


def test_downsample_decision_table_of_the_plan():
    # (box, need at 2, 3, 4, decision) for T = 107107, budget 152.96
    table = {(567, 520): (156.5, 74.0, 45.1, 3), (472, 436): (111.7, 54.1, 33.9, 2),
             (458, 458): (113.7, 55.0, 34.4, 2), (608, 608): (194.2, 90.8, 54.6, 3)}
    for (h, w), (n2, n3, n4, s) in table.items():
        for ss, n in ((2, n2), (3, n3), (4, n4)):
            assert cnmf_need_gib(HOUR, h, w, ss, _cfg()) == pytest.approx(n, abs=0.3)
        assert background_downsample(HOUR, h, w, _cfg()).ssub_B == s


def test_downsample_none_fits():
    d = background_downsample(HOUR, 567, 520, _cfg(budget=20))
    assert (d.ssub_B, d.fits) == (4, False)


def test_downsample_starts_at_the_configured_value():
    assert background_downsample(HOUR, 458, 458, _cfg(ssub_B=3)).ssub_B == 3


def test_cnmf_params_change_only_ssub_b():
    cfg = _cfg(); cfg["cnmf"].update(gSig=4, min_corr=0.8, min_pnr=8)
    base = cnmf_params(cfg, 29.75)
    assert cnmf_params(cfg, 29.75, (2000, 608, 608)) == base          # sessions that fit: same params, same hash
    big = cnmf_params(cfg, 29.75, (HOUR, 567, 520))
    assert big["ssub_B"] == 3 and {k: v for k, v in big.items() if k != "ssub_B"} == {k: v for k, v in base.items() if k != "ssub_B"}


def test_downsample_anomaly_levels():
    cfg = _cfg()
    assert downsample_anomaly(background_downsample(HOUR, 472, 436, cfg)) is None
    a = downsample_anomaly(background_downsample(HOUR, 567, 520, cfg))
    assert (a.code, a.level) == ("cnmf_downsampled", "warning")
    assert "ssub_B 3 instead of 2" in a.message and a.data["need_gib"] == pytest.approx(74.0, abs=0.3)
    a = downsample_anomaly(background_downsample(HOUR, 567, 520, _cfg(budget=20)))
    assert (a.code, a.level) == ("cnmf_downsampled", "error")
    assert "even at ssub_B 4" in a.message


def _mc_rec(metrics, shape=(16, 20)):
    part = RawPart(folder="x", chunks=[], n_frames=100, shape=shape, timestamps_file=None, ts_format=None,
                   start_ms=None, metadata_file=None)
    rec = _rec(100)
    rec.parts = [part]
    rec.stages = {"mc": StageRecord(state=State.DONE, metrics=metrics)}
    return rec


def test_movie_shape_from_the_mc_record():
    assert cnmf_movie_shape(_mc_rec({"n_frames": HOUR, "crop": [0, 0, 567, 520]})) == (HOUR, 567, 520)
    assert cnmf_movie_shape(_mc_rec({"n_frames": 90, "crop": None})) == (90, 16, 20)
    assert cnmf_movie_shape(_mc_rec({"crop": None})) == (100, 16, 20)
    assert cnmf_movie_shape(_rec(100)) is None
    rec = _mc_rec({"n_frames": 5, "crop": None})
    rec.stages["mc"].state = State.FAILED
    assert cnmf_movie_shape(rec) is None


def test_run_rejects_an_mc_video_that_differs_from_its_record(tmp_path):
    mov, _ = synthetic_movie(T=50, n_cells=2)
    paths, rec, cfg = _setup(tmp_path, mov)
    rec.stages = {"mc": StageRecord(state=State.DONE, metrics={"n_frames": 60, "crop": None})}
    rec.parts = [RawPart(folder="x", chunks=[], n_frames=60, shape=mov.shape[1:], timestamps_file=None,
                         ts_format=None, start_ms=None, metadata_file=None)]
    with pytest.raises(ValueError, match="run MC again"):
        cnmf.run(paths, rec, cfg)
    assert not (tmp_path / "tmp").exists()


def test_cnmf_params_reach_caiman_ssub_b():
    from caiman.source_extraction.cnmf.params import CNMFParams

    cfg = _cfg(); cfg["cnmf"].update(gSig=4, min_corr=0.8, min_pnr=8)
    opts = CNMFParams(params_dict=cnmf_params(cfg, 29.75, (HOUR, 567, 520)))
    assert opts.init["ssub_B"] == 3


def _boom_after_decision(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("heavy call reached")

    monkeypatch.setattr(cnmf, "decode_to_memmap", boom)


def test_run_logs_the_decision_before_the_heavy_work(tmp_path, monkeypatch, caplog):
    import logging

    mov, _ = synthetic_movie(T=50, n_cells=2)
    paths, rec, cfg = _setup(tmp_path, mov)
    cfg["machine"]["cnmf_ram_budget_gb"] = 8.0000001  # base alone is 8 GiB: nothing fits
    _boom_after_decision(monkeypatch)
    with caplog.at_level(logging.INFO, logger=cnmf.__name__):
        with pytest.raises(RuntimeError, match="heavy call reached"):
            cnmf.run(paths, rec, cfg)
    recs = [r for r in caplog.records if "memory decision" in r.getMessage()]
    assert len(recs) == 1 and recs[0].levelno == logging.WARNING
    m = recs[0].getMessage()
    assert "configured 2, chosen 4" in m and "need_gib" in m and "budget_gib 8.0" in m and "fits False" in m


def test_run_decides_from_the_video_without_an_mc_record(tmp_path, monkeypatch):
    mov, _ = synthetic_movie(T=50, n_cells=2)
    paths, rec, cfg = _setup(tmp_path, mov)
    cfg["machine"]["cnmf_ram_budget_gb"] = 8.0000001
    seen = {}
    real = cnmf.background_downsample

    def spy(T, h, w, c):
        seen["shape"] = (T, h, w)
        return real(T, h, w, c)

    monkeypatch.setattr(cnmf, "background_downsample", spy)
    _boom_after_decision(monkeypatch)
    with pytest.raises(RuntimeError, match="heavy call reached"):
        cnmf.run(paths, rec, cfg)
    assert seen["shape"] == tuple(mov.shape)
