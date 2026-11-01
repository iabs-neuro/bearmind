import errno
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
from bearmind.registry import SessionRecord  # noqa: E402
from bearmind.stages import WaitingForInput  # noqa: E402
from bearmind.stages import cnmf  # noqa: E402
from bearmind.stages.cnmf import cnmf_params, patch_workers  # noqa: E402
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


def test_move_falls_back_to_copy_across_drives(tmp_path, monkeypatch):
    src = tmp_path / "a" / "x.mp4"
    src.parent.mkdir()
    src.write_bytes(b"video")
    dst = tmp_path / "b" / "y.mp4"
    real = os.replace

    def cross_drive(a, b):
        if Path(a) == src:
            raise OSError(errno.EXDEV, "The system cannot move the file to a different disk drive")
        return real(a, b)

    monkeypatch.setattr(cnmf.os, "replace", cross_drive)
    cnmf._move(src, dst)
    assert dst.read_bytes() == b"video"
    assert not src.exists()
    assert sorted(p.name for p in dst.parent.iterdir()) == ["y.mp4"]


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

    res = cnmf.run(paths, rec, cfg)

    assert set(res.outputs) == {"estimates", "preview_cnmf_qc", "residual_png"}
    assert res.outputs["estimates"] == paths.rel(paths.estimates(SID))
    assert res.outputs["preview_cnmf_qc"] == paths.rel(paths.preview(SID, "cnmf_qc"))
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
