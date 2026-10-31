import numpy as np
import pytest
import tifffile
from scipy.ndimage import gaussian_filter

from bearmind import config
from bearmind.paths import SeriesPaths
from bearmind.registry import RawPart, SessionRecord
from bearmind.stages import mc as mc_stage
from bearmind.video import CropBox, SessionReader, VideoArray

SID = "NOF_H01_1D"
T, FULL, BOX = 300, 72, 64


def _raw_frames():
    rng = np.random.default_rng(1)
    base = gaussian_filter(rng.random((FULL, FULL)), 2.0)
    base = 30 + 190 * (base - base.min()) / np.ptp(base)
    frames = []
    for t in range(T):
        s = 3 if 120 <= t < 150 else 0  # known x shift of 3 px for a short stretch
        f = np.roll(base, s, axis=1) + rng.normal(0, 2, base.shape)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return np.stack(frames)


def _setup(tmp_path, make_avi):
    root = tmp_path / "series"
    folder = root / "1_Raw" / "HM_NOF_1D" / SID
    folder.mkdir(parents=True)
    make_avi(folder / f"{SID}_0.avi", _raw_frames())
    part = RawPart(folder=f"1_Raw/HM_NOF_1D/{SID}", chunks=[f"{SID}_0.avi"], n_frames=T,
                   shape=(FULL, FULL), timestamps_file=None, ts_format=None, start_ms=None,
                   metadata_file=None)
    rec = SessionRecord(sid=SID, mouse="NOF_H01", mode="bonsai", parts=[part], n_frames=T, fps=20.0,
                        fps_info={}, anomalies=[], decisions=[], excluded=False, stages={})
    paths = SeriesPaths(root)
    cfg = config.resolve(paths, SID)
    cfg["mc"].update(max_shifts=[5, 5], gSig_filt=[3, 3], strides=[32, 32], overlaps=[8, 8])
    cfg["machine"] = dict(cfg["machine"], n_cpus=3, temp_dir=str(tmp_path / "tmp"))
    cfg["crop"] = {**cfg.get("crop", {}), "box": [BOX, BOX], "box_pos": [4, 4]}
    return paths, rec, cfg


@pytest.mark.slow
def test_mc_stage(tmp_path, make_avi, monkeypatch):
    monkeypatch.setattr(mc_stage, "CHUNK", 128)  # several chunks, last one partial
    paths, rec, cfg = _setup(tmp_path, make_avi)
    res = mc_stage.run(paths, rec, cfg)

    assert set(res.outputs) == {"mc_video", "mc_shifts", "preview_raw", "preview_raw_dff",
                                "preview_mc", "preview_mc_dff"}
    for rel in res.outputs.values():
        assert (paths.root / rel).exists(), rel
    assert res.metrics["n_frames"] == T
    assert res.metrics["crop"] == [4, 4, BOX, BOX]
    assert res.metrics["max_shift_px"] == pytest.approx(3, abs=0.5)
    assert res.metrics["mean_template_corr"] > 0.5

    # rebuild the cropped input and apply the saved shifts with CaImAn's own function
    import caiman.motion_correction as cmc
    tif = tmp_path / "in.tif"
    with tifffile.TiffWriter(str(tif), bigtiff=True) as tw:
        for ch in SessionReader(paths.root, rec, CropBox(4, 4, BOX, BOX)).iter_chunks(100):
            for fr in ch:
                tw.write(fr, contiguous=True)
    shifts = np.load(paths.mc_shifts(SID))
    m = cmc.MotionCorrect([str(tif)], pw_rigid=False, shifts_opencv=True, border_nan="copy")
    m.shifts_rig = [tuple(s) for s in shifts]
    ref = np.clip(np.rint(m.apply_shifts_movie(str(tif))), 0, 255).astype(np.uint8)
    got = VideoArray(paths.mc_video(SID))[0:T]
    assert got.shape == ref.shape
    assert np.array_equal(got, ref)

    assert not (tmp_path / "tmp" / SID).exists() or not any((tmp_path / "tmp" / SID).iterdir())


@pytest.mark.slow
def test_mc_stage_failure_cleans_up(tmp_path, make_avi, monkeypatch):
    import caiman.motion_correction as cmc
    paths, rec, cfg = _setup(tmp_path, make_avi)

    def boom(self, *a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(cmc.MotionCorrect, "motion_correct", boom)
    with pytest.raises(RuntimeError, match="boom"):
        mc_stage.run(paths, rec, cfg)
    assert not (tmp_path / "tmp" / SID).exists() or not any((tmp_path / "tmp" / SID).iterdir())


def test_crop_from_config():
    cfg = {"crop": {"box": [10, 12], "box_pos": [1, 2]}}
    assert mc_stage._crop(cfg) == CropBox(1, 2, 10, 12)
    assert mc_stage._crop({"crop": {"box": [10, 12], "box_pos": None}}) is None
    assert mc_stage._crop({"crop": {"box": [10, 12]}}) is None


@pytest.mark.slow
def test_mc_stage_writer_failure_releases_temp(tmp_path, make_avi, monkeypatch):
    paths, rec, cfg = _setup(tmp_path, make_avi)
    monkeypatch.setattr(mc_stage, "CHUNK", 128)

    def bad_write(chunks, path, fps):
        next(iter(chunks))  # consume the first chunk, leave the generator suspended
        raise OSError("disk full")

    monkeypatch.setattr(mc_stage, "write_ffv1", bad_write)
    with pytest.raises(OSError, match="disk full"):
        mc_stage.run(paths, rec, cfg)
    assert not (tmp_path / "tmp" / SID).exists()


def test_mc_stage_rejects_pw_rigid(tmp_path, make_avi):
    paths, rec, cfg = _setup(tmp_path, make_avi)
    cfg["mc"]["pw_rigid"] = True
    with pytest.raises(ValueError, match="pw_rigid"):
        mc_stage.run(paths, rec, cfg)
