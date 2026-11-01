import numpy as np
import pytest

from bearmind.registry import RawPart, SessionRecord
from bearmind.video import (CropBox, SessionReader, VideoArray, decode_to_memmap, last_frame_readable,
                            video_info, write_ffv1, write_preview_mp4)


def _frames(n, h=24, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, h, w), dtype=np.uint8)


def _rec(parts, n_frames):
    return SessionRecord(sid="s", mouse="m", mode="x", parts=parts, n_frames=n_frames, fps=20.0,
                         fps_info={}, anomalies=[], decisions=[], excluded=False, stages={})


def _part(chunks, n, **kw):
    return RawPart(folder="d", chunks=chunks, n_frames=n, shape=(24, 32), timestamps_file=None,
                   ts_format=None, start_ms=None, metadata_file=None, **kw)


def test_ffv1_lossless(tmp_path):
    frames = _frames(37)
    n = write_ffv1([frames[:20], frames[20:]], tmp_path / "x.avi", fps=20.0)
    assert n == 37 and np.array_equal(VideoArray(tmp_path / "x.avi")[0:37], frames)
    assert not list(tmp_path.glob("*.partial*"))
    assert video_info(tmp_path / "x.avi") == (37, 24, 32, 20.0)
    assert last_frame_readable(tmp_path / "x.avi")


def test_write_ffv1_cleans_partial_on_error(tmp_path):
    def gen():
        yield _frames(5)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        write_ffv1(gen(), tmp_path / "x.avi", fps=20.0)
    assert list(tmp_path.iterdir()) == []


def test_session_reader_rules(tmp_path, make_avi):
    (tmp_path / "d").mkdir()
    f = _frames(30)
    for i in range(3):
        make_avi(tmp_path / "d" / f"{i}.avi", f[i * 10:(i + 1) * 10])
    # drop tail of chunks 0 and 1 -> 9 + 9 + 10 = 28
    part = _part(["0.avi", "1.avi", "2.avi"], 28, drop_chunk_tail=True)
    reader = SessionReader(tmp_path, _rec([part], 28))
    assert reader.n_frames == 28 and reader.shape == (28, 24, 32)
    out = np.concatenate(list(reader.iter_chunks(7)))
    assert out.shape[0] == 28
    assert np.array_equal(out, np.concatenate([f[0:9], f[10:19], f[20:30]]))
    assert all(c.shape[0] <= 7 for c in reader.iter_chunks(7))
    # drop_last_frame and truncation to rec.n_frames
    part2 = _part(["0.avi", "1.avi", "2.avi"], 29, drop_last_frame=True)
    out2 = np.concatenate(list(SessionReader(tmp_path, _rec([part2], 25)).iter_chunks(1000)))
    assert np.array_equal(out2, f[:25])


def test_session_reader_stops_chunks_at_recorded_counts(tmp_path, make_avi):
    (tmp_path / "d").mkdir()
    f = _frames(30)
    for i in range(3):
        make_avi(tmp_path / "d" / f"{i}.avi", f[i * 10:(i + 1) * 10])
    part = _part(["0.avi", "1.avi", "2.avi"], 30, chunk_frames=[10, 10, 10])
    assert np.array_equal(np.concatenate(list(SessionReader(tmp_path, _rec([part], 30)).iter_chunks(7))), f)
    # a chunk that decodes fewer frames than the scan counted would shift every later frame
    part = _part(["0.avi", "1.avi", "2.avi"], 31, chunk_frames=[10, 11, 10])
    with pytest.raises(ValueError, match="decoded 10 of 11 frames"):
        list(SessionReader(tmp_path, _rec([part], 31)).iter_chunks(7))


def test_session_reader_never_decodes_broken_last_frame(tmp_path, make_avi):
    import av
    (tmp_path / "d").mkdir()
    f = _frames(12)
    p = make_avi(tmp_path / "d" / "0.avi", f)
    with av.open(str(p)) as c:
        pk = [x for x in c.demux(c.streams.video[0]) if x.size][-1]
    p.write_bytes(p.read_bytes()[:pk.pos + pk.size // 2])
    with pytest.raises(av.error.InvalidDataError):  # decoding the cut frame itself fails
        with av.open(str(p)) as c:
            list(c.decode(c.streams.video[0]))
    part = _part(["0.avi"], 11, chunk_frames=[12], drop_last_frame=True)
    out = np.concatenate(list(SessionReader(tmp_path, _rec([part], 11)).iter_chunks(5)))
    assert np.array_equal(out, f[:11])


def test_crop_applied(tmp_path, make_avi):
    (tmp_path / "d").mkdir()
    f = _frames(10)
    make_avi(tmp_path / "d" / "0.avi", f)
    reader = SessionReader(tmp_path, _rec([_part(["0.avi"], 10)], 10), CropBox(2, 3, 10, 12))
    out = np.concatenate(list(reader.iter_chunks(4)))
    assert out.shape == (10, 10, 12) and reader.shape == (10, 10, 12)
    assert np.array_equal(out, f[:, 2:12, 3:15])
    s = reader.sample(4, seed=1)
    assert s.shape == (4, 10, 12)


def test_videoarray_random_access(tmp_path):
    f = _frames(40)
    write_ffv1([f], tmp_path / "x.avi", fps=20.0)
    v = VideoArray(tmp_path / "x.avi")
    assert v.shape == (40, 24, 32) and v.dtype == np.uint8 and len(v) == 40
    for i in (15, 0, 39, 7, 22, -1):
        assert np.array_equal(v[i], f[i])
    assert np.array_equal(v[5:30:4], f[5:30:4])
    assert np.array_equal(v[::-1][:2], f[::-1][:2])


def test_preview_mp4(tmp_path):
    f = _frames(11, 25, 33)  # odd dims
    write_preview_mp4([f[:6], f[6:]], tmp_path / "p.mp4", fps=20.0, tbin=2)
    import av
    with av.open(str(tmp_path / "p.mp4")) as c:
        s = c.streams.video[0]
        assert s.codec_context.width % 2 == 0 and s.codec_context.height % 2 == 0
        assert sum(1 for _ in c.decode(s)) == 6  # ceil(11/2)


def test_last_frame_readable_false_on_garbage(tmp_path):
    p = tmp_path / "bad.avi"
    p.write_bytes(b"not a video")
    assert not last_frame_readable(p)


def test_open_movie_avi(tmp_path):
    import bm_qc
    f = _frames(6)
    write_ffv1([f], tmp_path / "x.AVI", fps=20.0)
    assert isinstance(bm_qc.open_movie(tmp_path / "x.AVI"), VideoArray)


def test_videoarray_spatial_slicing(tmp_path):
    f = _frames(40)
    write_ffv1([f], tmp_path / "x.avi", fps=20.0)
    v = VideoArray(tmp_path / "x.avi")
    assert np.array_equal(v[3:35:4, 2:12, 5:20], f[3:35:4, 2:12, 5:20])
    assert np.array_equal(v[10:20, 4:9, :], f[10:20, 4:9, :])
    assert np.array_equal(v[7, 2:5, 3:6], f[7, 2:5, 3:6])
    assert np.array_equal(v[30:5:-5, 1:3], f[30:5:-5, 1:3])


def test_open_movie_avi_strip(tmp_path):
    import bm_qc
    f = _frames(20)
    write_ffv1([f], tmp_path / "x.avi", fps=20.0)
    mov = bm_qc.open_movie(tmp_path / "x.avi")
    assert np.array_equal(mov[slice(2, 18), 3:10, :], f[2:18, 3:10, :])


def test_preview_partial_cleanup(tmp_path):
    def gen():
        yield _frames(4)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        write_preview_mp4(gen(), tmp_path / "p.mp4", fps=20.0)
    assert list(tmp_path.iterdir()) == []
    write_preview_mp4([_frames(4)], tmp_path / "p.mp4", fps=20.0)
    assert [p.name for p in tmp_path.iterdir()] == ["p.mp4"]


def test_write_ffv1_empty(tmp_path):
    with pytest.raises(ValueError, match="no frames to write"):
        write_ffv1([], tmp_path / "x.avi", fps=20.0)
    assert list(tmp_path.iterdir()) == []


def test_decode_to_memmap_equals_videoarray(tmp_path):
    frames = _frames(37)
    write_ffv1([frames], tmp_path / "x.avi", fps=20.0)
    mm = decode_to_memmap(tmp_path / "x.avi", tmp_path / "x_u8.dat", block=10)  # 10+10+10+7
    assert isinstance(mm, np.memmap) and mm.dtype == np.uint8 and mm.shape == (37, 24, 32)
    assert mm.flags["C_CONTIGUOUS"] and not mm.flags["WRITEABLE"]
    assert np.array_equal(mm, VideoArray(tmp_path / "x.avi")[:])
    assert np.array_equal(mm, frames)
    del mm


def test_decode_to_memmap_removes_output_on_error(tmp_path, monkeypatch):
    import bearmind.video as video

    write_ffv1([_frames(12)], tmp_path / "x.avi", fps=20.0)

    def broken(path):
        yield _frames(1)[0]
        raise RuntimeError("decode error")

    monkeypatch.setattr(video, "_iter_avi", broken)
    with pytest.raises(RuntimeError, match="decode error"):
        decode_to_memmap(tmp_path / "x.avi", tmp_path / "x_u8.dat")
    assert not (tmp_path / "x_u8.dat").exists()


def test_decode_to_memmap_rejects_short_decode(tmp_path, monkeypatch):
    import bearmind.video as video

    write_ffv1([_frames(12)], tmp_path / "x.avi", fps=20.0)
    monkeypatch.setattr(video, "_iter_avi", lambda path: iter(_frames(5)))
    with pytest.raises(ValueError, match="decoded 5 frames"):
        decode_to_memmap(tmp_path / "x.avi", tmp_path / "x_u8.dat")
    assert not (tmp_path / "x_u8.dat").exists()


def _deny_first_replace(monkeypatch):
    """os.replace refuses once (a player or a GUI holds the target), as on Windows; then works."""
    import os

    from bearmind import fileio

    monkeypatch.setattr(fileio, "REPLACE_STEP_S", 0.0)
    real, denied = os.replace, []

    def replace(src, dst):
        if not denied:
            denied.append(str(dst))
            raise PermissionError(13, "the target is open in another program")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    return denied


@pytest.mark.parametrize("writer", ["ffv1", "mp4"])
def test_writers_retry_a_refused_rename(tmp_path, monkeypatch, writer):
    """10b: the final rename of the .partial file goes through fileio.replace (retried on PermissionError)."""
    denied = _deny_first_replace(monkeypatch)
    out = tmp_path / ("v.avi" if writer == "ffv1" else "v.mp4")
    if writer == "ffv1":
        write_ffv1([_frames(4)], out, 20.0)
    else:
        write_preview_mp4([_frames(4)], out, 20.0)
    assert denied == [str(out)] and out.is_file()
    assert not list(tmp_path.glob("*.partial*"))


def test_mc_shifts_written_through_partial(tmp_path, monkeypatch):
    from bearmind.stages import mc as mc_stage

    denied = _deny_first_replace(monkeypatch)
    out = tmp_path / "s_MC_shifts.npy"
    mc_stage._save_shifts(out, np.ones((5, 2), np.float32))
    assert denied == [str(out)] and np.load(out).shape == (5, 2)
    monkeypatch.setattr(np, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    out.unlink()
    with pytest.raises(OSError, match="disk full"):
        mc_stage._save_shifts(out, np.ones((5, 2), np.float32))
    assert not out.exists() and not list(tmp_path.glob("*.partial*"))
