import numpy as np
import pytest

from bearmind.registry import RawPart, SessionRecord
from bearmind.video import (CropBox, SessionReader, VideoArray, last_frame_readable,
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
