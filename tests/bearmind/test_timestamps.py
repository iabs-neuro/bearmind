import numpy as np
import pytest

from bearmind.timestamps import read_timestamps, robust_fps, stitch, write_timestamps

UCLA_HDR = "Frame Number,Time Stamp (ms),Buffer Index\n"


def test_bonsai_fps_with_gap(tmp_path):
    ticks = 638331574485491072 + np.arange(1000, dtype=np.int64) * 506000
    ticks[500:] += 30_000_000
    f = tmp_path / "S_Mini_TS.csv"
    np.savetxt(f, ticks, fmt="%d")
    ts = read_timestamps(f)
    est = robust_fps(ts.t_ms)
    assert ts.fmt == "bonsai_ticks"
    assert ts.raw.dtype == np.int64 and ts.start_epoch_ms is None and ts.buffer is None
    assert abs(est.fps - 19.763) < 0.01 and est.n_gaps == 1 and est.max_gap_ms > 3000


def test_ucla_negative_start(tmp_path):
    f = tmp_path / "timeStamps.csv"
    f.write_text(UCLA_HDR + "0,-14,0\n1,22,0\n2,53,0\n3,86,0\n")
    ts = read_timestamps(f, start_epoch_ms=1728989165178.0)
    assert ts.fmt == "ucla_ms" and ts.raw[0] == -14 and abs(robust_fps(ts.t_ms).fps - 30.0) < 1.5
    assert ts.t_ms[0] == 1728989165178.0 - 14
    assert list(ts.buffer) == [0, 0, 0, 0]
    ts2 = read_timestamps(f)
    assert ts2.t_ms[0] == -14.0 and ts2.start_epoch_ms is None


def test_roundtrip_bonsai_subset(tmp_path):
    ticks = 638331574485491072 + np.arange(30, dtype=np.int64) * 506000
    f = tmp_path / "a.csv"
    np.savetxt(f, ticks, fmt="%d")
    ts = read_timestamps(f)
    out = tmp_path / "b.csv"
    write_timestamps(ts, np.arange(10, 20), out)
    assert b"\r" not in out.read_bytes()
    back = read_timestamps(out)
    assert np.array_equal(back.raw, ts.raw[10:20])


def test_unknown_format(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a;b\n1;2\n")
    with pytest.raises(ValueError, match="unknown timestamp format"):
        read_timestamps(f)


def test_empty_file(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text("")
    with pytest.raises(ValueError, match="unknown timestamp format"):
        read_timestamps(f)


def test_robust_fps_too_few():
    with pytest.raises(ValueError, match="not enough timestamps"):
        robust_fps(np.array([0.0, 10.0]))


def _ucla_part(tmp_path, name, start, n):
    f = tmp_path / name
    rows = "".join(f"{i},{i * 33},{i % 3}\n" for i in range(n))
    f.write_text(UCLA_HDR + rows)
    return read_timestamps(f, start_epoch_ms=start)


def test_stitch_ucla_with_pause(tmp_path):
    s0 = 1_700_000_000_000.0
    a = _ucla_part(tmp_path, "a.csv", s0, 100)
    # second part starts 5 s after the last frame of the first
    b = _ucla_part(tmp_path, "b.csv", a.t_ms[-1] + 5000, 50)
    st = stitch([b, a])  # given out of order
    assert len(st.raw) == 150 and st.start_epoch_ms == s0 and st.fmt == "ucla_ms"
    assert np.array_equal(st.raw[:100], a.raw)
    assert st.raw[100] == round(b.t_ms[0] - s0)
    assert st.raw[100] - st.raw[99] == 5000
    assert len(st.buffer) == 150
    out = tmp_path / "out.csv"
    write_timestamps(st, np.arange(150), out)
    back = read_timestamps(out, start_epoch_ms=s0)
    assert np.array_equal(back.raw, st.raw) and np.allclose(back.t_ms, st.t_ms)
    assert np.array_equal(back.buffer, st.buffer)
    assert out.read_text().splitlines()[0] == UCLA_HDR.strip()
    assert out.read_text().splitlines()[1].startswith("0,")


def test_stitch_errors(tmp_path):
    a = _ucla_part(tmp_path, "a.csv", None, 5)
    b = _ucla_part(tmp_path, "b.csv", 1e12, 5)
    with pytest.raises(ValueError, match="without recording start"):
        stitch([a, b])
    f = tmp_path / "bon.csv"
    np.savetxt(f, np.arange(5, dtype=np.int64) * 10**5, fmt="%d")
    with pytest.raises(ValueError):
        stitch([b, read_timestamps(f)])


def test_session_timestamps_matches_export(make_series, tmp_path):
    """The stitched timestamps the export stage wrote: parts in time order, each cut to its frames, UCLA
    rows counted from the first part's start."""
    from bearmind import scan
    from bearmind.paths import SeriesPaths
    from bearmind.stages import export
    from bearmind.timestamps import session_timestamps

    assert export.session_timestamps is session_timestamps  # the export stage calls this function
    root = make_series(tmp_path / "split", "legacy_split")  # two parts of 30 frames, 1 h apart
    rec = scan.scan_series(root)[0]
    ts = session_timestamps(SeriesPaths(root), rec)
    assert ts.fmt == "ucla_ms" and len(ts.raw) == rec.n_frames == 60
    assert ts.raw.tolist() == [i * 50 for i in range(30)] + [3600_000 + i * 50 for i in range(30)]
    assert np.array_equal(ts.t_ms, rec.parts[0].start_ms + ts.raw)

    root = make_series(tmp_path / "bonsai", "bonsai", ts_rows=33)  # 30 frames, 33 timestamp rows
    rec = scan.scan_series(root)[0]
    ts = session_timestamps(SeriesPaths(root), rec)
    assert ts.fmt == "bonsai_ticks" and ts.raw.tolist() == [i * 500000 for i in range(30)]


def test_stitch_bonsai_single():
    from bearmind.timestamps import Timestamps
    raw = np.arange(5, dtype=np.int64) * 10**5
    t = Timestamps(raw, raw / 1e4, "bonsai_ticks")
    s = stitch([t])
    assert np.array_equal(s.raw, raw) and s.raw is not raw
