from pathlib import Path

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


def _jittered_clock(n, dt, every=36):
    """Frame k at k * dt, but every `every`-th frame and the one after it are reported late by 0.75 and 0.49 of
    a frame (delivery delay, the camera keeps its clock): intervals 1.75, 0.74, 0.51 of a frame, as in the LT1
    J06 timestamps. The long interval is above 1.5 of the median, the short ones are not."""
    t = np.arange(n, dtype=np.float64) * dt
    starts = np.arange(every, n - 2, every)
    t[starts] += 0.75 * dt
    t[starts + 1] += 0.49 * dt
    return t, len(starts)


def test_robust_fps_paired_jitter_unbiased():
    """Jitter that keeps the frame clock does not change the estimate: the long intervals are counted with
    the short ones (an estimate from the intervals in [0.5 m, 1.5 m] only was 2 % too high here)."""
    dt = 1000.0 / 29.78
    t, n_bursts = _jittered_clock(18000, dt)
    est = robust_fps(t)
    assert est.fps == pytest.approx(29.78, rel=1e-3)
    # n_outliers still counts the intervals outside [0.5 m, 1.5 m]: one long interval per burst
    assert est.n_outliers == n_bursts and est.n_gaps == 0
    assert est.median_interval_ms == pytest.approx(dt)


def test_robust_fps_excludes_gaps_not_jitter():
    """Pauses and dropped frames (intervals above gap_factor * median) stay out of the estimate; the jitter
    around them is counted."""
    dt = 50.0  # 20 fps
    t, _ = _jittered_clock(3000, dt)
    t[1000:] += 5000.0  # a 5 s pause between parts
    t[2000:] += 2 * dt  # two dropped frames: one interval of 3 frames
    est = robust_fps(t)
    assert est.fps == pytest.approx(20.0, rel=1e-3)
    assert est.n_gaps == 2 and est.max_gap_ms == pytest.approx(5000.0 + dt)


def test_robust_fps_plain_clock_unchanged():
    """Without outliers and gaps the estimate is 1000 / mean interval, as before."""
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.normal(33.4, 0.3, 5000))
    est = robust_fps(t)
    assert est.fps == pytest.approx(1000.0 / np.diff(t).mean(), rel=1e-12)
    assert est.n_outliers == 0 and est.n_gaps == 0


J06_1D_TS = Path(r"H:\Projects\LT1\CalciumData\1_Raw\LT1_J06_1D\LT1_J06_1D_Mini_TS.csv")


@pytest.mark.realdata
@pytest.mark.skipif(not J06_1D_TS.exists(), reason="LT1 raw data not on this machine")
def test_robust_fps_lt1_j06_1d():
    """LT1_J06_1D (bonsai, paired jitter): the rate of the timestamps, (frames - 1) / duration = 29.780. The
    estimate from the intervals in [0.5 m, 1.5 m] gave 30.241. The file is only read."""
    ts = read_timestamps(J06_1D_TS)
    span = (len(ts.t_ms) - 1) * 1000.0 / (ts.t_ms[-1] - ts.t_ms[0])
    est = robust_fps(ts.t_ms)
    assert est.fps == pytest.approx(29.78, abs=0.01)
    assert est.fps == pytest.approx(span, abs=0.01) and est.n_gaps == 0


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
