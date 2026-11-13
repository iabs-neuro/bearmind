"""Frame loss at the gaps of a part's timestamps by the window test (spec 2026-11-12-scan-approval-design, section 4)."""
import numpy as np
import pytest

from bearmind import scan

T = 1000.0 / 29.78


def clock(n):
    return np.arange(n) * T


def lose(t, at, n_lost, extra=0.05):
    """The frames at[0 .. n_lost - 1] are not recorded; the later ones keep the clock (plus a small delay, so the
    interval is above 2 m for one lost frame)."""
    t = np.asarray(t, float)
    keep = np.ones(len(t), bool)
    keep[at:at + n_lost] = False
    out = t[keep].copy()
    out[at:] += extra * T
    return out


def test_pure_jitter_is_no_loss():
    """One frame late by 1.05 periods and the next ones early: an interval of 2.05 m, the clock goes on."""
    t = clock(2000)
    t[1000] += 1.05 * T
    t[1001] += 1.0 * T
    t[1002] += 0.5 * T
    r = scan.part_gap_losses(t)
    assert r.gaps == [] and r.n_jitter == 1 and not r.damaged


def test_one_lost_frame_is_probable():
    r = scan.part_gap_losses(lose(clock(2000), 1000, 1))
    assert len(r.gaps) == 1 and r.n_jitter == 0
    g = r.gaps[0]
    assert g.frames == 1 and g.index == 999
    assert g.certain is False  # 2.05 m, excess 1.0
    assert g.lost_ms == pytest.approx(T, rel=0.01)
    assert g.at_ms == pytest.approx(999 * T)  # where the gap begins, from the part's first frame


def test_three_lost_frames_are_certain():
    r = scan.part_gap_losses(lose(clock(2000), 1000, 3))
    assert [(g.frames, g.certain) for g in r.gaps] == [(3, True)]
    assert r.gaps[0].lost_ms == pytest.approx(3 * T, rel=0.01)


def test_certain_by_the_interval_alone():
    """An interval above 2.5 m is certain even in a short part."""
    g = scan.part_gap_losses(lose(clock(150), 70, 2)).gaps[0]
    assert g.interval_ms > 2.5 * T and g.certain and g.frames == 2


def test_gap_at_the_start_and_the_end_uses_the_intervals_that_exist():
    start = scan.part_gap_losses(lose(clock(2000), 3, 1))
    assert [(g.index, g.frames) for g in start.gaps] == [(2, 1)]
    end = scan.part_gap_losses(lose(clock(2000), 1996, 1))
    assert [g.frames for g in end.gaps] == [1] and end.gaps[0].index == 1995


def test_damaged_row_skips_the_part_without_crashing():
    t = clock(500)
    t[250] = t[249] - 5.0  # a row that goes back
    r = scan.part_gap_losses(t)
    assert r.damaged and r.gaps == [] and r.n_jitter == 0
    t2 = clock(500)
    t2[250] = t2[249]  # a repeated row: interval 0
    assert scan.part_gap_losses(t2).damaged


def test_short_parts_do_not_crash():
    for n in (0, 1, 2, 3):
        r = scan.part_gap_losses(clock(n))
        assert r.gaps == [] and not r.damaged


def test_close_gaps_form_one_cluster_counted_once():
    """Two lost frames in two gaps 10 intervals apart: one window, one excess, 2 frames (not 4)."""
    t = lose(lose(clock(2000), 1000, 1), 1010, 1)
    r = scan.part_gap_losses(t)
    assert len(r.gaps) == 1
    g = r.gaps[0]
    assert g.n_gaps == 2 and g.frames == 2 and g.index == 999
    assert g.lost_ms == pytest.approx(2 * T, rel=0.02)


def test_far_gaps_are_separate_clusters():
    t = lose(lose(clock(2000), 800, 1), 1200, 1)
    assert [(g.n_gaps, g.frames) for g in scan.part_gap_losses(t).gaps] == [(1, 1), (1, 1)]


def test_two_close_jitter_gaps_are_no_loss():
    t = clock(2000)
    for j in (1000, 1010):
        t[j] += 1.05 * T
        t[j + 1] += 1.0 * T
        t[j + 2] += 0.5 * T
    r = scan.part_gap_losses(t)
    assert r.gaps == [] and r.n_jitter == 2


def test_a_cluster_is_certain_when_any_gap_is_above_2_5_m():
    t = lose(lose(clock(2000), 1000, 3), 1010, 1)
    g = scan.part_gap_losses(t).gaps[0]
    assert g.n_gaps == 2 and g.frames == 4 and g.certain


def test_short_part_keeps_its_loss():
    """The period is the mean of the non-gap intervals, so the gap does not inflate its own period: one 200 ms gap in
    30 frames at 30 fps is 5 lost frames."""
    t = np.arange(30) * (1000.0 / 30)
    t[15:] += 200.0 - 1000.0 / 30
    r = scan.part_gap_losses(t)
    assert [(g.frames, g.certain) for g in r.gaps] == [(5, True)]


def test_position_text():
    assert scan.position_text(12 * 60000 + 3400) == "12:03.4"
    assert scan.position_text(500) == "0:00.5"
    assert scan.position_text(3600000 + 62000) == "1:01:02.0"
