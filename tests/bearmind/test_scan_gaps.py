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
    """An interval above 2.5 m is certain although the excess is below 1.5. The short part's period u takes in the
    gap, so the excess falls short of the 2 lost frames: the rule counts 1, which is why certain exists."""
    r = scan.part_gap_losses(lose(clock(150), 70, 2))
    g = r.gaps[0]
    assert g.interval_ms > 2.5 * T and g.excess < 1.5 and g.certain


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


def test_two_lost_frames_two_gaps_in_one_window_are_counted_in_each_window():
    """Each gap has its own window, so two gaps closer than K intervals both see the excess of the pair. The
    reference script does the same; the tests pins it so a change is deliberate."""
    t = lose(lose(clock(2000), 1000, 1), 1010, 1)
    r = scan.part_gap_losses(t)
    assert len(r.gaps) == 2 and all(g.frames == 2 for g in r.gaps)


def test_position_text():
    assert scan.position_text(12 * 60000 + 3400) == "12:03.4"
    assert scan.position_text(500) == "0:00.5"
    assert scan.position_text(3600000 + 62000) == "1:01:02.0"
