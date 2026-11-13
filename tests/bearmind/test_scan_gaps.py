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
    assert g.interval_ms < 2.5 * T and g.certain  # certain by the excess (2 >= 1.5) alone
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


def run_session(*parts):
    return scan.session_gap_losses([np.asarray(p, float) for p in parts])


def jitter_event(t, j):
    t = np.array(t, float)
    t[j] += 1.05 * T
    t[j + 1] += 1.0 * T
    t[j + 2] += 0.5 * T
    return t


@pytest.mark.parametrize("n", [30, 61, 100, 130, 200, 500])
def test_one_late_frame_is_never_a_loss_whatever_the_part_length(n):
    """A late frame (long interval, then a short one) gives no loss: not in a short part with no period to measure,
    not in a long one."""
    r = scan.part_gap_losses(jitter_event(clock(n), n // 2))
    assert r.gaps == [] and not r.damaged


def test_jitter_every_50_intervals_chains_into_one_cluster_with_no_loss():
    t = clock(2000)
    for j in range(100, 1900, 50):
        t = jitter_event(t, j)
    r = scan.part_gap_losses(t)
    assert r.gaps == [] and r.n_jitter == 36


def test_period_is_pooled_over_the_session_a_short_part_borrows_it():
    """A 30-frame part has no interval outside its window; the long parts of the session (one camera clock) give
    the period, so its one lost frame is found."""
    short = lose(clock(30), 15, 1)
    for others in ([clock(2000), clock(2000)], [clock(2000)]):
        r = run_session(short, *others)
        assert [(g.part, g.frames, g.tested) for g in r.gaps] == [(0, 1, True)]
    r = run_session(clock(2000), short)
    assert [(g.part, g.frames) for g in r.gaps] == [(1, 1)]


def test_too_few_intervals_outside_the_gaps_skips_the_test_but_never_hides_a_pause():
    """A single short part with a 10 s pause: no period can be measured, the pause is reported untested."""
    t = clock(100)
    t[50:] += 10000.0
    r = scan.part_gap_losses(t)
    assert r.too_few and len(r.gaps) == 1
    g = r.gaps[0]
    assert g.tested is False and g.certain and g.frames == round(g.interval_ms / T - 1)
    assert g.frames == 298 and g.at_ms == pytest.approx(49 * T)
    # a small gap in the same part is below 2.5 m: nothing is said without a test
    assert scan.part_gap_losses(lose(clock(100), 50, 1)).gaps == []


def test_damaged_part_still_reports_a_long_pause_untested():
    t = clock(500)
    t[300:] += 10000.0
    t[400] = t[399] - 5.0
    r = scan.part_gap_losses(t)
    assert r.damaged and [(g.tested, g.frames) for g in r.gaps] == [(False, 298)]


def test_a_damaged_part_does_not_feed_the_pooled_period():
    bad = clock(2000)
    bad[10] = bad[9]
    r = run_session(bad, lose(clock(2000), 1000, 1))
    assert r.damaged == [0] and [(g.part, g.frames, g.tested) for g in r.gaps] == [(1, 1, True)]


def test_the_period_is_the_mean_not_the_median_of_the_outside_intervals():
    """Intervals 0.9 T, 0.9 T, 1.2 T repeating: mean T, median 0.9 T. A median period would count about 6 lost frames
    for the one frame that is lost."""
    steps = np.tile([0.9, 0.9, 1.2], 700) * T
    t = np.r_[0.0, np.cumsum(steps)]
    # frame 1001 is not recorded: its two intervals (0.9 T + 1.2 T = 2.1 T, above 2 m = 1.8 T) merge
    t = np.delete(t, 1001)
    g = scan.part_gap_losses(t).gaps
    assert [(x.frames, x.n_gaps) for x in g] == [(1, 1)]
    assert g[0].excess == pytest.approx(1.0, abs=0.3)


def test_interval_above_2_5_m_is_certain_with_excess_near_1_and_below_it_is_probable():
    for late, certain in ((0.6, True), (0.4, False)):
        t = lose(clock(2000), 1000, 1, extra=0.0)
        t[1000] += late * T  # the interval before it grows to (2 + late) T, the next shrinks: the clock goes on
        g = scan.part_gap_losses(t).gaps[0]
        assert g.excess == pytest.approx(1.0, abs=0.1) and g.frames == 1
        assert g.certain is certain, late


def test_gaps_60_intervals_apart_share_a_window_61_apart_do_not():
    """Windows are K = 30 intervals each side: gaps at distance <= 2K = 60 share an interval, so they merge."""
    def two(distance):
        t = lose(clock(2000), 1000, 1)
        return scan.part_gap_losses(lose(t, 1000 + distance, 1)).gaps
    assert [(g.n_gaps, g.frames) for g in two(60)] == [(2, 2)]
    assert [(g.n_gaps, g.frames) for g in two(61)] == [(1, 1), (1, 1)]


def test_position_text():
    assert scan.position_text(12 * 60000 + 3400) == "12:03.4"
    assert scan.position_text(500) == "0:00.5"
    assert scan.position_text(3600000 + 62000) == "1:01:02.0"


def test_a_short_session_without_a_gap_has_nothing_skipped():
    r = scan.part_gap_losses(clock(30))
    assert not r.too_few and r.gaps == []
