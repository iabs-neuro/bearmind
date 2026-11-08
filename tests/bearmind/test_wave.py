"""The calcium-wave hint of the inspection page (bearmind.wave)."""
from types import SimpleNamespace

import numpy as np
import pytest

from bearmind import wave


def _noise(n, T, seed=0):
    return np.random.default_rng(seed).normal(size=(n, T))


def _robust_sd(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _event_trace(T=1000, at=800, amp=10.0, tau=20.0):
    """A denoised trace as CNMF-E leaves it: an exact-zero baseline and one event that decays to the end. More
    than half of its frames are 0, so its MAD is 0."""
    c = np.zeros(T)
    c[at:] = amp * np.exp(-np.arange(T - at) / tau)
    return c


def test_mad0_trace_active_only_near_its_event():
    c = _event_trace()[None, :]
    assert np.median(np.abs(c - np.median(c))) == 0
    near = int(np.count_nonzero(c[0] > 3.0 * 0.5))  # above 3 noise SDs of 0.5: the first frames of the event
    assert near == 38
    assert wave.wave_frac(c, [0], sn=[0.5]) == near / 1000  # one component: its share of active frames
    # the noise from the residual when CaImAn's estimate is missing
    est = SimpleNamespace(C=c, YrA=0.5 * _noise(1, 1000, seed=2), neurons_sn=np.array([np.nan]))
    sn = wave.component_noise(est)
    assert sn[0] == pytest.approx(0.5, rel=0.1)
    assert wave.wave_frac(c, [0], sn=sn) == np.count_nonzero(c[0] > 3.0 * sn[0]) / 1000
    # no noise estimate at all and a noise-free trace (its first difference has MAD 0 too): no floor, so the
    # whole decay tail counts; CaImAn's estimates always carry neurons_sn and YrA
    assert wave.wave_frac(c, [0]) == 200 / 1000


def test_clipped_noise_is_not_a_wave():
    """MAD 0 by clipping: 58 % of the frames exactly 0, the rest noise. Without a floor every positive frame was
    active, and 42 % of 50 such traces made nearly every frame a wave frame."""
    C = np.maximum(_noise(50, 1000) - 0.2, 0.0)
    assert np.all(np.median(np.abs(C - np.median(C, axis=1, keepdims=True)), axis=1) == 0)
    assert wave.wave_frac(C, np.arange(50), sn=np.ones(50)) == 0.0  # floor from the noise estimate
    # floor from the first difference: it underestimates the SD of clipped noise (pairs of zeros differ by 0), so
    # a few frames are left, against 97 % without a floor
    diff_floor = wave.wave_frac(C, np.arange(50))
    assert 0 < diff_floor < 0.01
    assert wave.wave_frac(C, np.arange(50), sn=np.full(50, np.nan)) == diff_floor  # not finite: the difference
    assert wave.wave_frac(C, np.arange(50), sn=np.zeros(50)) == diff_floor  # not positive: the difference


def test_noise_floor_and_spread():
    """The threshold is median + k_mad x max(1.4826 x MAD, sn): a large sn raises it, a small one does not lower
    it below the trace's own spread."""
    T = 1000
    C = _noise(50, T)
    frames = np.arange(100, 1000, 90)[:10]
    C[:, frames] += 20.0
    assert wave.wave_frac(C, np.arange(50), sn=np.full(50, 0.01)) == 10 / T  # the spread (about 1) rules
    assert wave.wave_frac(C, np.arange(50), sn=np.full(50, 10.0)) == 0.0  # 20 is below 3 x 10
    with pytest.raises(ValueError):
        wave.wave_frac(C, np.arange(50), sn=np.ones(49))  # one value per row of C


def test_component_noise_sources():
    C = np.zeros((3, 500))
    yra = 0.5 * _noise(3, 500, seed=3)
    sn = wave.component_noise(SimpleNamespace(C=C, YrA=yra, neurons_sn=np.array([0.2, np.nan, 0.0])))
    assert sn[0] == 0.2  # CaImAn's estimate
    for i in (1, 2):  # not finite, not positive: the robust SD of the residual
        assert sn[i] == pytest.approx(_robust_sd(yra[i]))
    other = wave.component_noise(SimpleNamespace(C=C, YrA=yra, neurons_sn=np.ones(2)))  # not one per component
    assert np.allclose(other, [_robust_sd(r) for r in yra])
    assert np.isnan(wave.component_noise(SimpleNamespace(C=C))).all()  # nothing to estimate it from


def test_wave_frac_synthetic():
    T, n = 1000, 50
    C = _noise(n, T)
    frames = np.arange(100, 1000, 90)[:10]
    C[:, frames] += 20.0  # all 50 fire together in these 10 frames, far above 3 robust SDs
    assert wave.wave_frac(C, np.arange(n)) == 10 / T
    # only the components in idx count: with 150 that never fire, 50 of 200 (25 %) fire in those frames
    both = np.vstack([C, _noise(150, T, seed=1)])
    assert wave.wave_frac(both, np.arange(200)) == 0.0
    assert wave.wave_frac(both, np.arange(n)) == 10 / T
    assert wave.wave_frac(both, np.arange(200), active_frac=0.25) == 10 / T  # at least active_frac
    # a frame where fewer than active_frac fire is not a wave frame: 14 of 50 is below 15
    part = C.copy()
    part[:36, frames[0]] = 0.0
    assert wave.wave_frac(part, np.arange(n)) == 9 / T
    # idx as CaImAn stores it: a list
    assert wave.wave_frac(C, list(range(n))) == 10 / T


def test_wave_frac_flat_traces():
    T = 200
    C = np.ones((20, T))
    C[5] = 0.0
    with np.errstate(all="raise"):
        assert wave.wave_frac(C, np.arange(20)) == 0.0
    assert wave.wave_frac(C, np.array([], dtype=int)) == 0.0  # no kept components
    assert wave.wave_frac(np.zeros((3, 0)), np.arange(3)) == 0.0  # no frames


def test_wave_text_english():
    text = wave.wave_text(0.012, 0.3)
    assert text == "Looks like a calcium wave: in 1.2 % of frames more than 30 % of the components fire at once."
    assert text.isascii()
    assert wave.wave_text(0.25, 0.25) == ("Looks like a calcium wave: in 25.0 % of frames more than 25 % of the "
                                          "components fire at once.")
