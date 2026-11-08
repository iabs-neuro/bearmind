"""The calcium-wave hint of the inspection page (bearmind.wave)."""
import numpy as np

from bearmind import wave


def _noise(n, T, seed=0):
    return np.random.default_rng(seed).normal(size=(n, T))


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
