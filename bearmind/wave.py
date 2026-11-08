"""The calcium-wave hint of the inspection page.

A calcium wave (or a whole-field flash) makes many components rise in the same frames. A component is active
in a frame when its trace is above median + k_mad x 1.4826 x MAD of that trace (1.4826 x MAD estimates the SD
of normal noise and, unlike the SD, is not raised by the transients); a frame is a wave frame when at least
active_frac of the kept components are active in it. The inspection server shows the hint when the share of
wave frames is above inspection.wave.warn_frac. Nothing else reads it: Monitor and the stage checks do not.
"""
from __future__ import annotations

import math

import numpy as np

MAD_TO_SD = 1.4826
# Rows per block: the median of an hour-long session (107 107 frames) is taken on copies of this many traces at
# a time, so the hint does not double the memory of a prepared session.
_BLOCK_ROWS = 64


def wave_frac(C: np.ndarray, idx: np.ndarray, k_mad: float = 3.0, active_frac: float = 0.3) -> float:
    """Share of frames in which at least active_frac of the components in idx are above median + k_mad * 1.4826
    * MAD of their own trace. 0.0 without components or frames. A flat trace is never above its median, so it is
    never active; a trace that sits at one value in more than half of its frames (MAD 0, such as a sparse trace
    with an exact-zero baseline) is active in every frame above that value."""
    rows = np.asarray(idx, dtype=np.intp).ravel()
    T = C.shape[1] if np.ndim(C) == 2 else 0
    if rows.size == 0 or T == 0:
        return 0.0
    # a small margin: 0.07 x 100 is 7.000000000000001 in floating point, and "at least 7 % of 100" means 7
    need = max(1, math.ceil(active_frac * rows.size - 1e-9))
    n_active = np.zeros(T, dtype=np.int64)
    for i in range(0, rows.size, _BLOCK_ROWS):
        X = np.asarray(C[rows[i:i + _BLOCK_ROWS]], dtype=np.float64)
        med = np.median(X, axis=1, keepdims=True)
        mad = np.median(np.abs(X - med), axis=1, keepdims=True)
        n_active += np.count_nonzero(X > med + k_mad * MAD_TO_SD * mad, axis=0)
    return float(np.count_nonzero(n_active >= need)) / T


def _percent(x: float) -> str:
    p = 100.0 * x
    # one decimal, but never "0.0 %" for a share that is not zero
    return f"{p:.2g}" if 0 < p < 0.05 else f"{p:.1f}"


def wave_text(frac: float, active_frac: float) -> str:
    """The hint as the page shows it (English)."""
    return (f"Looks like a calcium wave: in {_percent(frac)} % of frames more than "
            f"{round(100.0 * active_frac, 1):g} % of the components fire at once.")
