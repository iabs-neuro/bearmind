"""The calcium-wave hint of the inspection page.

A calcium wave (or a whole-field flash) makes many components rise in the same frames. A component is active
in a frame when its trace C is above median + k_mad x max(1.4826 x MAD, sn): 1.4826 x MAD estimates the SD of
normal noise and, unlike the SD, is not raised by the transients; sn, the component's noise SD, is a floor. A
denoised CNMF-E trace often sits at exactly one value (its zero baseline) in more than half of its frames; its
MAD is then 0, and without the floor every frame of every decay tail would count as active. A frame is a wave
frame when at least active_frac of the kept components are active in it. The inspection server shows the hint
when the share of wave frames is above inspection.wave.warn_frac. Nothing else reads it: Monitor and the stage
checks do not.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MAD_TO_SD = 1.4826
# A cached share of another version of the rule is computed again.
RULE_VERSION = 1
# Rows per block and blocks at a time: medians of an hour-long session (107 107 frames) are taken on copies of at
# most 64 traces, so the hint does not double the memory of a prepared session. numpy releases the GIL in its
# partitions and element-wise operations, so four threads cut the time of an hour-long session from 0.18 s to
# 0.08 s; a page opened without preparation computes the share before it is shown.
_BLOCK_ROWS = 16
_THREADS = 4


def _robust_sd(X: np.ndarray) -> np.ndarray:
    """1.4826 x MAD of each row of X (X is overwritten)."""
    med = np.median(X, axis=1, keepdims=True)
    np.subtract(X, med, out=X)
    np.abs(X, out=X)
    return MAD_TO_SD * np.median(X, axis=1, overwrite_input=True)


def _valid(sn: np.ndarray) -> np.ndarray:
    return np.isfinite(sn) & (sn > 0)


def component_noise(est) -> np.ndarray:
    """Noise SD of each component of est.C (one value per row): CaImAn's estimate est.neurons_sn where it is
    finite and positive; otherwise 1.4826 x MAD of the component's residual est.YrA (the raw trace minus C);
    NaN where neither exists (wave_frac then takes the noise from the trace's first difference)."""
    n = est.C.shape[0]
    out = np.full(n, np.nan)
    sn = getattr(est, "neurons_sn", None)
    if sn is not None:
        s = np.asarray(sn, dtype=np.float64).ravel()
        if s.size == n:  # a vector of another length belongs to other components
            out[:] = s
    missing = np.flatnonzero(~_valid(out))
    yra = getattr(est, "YrA", None)
    if missing.size and yra is not None and np.shape(yra) == est.C.shape:
        for i in range(0, missing.size, _BLOCK_ROWS):
            rows = missing[i:i + _BLOCK_ROWS]
            out[rows] = _robust_sd(np.array(yra[rows], dtype=np.float64))
    return out


def _active_counts(C, rows: np.ndarray, sn_rows: np.ndarray | None, k_mad: float) -> np.ndarray:
    """Per frame, how many of the components `rows` are active."""
    X = np.asarray(C[rows], dtype=np.float64)
    med = np.median(X, axis=1, keepdims=True)
    dev = np.subtract(X, med)
    np.abs(dev, out=dev)
    spread = MAD_TO_SD * np.median(dev, axis=1, overwrite_input=True)
    del dev
    noise = sn_rows.copy() if sn_rows is not None else np.full(rows.size, np.nan)
    bad = ~_valid(noise)
    if bad.any():
        noise[bad] = _robust_sd(np.diff(X[bad], axis=1)) / math.sqrt(2) if X.shape[1] > 1 else 0.0
    floor = np.maximum(spread, np.nan_to_num(noise, nan=0.0))
    return np.count_nonzero(X > med + k_mad * floor[:, None], axis=0)


def wave_frac(C: np.ndarray, idx: np.ndarray, k_mad: float = 3.0, active_frac: float = 0.3,
              sn: np.ndarray | None = None) -> float:
    """Share of frames in which at least active_frac of the components in idx are above median + k_mad *
    max(1.4826 * MAD, sn) of their own trace. 0.0 without components or frames.

    sn: noise SD per row of C (component_noise); where it is missing, not finite or not positive, the floor is
    1.4826 * MAD of the trace's first difference / sqrt(2) (the difference of white noise has sqrt(2) times its
    SD). A flat trace is never active. A noise-free trace without sn whose first difference is mostly 0 has no
    floor: then any frame above its median counts."""
    rows = np.asarray(idx, dtype=np.intp).ravel()
    T = C.shape[1] if np.ndim(C) == 2 else 0
    if sn is not None:
        sn = np.asarray(sn, dtype=np.float64).ravel()
        if sn.size != C.shape[0]:
            raise ValueError(f"sn has {sn.size} values for {C.shape[0]} components")
    if rows.size == 0 or T == 0:
        return 0.0
    # a small margin: 0.07 x 100 is 7.000000000000001 in floating point, and "at least 7 % of 100" means 7
    need = max(1, math.ceil(active_frac * rows.size - 1e-9))
    blocks = [rows[i:i + _BLOCK_ROWS] for i in range(0, rows.size, _BLOCK_ROWS)]

    def count(block):
        return _active_counts(C, block, sn[block] if sn is not None else None, k_mad)

    if len(blocks) == 1:
        counts = [count(blocks[0])]
    else:
        with ThreadPoolExecutor(min(_THREADS, len(blocks)), thread_name_prefix="bearmind-wave") as ex:
            counts = list(ex.map(count, blocks))
    n_active = np.sum(counts, axis=0, dtype=np.int64)
    return float(np.count_nonzero(n_active >= need)) / T


def _percent(x: float) -> str:
    p = 100.0 * x
    # one decimal, but never "0.0 %" for a share that is not zero
    return f"{p:.2g}" if 0 < p < 0.05 else f"{p:.1f}"


def wave_text(frac: float, active_frac: float) -> str:
    """The hint as the page shows it (English)."""
    return (f"Looks like a calcium wave: in {_percent(frac)} % of frames more than "
            f"{round(100.0 * active_frac, 1):g} % of the components fire at once.")
