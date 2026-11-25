"""Preserve peaks while limiting the number of trace samples sent to Bokeh."""
import numpy as np


def _minmax_positions(Y, n_out):
    """Return bin starts, time-ordered extrema, and the bin length for each trace."""
    k, n = Y.shape
    b = -(-n // max(n_out // 2, 1))
    nb = -(-n // b)
    pad = nb * b - n
    # Padding uses the last value. Argmin and argmax choose the first occurrence.
    Z = np.concatenate([Y, np.repeat(Y[:, -1:], pad, axis=1)], axis=1) if pad else Y
    Z = Z.reshape(k, nb, b)
    amin, amax = Z.argmin(axis=2), Z.argmax(axis=2)
    starts = (np.arange(nb) * b)[None, :]
    return starts, starts + np.minimum(amin, amax), starts + np.maximum(amin, amax), b


def decimate_minmax(y: np.ndarray, n_out: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Return the frame indices and values of the minimum and maximum of each display bin."""
    y = np.asarray(y)
    if len(y) <= n_out:
        return np.arange(len(y)), y
    _, first, second, _ = _minmax_positions(y[None, :], n_out)
    frames = np.stack([first[0], second[0]], axis=1).ravel()
    return frames, y[frames]
