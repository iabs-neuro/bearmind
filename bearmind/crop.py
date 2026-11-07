"""Automatic crop: the field of view with calcium activity inside the lit area, one box size per mouse."""
from dataclasses import dataclass

import numpy as np

from bearmind.video import CropBox

# Version of the field rule below (its code and constants): a stage parameter of crop, so a change re-runs crop
# on processed series.
FOV_RULE = "activity_v2"
MIN_CONTRAST = 5.0  # grey levels, uint8 scale
RIM_PX = 10  # the lens rim carries edge artefacts in the seed maps
ACTIVITY_SIGMA_PX = 10  # smoothing of the activity map: a few cell diameters, so the field follows tissue, not cells
MIN_FRAC_OF_LIT = 0.25  # a smaller activity field is not trusted: the lit area is used
# Positive over negative mass of the activity map: noise alone is symmetric around 0 (0.2-3.1 on synthetic
# noise and hot-spot movies, 3.2 on the weak real session LT1_K02_1D; 39-572 on six active real sessions).
# Below it the transients are not told from noise and the lit area is used, so noise never shapes the field;
# set in the middle of that gap, since falling back to the lit area is the safe direction.
MIN_POS_NEG = 8.0
# Seeds this strong are kept inside the field whatever the activity map says: the sign-inverted movie of a very
# active cell has transients of its own and a ring from the CNMF-E filter, which can cancel the cell's excess
# after smoothing (an isolated strong cell beyond the active area would be cut).
STRONG_SEED_CORR, STRONG_SEED_PNR = 0.95, 15.0


def lit_fov(mean: np.ndarray, min_fov_frac: float = 0.05, threshold_frac: float = 0.2):
    """((top, left, height, width), mask) of the largest lit component of the mean frame, or None.

    Threshold sits at threshold_frac of the p1..p99 range of the smoothed mean image:
    Otsu would keep only the brightest part and cut dim but live tissue at the edges.
    """
    from scipy import ndimage  # here, not at the top: the runner imports this module for the box helpers

    smooth = ndimage.gaussian_filter(np.asarray(mean, dtype=np.float32), sigma=5)
    p1, p99 = np.percentile(smooth, [1, 99])
    # Range under a few grey levels is noise only (also covers a constant image).
    if float(p99 - p1) < MIN_CONTRAST:
        return None
    lab, n = ndimage.label(smooth > p1 + threshold_frac * (p99 - p1))
    if n == 0:
        return None
    sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < min_fov_frac * lab.size:
        return None
    sl = ndimage.find_objects(lab)[k - 1]
    top, left = int(sl[0].start), int(sl[1].start)
    return (top, left, int(sl[0].stop - top), int(sl[1].stop - left)), lab == k


def detect_fov(frames: np.ndarray, min_fov_frac: float = 0.05, threshold_frac: float = 0.2):
    """(top, left, height, width) of the lit area of the mean frame, or None (the brightness rule)."""
    found = lit_fov(frames.astype(np.float32).mean(axis=0), min_fov_frac, threshold_frac)
    return None if found is None else found[0]


def lit_inner(lit_mask: np.ndarray) -> np.ndarray:
    """The lit mask eroded by RIM_PX (Euclidean); the frame edge counts as an edge too."""
    from scipy import ndimage

    return ndimage.distance_transform_edt(np.pad(np.asarray(lit_mask, bool), 1))[1:-1, 1:-1] > RIM_PX


def activity_map(maps: dict, lit_mask: np.ndarray) -> np.ndarray:
    """Where calcium transients are: the CNMF-E seed score cn * pnr over its value on the sign-inverted movie.

    Transients are positive, so the inverted movie gives the score of noise alone at every pixel. The excess
    is smoothed by ACTIVITY_SIGMA_PX, set to 0 outside lit_inner(lit_mask), and clipped from above at the
    99th percentile of its positive values inside: one hot spot cannot carry the mass that places the field.
    """
    from scipy import ndimage

    excess = (maps["cn"] * maps["pnr"] - maps["cn_neg"] * maps["pnr_neg"]).astype(np.float32)
    e = ndimage.gaussian_filter(excess, ACTIVITY_SIGMA_PX)
    e[~lit_inner(lit_mask)] = 0
    pos = e[e > 0]
    if pos.size:
        np.minimum(e, np.percentile(pos, 99), out=e)
    return e


def pos_neg_ratio(activity: np.ndarray) -> float:
    """Positive over negative mass of the activity map; 0 without positive mass, capped at 1e6."""
    a = np.asarray(activity, dtype=np.float64)
    pos, neg = float(np.maximum(a, 0).sum()), float(np.maximum(-a, 0).sum())
    if pos <= 0:
        return 0.0
    return min(pos / neg, 1e6) if neg > 0 else 1e6


def strong_seeds(maps: dict, inner: np.ndarray, gsig: int) -> np.ndarray:
    """(n, 2) int32 (y, x): CNMF-E seed candidates at STRONG_SEED_CORR / STRONG_SEED_PNR inside `inner`, the
    strongest within 2 * gSig (about one per cell)."""
    import bm_qc  # CaImAn: only the stage process and the tests that compute maps get here

    pts = bm_qc.count_seeds(maps["cn"], maps["pnr"], STRONG_SEED_CORR, STRONG_SEED_PNR, 4 * gsig + 1,
                            valid=inner, min_dist=2 * gsig)
    return np.asarray(pts, dtype=np.int32).reshape(-1, 2)


def activity_fov(activity: np.ndarray, lit_box, mass_cut: float, margin: int, protect=None):
    """(top, left, height, width) leaving at most `mass_cut` of the positive activity outside on each side,
    widened by `margin`, then widened to cover each (y, x) of `protect` +/- `margin`, and clipped to `lit_box`.

    None (the lit area is the field) when there is no activity, when its positive mass is under MIN_POS_NEG
    times its negative mass (noise), or when the activity field before `protect` is smaller than
    MIN_FRAC_OF_LIT of the lit box. `protect` only ever grows a field.
    """
    a = np.asarray(activity, dtype=np.float64)
    if pos_neg_ratio(a) < MIN_POS_NEG:
        return None
    w = np.maximum(a, 0)
    spans, limits = [], []
    for marg, start, length in ((w.sum(1), lit_box[0], lit_box[2]), (w.sum(0), lit_box[1], lit_box[3])):
        c = np.cumsum(marg)
        c /= c[-1]
        lo = int(np.searchsorted(c, mass_cut))  # first index with a cumulative share >= mass_cut
        hi = min(int(np.searchsorted(c, 1 - mass_cut)), len(c) - 1)
        spans.append([lo - int(margin), hi + int(margin)])
        limits.append((int(start), int(start + length) - 1))

    def clipped():
        return [(max(lo, first), min(hi, last)) for (lo, hi), (first, last) in zip(spans, limits)]

    (top, bottom), (left, right) = clipped()
    h, wd = bottom - top + 1, right - left + 1
    if h <= 0 or wd <= 0 or h * wd < MIN_FRAC_OF_LIT * lit_box[2] * lit_box[3]:
        return None
    for point in (() if protect is None else protect):
        for span, v in zip(spans, point):
            span[0], span[1] = min(span[0], int(v) - int(margin)), max(span[1], int(v) + int(margin))
    (top, bottom), (left, right) = clipped()
    return int(top), int(left), int(bottom - top + 1), int(right - left + 1)


@dataclass
class ActivityField:
    activity: np.ndarray  # activity_map
    field: tuple | None   # activity_fov with the strong seeds protected; None: the lit area is the field
    reason: str           # activity | fallback_pos_neg | fallback_small
    pos_neg: float        # pos_neg_ratio(activity)
    seeds: np.ndarray     # strong_seeds: (n, 2) y, x


def activity_field(maps: dict, lit_box, lit_mask: np.ndarray, gsig: int, mass_cut: float,
                   margin: int) -> ActivityField:
    """The activity field of a session from its seed maps (computed with mirror=True), and why it is what it is."""
    activity = activity_map(maps, lit_mask)
    ratio = pos_neg_ratio(activity)
    seeds = strong_seeds(maps, lit_inner(lit_mask), gsig)
    if ratio < MIN_POS_NEG:
        return ActivityField(activity, None, "fallback_pos_neg", ratio, seeds)
    field = activity_fov(activity, lit_box, mass_cut, margin, protect=seeds)
    return ActivityField(activity, field, "activity" if field is not None else "fallback_small", ratio, seeds)


def mouse_box_size(fovs, frame_shape, margin: int) -> tuple[int, int]:
    """Element-wise max (h, w) over sessions plus margin on both sides, capped at the frame."""
    h = max(f[2] for f in fovs) + 2 * margin
    w = max(f[3] for f in fovs) + 2 * margin
    return min(h, int(frame_shape[0])), min(w, int(frame_shape[1]))


def fitted_size(size, frame_shape) -> list[int]:
    """(h, w) of a box of `size` as place_box puts it into a frame of `frame_shape`."""
    return [min(int(size[0]), int(frame_shape[0])), min(int(size[1]), int(frame_shape[1]))]


def mouse_frame_shape(metrics) -> tuple[int, int] | None:
    """Smallest frame over the sessions' crop metrics (the frame every box of the mouse must fit)."""
    shapes = [m["frame_shape"] for m in metrics if m.get("frame_shape")]
    if not shapes:
        return None
    return min(int(s[0]) for s in shapes), min(int(s[1]) for s in shapes)


def place_box(fov, size, frame_shape) -> CropBox:
    """Box of fixed size centred on the FOV centre, shifted to lie inside the frame."""
    out = []
    for start, length, box, full in ((fov[0], fov[2], size[0], frame_shape[0]),
                                     (fov[1], fov[3], size[1], frame_shape[1])):
        box = min(int(box), int(full))
        pos = int(round(start + length / 2 - box / 2))
        out.append((max(0, min(pos, int(full) - box)), box))
    return CropBox(out[0][0], out[1][0], out[0][1], out[1][1])
