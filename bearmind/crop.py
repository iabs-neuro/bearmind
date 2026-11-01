"""Rough automatic crop: find the lit field of view, keep one box size per mouse."""
import numpy as np

from bearmind.video import CropBox

MIN_CONTRAST = 5.0  # grey levels, uint8 scale


def detect_fov(frames: np.ndarray, min_fov_frac: float = 0.05, threshold_frac: float = 0.2):
    """(top, left, height, width) of the lit area of the mean frame, or None.

    Threshold sits at threshold_frac of the p1..p99 range of the smoothed mean image:
    Otsu would keep only the brightest part and cut dim but live tissue at the edges.
    """
    from scipy import ndimage  # here, not at the top: the runner imports this module for the box helpers

    mean = frames.astype(np.float32).mean(axis=0)
    smooth = ndimage.gaussian_filter(mean, sigma=5)
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
    return top, left, int(sl[0].stop - top), int(sl[1].stop - left)


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
