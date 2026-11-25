"""Detect stripe bursts on original frames and apply a reviewed handling policy before motion correction."""
from collections import deque
import math

import cv2
import numpy as np
from scipy.ndimage import median_filter


REL_MAX = 1.5
VZ_MAX = 20.0
VZ_WINDOW = 301
DEFAULT_ACTION = "auto"


def normalize(ranges, n_frames):
    pairs = []
    for span in ranges:
        if not isinstance(span, (list, tuple)) or len(span) != 2 or any(type(i) is not int for i in span):
            raise ValueError("bad_frames must contain pairs of integer frame indices.")
        a, b = span
        if not 0 <= a <= b < n_frames:
            raise ValueError(f"Bad frame range {a}-{b} is outside 0-{n_frames - 1} or reversed.")
        pairs.append((a, b))
    merged = []
    for a, b in sorted(pairs):
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    if merged == [(0, n_frames - 1)]:
        raise ValueError("At least one frame must remain outside the bad frame ranges.")
    return merged


def ranges_from_indices(indices):
    out = []
    for i in sorted(set(indices)):
        if out and out[-1][1] + 1 == i:
            out[-1] = out[-1][0], int(i)
        else:
            out.append((int(i), int(i)))
    return out


def parse(text):
    result = []
    for token in text:
        ends = token.split("-")
        if len(ends) > 2 or not all(part.isascii() and part.isdigit() for part in ends):
            raise ValueError(f"Invalid frame range {token!r}; use an index or FIRST-LAST.")
        result.append((int(ends[0]), int(ends[-1])))
    return result


def label(ranges):
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def frame_mask(n_frames, ranges):
    keep = np.ones(n_frames, dtype=bool)
    for a, b in normalize(ranges, n_frames):
        keep[a:b + 1] = False
    return keep


def filter_frames(frames, interpolate=(), drop=()):
    """Stream corrected frames. Bad frames need no storage, even for a long run."""
    inserts = deque(interpolate)
    removals = deque(drop)
    previous = None
    for i, image in enumerate(frames):
        if removals and i >= removals[0][0]:
            if i == removals[0][1]:
                removals.popleft()
            continue
        if inserts and inserts[0][0] <= i <= inserts[0][1]:
            continue
        if inserts and i > inserts[0][1]:
            a, b = inserts.popleft()
            size = b - a + 1
            if previous is None:
                for _ in range(size):
                    yield image
            else:
                before = previous.astype(np.float32)
                delta = image.astype(np.float32) - before
                for k in range(1, size + 1):
                    yield np.rint(before + delta * (k / (size + 1))).astype(np.uint8)
        yield image
        previous = image
    if inserts and previous is not None:
        a, b = inserts.popleft()
        for _ in range(b - a + 1):
            yield previous


def settings(cfg, n_frames):
    action = cfg.get("bad_frames_action", DEFAULT_ACTION)
    if action not in ("auto", "interpolate", "drop", "report"):
        raise ValueError("bad_frames_action must be auto, interpolate, drop, or report.")
    duration = cfg.get("bad_frames_max_interpolate_s", 1.0)
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("bad_frames_max_interpolate_s must be a positive number.")
    enabled = cfg.get("bad_frames_auto", True)
    if type(enabled) is not bool:
        raise ValueError("bad_frames_auto must be true or false.")
    manual = normalize(cfg.get("bad_frames") or (), n_frames)
    return action, duration, enabled, manual


def duration_s(a, b, timestamps, fps):
    if timestamps is not None and b + 1 < len(timestamps) and timestamps[b + 1] > timestamps[a]:
        return float(timestamps[b + 1] - timestamps[a])
    if timestamps is not None and b > a and b < len(timestamps) and timestamps[b] > timestamps[a]:
        return float(timestamps[b] - timestamps[a]) + 1.0 / fps
    return (b - a + 1) / fps


def repair_plan(suspects, manual, action, limit, timestamps, fps, n_frames):
    """Measured bursts with their action. Auto interpolates short bursts; long ones need review."""
    all_ranges = normalize([*ranges_from_indices(suspects), *manual], n_frames)
    plan = []
    for a, b in all_ranges:
        seconds = duration_s(a, b, timestamps, fps)
        effect = action if action != "auto" else ("interpolate" if seconds <= limit else "report")
        plan.append({"first": a, "last": b, "duration_s": round(seconds, 6), "action": effect})
    return plan


class StripeMeter:
    def __init__(self):
        self._ratio = []
        self._peak = []

    def add(self, frame):
        if min(frame.shape) < 2:
            self._ratio.append(0.0)
            self._peak.append(0.0)
            return
        vertical = cv2.absdiff(frame[1:], frame[:-1])
        horizontal = cv2.absdiff(frame[:, 1:], frame[:, :-1])
        rows = vertical.mean(axis=1)
        self._ratio.append(float(rows.mean()) / max(float(horizontal.mean()), .001))
        self._peak.append(float(rows.max()))

    def suspects(self):
        if not self._ratio:
            return []
        ratios = np.asarray(self._ratio)
        peaks = np.asarray(self._peak)
        base = np.median(ratios[ratios > 0]) if np.any(ratios > 0) else 0.0
        local = median_filter(peaks, size=min(VZ_WINDOW, len(peaks) // 2 * 2 + 1), mode="nearest")
        residual = peaks - local
        scatter = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        return np.flatnonzero(((ratios > REL_MAX * base) & (base > 0))
                              | (residual > VZ_MAX * max(float(scatter), .001))).tolist()
