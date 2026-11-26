"""Detect stripe bursts on original frames and apply a reviewed handling policy before motion correction."""
from collections import deque
from itertools import chain
import math

import cv2
import numpy as np
from scipy.ndimage import median_filter


REL_MAX = 1.5
VZ_MAX = 20.0
VZ_WINDOW = 301
DEFAULT_ACTION = "auto"
FRAME_REPAIR_RULE = "mirror_horizontal_edges_with_padding_v3"


def normalize(ranges, n_frames, *, allow_all=False):
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
    if not allow_all and merged == [(0, n_frames - 1)]:
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


def _kept_frames(frames, drop):
    removals = deque(drop)
    for i, image in enumerate(frames):
        if removals and i >= removals[0][0]:
            if i == removals[0][1]:
                removals.popleft()
        else:
            yield i, image


def _reflected_index(distance, count):
    if count == 1:
        return 0
    period = 2 * (count - 1)
    pos = distance % period
    return min(pos, period - pos)


def filter_frames(frames, interpolate=(), drop=()):
    """Reflect a missing prefix and interpolate interior gaps between retained frames."""
    inserts = deque(interpolate)
    stream = _kept_frames(frames, drop)
    previous = None
    while (item := next(stream, None)) is not None:
        i, image = item
        if inserts and inserts[0][0] <= i <= inserts[0][1]:
            continue
        if inserts and i > inserts[0][1]:
            a, b = inserts.popleft()
            size = b - a + 1
            if previous is None:
                good = [image]
                for _ in range(size):
                    following = next(stream, None)
                    if following is None:
                        break
                    if inserts and following[0] >= inserts[0][0]:
                        stream = chain((following,), stream)
                        break
                    good.append(following[1])
                for distance in range(size, 0, -1):
                    yield good[_reflected_index(distance, len(good))]
                yield from good
                previous = good[-1]
                continue
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
    trim_from = cfg.get("bad_frames_trim_tail_from")
    if trim_from is not None and (type(trim_from) is not int or not 0 < trim_from < n_frames):
        raise ValueError(f"bad_frames_trim_tail_from must be a frame index from 1 to {n_frames - 1}.")
    return action, duration, enabled, manual, trim_from


def duration_s(a, b, timestamps, fps):
    if timestamps is not None and b + 1 < len(timestamps) and timestamps[b + 1] > timestamps[a]:
        return float(timestamps[b + 1] - timestamps[a])
    if timestamps is not None and b > a and b < len(timestamps) and timestamps[b] > timestamps[a]:
        return float(timestamps[b] - timestamps[a]) + 1.0 / fps
    return (b - a + 1) / fps


def repair_plan(suspects, manual, action, limit, timestamps, fps, n_frames, *, trim_from=None):
    """Plan repairs in source frame indices, with an explicit tail trim taking precedence."""
    all_ranges = normalize([*ranges_from_indices(suspects), *manual], n_frames, allow_all=True)
    end = trim_from if trim_from is not None else n_frames
    plan = []
    for a, b in all_ranges:
        if a >= end:
            break
        b = min(b, end - 1)
        seconds = duration_s(a, b, timestamps, fps)
        effect = action if action != "auto" else ("interpolate" if seconds <= limit else "report")
        if a == 0 and b == end - 1:
            if effect == "drop":
                raise ValueError("At least one frame must remain outside the bad frame ranges.")
            effect = "report"  # no intact frame to interpolate from
        plan.append({"first": a, "last": b, "duration_s": round(seconds, 6), "action": effect})
    if trim_from is not None:
        seconds = duration_s(trim_from, n_frames - 1, timestamps, fps)
        plan.append({"first": trim_from, "last": n_frames - 1, "duration_s": round(seconds, 6), "action": "drop"})
    return plan


def pad_interpolation(plan, timestamps, fps, n_frames):
    """Extend interpolation by one frame on either side, within the retained video."""
    ordered = sorted(plan, key=lambda e: e["first"])
    padded = []
    for i, event in enumerate(ordered):
        first, last, action = event["first"], event["last"], event["action"]
        if action == "interpolate":
            previous = ordered[i - 1] if i else None
            following = ordered[i + 1] if i + 1 < len(ordered) else None
            low = previous["last"] + 1 if previous and previous["action"] != action else 0
            high = following["first"] - 1 if following and following["action"] != action else n_frames - 1
            first, last = max(low, first - 1), min(high, last + 1)
        if padded and action == "interpolate" and padded[-1][2] == action and first <= padded[-1][1] + 1:
            first, last = padded.pop()[0], last
        padded.append((first, last, action))
    unavailable = sum(last - first + 1 for first, last, action in padded if action != "report")
    if unavailable >= n_frames:  # interpolation requires an intact reference frame
        return ordered
    return [{"first": a, "last": b, "duration_s": round(duration_s(a, b, timestamps, fps), 6), "action": action}
            for a, b, action in padded]


def _horizontal_edges(frame):
    """Count alternating horizontal edges shared by the central image columns."""
    h, w = frame.shape
    if min(h, w) < 16:
        return False
    center = frame[h // 10:h - h // 10, w // 5:w - w // 5:3]
    profiles = [strip.mean(axis=1, dtype=np.float32) for strip in np.array_split(center, 3, axis=1)]
    slope = np.median(np.diff(profiles, axis=1), axis=0)
    strength = np.abs(slope)
    strong = strength > max(1.5, 4 * np.median(strength))
    signs = np.signbit(slope[strong])
    return np.count_nonzero(signs[1:] != signs[:-1]) >= 5


class StripeMeter:
    def __init__(self):
        self._ratio = []
        self._peak = []
        self._horizontal = []
        self._brightness = []

    def add(self, frame):
        self._horizontal.append(_horizontal_edges(frame))
        self._brightness.append(cv2.mean(frame)[0])
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
        marked = (((ratios > REL_MAX * base) & (base > 0))
                  | (residual > VZ_MAX * max(float(scatter), .001))
                  | np.asarray(self._horizontal))
        levels = np.asarray(self._brightness)
        if np.any(levels > 0):
            illuminated = np.flatnonzero(levels >= 0.1 * np.median(levels[levels > 0]))
            marked[:illuminated[0]] = True
        return np.flatnonzero(marked).tolist()
