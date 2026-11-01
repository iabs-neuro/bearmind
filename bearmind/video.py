"""Video I/O: session raw chunks as one frame stream, FFV1/mp4 writers, array view over an AVI."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Iterator

import av
import numpy as np

from .registry import SessionRecord


@dataclass
class CropBox:
    top: int
    left: int
    height: int
    width: int


def _grey(frame) -> np.ndarray:
    """uint8 2-D grey frame: the Y (luma) plane, the grey signal the camera wrote (PyAV format "gray").

    Legacy UCLA files (FFV1 gray): the same values as the old moviepy path, which kept channel 0 of an rgb24
    decode. Bonsai files (mpeg4 yuv420p): the old path is 2-3 grey levels lower in every pixel, because the
    rgb24 conversion mixes the chroma into channel 0 (Task 18, NOF_H01_1D: +2.5 on average).
    """
    return frame.to_ndarray(format="gray")


def _iter_avi(path: Path, threads: bool = True) -> Iterator[np.ndarray]:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        if threads:
            s.thread_type = "AUTO"  # frame threading: ~3x faster FFV1 decode, same frames
        for fr in c.decode(s):
            yield _grey(fr)


def _head_frames(path: Path, n: int, broken_tail: bool) -> Iterator[np.ndarray]:
    """The first n frames of an AVI; fewer decoded frames is an error, since later frames would shift.

    With `broken_tail` the frame after them is unreadable. Decoding without frame threading then outputs
    each frame before the next packet is read, so stopping at n never sends the broken packet to the
    decoder (it raises InvalidDataError there, or a threaded decoder drops it silently).
    """
    if n <= 0:
        return
    k = 0
    for fr in _iter_avi(path, threads=not broken_tail):
        yield fr
        k += 1
        if k == n:
            return
    raise ValueError(f"{path}: decoded {k} of {n} frames")


def decode_to_memmap(path: Path, out: Path, block: int = 1000) -> np.memmap:
    """Whole AVI as a uint8 C-order (T, h, w) file at `out`, decoded in one sequential pass; returned read-only.

    Stages that read a movie many times in strips use this instead of VideoArray, which re-decodes per request.
    """
    path, out = Path(path), Path(out)
    n, h, w, _ = video_info(path)
    try:
        k = 0
        with open(out, "wb") as f:  # sequential writes: np.memmap('w+') would zero-fill the whole file first
            buf: list[np.ndarray] = []
            for fr in _iter_avi(path):
                buf.append(fr)
                if len(buf) == block:
                    f.write(np.stack(buf).tobytes())
                    k += len(buf)
                    buf = []
            if buf:
                f.write(np.stack(buf).tobytes())
                k += len(buf)
        if k != n:
            raise ValueError(f"{path}: decoded {k} frames, the container reports {n}")
        return np.memmap(out, dtype=np.uint8, mode="r", shape=(n, h, w))
    except BaseException:
        out.unlink(missing_ok=True)
        raise


def video_info(path: Path) -> tuple[int, int, int, float]:
    """(n_frames, h, w, fps). Counts packets when the container header has no frame count."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        h, w = s.codec_context.height, s.codec_context.width
        rate = s.average_rate or s.guessed_rate
        fps = float(rate) if rate else float("nan")
        n = s.frames
        if not n:
            n = sum(1 for p in c.demux(s) if p.size)
    return int(n), int(h), int(w), fps


def last_frame_readable(path: Path) -> bool:
    """True iff the last frame of the file decodes (detects files truncated mid-write)."""
    try:
        n = video_info(path)[0]
        if n < 1:
            return False
        VideoArray(path)[n - 1]
        return True
    except Exception:
        return False


class SessionReader:
    """A session's raw chunks as one uint8 (n, h, w) stream, with the registry's drop rules and optional crop."""

    def __init__(self, series_root: Path, rec: SessionRecord, crop: CropBox | None = None):
        self.root = Path(series_root)
        self.rec = rec
        self.crop = crop
        h, w = rec.parts[0].shape
        if crop is not None:
            h, w = crop.height, crop.width
        self.n_frames = int(rec.n_frames)
        self.shape = (self.n_frames, h, w)

    def _iter_frames(self) -> Iterator[np.ndarray]:
        left = self.n_frames
        for part in self.rec.parts:
            # a part stops at its own frame count, so a surplus at its end cannot shift the next part
            part_left = int(part.n_frames)
            last = len(part.chunks) - 1
            for i, name in enumerate(part.chunks):
                if left <= 0:
                    return
                if part_left <= 0:
                    break
                path = self.root / part.folder / name
                # registries scanned before chunk_frames was recorded: ask the container
                count = part.chunk_frames[i] if part.chunk_frames else video_info(path)[0]
                broken = i == last and part.drop_last_frame
                drop = broken or (i < last and part.drop_chunk_tail)
                n = min(count - 1 if drop else count, part_left, left)
                for fr in _head_frames(path, n, broken):
                    yield fr
                part_left -= n
                left -= n

    def _crop(self, fr: np.ndarray) -> np.ndarray:
        c = self.crop
        return fr if c is None else fr[c.top:c.top + c.height, c.left:c.left + c.width]

    def iter_chunks(self, chunk: int = 1000) -> Iterator[np.ndarray]:
        buf: list[np.ndarray] = []
        for fr in self._iter_frames():
            buf.append(self._crop(fr))
            if len(buf) == chunk:
                yield np.stack(buf)
                buf = []
        if buf:
            yield np.stack(buf)

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """n frames at random positions (in stream order), one pass over the stream."""
        n = min(n, self.n_frames)
        idx = set(np.random.default_rng(seed).choice(self.n_frames, n, replace=False).tolist())
        out = []
        for i, fr in enumerate(self._iter_frames()):
            if i in idx:
                out.append(self._crop(fr))
                if len(out) == n:
                    break
        return np.stack(out) if out else np.empty((0,) + self.shape[1:], np.uint8)


def _rate(fps: float) -> Fraction:
    return Fraction(fps).limit_denominator(1000)


def write_ffv1(chunks: Iterable[np.ndarray], path: Path, fps: float) -> int:
    """Lossless grey FFV1 in AVI. Writes <stem>.partial.avi and renames on success."""
    path = Path(path)
    tmp = path.with_suffix(".partial.avi")
    n = 0
    try:
        with av.open(str(tmp), "w", format="avi") as c:
            s = None
            for block in chunks:
                for f in block:
                    if s is None:
                        s = c.add_stream("ffv1", rate=_rate(fps))
                        s.height, s.width = f.shape
                        s.pix_fmt = "gray"
                    vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="gray")
                    for p in s.encode(vf):
                        c.mux(p)
                    n += 1
            if s is None:
                raise ValueError("no frames to write")
            for p in s.encode():
                c.mux(p)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return n


def write_preview_mp4(chunks: Iterable[np.ndarray], path: Path, fps: float, crf: int = 23,
                      speed: int = 4, tbin: int = 2) -> None:
    """H.264 preview. Each output frame is the mean of `tbin` input frames; playback is `speed` x faster."""
    path = Path(path)
    tmp = path.with_suffix(".partial.mp4")
    try:
        _write_mp4(chunks, tmp, fps, crf, speed, tbin)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_mp4(chunks, path, fps, crf, speed, tbin) -> None:
    out_rate = _rate(fps * speed / tbin)
    with av.open(str(path), "w", format="mp4") as c:
        s = None

        def emit(mean: np.ndarray) -> None:
            nonlocal s
            h, w = mean.shape
            if s is None:
                s = c.add_stream("libx264", rate=out_rate)
                s.pix_fmt = "yuv420p"
                s.options = {"crf": str(crf)}
                s.height, s.width = h - h % 2, w - w % 2  # yuv420p needs even dims
            img = np.rint(mean).astype(np.uint8)[:s.height, :s.width]
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="gray").reformat(format="yuv420p")
            for p in s.encode(vf):
                c.mux(p)

        acc, k = None, 0
        for block in chunks:
            for f in block:
                acc = f.astype(np.float32) if acc is None else acc + f
                k += 1
                if k == tbin:
                    emit(acc / k)
                    acc, k = None, 0
        if k:
            emit(acc / k)
        if s is not None:
            for p in s.encode():
                c.mux(p)


class VideoArray:
    """Read-only (T, h, w) uint8 view over an AVI; int and time-slice access, seeking per request."""

    dtype = np.dtype(np.uint8)

    def __init__(self, path):
        self.path = Path(path)
        self._n, h, w, _ = video_info(self.path)
        self.shape = (self._n, h, w)
        self.ndim = 3

    def __len__(self) -> int:
        return self._n

    def _decode_from(self, start: int, threads: bool = True) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (index, frame) from `start` on; FFV1 frames are all keyframes, so the seek lands on it."""
        with av.open(str(self.path)) as c:
            s = c.streams.video[0]
            if threads:  # frame threading speeds up runs of frames but delays the first one (~2x slower single reads)
                s.thread_type = "AUTO"
            tb = float(s.time_base)
            rate = float(s.average_rate or s.guessed_rate)
            if start > 0:
                c.seek(int(round(start / rate / tb)), stream=s, backward=True)
            for fr in c.decode(s):
                idx = int(round(fr.pts * tb * rate))
                if idx >= start:
                    yield idx, _grey(fr)

    def _read(self, tkey, spatial: tuple):
        """Frames for `tkey` (int or slice); `spatial` is applied per frame so no full-size stack is built."""
        n = self._n
        sp = spatial if spatial else (slice(None),)
        if isinstance(tkey, (int, np.integer)):
            i = int(tkey) + (n if tkey < 0 else 0)
            if not 0 <= i < n:
                raise IndexError(tkey)
            for idx, fr in self._decode_from(i, threads=False):
                if idx == i:
                    return fr[sp]
            raise IndexError(f"frame {i} not decodable in {self.path}")
        if not isinstance(tkey, slice):
            raise TypeError(f"unsupported index {tkey!r}")
        r = range(*tkey.indices(n))
        oshape = np.broadcast_to(np.zeros((), bool), self.shape[1:])[sp].shape
        out = np.empty((len(r),) + oshape, np.uint8)
        if len(r) == 0:
            return out
        if r.step < 0:
            for k, i in enumerate(r):
                out[k] = self._read(i, spatial)
            return out
        pos = {i: k for k, i in enumerate(r)}
        for idx, fr in self._decode_from(r.start):
            k = pos.get(idx)
            if k is not None:
                out[k] = fr[sp]
            if idx >= r[-1]:
                break
        return out

    def __getitem__(self, key):
        if isinstance(key, tuple):  # (time, rows, cols)
            return self._read(key[0], key[1:])
        return self._read(key, ())
