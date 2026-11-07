"""MC video view: one frame, a slider and a play button. Decoding runs in a worker thread, so the GUI never waits.

A frame request is a single slot in the worker: a new request replaces the one that is still waiting, so a fast
scrub decodes only the frames the slider rests on, and the last request is always decoded. Opening the file
(which counts packets when the header has no frame count) also happens in the worker. The worker is stopped
and joined when the session changes or the widget closes or is destroyed; VideoArray holds no file handle
between requests, so no lock stays on the FFV1 file after that (the batch may need it on Windows).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget

from bearmind.video import VideoArray

log = logging.getLogger("bearmind")

CACHE_FRAMES = 64
PLAY_TICK_MS = 15


def position_text(index: int, total: int, fps: float) -> str:
    """'frame 1234 / 18000 · 00:41.1': frames count from 1, the time is that of the frame start."""
    t = index / fps if fps > 0 else 0.0
    tenths = int(t * 10 + 1e-6)  # floor: rounding 59.96 s up would show 00:60.0
    return f"frame {index + 1} / {total} · {tenths // 600:02d}:{tenths % 600 // 10:02d}.{tenths % 10}"


class _Worker(QThread):
    opened = Signal(object, int, int, int)   # token, frames, height, width
    failed = Signal(object, str)             # token, message
    decoded = Signal(object, int, object)    # token, index, uint8 (h, w)

    def __init__(self, path: Path, token, parent=None):
        super().__init__(parent)
        self._path, self._token = path, token
        self._cond = threading.Condition()
        self._want: int | None = None
        self._stop = False

    def request(self, index: int) -> None:
        with self._cond:
            self._want = index
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()

    def run(self) -> None:
        try:
            arr = VideoArray(self._path)
            if len(arr) < 1:
                raise ValueError("no frames")
        except Exception as e:
            self.failed.emit(self._token, f"{type(e).__name__}: {e}")
            return
        self.opened.emit(self._token, len(arr), arr.shape[1], arr.shape[2])
        while True:
            with self._cond:
                while self._want is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                i, self._want = self._want, None
            try:
                frame = arr[i]
            except Exception as e:
                log.warning("MC video frame %s of %s not decoded: %s", i, self._path, e)
                continue
            self.decoded.emit(self._token, i, frame)


class VideoView(QWidget):
    frame_shown = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: _Worker | None = None
        self._token = 0
        self._path: Path | None = None
        self._fps = 20.0
        self._contrast: tuple[float, float] | None = None
        self._total = 0
        self._index = 0
        self._shown = -1
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._image: QImage | None = None
        self._play_t0 = 0.0
        self._play_i0 = 0

        self.image_label = QLabel(alignment=Qt.AlignCenter)
        self.image_label.setMinimumSize(QSize(160, 120))
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setStyleSheet("background: #808080; color: #202020;")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.play_button = QPushButton("▶")
        self.play_button.setToolTip("Play")
        self.play_button.setEnabled(False)
        self.play_button.setFixedWidth(40)
        self.position_label = QLabel("")
        bar = QHBoxLayout()
        bar.addWidget(self.play_button)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.position_label)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.image_label, 1)
        lay.addLayout(bar)

        self._timer = QTimer(self)
        self._timer.setInterval(PLAY_TICK_MS)
        self._timer.timeout.connect(self._tick)
        self.slider.valueChanged.connect(self.seek)
        self.play_button.clicked.connect(self._toggle_play)

    # ---- public ----
    def set_video(self, path: Path | None, fps: float, contrast: tuple[float, float] | None = None) -> None:
        self._stop_worker()
        self._token += 1
        self._path = Path(path) if path is not None else None
        self._fps = float(fps) if fps and fps > 0 else 20.0
        self._contrast = contrast
        self._total, self._index, self._shown = 0, 0, -1
        self._cache.clear()
        self._image = None
        self.image_label.setPixmap(QPixmap())
        self.position_label.setText("")
        self._set_controls(False)
        if self._path is None:
            self.image_label.setText("")
            return
        self.image_label.setText("")
        self._worker = _Worker(self._path, self._token, self)
        self._worker.opened.connect(self._on_opened)
        self._worker.failed.connect(self._on_failed)
        self._worker.decoded.connect(self._on_decoded)
        self._worker.start()

    def seek(self, index: int) -> None:
        if self._total < 1:
            self._index = max(0, int(index))  # remembered until the file is open
            return
        index = min(max(0, int(index)), self._total - 1)
        self._index = index
        self.position_label.setText(position_text(index, self._total, self._fps))
        if self.slider.value() != index:
            self.slider.blockSignals(True)
            self.slider.setValue(index)
            self.slider.blockSignals(False)
        if index in self._cache:
            self._cache.move_to_end(index)
            self._show(index, self._cache[index])
        elif self._worker is not None:
            self._worker.request(index)

    def current_index(self) -> int:
        return self._index

    def shown_image(self) -> QImage | None:
        """The frame on screen at full resolution, with the contrast applied (None before the first frame)."""
        return self._image

    # ---- worker results ----
    def _on_opened(self, token, n: int, h: int, w: int) -> None:
        if token != self._token:
            return
        self._total = n
        self.slider.blockSignals(True)
        self.slider.setRange(0, n - 1)
        self.slider.blockSignals(False)
        self._set_controls(True)
        self.seek(self._index)

    def _on_failed(self, token, msg: str) -> None:
        if token != self._token:
            return
        log.warning("MC video %s not opened: %s", self._path, msg)
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(f"MC video not found: {self._path}")
        self._set_controls(False)

    def _on_decoded(self, token, index: int, frame) -> None:
        if token != self._token:
            return
        self._cache[index] = frame
        while len(self._cache) > CACHE_FRAMES:
            self._cache.popitem(last=False)
        if index == self._index:  # a stale request that was already decoding is cached but not shown
            self._show(index, frame)

    # ---- display ----
    def _show(self, index: int, frame: np.ndarray) -> None:
        if index == self._shown:
            return
        self._image, self._shown = self._to_image(frame), index
        self._fit()
        self.frame_shown.emit(index)

    def _to_image(self, frame: np.ndarray) -> QImage:
        if self._contrast is not None:
            lo, hi = self._contrast
            frame = np.clip((frame.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-9)), 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        h, w = frame.shape
        return QImage(frame.data, w, h, w, QImage.Format_Grayscale8).copy()  # copy: detach from the numpy buffer

    def _fit(self) -> None:
        if self._image is not None:
            self.image_label.setPixmap(QPixmap.fromImage(self._image).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._fit()

    def _set_controls(self, on: bool) -> None:
        self.slider.setEnabled(on)
        self.play_button.setEnabled(on)
        if not on:
            self._pause()

    # ---- playing ----
    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._pause()
            return
        if self._total < 1:
            return
        if self._index >= self._total - 1:
            self.seek(0)
        self._play_t0, self._play_i0 = time.monotonic(), self._index
        self._timer.start()
        self.play_button.setText("❚❚")
        self.play_button.setToolTip("Pause")

    def _pause(self) -> None:
        self._timer.stop()
        self.play_button.setText("▶")
        self.play_button.setToolTip("Play")

    def _tick(self) -> None:
        # The frame follows the wall clock, so a slow decode skips frames and the time does not lag.
        target = self._play_i0 + int((time.monotonic() - self._play_t0) * self._fps)
        if target >= self._total - 1:
            self.seek(self._total - 1)
            self._pause()
        elif target != self._index:
            self.seek(target)

    # ---- lifetime ----
    def _stop_worker(self) -> None:
        self._pause()
        w, self._worker = self._worker, None
        if w is not None:
            w.stop()
            w.wait()  # at most one frame decode; the file is free after this

    def closeEvent(self, e) -> None:
        self._stop_worker()
        super().closeEvent(e)

    def __del__(self) -> None:
        try:
            self._stop_worker()
        except RuntimeError:  # the Qt object is already gone
            pass
