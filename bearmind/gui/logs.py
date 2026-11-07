"""Log tail view: the last lines of a log file, read from the end, then only the bytes that were added."""
import os
from pathlib import Path

from PySide6.QtWidgets import QPlainTextEdit

MAX_LINES = 500
BLOCK = 64 * 1024
MAX_READ = 1024 * 1024  # a log of a long batch is tens of MB; the window never needs more than its tail


def _tail(path: Path) -> tuple[list[str], int, bytes]:
    """The last MAX_LINES complete lines, the offset after them, and the unfinished last line (bytes)."""
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        start, data = size, b""
        while start > 0 and data.count(b"\n") <= MAX_LINES and size - start < MAX_READ:
            step = min(BLOCK, start, MAX_READ - (size - start))
            start -= step
            fh.seek(start)
            data = fh.read(step) + data
    cut = data.rfind(b"\n") + 1
    body, partial = data[:cut], data[cut:]
    lines = body.split(b"\n")[:-1] if body else []
    if start > 0 and lines:
        lines = lines[1:]  # the first line may be cut in the middle
    return [x.decode("utf-8", "replace").rstrip("\r") for x in lines[-MAX_LINES:]], size - len(partial), partial


class LogView(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(MAX_LINES)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        self.path: Path | None = None
        self._pos: int | None = None  # None: the file was not there at the last read
        self._partial = b""

    def show_log(self, path: Path) -> None:
        self.path = Path(path)
        self._load()

    def _load(self) -> None:
        try:
            lines, self._pos, self._partial = _tail(self.path)
        except OSError:
            lines, self._pos, self._partial = [], None, b""
        self.setPlainText("\n".join(lines))
        self._to_end()

    def poll(self) -> None:
        if self.path is None:
            return
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        # a file that shrank was rewritten; one that grew by more than we ever show is cheaper to reload
        if self._pos is None or size < self._pos + len(self._partial) or size - self._pos > MAX_READ:
            self._load()
            return
        if size == self._pos + len(self._partial):
            return
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self._pos)
                data = fh.read(size - self._pos)  # starts with the unfinished line of the last read
        except OSError:
            return
        cut = data.rfind(b"\n") + 1
        self._pos += cut
        self._partial = data[cut:]
        if cut:
            # a person reading further up keeps the place; at the end, the view follows the new lines
            bar = self.verticalScrollBar()
            at_end = bar.value() >= bar.maximum()
            # Python logging writes CRLF on Windows; a lone \r would show as an extra empty line
            text = data[:cut].decode("utf-8", "replace").replace("\r\n", "\n").rstrip("\n")
            self.appendPlainText(text)
            if at_end:
                self._to_end()

    def _to_end(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
