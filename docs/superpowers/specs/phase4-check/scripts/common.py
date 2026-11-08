"""Shared helpers of the phase 4 check drivers (offscreen Qt, event pumping, snapshots under 300 KB)."""
import io
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
REPO = r"C:\Users\User\PycharmProjects\bearmind"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from PySide6.QtCore import QBuffer, QCoreApplication, QEvent, QEventLoop, QIODevice  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
MAX_PNG = 300 * 1024


def pump(seconds: float, until=None) -> bool:
    """Run the event loop for up to `seconds`; stop early when until() is true. Returns until() (or True)."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        # processEvents alone never runs deleteLater outside app.exec(): the old gSig buttons would stay painted
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        if until is not None and until():
            return True
        time.sleep(0.005)
    return bool(until()) if until is not None else True


def save_png(widget, path: Path) -> dict:
    """widget.grab() as PNG under 300 KB: as is, else a 256-colour palette, else scaled down in steps."""
    from PIL import Image

    pix = widget.grab()
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    pix.save(buf, "PNG")
    raw = bytes(buf.data())
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    how = "as is"
    data = raw
    if len(data) > MAX_PNG:
        scale = 1.0
        while True:
            im = img if scale == 1.0 else img.resize((int(img.width * scale), int(img.height * scale)),
                                                     Image.Resampling.LANCZOS)
            q = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            out = io.BytesIO()
            q.save(out, "PNG", optimize=True)
            data = out.getvalue()
            how = "palette 256" + ("" if scale == 1.0 else f", scaled {scale:.2f}")
            if len(data) <= MAX_PNG or scale < 0.4:
                break
            scale -= 0.1
    path.write_bytes(data)
    return {"file": path.name, "bytes": len(data), "how": how, "px": [pix.width(), pix.height()]}


def ms(xs) -> dict:
    import statistics

    xs = [1000 * x for x in xs]
    return {"n": len(xs), "median_ms": round(statistics.median(xs), 1), "max_ms": round(max(xs), 1),
            "min_ms": round(min(xs), 1)}
