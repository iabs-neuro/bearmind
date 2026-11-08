"""Snapshots of the Thresholds tab on series1 taken inside app.exec() (the real event loop: deleteLater runs).

Also types "0.85" and "0,85" into the min_corr spin box (no write: actions writes are refused).
usage: python snaps_series1_loop.py OUT_DIR
"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import app, save_png  # noqa: E402

from PySide6.QtCore import QLocale, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from bearmind import actions  # noqa: E402

SERIES = Path(r"H:\Projects\_bearmind_test\series1")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
refused, errors, messages, res = [], [], [], {}


def _refusing(name):
    def f(*a, **k):
        refused.append(name)
        raise RuntimeError(f"check is read only: {name} refused")
    return f


for _n in ("set_thresholds", "apply_to_mouse", "accept_suggested", "rerun_from", "set_excluded", "gate_not_needed",
           "gate_cleaned", "retry_failed", "set_previews", "accept_mismatch", "export_summary"):
    if hasattr(actions, _n):
        setattr(actions, _n, _refusing(_n))
sys.excepthook = lambda t, e, tb: errors.append("".join(traceback.format_exception(t, e, tb)))

from bearmind.gui.app import MainWindow  # noqa: E402

win = MainWindow(SERIES, message=lambda title, text: messages.append(text))
win.resize(1400, 860)
win.show()
tab = None
res["locale"] = QLocale().name()


def step(fn, delay_ms):
    def run():
        try:
            fn()
        except Exception:
            errors.append(traceback.format_exc())
            app.quit()
    QTimer.singleShot(delay_ms, run)


def s_open():
    global tab
    win.show_tab("Thresholds")
    tab = win.thresholds


def s_select(sid):
    def f():
        tab.session_list.setCurrentItem(tab._tree_items[sid])
    return f


def s_snap(sid):
    def f():
        res[sid] = {"snap": save_png(win, OUT / f"series1-{sid}.png"),
                    "strip_children": sum(1 for c in tab.gsig_strip.children() if c.metaObject().className()
                                          == "QToolButton"),
                    "strip_buttons": [b.text() for b in tab.gsig_buttons.values()],
                    "video_px": [tab.video.width(), tab.video.height()],
                    "corr_px": [tab.corr_view.width(), tab.corr_view.height()]}
    return f


def s_spin():
    spin = tab.min_corr_spin
    out = {"start": spin.value(), "text": spin.text()}
    for typed in ("0.85", "0,85"):
        le = spin.lineEdit()
        le.selectAll()
        QTest.keyClicks(le, typed)
        shown_text = le.text()
        QTest.keyClick(le, Qt.Key.Key_Return)
        out[typed] = {"line_edit_after_typing": shown_text, "value_after_enter": spin.value(),
                      "slider_value": tab.min_corr_slider.value()}
        tab.set_thresholds(out["start"], tab.current()[3])  # back to the start value
    res["spin_typing"] = out
    pspin = tab.min_pnr_spin
    p0 = pspin.value()
    outp = {"start": p0, "text": pspin.text(), "max": pspin.maximum()}
    for typed in ("12.5", "12,5", "9.5"):
        le = pspin.lineEdit()
        le.selectAll()
        QTest.keyClicks(le, typed)
        shown_text = le.text()
        QTest.keyClick(le, Qt.Key.Key_Return)
        outp[typed] = {"line_edit_after_typing": shown_text, "value_after_enter": pspin.value(),
                       "current": list(tab.current()), "accept_enabled": tab.accept_button.isEnabled()}
        tab.set_thresholds(tab.current()[2], p0)
    res["pnr_spin_typing"] = outp


def s_done():
    tab.shutdown()
    win.monitor.timer.stop()
    win.close()
    app.quit()


plan = [(s_open, 500), (s_select("NOF_H01_1D"), 1500), (s_snap("NOF_H01_1D"), 4000),
        (s_select("RFC_F01_1D"), 4500), (s_snap("RFC_F01_1D"), 7000), (s_spin, 7500), (s_done, 8500)]
for fn, at in plan:
    step(fn, at)
QTimer.singleShot(30000, app.quit)  # bound
app.exec()
res.update(refused=refused, errors=errors, messages=messages)
(OUT / "snaps_loop.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps(res, indent=1, default=str).encode("ascii", "backslashreplace").decode())
