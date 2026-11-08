"""Steps 1-2 of the phase 4 check: the window on series1, read only.

Snapshots of the Thresholds tab on NOF_H01_1D and RFC_F01_1D, slider latencies (20 changes per kind), the time to
show the next session. Every bearmind.actions write function is replaced by one that refuses and records the
call, so nothing can be accepted by mistake. usage: python view_series1.py OUT_DIR
"""
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import app, ms, pump, save_png  # noqa: E402

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from bearmind import actions  # noqa: E402

SERIES = Path(r"H:\Projects\_bearmind_test\series1")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

refused = []


def _refusing(name):
    def f(*a, **k):
        refused.append(name)
        raise RuntimeError(f"check is read only: {name} refused")
    return f


for _n in ("set_thresholds", "apply_to_mouse", "accept_suggested", "rerun_from", "set_excluded", "gate_not_needed",
           "gate_cleaned", "retry_failed", "set_previews", "accept_mismatch", "export_summary"):
    if hasattr(actions, _n):
        setattr(actions, _n, _refusing(_n))

errors, messages = [], []
sys.excepthook = lambda t, e, tb: errors.append("".join(traceback.format_exception(t, e, tb)))

from bearmind.gui.app import MainWindow  # noqa: E402

res = {"series": str(SERIES)}
t0 = time.perf_counter()
win = MainWindow(SERIES, message=lambda title, text: messages.append(text))
res["window_open_s"] = round(time.perf_counter() - t0, 3)
win.resize(1400, 860)
win.show()
pump(1.0)
res["snap_monitor"] = save_png(win, OUT / "series1-monitor.png")

tab = win.thresholds
shown = []  # (perf time, frame index) of every video frame shown
tab.video.frame_shown.connect(lambda i: shown.append((time.perf_counter(), i)))
t0 = time.perf_counter()
win.show_tab("Thresholds")
pump(0.0)
res["tab_show_s"] = round(time.perf_counter() - t0, 3)
pump(8.0, lambda: bool(shown))


def tab_state():
    sid, g, c, p = tab.current()
    item = tab._by_sid.get(sid)
    return {"sid": sid, "gSig": g, "min_corr": c, "min_pnr": p, "state": item.state if item else None,
            "suggested": item.suggested if item else None, "chosen": item.chosen if item else None,
            "counters": tab.counters_label.text(), "source": tab.source_label.text(),
            "maps_title": tab.maps_title.text(), "video_title": tab.video_title.text(),
            "strip": [b.text() for b in tab.gsig_buttons.values()],
            "accept_enabled": tab.accept_button.isEnabled(), "accept_tip": tab.accept_button.toolTip(),
            "reset_enabled": tab.reset_button.isEnabled(),
            "tree": [(tab.session_list.topLevelItem(i).text(0),
                      [(tab.session_list.topLevelItem(i).child(j).text(0),
                        tab.session_list.topLevelItem(i).child(j).text(1))
                       for j in range(tab.session_list.topLevelItem(i).childCount())])
                     for i in range(tab.session_list.topLevelItemCount())],
            "corr_view_px": [tab.corr_view.width(), tab.corr_view.height()],
            "map_shape": list(tab._maps.cn[g].shape) if tab._maps is not None and g in tab._maps.cn else None,
            "video_frames_shown": len(shown)}


res["first_shown"] = tab_state()

# --- step 1: the two sessions, selected in the list like a click ---
res["sessions"] = {}
for sid in ("NOF_H01_1D", "RFC_F01_1D"):
    n0 = len(shown)
    t0 = time.perf_counter()
    tab.session_list.setCurrentItem(tab._tree_items[sid])
    t_sync = time.perf_counter() - t0
    pump(0.0)
    t_ui = time.perf_counter() - t0
    got = pump(15.0, lambda: len(shown) > n0)
    t_frame = (shown[n0][0] - t0) if got else None
    pump(0.5)
    st = tab_state()
    st.update(select_sync_s=round(t_sync, 3), select_ui_s=round(t_ui, 3),
              first_frame_s=None if t_frame is None else round(t_frame, 3))
    st["snap"] = save_png(win, OUT / f"series1-{sid}.png")
    res["sessions"][sid] = st

# --- step 2a: slider latencies on both sessions ---


def slider_latency(slider, values, drag: bool):
    out = []
    if drag:
        slider.setSliderDown(True)
    for v in values:
        t = time.perf_counter()
        slider.setValue(v)
        app.processEvents()  # the repaint of both maps
        out.append(time.perf_counter() - t)
    if drag:
        slider.setSliderDown(False)
        app.processEvents()
    return out


def release_latency(slider, values):
    out = []
    for v in values:
        slider.setSliderDown(True)
        slider.setValue(v)
        app.processEvents()
        t = time.perf_counter()
        slider.setSliderDown(False)  # sliderReleased: counters and the strip numbers
        app.processEvents()
        out.append(time.perf_counter() - t)
    return out


def key_latency(slider, n, key):
    out = []
    slider.setFocus()
    for _ in range(n):
        t = time.perf_counter()
        QTest.keyClick(slider, key)  # one step: mask, counters and strip numbers at once
        app.processEvents()
        out.append(time.perf_counter() - t)
    return out


res["latency"] = {}
for sid in ("NOF_H01_1D", "RFC_F01_1D"):
    tab.session_list.setCurrentItem(tab._tree_items[sid])
    pump(1.0)
    start = tab.current()
    c0 = tab.min_corr_slider.value()
    p0 = tab.min_pnr_slider.value()
    corr_vals = [c0 - 10 + i for i in range(20)]  # 0.01 steps around the start value
    pnr_vals = [p0 - 20 + 2 * i for i in range(20)]  # 0.2 steps around the start value
    lat = {
        "drag_min_corr": ms(slider_latency(tab.min_corr_slider, corr_vals, True)),
        "drag_min_pnr": ms(slider_latency(tab.min_pnr_slider, pnr_vals, True)),
        "release_min_corr": ms(release_latency(tab.min_corr_slider, corr_vals)),
        "release_min_pnr": ms(release_latency(tab.min_pnr_slider, pnr_vals)),
        "key_min_corr": ms(key_latency(tab.min_corr_slider, 20, Qt.Key.Key_Left)),
        "key_min_pnr": ms(key_latency(tab.min_pnr_slider, 20, Qt.Key.Key_Left)),
    }
    lat["map_shape"] = res["sessions"][sid]["map_shape"]
    lat["n_gsig"] = len(tab.gsig_buttons)
    # back to the start values: nothing is written, the screen shows the suggestion again
    tab.set_thresholds(start[2], start[3])
    lat["back_to_start"] = tab.current() == start
    res["latency"][sid] = lat

# --- step 2b: Next / Previous through the whole list ---
order = [i.sid for i in tab._items]
res["order"] = order
tab.session_list.setCurrentItem(tab._tree_items[order[0]])
pump(15.0, lambda: tab.current()[0] == order[0])
pump(2.0)
trans = []
for button, steps in ((tab.next_button, len(order) - 1), (tab.prev_button, len(order) - 1)):
    for _ in range(steps):
        before = tab.current()[0]
        n0 = len(shown)
        t0 = time.perf_counter()
        button.click()
        t_sync = time.perf_counter() - t0
        app.processEvents()
        t_ui = time.perf_counter() - t0
        got = pump(15.0, lambda: len(shown) > n0)
        t_frame = (shown[n0][0] - t0) if got else None
        trans.append({"button": button.text(), "from": before, "to": tab.current()[0], "sync_s": round(t_sync, 3),
                      "ui_s": round(t_ui, 3), "first_frame_s": None if t_frame is None else round(t_frame, 3)})
        pump(0.3)
res["transitions"] = trans
res["transition_summary"] = {
    "next_ui": ms([t["ui_s"] for t in trans if t["button"].startswith("Next")]),
    "prev_ui": ms([t["ui_s"] for t in trans if not t["button"].startswith("Next")]),
    "all_ui": ms([t["ui_s"] for t in trans]),
    "all_first_frame": ms([t["first_frame_s"] for t in trans if t["first_frame_s"] is not None]),
}

tab.shutdown()
win.monitor.timer.stop()
win.close()
pump(0.5)
res["refused_writes"] = refused
res["errors"] = errors
res["messages"] = messages
(OUT / "view_series1.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("transition_summary", "refused_writes", "errors", "messages")}, indent=1))
for sid, lat in res["latency"].items():
    print(sid, json.dumps(lat))
print("[OK] done")
