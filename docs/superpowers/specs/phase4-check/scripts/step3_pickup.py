"""Step 3 of the phase 4 check: the batch picks up thresholds accepted in the Thresholds tab.

The window (offscreen) is open on the check root. The runner is started with the window's launcher
(launcher.start_detached) and one session (--sessions SID). The script waits for "waits" in the Thresholds cell,
accepts in the tab, measures the delay from the Accept write to CNMF-E RUNNING, presses "Stop after current
stage" while CNMF-E runs, and waits for the runner to exit. Every wait is bounded. A guard stops the batch and
kills only the runner's own process tree if any stage other than cnmf of SID starts.

usage: python step3_pickup.py ROOT SID OUT_DIR
"""
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import app, pump, save_png  # noqa: E402

import psutil  # noqa: E402

from bearmind import launcher, registry  # noqa: E402
from bearmind import threshold_model as tm  # noqa: E402
from bearmind.paths import SeriesPaths  # noqa: E402

ROOT, SID, OUT = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
paths = SeriesPaths(ROOT)
res = {"root": str(ROOT), "sid": SID, "events": []}
errors, messages, status_lines = [], [], []
sys.excepthook = lambda t, e, tb: errors.append("".join(traceback.format_exception(t, e, tb)))
LOG = OUT / "step3.log"


def note(text, **kw):
    line = f"{datetime.now().isoformat(timespec='milliseconds')} {text}"
    if kw:
        line += " " + json.dumps(kw, default=str)
    res["events"].append(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line.encode("ascii", "backslashreplace").decode(), flush=True)


def save():
    res.update(errors=errors, messages=messages, status_lines=status_lines)
    (OUT / "step3.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


def stage(sid=SID, name="cnmf"):
    try:
        st = registry.load_record(paths, sid).stages.get(name)
    except (OSError, ValueError):
        return None
    return st


def heartbeat():
    try:
        return json.loads(paths.heartbeat.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


pid = None
proc = None
children = {}  # pid -> (name, create_time, cmdline head)


def track_children():
    if proc is None:
        return
    try:
        for c in proc.children(recursive=True):
            if c.pid not in children:
                try:
                    children[c.pid] = (c.name(), c.create_time(), " ".join(c.cmdline()[:4])[:200])
                except psutil.Error:
                    pass
    except psutil.Error:
        pass


def runner_alive():
    try:
        return proc is not None and proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def guard():
    """Any running stage other than cnmf of SID: stop request, then kill the runner's own tree by PID."""
    hb = heartbeat()
    cur = (hb or {}).get("current")
    if hb and hb.get("pid") == pid and cur and (cur.get("sid"), cur.get("stage")) != (SID, "cnmf"):
        note("[FAILED] guard: unexpected stage started; stopping and killing the runner tree", current=cur)
        launcher.request_stop(paths)
        track_children()
        for p in sorted(children):
            try:
                psutil.Process(p).kill()
            except psutil.Error:
                pass
        try:
            proc.kill()
        except psutil.Error:
            pass
        res["guard_fired"] = cur
        save()
        raise SystemExit(3)


from bearmind.gui.app import MainWindow  # noqa: E402

win = MainWindow(ROOT, message=lambda title, text: messages.append(text))
win.statusBar().messageChanged.connect(lambda t: t and status_lines.append(
    f"{datetime.now().isoformat(timespec='milliseconds')} {t}"))
win.resize(1400, 860)
win.show()
pump(2.5)
mon = win.monitor
res["monitor_before"] = {"batch": mon.batch_label.text(), "detail": mon.batch_detail.text(),
                         "thresholds_button": mon.thresholds_button.text()}
res["snap_monitor_before"] = save_png(win, OUT / "phase4-monitor-before.png")


def cell(sid, title):
    m = mon.model
    heads = [m.horizontalHeaderItem(c).text() for c in range(m.columnCount())]
    col = heads.index(title)
    for r in range(m.rowCount()):
        if m.item(r, 0).text() == sid:
            return m.item(r, col).text()
    return None


# --- start the runner with the window's launcher, one session ---
cmd = launcher.runner_command(ROOT) + ["--sessions", SID]
res["runner_cmd"] = cmd
try:
    pid = launcher.start_detached(ROOT, cmd=cmd)
except Exception as e:
    note(f"[FAILED] runner did not start: {type(e).__name__}: {e}")
    save()
    raise SystemExit(2)
proc = psutil.Process(pid)
t_start = time.time()
note("runner started", pid=pid)
res["runner_pid"] = pid

# --- wait for "waits" in the Thresholds cell (bounded 180 s) ---
seen_wait = None
end = time.time() + 180
while time.time() < end:
    pump(0.5)
    guard()
    track_children()
    st = stage()
    if st is not None and st.state.value == "waiting":
        if seen_wait is None:
            seen_wait = time.time()
            note("registry: cnmf WAITING", error=st.error, metrics=st.metrics)
        if cell(SID, "Thresholds") == "waits":
            break
    if not runner_alive():
        note("[FAILED] runner exited before the wait")
        break
res["waiting_seen_after_start_s"] = None if seen_wait is None else round(seen_wait - t_start, 1)
res["monitor_waits"] = {"thresholds_cell": cell(SID, "Thresholds"), "cnmf_cell": cell(SID, "CNMF-E"),
                        "batch": mon.batch_label.text(), "detail": mon.batch_detail.text(),
                        "thresholds_button": mon.thresholds_button.text(), "heartbeat": heartbeat()}
note("monitor", **res["monitor_waits"])
res["snap_monitor_waits"] = save_png(win, OUT / "phase4-monitor-waits.png")
if res["monitor_waits"]["thresholds_cell"] != "waits":
    note("[FAILED] no 'waits' in the Thresholds cell; stopping the batch")
    launcher.request_stop(paths)
    save()
    end = time.time() + 120
    while time.time() < end and runner_alive():
        pump(1.0)
    raise SystemExit(4)

# --- the Thresholds tab, opened with Monitor's "Thresholds: 1 session" button ---
tab = win.thresholds
shown = []
tab.video.frame_shown.connect(lambda i: shown.append(i))
mon.thresholds_button.click()
pump(10.0, lambda: tab.isVisible() and tab.current()[0] == SID and tab._maps is not None and bool(shown))
pump(1.0)
sid_, g, c, p = tab.current()
item = tab._by_sid.get(SID)
res["tab_before_accept"] = {"current": [sid_, g, c, p], "state": item.state if item else None,
                            "suggested": item.suggested if item else None, "chosen": item.chosen if item else None,
                            "counters": tab.counters_label.text(), "source": tab.source_label.text(),
                            "strip": [b.text() for b in tab.gsig_buttons.values()],
                            "accept_enabled": tab.accept_button.isEnabled(), "frames_shown": len(shown),
                            "strip_children": sum(1 for ch in tab.gsig_strip.children()
                                                  if ch.metaObject().className() == "QToolButton")}
note("tab before accept", **res["tab_before_accept"])
res["snap_tab_before"] = save_png(win, OUT / "phase4-thresholds-before-accept.png")
if sid_ != SID or not tab.accept_button.isEnabled():
    note("[FAILED] the tab does not show the session ready to accept; stopping the batch")
    launcher.request_stop(paths)
    save()
    end = time.time() + 120
    while time.time() < end and runner_alive():
        pump(1.0)
    raise SystemExit(5)

# --- Accept -> next ---
tab._confirm = lambda *a, **k: (note("[WARN] a question was asked before the write", args=a) or True)
t_click = datetime.now()
tab.accept_button.click()
t_after = datetime.now()
rec = registry.load_record(paths, SID)
dec = [d for d in rec.decisions if d.what == "thresholds"]
res["accept"] = {"click": t_click.isoformat(timespec="milliseconds"),
                 "returned": t_after.isoformat(timespec="milliseconds"),
                 "decision": None if not dec else {"value": dec[-1].value, "who": dec[-1].who, "when": dec[-1].when}}
note("accepted", **res["accept"])
pump(1.0)
res["tab_after_accept"] = {"current": list(tab.current()), "source": tab.source_label.text(),
                           "status": win.statusBar().currentMessage(),
                           "tree": [(tab._tree_items[s].text(0), tab._tree_items[s].text(1)) for s in tab._tree_items]}
note("tab after accept", **res["tab_after_accept"])
res["snap_tab_after"] = save_png(win, OUT / "phase4-thresholds-after-accept.png")

# --- pickup: cnmf RUNNING (bounded 90 s, poll 0.2 s) ---
t_dec = datetime.fromisoformat(dec[-1].when) if dec else t_click
running_seen = None
end = time.time() + 90
while time.time() < end:
    pump(0.2)
    track_children()
    st = stage()
    if st is not None and st.state.value in ("running", "done", "failed"):
        running_seen = datetime.now()
        res["cnmf_after_accept"] = {"state": st.state.value, "started": st.started, "params": {
            k: st.params.get(k) for k in ("gSig", "min_corr", "min_pnr")}}
        break
    if not runner_alive():
        break
if running_seen is None:
    note("[FAILED] CNMF-E did not start within 90 s; stopping the batch")
    launcher.request_stop(paths)
else:
    started = datetime.fromisoformat(res["cnmf_after_accept"]["started"])
    res["pickup"] = {"decision_when": t_dec.isoformat(timespec="milliseconds"),
                     "cnmf_started": res["cnmf_after_accept"]["started"],
                     "delay_registry_s": round((started - t_dec).total_seconds(), 3),
                     "delay_seen_by_poll_s": round((running_seen - t_dec).total_seconds(), 3),
                     "wait_to_accept_s": None if seen_wait is None else round(t_click.timestamp() - seen_wait, 1)}
    note("pickup", **res["pickup"], cnmf=res["cnmf_after_accept"])
    guard()

    # --- Stop after current stage, pressed while CNMF-E runs ---
    win.show_tab("Monitor")
    pump(3.0, lambda: mon.batch_label.text() == "Running")
    pump(1.0)
    res["monitor_running"] = {"batch": mon.batch_label.text(), "detail": mon.batch_detail.text(),
                              "cnmf_cell": cell(SID, "CNMF-E"), "stop_enabled": mon.stop_button.isEnabled()}
    note("monitor running", **res["monitor_running"])
    res["snap_monitor_running"] = save_png(win, OUT / "phase4-monitor-running.png")
    t_stop = datetime.now()
    mon.stop_button.click()
    res["stop"] = {"clicked": t_stop.isoformat(timespec="milliseconds"),
                   "stop_request_file": paths.stop_request.is_file()}
    note("stop clicked", **res["stop"])
    pump(3.0, lambda: mon.batch_label.text().startswith("Stopping"))
    res["monitor_stopping"] = {"batch": mon.batch_label.text(), "detail": mon.batch_detail.text()}
    res["snap_monitor_stopping"] = save_png(win, OUT / "phase4-monitor-stopping.png")
    note("monitor stopping", **res["monitor_stopping"])

# --- wait for the runner to exit (bounded 20 min, poll 2 s) ---
end = time.time() + 1200
last = 0
while time.time() < end and runner_alive():
    pump(2.0)
    track_children()
    guard()
    if time.time() - last > 60:
        st = stage()
        note("waiting for the runner to exit", cnmf=None if st is None else st.state.value,
             children=len(children), heartbeat_current=(heartbeat() or {}).get("current"))
        last = time.time()
t_exit = datetime.now()
res["runner_exited"] = not runner_alive()
note("runner exit", exited=res["runner_exited"])
pump(3.0)
st = stage()
res["cnmf_final"] = None if st is None else {"state": st.state.value, "started": st.started, "finished": st.finished,
                                             "metrics": st.metrics, "error": st.error,
                                             "params": {k: st.params.get(k) for k in ("gSig", "min_corr", "min_pnr")}}
rec = registry.load_record(paths, SID)
res["stages_final"] = {k: v.state.value for k, v in rec.stages.items()}
res["heartbeat_final"] = heartbeat()
res["stop_request_left"] = paths.stop_request.is_file()
alive = []
for p, (name, ctime, cmdl) in children.items():
    try:
        q = psutil.Process(p)
        if abs(q.create_time() - ctime) < 0.01:
            alive.append((p, name, cmdl))
    except psutil.Error:
        pass
res["children_seen"] = {str(k): v for k, v in children.items()}
res["children_alive_after"] = alive
res["runner_alive_after"] = runner_alive()
res["monitor_after"] = {"batch": mon.batch_label.text(), "detail": mon.batch_detail.text(),
                        "cnmf_cell": cell(SID, "CNMF-E"), "thresholds_cell": cell(SID, "Thresholds")}
res["snap_monitor_after"] = save_png(win, OUT / "phase4-monitor-stopped.png")
note("after", children_seen=len(children), children_alive=alive, monitor=res["monitor_after"],
     cnmf=res["cnmf_final"])
tab.shutdown()
mon.timer.stop()
win.close()
pump(0.5)
save()
print("[OK] done")
