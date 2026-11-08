r"""Step 2 of the phase 5 check: the inspection server of phase5_short and the gate.

Starts `python -m bearmind inspect ROOT NOF_H01_1D --no-browser` as a child process (the background server),
GETs its session page (200), connects to that page's session as a browser does (bokeh.client with the page's
session id), deletes one component, merges two, clicks Save results and then Next, all as browser events
(send_event -> PATCH-DOC to the server). Checks the manual result, the decision "cleaned", the side files, the
prepared next session (the lock's "prepared") and the Next label. Then closes every connection and waits for the
server to exit by itself after inspection.idle_exit_s (30 s in the copy's series.yaml); kills it by PID if it
has not ended within 120 s. Every wait is bounded.

usage: python step2_server.py OUT_DIR

Known problem of this script (the run of 2026-11-08 hung after Next and was stopped by PID): every bokeh.client
connection here shares one IOLoop, and one connection can stop the loop another one waits on. The tests of the
server give every tab its own IOLoop (tests/bearmind/test_inspect_server.py, `tab`); probe_tabs.py does so too.
Give each connection its own IOLoop before running this again.
"""
import torch  # noqa: F401  (torch before bm_examinator: shm.dll)

import asyncio
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = r"C:\Users\User\PycharmProjects\bearmind"
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import psutil  # noqa: E402
from bokeh.client import pull_session  # noqa: E402
from bokeh.events import ButtonClick  # noqa: E402
from bokeh.models import Button, ColumnDataSource, Div, GlyphRenderer  # noqa: E402
from bokeh.util.token import get_session_id  # noqa: E402
from tornado.ioloop import IOLoop  # noqa: E402

from bearmind import inspect_lock, registry  # noqa: E402
from bearmind.paths import SeriesPaths  # noqa: E402

ROOT = Path(r"H:\Projects\_bearmind_test\phase5_short")
SID, NEXT_SID = "NOF_H01_1D", "NOF_H03_3D"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
paths = SeriesPaths(ROOT)
res = {"events": [], "checks": {}}
T0 = time.perf_counter()


def note(text, **kw):
    line = f"{datetime.now().isoformat(timespec='milliseconds')} +{time.perf_counter() - T0:7.2f}s {text}"
    if kw:
        line += " " + json.dumps(kw, default=str, ensure_ascii=True)
    res["events"].append(line)
    print(line.encode("ascii", "backslashreplace").decode(), flush=True)


def check(name, ok, **kw):
    res["checks"][name] = {"ok": bool(ok), **kw}
    note(f"[{'OK' if ok else 'FAILED'}] {name}", **kw)


def save():
    (OUT / "t7_step2.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


def until(cond, timeout, step=0.1):
    deadline = time.monotonic() + timeout
    while True:
        v = cond()
        if v:
            return v
        if time.monotonic() > deadline:
            return None
        time.sleep(step)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def folder_hashes(d):
    return {p.name: [p.stat().st_size, p.stat().st_mtime_ns, sha(p)] for p in sorted(Path(d).iterdir()) if p.is_file()}


def models(doc, kind=object):
    return [m for r in doc.roots for m in r.references() if isinstance(m, kind)]


def button(doc, label):
    bs = [b for b in models(doc, Button) if b.label == label]
    return bs[0] if len(bs) == 1 else None


def div(doc, name):
    ds = [d for d in models(doc, Div) if d.name == name]
    return ds[0] if len(ds) == 1 else None


def rows_view(doc):
    src, = [s for s in models(doc, ColumnDataSource) if "idx" in s.data]
    view = [r.view for r in models(doc, GlyphRenderer) if r.data_source is src][0]
    return src, list(view.filter.indices)


# --- the client side: one IOLoop for every connection of this script
io = IOLoop(make_current=False)
asyncio.set_event_loop(io.asyncio_loop)
sessions = []


def connect(url, session_id):
    s = pull_session(session_id=session_id, url=url + "/", io_loop=io)
    sessions.append(s)
    return s


def server_view(url, session_id):
    """The server's document of the session now (a new connection pulls it whole), then closed."""
    s = pull_session(session_id=session_id, url=url + "/", io_loop=io)
    try:
        doc = s.document
        out = {"title": doc.title, "buttons": sorted(b.label for b in models(doc, Button) if b.visible)}
        st = div(doc, "status")
        out["status"] = st.text if st is not None else None
        nb = [b for b in models(doc, Button) if b.name == "next"]
        out["next"] = (nb[0].label, nb[0].visible, nb[0].disabled) if nb else None
        nt = div(doc, "next_text")
        out["next_text"] = (nt.text, nt.visible) if nt is not None else None
        w = div(doc, "wave")
        out["wave"] = (w.text, w.visible) if w is not None else None
        try:
            src, shown = rows_view(doc)
            out["n_rows_shown"] = len(shown)
            out["shown_components"] = [int(src.data["idx"][i]) for i in shown]
        except Exception as e:  # a message page has no rows
            out["n_rows_shown"] = None
        return out
    finally:
        s.close()


def send_click(session, label=None, name=None):
    doc = session.document
    b = button(doc, label) if label else [m for m in models(doc, Button) if m.name == name][0]
    doc.callbacks.send_event(ButtonClick(b))
    session.force_roundtrip()  # flushes the PATCH-DOC; the server handles the click before it replies


def select(session, url, session_id, rows):
    """The browser's selection of these positions of the list on screen. The positions are read from a fresh
    pull of the server's document: this client may have missed the server's patches after a change (it drops
    patches that arrive while it waits for a reply), but the data-source rows (slots) never move."""
    fresh = pull_session(session_id=session_id, url=url + "/", io_loop=io)
    try:
        fsrc, shown = rows_view(fresh.document)
        slots = [shown[r] for r in rows]
        comps = [int(fsrc.data["idx"][s]) for s in slots]
    finally:
        fresh.close()
    src, _ = rows_view(session.document)
    src.selected.indices = []
    session.force_roundtrip()
    src.selected.indices = slots
    session.force_roundtrip()
    return comps


server = None
log_path = OUT / "t7_server_console.log"
try:
    est_before = folder_hashes(ROOT / "4_Estimates")
    insp_dir = paths.inspection_dir(SID)
    res["inspection_dir_before"] = sorted(os.listdir(insp_dir)) if insp_dir.exists() else None
    res["cwd_before"] = sorted(os.listdir(REPO))
    rec0 = registry.load_record(paths, SID)
    res["decisions_before"] = [(d.what, d.value, d.who, d.when) for d in rec0.decisions]
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = [sys.executable, "-u", "-m", "bearmind", "inspect", str(ROOT), SID, "--no-browser"]
    with open(log_path, "wb") as logf:
        server = subprocess.Popen(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT, env=env,
                                  creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    res["server_pid"] = server.pid
    res["server_cmd"] = cmd
    note("server started", pid=server.pid)

    def started_line():
        try:
            m = re.search(r"\[OK\] inspection server: (http://\S+)", log_path.read_text("utf-8", "replace"))
        except OSError:
            return None
        if server.poll() is not None:
            return "exited"
        return m.group(1) if m else None
    url = until(started_line, 180, 0.05)
    t_started = time.perf_counter() - T0
    check("server printed its address", url and url != "exited", url=url, seconds=round(t_started, 2))
    if not url or url == "exited":
        raise SystemExit("server did not start")
    st = inspect_lock.server_state(paths)
    check("lock names the server", st.alive and st.url == url and st.pid == server.pid,
          lock={"alive": st.alive, "url": st.url, "pid": st.pid, "current": st.current, "prepared": st.prepared})

    # --- the page of the session (a browser's GET), 200
    surl = inspect_lock.session_url(url, SID)
    t = time.perf_counter()
    with urllib.request.urlopen(surl, timeout=300) as r:
        status = r.status
        body = r.read().decode("utf-8", "replace")
        cache_control = r.headers.get("Cache-Control")
    t_get = time.perf_counter() - t
    check("session page answers 200", status == 200, url=surl, seconds=round(t_get, 3), bytes=len(body),
          cache_control=cache_control)
    tok = re.search(r'"token"\s*:\s*"([^"]+)"', body) or re.search(r"token\s*=\s*'([^']+)'", body)
    session_id = get_session_id(tok.group(1))
    note("page session id taken from the page", session_id=session_id[:12] + "...")

    # --- the browser connects to that session
    t = time.perf_counter()
    tab = connect(url, session_id)
    doc = tab.document
    t_connect = time.perf_counter() - t
    check("connected page shows the session", doc.title == f"{SID} - manual inspection"
          and button(doc, "Save results") is not None, title=doc.title, seconds=round(t_connect, 3))
    view0 = server_view(url, session_id)
    res["view_opened"] = {k: v for k, v in view0.items() if k != "shown_components"}
    res["n_rows_opened"] = view0["n_rows_shown"]
    note("opened", n_rows=view0["n_rows_shown"], wave=view0["wave"], buttons=view0["buttons"])
    until(lambda: inspect_lock.server_state(paths).prepared == NEXT_SID, 120, 0.2)
    st = inspect_lock.server_state(paths)
    check("next session prepared while the first is open", st.prepared == NEXT_SID,
          current=st.current, prepared=st.prepared)

    # --- Delete one component, Merge two
    deleted = select(tab, url, session_id, [0])
    send_click(tab, "Delete sel.")
    v = server_view(url, session_id)
    check("Delete removed the row", v["n_rows_shown"] == view0["n_rows_shown"] - 1
          and deleted[0] not in v["shown_components"], deleted=deleted, n_rows=v["n_rows_shown"])
    merged = select(tab, url, session_id, [0, 1])
    send_click(tab, "Merge sel.")
    v = server_view(url, session_id)
    check("Merge replaced two rows by one", v["n_rows_shown"] == view0["n_rows_shown"] - 2
          and not set(merged) & set(v["shown_components"]), merged=merged, n_rows=v["n_rows_shown"])
    res["after_edits_shown"] = v["shown_components"]

    # --- Save results
    t = time.perf_counter()
    send_click(tab, "Save results")
    t_save_roundtrip = time.perf_counter() - t
    result = paths.manual_result(SID)
    ok_file = until(lambda: result.is_file(), 60, 0.05)
    rec = registry.load_record(paths, SID)
    decisions = [(d.what, d.value, d.who, d.when) for d in rec.decisions if d.what == "manual_inspection"]
    check("result file written", bool(ok_file), file=str(result), seconds_roundtrip=round(t_save_roundtrip, 3))
    check("decision cleaned recorded", decisions[-1:] and decisions[-1][1] == "cleaned", decisions=decisions,
          gate_state=rec.stages["gate"].state.value)
    v = until(lambda: (lambda x: x if x["next"] and x["next"][1] else None)(server_view(url, session_id)), 60, 0.3)
    check("page says saved and offers Next", v is not None and v["status"]
          == f"Saved. Decision \u201ccleaned\u201d recorded for {SID}. You can close this tab."
          and v["next"][0] == f"Next: {NEXT_SID} \u2192", status=v and v["status"], next=v and v["next"])
    st = inspect_lock.server_state(paths)
    check("lock: next session prepared after save", st.prepared == NEXT_SID, current=st.current,
          prepared=st.prepared)

    # --- side files (background thread of the server)
    want = [f"{SID}_traces.csv", f"{SID}_filters", f"{SID}_filters_session.mat"]
    ok_side = until(lambda: all((insp_dir / w).exists() for w in want), 180, 0.2)
    res["inspection_dir_after_save"] = sorted(os.listdir(insp_dir))
    check("side files written in 5_Inspection", bool(ok_side), files=res["inspection_dir_after_save"],
          n_filters=len(list((insp_dir / f"{SID}_filters").iterdir())) if ok_side else None)

    # --- the result file itself
    with open(result, "rb") as f:
        est = pickle.load(f)
    with open(ROOT / "4_Estimates" / f"{SID}_estimates.pickle", "rb") as f:
        src = pickle.load(f)
    n_src, n_res = src.A.shape[1], est.A.shape[1]
    kept = list(map(int, est.idx_components))
    res["result"] = {"bytes": result.stat().st_size, "A_cols": n_res, "n_kept": len(kept),
                     "n_bad": len(est.idx_components_bad), "C_dtype": str(est.C.dtype),
                     "compressed": hasattr(est, "idx_components_bad_original"),
                     "newer_than_estimates": result.stat().st_mtime_ns
                     > (ROOT / "4_Estimates" / f"{SID}_estimates.pickle").stat().st_mtime_ns,
                     "src_A_cols": n_src, "src_n_kept": len(src.idx_components)}
    check("result: one deleted, two merged into one", n_res == n_src - 1
          and len(kept) == len(src.idx_components) - 2 and deleted[0] in list(map(int, est.idx_components_bad))
          and not res["result"]["compressed"], **res["result"])
    del est, src

    # --- Next
    t = time.perf_counter()
    send_click(tab, name="next")
    v = until(lambda: (lambda x: x if x["title"] == f"{NEXT_SID} - manual inspection" and x["n_rows_shown"]
                       else None)(server_view(url, session_id)), 60, 0.1)
    t_next = time.perf_counter() - t
    check("Next opened the prepared session", v is not None, title=v and v["title"], n_rows=v and v["n_rows_shown"],
          seconds_until_seen=round(t_next, 3))
    until(lambda: inspect_lock.server_state(paths).current == [NEXT_SID], 10, 0.2)
    st = inspect_lock.server_state(paths)
    res["lock_after_next"] = {"current": st.current, "prepared": st.prepared}
    note("lock after Next", **res["lock_after_next"])

    # --- nothing written into 4_Estimates or the cwd
    check("4_Estimates unchanged", folder_hashes(ROOT / "4_Estimates") == est_before)
    check("nothing new in the cwd (repo root)", sorted(os.listdir(REPO)) == res["cwd_before"])
    rec = registry.load_record(paths, NEXT_SID)
    check(f"{NEXT_SID} still waits (not saved)",
          not [d for d in rec.decisions if d.what == "manual_inspection"])

    # --- close the tab; the server stops by itself after idle_exit_s
    for s in sessions:
        s.close()
    t = time.perf_counter()
    code = None
    try:
        code = server.wait(timeout=120)
    except subprocess.TimeoutExpired:
        pass
    t_exit = time.perf_counter() - t
    check("server exits by itself after the tab is closed", code == 0, code=code, seconds_after_close=round(t_exit, 1))
    st = inspect_lock.server_state(paths)
    res["lock_after_exit"] = {"exists": paths.inspection_lock.exists(), "alive": st.alive, "url": st.url}
    check("lock released", not st.alive, **res["lock_after_exit"])
except BaseException as e:  # noqa: BLE001
    res["error"] = "".join(traceback.format_exception(e))
    note("[FAILED] driver error", error=repr(e))
finally:
    for s in sessions:
        try:
            s.close()
        except Exception:
            pass
    if server is not None and server.poll() is None:
        note("killing the server by PID", pid=server.pid)
        try:
            for c in psutil.Process(server.pid).children(recursive=True):
                c.kill()
        except psutil.Error:
            pass
        server.kill()
        server.wait(30)
    res["server_returncode"] = server.returncode if server is not None else None
    res["server_alive_at_end"] = bool(server is not None and psutil.pid_exists(server.pid)
                                      and psutil.Process(server.pid).status() != psutil.STATUS_ZOMBIE)
    try:
        res["server_console"] = log_path.read_text("utf-8", "replace").splitlines()[-40:]
    except OSError:
        pass
    save()
    io.close(all_fds=True)
    print("[OK] written", OUT / "t7_step2.json")
