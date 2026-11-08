r"""Step 3 of the phase 5 check: the batch after the gate.

`python -m bearmind run ROOT --once` as a child process. Allowed to run: gate, autoinspect and export of
NOF_H01_1D, and the gate check of NOF_H03_3D (it must wait). A guard reads the registry every 0.25 s and kills
the runner's process tree if any other stage starts (MC and CNMF-E must not run). The whole run is bounded
(900 s). Afterwards: which estimates autoinspection read (gate params in the registry, the processed pickle's
name, the component count), the export outputs, the session log lines of this run, and that no process of the
runner is left.

usage: python t7_step3.py OUT_DIR
"""
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = r"C:\Users\User\PycharmProjects\bearmind"
sys.path.insert(0, REPO)
import torch  # noqa: F401,E402  (unpickling estimates imports CaImAn)
import numpy as np  # noqa: E402
import psutil  # noqa: E402

from bearmind import registry, runner  # noqa: E402
from bearmind.paths import SeriesPaths  # noqa: E402
from bearmind.registry import State  # noqa: E402

ROOT = Path(r"H:\Projects\_bearmind_test\phase5_short")
SID, OTHER = "NOF_H01_1D", "NOF_H03_3D"
ALLOWED = {(SID, "gate"), (SID, "autoinspect"), (SID, "export"), (OTHER, "gate")}
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
paths = SeriesPaths(ROOT)
res = {"events": [], "checks": {}, "transitions": []}
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


def states():
    recs, broken = registry.read_all(paths)
    return {(r.sid, s): st.state.value for r in recs for s, st in r.stages.items()}


def save():
    (OUT / "t7_step3.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


proc = None
children = {}
log_path = OUT / "t7_runner_console.log"
try:
    recs, _ = registry.read_all(paths)
    res["next_ready_before"] = runner.next_ready(recs, paths)
    note("next_ready before", value=res["next_ready_before"])
    log_sizes = {p.name: p.stat().st_size for p in paths.logs_dir.iterdir()} if hasattr(paths, "logs_dir") else {}
    if not log_sizes:
        log_sizes = {p.name: p.stat().st_size for p in (ROOT / "_bearmind" / "logs").iterdir()}
    res["log_sizes_before"] = log_sizes
    prev = states()
    cmd = [sys.executable, "-u", "-m", "bearmind", "run", str(ROOT), "--once"]
    t_start = datetime.now()
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    res["runner_pid"], res["runner_cmd"] = proc.pid, cmd
    note("runner started", pid=proc.pid)
    pp = psutil.Process(proc.pid)
    deadline = time.monotonic() + 900
    violation = None
    while proc.poll() is None:
        try:
            for c in pp.children(recursive=True):
                if c.pid not in children:
                    try:
                        children[c.pid] = (c.name(), c.create_time(), " ".join(c.cmdline()[:6])[:240])
                    except psutil.Error:
                        pass
        except psutil.Error:
            pass
        try:
            now = states()
        except Exception:  # a record being rewritten: look again
            time.sleep(0.05)
            continue
        for k, v in now.items():
            if prev.get(k) != v:
                res["transitions"].append((datetime.now().isoformat(timespec="milliseconds"), k[0], k[1],
                                           prev.get(k), v))
                note("state", sid=k[0], stage=k[1], old=prev.get(k), new=v)
                if v == "running" and k not in ALLOWED:
                    violation = k
        prev = now
        if violation or time.monotonic() > deadline:
            note("[FAILED] stopping the runner", violation=violation, timeout=time.monotonic() > deadline)
            for c in pp.children(recursive=True):
                c.kill()
            proc.kill()
            break
        time.sleep(0.25)
    proc.wait(60)
    res["runner_returncode"] = proc.returncode
    res["runner_seconds"] = round(time.perf_counter() - T0, 1)
    res["violation"] = violation
    check("runner ended by itself (--once), no forbidden stage", violation is None and proc.returncode == 0,
          code=proc.returncode, seconds=res["runner_seconds"])
    now = states()
    for k, v in now.items():
        if prev.get(k) != v:
            res["transitions"].append((datetime.now().isoformat(timespec="milliseconds"), k[0], k[1], prev.get(k), v))

    # --- the registry after the run
    rec = registry.load_record(paths, SID)
    other = registry.load_record(paths, OTHER)
    res["stages_after"] = {SID: {s: st.state.value for s, st in rec.stages.items()},
                           OTHER: {s: st.state.value for s, st in other.stages.items()}}
    g = rec.stages["gate"]
    manual = paths.manual_result(SID)
    mst = manual.stat()
    res["gate"] = {"state": g.state.value, "metrics": g.metrics, "params": g.params}
    check("gate passed with the decision cleaned", g.state is State.DONE and g.metrics.get("gate") == "cleaned"
          and (g.params.get("decision") or {}).get("value") == "cleaned"
          and g.params.get("manual_result") == {"size": mst.st_size, "mtime_ns": mst.st_mtime_ns},
          metrics=g.metrics, manual_result_param=g.params.get("manual_result"),
          manual_file={"size": mst.st_size, "mtime_ns": mst.st_mtime_ns})
    a = rec.stages["autoinspect"]
    res["autoinspect"] = {"state": a.state.value, "metrics": a.metrics, "outputs": a.outputs,
                          "started": a.started, "finished": a.finished}
    with open(manual, "rb") as f:
        m = pickle.load(f)
    n_manual = len(m.idx_components)
    del m
    processed = ROOT / a.outputs["processed"]
    with open(processed, "rb") as f:
        pe = pickle.load(f)
    res["processed"] = {"file": str(processed), "name_attr": getattr(pe, "name", None),
                        "components": int(pe.A.shape[1]),
                        "metrics_df_rows": None if getattr(pe, "metrics_df", None) is None else len(pe.metrics_df)}
    del pe
    check("autoinspection read the manual result", a.state is State.DONE
          and a.metrics.get("n_initial") == n_manual and str(res["processed"]["name_attr"]).replace("\\", "/")
          .endswith(f"5_Inspection/{SID}/{SID}_manual_estimates.pickle"),
          n_initial=a.metrics.get("n_initial"), n_kept_in_manual_result=n_manual,
          processed_name=res["processed"]["name_attr"])
    x = rec.stages["export"]
    res["export"] = {"state": x.state.value, "metrics": x.metrics, "outputs": x.outputs,
                     "started": x.started, "finished": x.finished}
    check("export done, files present", x.state is State.DONE
          and all((ROOT / v).is_file() for v in x.outputs.values()), outputs=x.outputs, metrics=x.metrics)
    og = other.stages.get("gate")
    check(f"{OTHER} waits at the gate, nothing else ran for it",
          og is not None and og.state is State.WAITING
          and all(other.stages[s].state is State.PENDING for s in ("autoinspect", "export")),
          gate=og.state.value if og else None, gate_error=og.error if og else None)
    for s in ("crop", "mc", "thresholds", "cnmf"):
        for r_ in (rec, other):
            st = r_.stages[s]
            res.setdefault("main_stages_untouched", {})[f"{r_.sid}.{s}"] = st.finished
    # --- the log lines of this run
    lines = {}
    for name in (f"{SID}.log", f"{OTHER}.log", "series.log"):
        p = ROOT / "_bearmind" / "logs" / name
        with open(p, "rb") as f:
            f.seek(log_sizes.get(name, 0))
            text = f.read().decode("utf-8", "replace").splitlines()
        lines[name] = [ln for ln in text if (" INFO " in ln or " ERROR " in ln or " WARNING " in ln
                                             or "Preparing traces" in ln or "fatal" in ln.lower()
                                             or "Traceback" in ln)][:80]
    res["log_lines"] = lines
    for name, ls in lines.items():
        for ln in ls:
            print("   ", name, ln.encode("ascii", "backslashreplace").decode()[:220])
    hb = ROOT / "_bearmind" / "heartbeat.json"
    res["heartbeat"] = json.loads(hb.read_text("utf-8")) if hb.exists() else None
except BaseException as e:  # noqa: BLE001
    res["error"] = "".join(traceback.format_exception(e))
    note("[FAILED] driver error", error=repr(e))
finally:
    if proc is not None and proc.poll() is None:
        note("killing the runner by PID", pid=proc.pid)
        try:
            for c in psutil.Process(proc.pid).children(recursive=True):
                c.kill()
        except psutil.Error:
            pass
        proc.kill()
        proc.wait(30)
    alive = []
    for pid, (name, ct, cmdl) in children.items():
        try:
            if abs(psutil.Process(pid).create_time() - ct) < 1.0:
                alive.append((pid, name, cmdl))
        except psutil.Error:
            pass
    res["children_seen"] = {str(k): v for k, v in children.items()}
    res["children_alive_after"] = alive
    note("children of the runner", seen=len(children), alive_after=len(alive))
    try:
        res["runner_console_tail"] = log_path.read_text("utf-8", "replace").splitlines()[-40:]
    except OSError:
        pass
    save()
    print("[OK] written", OUT / "t7_step3.json")
