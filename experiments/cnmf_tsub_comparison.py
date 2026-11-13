"""CNMF-E at tsub 1 vs tsub k on two LT1 sessions (the author's question of 2026-11-12).

Runs the production CNMF-E path of bearmind.stages.cnmf (cnmf_params, decode_to_memmap, write_memmap, the
env the stage sets, pool.start_pool, CNMF(...).fit, evaluate_components). Only tsub, min_corr, min_pnr and
the pool size are overridden. One run at a time, each in a child process inside a Windows job object (peak
commit of the process tree, the number the runner logs), with a free-RAM watchdog. Then every variant is
compared with R1 (tsub 1, full thresholds).

Commands:
  run --sid SID [--variants R1 TK ...]   prepare the session's memmap once and run the variants in order
  cleanup --sid SID                       delete the session's memmap (the estimates stay)
  compare                                 tables, CSV and figures into docs/superpowers/specs/assets/cnmf-tsub/
  autoinspect --sid SID [--variants ...]  optional: the pipeline's autoinspection on each result

Nothing is written into the test series: memmaps, estimates and logs go to the work folder (CNMF_TSUB_WORK),
tables and figures to the docs assets folder.
"""
import os
import sys
from pathlib import Path

# BEARMIND_REPO: the repo whose bearmind package is imported (a frozen copy of this script can run elsewhere)
REPO = Path(os.environ.get("BEARMIND_REPO") or Path(__file__).resolve().parents[1])
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import bearmind  # noqa: E402,F401  first: sets OPENBLAS_CORETYPE before numpy loads, as in the stage process

import argparse  # noqa: E402
import csv  # noqa: E402
import faulthandler  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import multiprocessing  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

SERIES = Path(r"H:\Projects\_bearmind_test\lt1_J03_J06")
WORK = Path(os.environ.get("CNMF_TSUB_WORK") or
            r"C:\Users\User\AppData\Local\Temp\claude\3fb66d12025b3383\8c6833f1-37e2-4703-88da-d2e058320a8f"
            r"\scratchpad\cnmf_tsub")
MAPS_CSV = REPO / "docs" / "superpowers" / "specs" / "assets" / "maps-validation" / "maps_subsample_results.csv"
ASSETS = REPO / "docs" / "superpowers" / "specs" / "assets" / "cnmf-tsub"
BATCH_ROOT = Path(r"H:\Projects\LT1\CalciumData")  # the production batch on this machine: its stage is recorded
SESSIONS = ("LT1_J03_2D", "LT1_J06_2D")

# The production pool (31) needs ~97 GB on J06_2D (runner log), more than the batch leaves free; every variant
# but R1b runs at POOL, R1b at POOL_B. 88-90 patches: 16 workers take 6 rounds, 12 take 8.
# R1r (R1 again, same pool) was added after R1 at pool 16 came out different from the stored pool-31 result.
POOL = 16
POOL_B = 12
START_FREE_GB = 60.0   # start a run only with this much free RAM
KILL_FREE_GB = 15.0    # kill the run when free RAM falls below this
SAMPLE_S = 2.0
RAM_RETRIES = 3
WORKER_RETRIES = 1     # as the runner: one automatic retry after a dead pool worker
POOL_WORKER_DIED = 75  # bearmind.pool.POOL_WORKER_DIED
DEFAULT_ORDER = ("R1", "TK", "T2", "TKf", "R1b", "R1r", "TKi1", "R1p31")
GB = 2 ** 30           # "GB" here and in the runner log is 2**30 bytes


def now():
    return datetime.now().isoformat(timespec="seconds")


def say(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


# ------------------------------------------------------------------------------------------------------------
# configuration and variants
# ------------------------------------------------------------------------------------------------------------

def series_paths():
    from bearmind.paths import SeriesPaths
    return SeriesPaths(SERIES)


def session_cfg(sid):
    """The series config layers as the runner resolves them (read only)."""
    from bearmind import config
    return config.resolve(series_paths(), sid)


def registry(sid):
    with open(series_paths().registry_file(sid), encoding="utf-8") as f:
        return json.load(f)


def maps_rows(sid):
    with open(MAPS_CSV, encoding="utf-8", newline="") as f:
        return {r["variant"]: r for r in csv.DictReader(f) if r["sid"] == sid and r["gsig"] == "4"}


def variant_specs(sid):
    """{name: spec}: tsub, thresholds and pool of each variant; thresholds re-read from the maps CSV."""
    cfg = session_cfg(sid)
    full = (float(cfg["cnmf"]["min_corr"]), float(cfg["cnmf"]["min_pnr"]))
    rows = maps_rows(sid)
    t2, tk = rows["tsub2"], rows["tsubK"]
    for r in (t2, tk):  # the CSV's full suggestion is the threshold pair the session runs with
        if abs(float(r["full_min_corr"]) - full[0]) > 1e-9 or abs(float(r["full_min_pnr"]) - full[1]) > 1e-9:
            raise ValueError(f"{sid}: maps CSV full thresholds differ from the session config {full}")
    k = int(tk["tsub"])
    if int(t2["tsub"]) != 2 or k < 3:
        raise ValueError(f"{sid}: unexpected tsub in the maps CSV ({t2['tsub']}, {tk['tsub']})")

    def spec(tsub, corr, pnr, pool, thr, **extra):
        s = {"tsub": int(tsub), "min_corr": float(corr), "min_pnr": float(pnr), "pool": int(pool), "thr": thr}
        if extra:
            s["extra"] = extra
        return s

    return {
        "R1": spec(1, *full, POOL, "full maps"),
        # R1 again at the same pool: tells run-to-run variation from the pool-size effect (R1b)
        "R1r": spec(1, *full, POOL, "full maps"),
        # R1 at the production pool, only when the batch leaves room (J03_2D: ~72 GB at pool 31 in the runner
        # log): is the difference from the stored result the pool size?
        "R1p31": spec(1, *full, 31, "full maps", start_free_gb=110.0),
        "R1b": spec(1, *full, POOL_B, "full maps"),
        "T2": spec(2, t2["var_min_corr"], t2["var_min_pnr"], POOL, "tsub2 maps"),
        "TK": spec(k, tk["var_min_corr"], tk["var_min_pnr"], POOL, f"tsub{k} maps"),
        "TKf": spec(k, *full, POOL, "full maps"),
        # mechanism check, not a production setting: TK without the residual search of the initialization
        # (init_iter 1). At tsub > 1 that search compares the residual's PNR with min_pnr not scaled by
        # sqrt(tsub) (initialization.py:1267-1273), the first search with min_pnr * sqrt(tsub) (:1208-1211).
        "TKi1": spec(k, tk["var_min_corr"], tk["var_min_pnr"], POOL, f"tsub{k} maps", init_iter=1),
    }


def production_params(sid):
    """cnmf_params exactly as the stage builds them; checked against the params of the stored production run."""
    from bearmind.stages import cnmf as st
    from bearmind.video import video_info
    cfg = session_cfg(sid)
    reg = registry(sid)
    T, h, w, _ = video_info(series_paths().mc_video(sid))
    params = st.cnmf_params(cfg, reg["fps"], (T, h, w))
    stored = reg["stages"]["cnmf"]["params"]
    if json.loads(json.dumps(params)) != stored:
        raise ValueError(f"{sid}: params differ from the stored production run:\n{params}\n{stored}")
    return params, (T, h, w)


# ------------------------------------------------------------------------------------------------------------
# memmap (once per session, shared by all variants: the same bytes the stage writes on every run)
# ------------------------------------------------------------------------------------------------------------

def prepare(sid):
    from bearmind.stages import cnmf as st
    from bearmind.video import decode_to_memmap, video_info
    d = WORK / sid
    d.mkdir(parents=True, exist_ok=True)
    mc = series_paths().mc_video(sid)
    T, h, w, _ = video_info(mc)
    mm = d / st.memmap_name("Yr", (h, w), T)
    if mm.exists() and mm.stat().st_size == 4 * T * h * w:
        return mm
    say(f"{sid}: decoding {mc.name} ({T} x {h} x {w})")
    u8_path = d / "mc_u8.dat"
    t0 = time.perf_counter()
    u8 = decode_to_memmap(mc, u8_path)
    t1 = time.perf_counter()
    # the strip size does not change the bytes written (write_memmap docstring); 4 GB keeps RAM low
    mm = st.write_memmap(u8, d, 4.0)
    t2 = time.perf_counter()
    u8 = None
    gc.collect()
    u8_path.unlink(missing_ok=True)
    (d / "prepare.json").write_text(json.dumps({"sid": sid, "memmap": str(mm), "T": T, "h": h, "w": w,
                                                "decode_s": t1 - t0, "memmap_s": t2 - t1, "when": now()},
                                               indent=1), encoding="utf-8")
    say(f"{sid}: memmap ready in {t2 - t0:.0f} s: {mm.name}")
    return mm


def cleanup(sid):
    for f in (WORK / sid).glob("*.mmap"):
        f.unlink()
        say(f"deleted {f}")
    for f in (WORK / sid).glob("mc_u8.dat"):
        f.unlink()


# ------------------------------------------------------------------------------------------------------------
# the child process: one CNMF-E fit as the stage does it
# ------------------------------------------------------------------------------------------------------------

def child_entry(js, conn):
    log_file = open(js["log"], "ab", buffering=0)
    os.environ["PYTHONFAULTHANDLER"] = "1"  # as bearmind.isolate: pool workers dump native crashes too
    os.environ["MPLBACKEND"] = "Agg"
    faulthandler.enable(log_file, all_threads=True)
    for fd in (1, 2):
        os.dup2(log_file.fileno(), fd)
    sys.stdout = open(1, "w", encoding="utf-8", errors="backslashreplace", buffering=1, closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", errors="backslashreplace", buffering=1, closefd=False)
    if conn.recv() != "go":  # the parent has put this process into its job object
        return
    try:
        msg = ("ok", (autoinspect_variant if js.get("task") == "autoinspect" else fit_variant)(js))
    except BaseException as e:
        msg = ("error", "".join(traceback.format_exception(type(e), e, e.__traceback__)))
    try:
        conn.send(msg)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        conn.close()
        os._exit(0)


class _Marks(logging.Handler):
    """Times of CaImAn's main-process log records: they split the fit into patches and the rest."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.marks = []

    def emit(self, record):
        if len(self.marks) < 5000:
            self.marks.append((time.perf_counter(), record.getMessage()[:160]))


def fit_variant(js):
    import caiman as cm
    from caiman.source_extraction.cnmf.cnmf import CNMF
    from caiman.source_extraction.cnmf.params import CNMFParams

    from bearmind import pool

    params = dict(js["params"])
    params["gSig"], params["gSiz"] = tuple(params["gSig"]), tuple(params["gSiz"])  # as cnmf_params gives them
    tmp = Path(js["caiman_temp"])
    tmp.mkdir(parents=True, exist_ok=True)
    # the stage's env, set after numpy loaded (main process keeps its BLAS threads, workers get one each)
    os.environ["CAIMAN_TEMP"] = str(tmp)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    lg = logging.getLogger("caiman")
    lg.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    lg.addHandler(h)
    marks = _Marks()
    lg.addHandler(marks)

    n = int(js["pool"])
    dview = cnm = est = Yr = images = None
    try:
        Yr, dims, T = cm.load_memmap(js["mmap"])
        images = Yr.T.reshape((T,) + dims, order="F")
        t0 = time.perf_counter()
        dview = pool.start_pool(n)
        t1 = time.perf_counter()
        opts = CNMFParams(params_dict=params)
        cnm = CNMF(n, dview=dview, params=opts)
        avail = psutil.virtual_memory().available / GB
        t2 = time.perf_counter()
        cnm.fit(images)
        t3 = time.perf_counter()
        est = cnm.estimates
        est.evaluate_components(images, opts, dview=dview)
        t4 = time.perf_counter()
        pool.stop_pool(dview)
        dview = None

        est.cnmf_dict = params
        est.patch_workers = n
        est.mc_video = js["mc_video"]
        est.tsub_experiment = {k: js[k] for k in ("sid", "variant", "spec")}

        out = Path(js["out"])
        part = out.with_name(out.name + ".partial")
        with open(part, "wb") as f:
            pickle.dump(est, f)
        os.replace(part, out)

        t_patch_end = next((t for t, m in marks.marks if m.startswith("Elapsed time for processing patches")),
                           None)
        idx = getattr(est, "idx_components", None)
        return {
            "n_components": int(est.A.shape[1]),
            "n_accepted": None if idx is None else int(len(idx)),
            "T": int(T), "dims": [int(x) for x in dims],
            "pool_start_s": t1 - t0,
            "fit_s": t3 - t2,
            "patches_s": None if t_patch_end is None else t_patch_end - t2,
            "after_patches_s": None if t_patch_end is None else t3 - t_patch_end,
            "evaluate_s": t4 - t3,
            "avail_at_fit_gb": avail,
            "n_pixels_per_process": cnm.params.get("preprocess", "n_pixels_per_process"),
        }
    finally:
        if dview is not None:
            pool.stop_pool(dview)
        cnm = est = Yr = images = None
        gc.collect()


def autoinspect_variant(js):
    """The call of bearmind.stages.autoinspect.run on one result; its files go to the work folder."""
    import ae_launch
    from bearmind.stages.autoinspect import _REPO_ROOT, check_aligned
    params = dict(js["ai_params"])
    model = _REPO_ROOT / params.pop("model_path")
    out_dir = Path(js["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # fit_variant skips the stage's imax; autoinspection reads its shape (polygon.get_contours), so it is set
    # here as cnmf._imax does when the threshold maps have the gSig: pnr_to_imax of the stored PNR map
    import bm_qc
    with open(js["source"], "rb") as f:
        est = pickle.load(f)
    with np.load(js["maps"]) as d:
        est.imax = bm_qc.pnr_to_imax(np.asarray(d[f"pnr_{int(js['gSig'])}"], np.float32))
    src = out_dir / "estimates_with_imax.pickle"
    with open(src, "wb") as f:
        pickle.dump(est, f)
    del est
    check_aligned(src)
    t0 = time.perf_counter()
    result = ae_launch.run_auto_inspection(str(src), fps=int(js["fps_int"]), session_name=js["sid"],
                                           artifacts_dir=js["out_dir"], ml_model_path=str(model), **params)
    t1 = time.perf_counter()
    s = result["summary"]
    df = result["estimates"].metrics_df

    def count(col, val):
        return int((df[col] == val).sum()) if df is not None and col in df.columns else 0

    # footprints of the kept components (delete == 0, the stage's n_kept): the "final cells" compared later.
    # metrics_df.component_idx indexes the columns of the processed estimates (auto_inspector docstring).
    import scipy.sparse as sp
    est_p = result["estimates"]
    kept = np.sort(df.loc[df["delete"] == 0, "component_idx"].to_numpy(int)) if df is not None else np.zeros(0, int)
    if len(kept) and kept.max() >= est_p.A.shape[1]:
        raise ValueError(f"component_idx {kept.max()} outside A with {est_p.A.shape[1]} columns")
    sp.save_npz(out_dir / "kept_A.npz", sp.csc_matrix(est_p.A)[:, kept].astype(np.float32))
    np.save(out_dir / "kept_C.npy", np.asarray(est_p.C)[kept].astype(np.float32))
    src.unlink(missing_ok=True)  # a 100-200 MB copy; the variant's own estimates stay
    n0, nd = int(s["n_initial"]), int(s["n_deleted"])
    return {"n_initial": n0, "n_after_merge": int(s["n_final"]), "n_deleted": nd,
            "n_merge_groups": count("decision", "from_merge"), "n_kept": count("delete", 0),
            "deleted_frac": nd / n0 if n0 else 0.0, "autoinspect_s": t1 - t0}


# ------------------------------------------------------------------------------------------------------------
# the parent: one run at a time, job object, free-RAM watchdog
# ------------------------------------------------------------------------------------------------------------

def batch_state():
    try:
        with open(BATCH_ROOT / "_bearmind" / "heartbeat.json", encoding="utf-8") as f:
            hb = json.load(f)
        cur = hb.get("current") or {}
        return {"pid": hb.get("pid"), "sid": cur.get("sid"), "stage": cur.get("stage"), "status": hb.get("status"),
                "time": hb.get("time")}
    except Exception as e:
        return {"error": repr(e)}


def tree_private_gb(pid):
    total = 0
    try:
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
    except psutil.Error:
        return 0.0
    for p in procs:
        try:
            total += p.memory_info().private
        except psutil.Error:
            pass
    return total / GB


def launch(js, mem_csv):
    """One child run; status ok / ram_kill / worker_died / error / crash, with peak commit and wall time."""
    from bearmind.isolate import _Job
    ctx = multiprocessing.get_context("spawn")
    here, there = ctx.Pipe()
    proc = ctx.Process(target=child_entry, args=(js, there), name=f"cnmf_tsub {js['variant']}")
    t0 = time.monotonic()
    started = now()
    proc.start()
    there.close()
    job = None
    try:
        job = _Job(proc.pid)
    except OSError as e:
        say(f"[FAILED] job object: {e}; peak commit from the 2 s samples only")
    here.send("go")
    b0 = batch_state()
    status, msg, min_free, peak_sampled = None, None, float("inf"), 0.0
    batch_pid = b0.get("pid")
    new = not mem_csv.exists()
    with open(mem_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["attempt_start", "t_s", "free_gb", "tree_private_gb", "job_peak_gb", "batch_private_gb"])
        while True:
            if here.poll(SAMPLE_S):
                try:
                    msg = here.recv()
                except (EOFError, OSError):
                    msg = None
                break
            if not proc.is_alive():
                try:
                    msg = here.recv() if here.poll(0) else None
                except (EOFError, OSError):
                    msg = None
                break
            free = psutil.virtual_memory().available / GB
            tree = tree_private_gb(proc.pid)
            peak_sampled = max(peak_sampled, tree)
            jp = job.peak_gb() if job is not None else float("nan")
            bg = tree_private_gb(batch_pid) if batch_pid else float("nan")
            w.writerow([started, round(time.monotonic() - t0, 1), round(free, 2), round(tree, 2), round(jp, 2),
                        round(bg, 2)])
            min_free = min(min_free, free)
            if free < KILL_FREE_GB:
                status = "ram_kill"
                say(f"[FAILED] free RAM {free:.1f} GB < {KILL_FREE_GB} GB: killing the run")
                break
    if status is None:
        proc.join(60)
    peak = job.peak_gb() if job is not None else None
    if job is not None:
        job.kill(10)  # the child after its result, or the whole tree on a kill
        job.close()
    else:
        for p in psutil.Process(proc.pid).children(recursive=True) if proc.is_alive() else []:
            p.kill()
        proc.kill()
    proc.join(10)
    here.close()
    code = proc.exitcode
    if status is None:
        if msg is not None and msg[0] == "ok":
            status = "ok"
        elif code == POOL_WORKER_DIED:
            status = "worker_died"
        elif msg is not None and msg[0] == "error":
            status = "error"
        else:
            status = "crash"
    return {"status": status, "exit_code": code, "started": started, "finished": now(),
            "wall_s": round(time.monotonic() - t0, 1), "peak_commit_gb": peak, "peak_sampled_gb": peak_sampled,
            "min_free_gb": min_free, "batch_start": b0, "batch_end": batch_state(),
            "child": msg[1] if status == "ok" else None, "error": msg[1] if status == "error" else None}


def wait_free(min_gb):
    first = True
    while True:
        free = psutil.virtual_memory().available / GB
        if free >= min_gb:
            return free
        if first:
            say(f"waiting: free RAM {free:.1f} GB < {min_gb} GB (batch {batch_state()})")
            first = False
        time.sleep(30)


def run_variant(sid, name, spec, params, mmap):
    d = WORK / sid
    out = d / f"{name}_estimates.pickle"
    meta = d / f"{name}_run.json"
    if out.exists() and meta.exists():
        say(f"{sid} {name}: done before, skipped")
        return
    p = dict(params)
    extra = {k: v for k, v in spec.get("extra", {}).items() if k != "start_free_gb"}  # CaImAn params only
    p.update(tsub=spec["tsub"], min_corr=spec["min_corr"], min_pnr=spec["min_pnr"], **extra)
    js = {"sid": sid, "variant": name, "spec": spec, "params": json.loads(json.dumps(p)), "pool": spec["pool"],
          "mmap": str(mmap), "out": str(out), "log": str(d / f"{name}.log"),
          "caiman_temp": str(WORK / "caiman_temp" / f"{sid}_{name}"),
          "mc_video": series_paths().rel(series_paths().mc_video(sid))}
    attempts = []
    ram_kills = deaths = 0
    while True:
        free = wait_free(max(START_FREE_GB, spec.get("extra", {}).get("start_free_gb", 0)))
        say(f"{sid} {name}: start (tsub {spec['tsub']}, {spec['min_corr']} / {spec['min_pnr']}, pool "
            f"{spec['pool']}), free RAM {free:.1f} GB, batch {batch_state()}")
        a = launch(js, d / f"{name}_mem.csv")
        a["free_at_start_gb"] = free
        attempts.append({k: v for k, v in a.items() if k != "child"})
        say(f"{sid} {name}: {a['status']} after {a['wall_s']:.0f} s, peak commit {a['peak_commit_gb']} GB, "
            f"min free {a['min_free_gb']:.1f} GB")
        if a["status"] == "ok":
            break
        if a["status"] == "ram_kill" and ram_kills < RAM_RETRIES:
            ram_kills += 1
            time.sleep(300)
            continue
        if a["status"] == "worker_died" and deaths < WORKER_RETRIES:
            deaths += 1
            continue
        (d / f"{name}_failed.json").write_text(json.dumps({"spec": spec, "attempts": attempts}, indent=1,
                                                          default=str), encoding="utf-8")
        raise SystemExit(f"[FAILED] {sid} {name}: {a['status']}\n{a.get('error') or ''}")
    rec = {"sid": sid, "variant": name, "spec": spec, "params": js["params"], "attempts": attempts,
           "ram_kills": ram_kills, "worker_deaths": deaths, **{k: v for k, v in a.items() if k != "child"},
           **a["child"]}
    meta.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")


def cmd_run(sid, variants):
    specs = variant_specs(sid)
    params, shape = production_params(sid)
    say(f"{sid}: movie {shape}, specs {json.dumps(specs)}")
    (WORK / sid).mkdir(parents=True, exist_ok=True)
    (WORK / sid / "specs.json").write_text(json.dumps({"specs": specs, "production_params": params},
                                                      indent=1), encoding="utf-8")
    mmap = prepare(sid)
    for name in variants:
        run_variant(sid, name, specs[name], params, mmap)
    say(f"{sid}: [OK] variants {' '.join(variants)} done")


def cmd_autoinspect(sid, variants):
    """Optional: the pipeline's autoinspection (production parameters) on each result, one at a time."""
    cfg = session_cfg(sid)
    fps_int = round(registry(sid)["fps"])  # the stage passes integer fps
    d = WORK / sid
    for name in variants:
        # the stored production result is not run here: its autoinspection counts are in the series registry
        src = d / f"{name}_estimates.pickle"
        meta = d / f"ai_{name}.json"
        if meta.exists() or not src.exists():
            say(f"{sid} autoinspect {name}: {'done before' if meta.exists() else 'no estimates'}, skipped")
            continue
        js = {"task": "autoinspect", "sid": sid, "variant": f"ai_{name}", "source": str(src),
              "out_dir": str(d / f"ai_{name}"), "ai_params": dict(cfg["autoinspect"]), "fps_int": fps_int,
              "maps": str(series_paths().threshold_maps(sid)), "gSig": int(cfg["cnmf"]["gSig"]),
              "log": str(d / f"ai_{name}.log")}
        free = wait_free(START_FREE_GB)
        say(f"{sid} autoinspect {name}: start, free RAM {free:.1f} GB")
        a = launch(js, d / f"ai_{name}_mem.csv")
        say(f"{sid} autoinspect {name}: {a['status']} after {a['wall_s']:.0f} s, peak {a['peak_commit_gb']} GB")
        if a["status"] != "ok":
            say(f"[FAILED] {a.get('error') or a['status']}")
            continue
        meta.write_text(json.dumps({**{k: v for k, v in a.items() if k != "child"}, **a["child"]}, indent=1,
                                   default=str), encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------
# comparison with R1
# ------------------------------------------------------------------------------------------------------------

MASK_FRAC = 0.2     # footprint mask: pixels above this fraction of the component's maximum
MAX_DIST = 10.0     # centroids farther apart than this (px) never match (register_ROIs default max_dist)
IOU_CUTS = (0.5, 0.3)  # main cut-off and the looser one of register_ROIs (thresh_cost 0.7)
PARTIAL_IOU = 0.1   # an unmatched component overlapping one of the other run at least this much is "partial"
EDGE_PX = 10        # "at the edge": centroid closer than this to the frame border


def load_est(path):
    with open(path, "rb") as f:
        est = pickle.load(f)
    A = est.A.tocsc().astype(np.float32)
    n = A.shape[1]
    idx = getattr(est, "idx_components", None)
    acc = np.zeros(n, bool)
    acc[np.arange(n) if idx is None else np.asarray(idx, int)] = True
    d = {"A": A, "C": np.asarray(est.C, np.float32), "acc": acc, "n": n,
         "snr": np.asarray(est.SNR_comp, float), "rval": np.asarray(est.r_values, float),
         "pool": getattr(est, "patch_workers", None)}
    del est
    return d


def footprint_masks(A, dims):
    """Boolean masks (pixels x K, csc), mask sizes, mask-weighted centroids (row, col), peak pixel of each A."""
    h, w = dims
    A = A.tocsc()
    rows, cols, sizes, coms, peaks = [], [], [], [], []
    for k in range(A.shape[1]):
        lo, hi = A.indptr[k], A.indptr[k + 1]
        pix, val = A.indices[lo:hi], A.data[lo:hi]
        peaks.append(int(pix[np.argmax(val)]) if len(val) else -1)
        keep = val > MASK_FRAC * val.max() if len(val) else np.zeros(0, bool)
        p, v = pix[keep], val[keep]
        rows.append(p)
        cols.append(np.full(len(p), k))
        sizes.append(len(p))
        r, c = p % h, p // h  # CaImAn numbers pixels in F order
        coms.append((float(np.average(r, weights=v)), float(np.average(c, weights=v))) if len(p) else
                    (np.nan, np.nan))
    import scipy.sparse as sp
    r = np.concatenate(rows) if rows else np.zeros(0, int)
    c = np.concatenate(cols) if cols else np.zeros(0, int)
    M = sp.csc_matrix((np.ones(len(r), np.float32), (r, c)), shape=(h * w, A.shape[1]))
    return M, np.asarray(sizes), np.asarray(coms), np.asarray(peaks)


def match(f1, f2):
    """One-to-one matching (Hungarian on 1 - IoU, centroids < MAX_DIST); returns pairs, their IoU, IoU matrix."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    M1, s1, c1, _ = f1
    M2, s2, c2, _ = f2
    inter = (M1.T @ M2).toarray()
    union = s1[:, None] + s2[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    gated = iou.copy()
    gated[cdist(c1, c2) >= MAX_DIST] = 0
    i, j = linear_sum_assignment(1 - gated)
    keep = gated[i, j] > 0
    return i[keep], j[keep], gated[i[keep], j[keep]], iou


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def pair_metrics(e1, e2, i, j):
    """Pearson r of the C traces and of the footprints (over the union of their supports) of matched pairs."""
    tr, fp = [], []
    for a, b in zip(i, j):
        tr.append(_corr(e1["C"][a].astype(float), e2["C"][b].astype(float)))
        x = e1["A"][:, a].toarray().ravel()
        y = e2["A"][:, b].toarray().ravel()
        u = (x > 0) | (y > 0)
        fp.append(_corr(x[u].astype(float), y[u].astype(float)))
    return np.asarray(tr), np.asarray(fp)


def session_maps(sid):
    import bm_qc
    with np.load(series_paths().threshold_maps(sid)) as d:
        maps = {"cn": np.asarray(d["cn_4"], np.float32), "pnr": np.asarray(d["pnr_4"], np.float32),
                "mean": np.asarray(d["mean"], np.float32), "gSig": 4}
    maps["dark"] = ~bm_qc.valid_mask(maps, border_pix=0)  # dark area as the seed search excludes it
    return maps


def describe(e, f, maps, dims, thr):
    """Per-component table: accepted, SNR, r, mask size, centroid, edge distance, dark area, maps at the peak."""
    h, w = dims
    _, sizes, coms, peaks = f
    pr, pc = peaks % h, peaks // h
    cn, pnr = maps["cn"][pr, pc], maps["pnr"][pr, pc]
    rr, cc = np.clip(np.round(coms[:, 0]).astype(int), 0, h - 1), np.clip(np.round(coms[:, 1]).astype(int), 0, w - 1)
    edge = np.minimum.reduce([coms[:, 0], h - 1 - coms[:, 0], coms[:, 1], w - 1 - coms[:, 1]])
    return {"acc": e["acc"], "snr": e["snr"], "rval": e["rval"], "size": sizes, "row": coms[:, 0],
            "col": coms[:, 1], "edge_px": edge, "dark": maps["dark"][rr, cc], "cn_peak": cn, "pnr_peak": pnr,
            "seed_ok_full": (cn >= thr[0]) & (pnr >= thr[1]),
            "mean_rel": maps["mean"][rr, cc] / np.median(maps["mean"])}  # brightness at the centroid


def _med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def _q(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else np.nan


def compare_pair(sid, name, e0, f0, d0, e, f, dv):
    """Summary row of variant `name` against the reference (R1) and per-component rows of both sides."""
    i, j, iou_p, iou = match(f0, f)
    tr, fp = pair_metrics(e0, e, i, j)
    row = {"sid": sid, "variant": name}
    comp_rows = []
    for cut in IOU_CUTS:
        ok = iou_p >= cut
        ii, jj = i[ok], j[ok]
        s = f"{cut:.1f}"
        row[f"n_matched_iou{s}"] = int(ok.sum())
        row[f"frac_ref_matched_iou{s}"] = ok.sum() / max(e0["n"], 1)
        row[f"frac_var_matched_iou{s}"] = ok.sum() / max(e["n"], 1)
        # accepted components of one run found among all components of the other
        row[f"frac_ref_acc_found_iou{s}"] = np.isin(np.flatnonzero(e0["acc"]), ii).mean() if e0["acc"].any() else np.nan
        row[f"frac_var_acc_found_iou{s}"] = np.isin(np.flatnonzero(e["acc"]), jj).mean() if e["acc"].any() else np.nan
        # accepted vs accepted
        both = e0["acc"][ii] & e["acc"][jj]
        row[f"n_acc_both_iou{s}"] = int(both.sum())
        row[f"frac_ref_acc_in_var_acc_iou{s}"] = both.sum() / max(e0["acc"].sum(), 1)
        row[f"frac_var_acc_in_ref_acc_iou{s}"] = both.sum() / max(e["acc"].sum(), 1)
    ok = iou_p >= IOU_CUTS[0]
    row["median_iou_matched"] = _med(iou_p[ok])
    row["median_trace_r"] = _med(tr[ok])
    row["p10_trace_r"] = _q(tr[ok], 10)
    row["median_footprint_r"] = _med(fp[ok])
    row["p10_footprint_r"] = _q(fp[ok], 10)
    m0 = np.zeros(e0["n"], bool)
    m0[i[ok]] = True
    m1 = np.zeros(e["n"], bool)
    m1[j[ok]] = True
    for side, ee, dd, mm, other_iou, partner, tr_s, fp_s, iou_s in (
            ("ref", e0, d0, m0, iou, dict(zip(i[ok], j[ok])), dict(zip(i[ok], tr[ok])), dict(zip(i[ok], fp[ok])),
             dict(zip(i[ok], iou_p[ok]))),
            ("var", e, dv, m1, iou.T, dict(zip(j[ok], i[ok])), dict(zip(j[ok], tr[ok])), dict(zip(j[ok], fp[ok])),
             dict(zip(j[ok], iou_p[ok])))):
        best = other_iou.max(1) if other_iou.shape[1] else np.zeros(ee["n"])
        un = ~mm
        row[f"{side}_unmatched"] = int(un.sum())
        row[f"{side}_unmatched_partial"] = int((un & (best >= PARTIAL_IOU)).sum())
        row[f"{side}_unmatched_acc"] = int((un & dd["acc"]).sum())
        row[f"{side}_unmatched_median_snr"] = _med(dd["snr"][un])
        row[f"{side}_matched_median_snr"] = _med(dd["snr"][mm])
        row[f"{side}_unmatched_median_pnr_peak"] = _med(dd["pnr_peak"][un])
        row[f"{side}_matched_median_pnr_peak"] = _med(dd["pnr_peak"][mm])
        row[f"{side}_unmatched_edge_or_dark"] = int((un & ((dd["edge_px"] < EDGE_PX) | dd["dark"])).sum())
        row[f"{side}_unmatched_acc_median_snr"] = _med(dd["snr"][un & dd["acc"]])
        row[f"{side}_unmatched_acc_median_mean_rel"] = _med(dd["mean_rel"][un & dd["acc"]])
        row[f"{side}_matched_acc_median_mean_rel"] = _med(dd["mean_rel"][mm & dd["acc"]])
        row[f"{side}_unmatched_dim"] = int((un & (dd["mean_rel"] < 0.5)).sum())
        row[f"{side}_unmatched_seed_ok_full"] = int((un & dd["seed_ok_full"]).sum())
        for k in range(ee["n"]):
            comp_rows.append({
                "sid": sid, "comparison": f"{name}_vs_R1", "side": "R1" if side == "ref" else name, "comp": k,
                "matched": bool(mm[k]), "partner": partner.get(k, ""), "iou": round(float(iou_s.get(k, 0)), 3),
                "best_iou_any": round(float(best[k]), 3),
                "trace_r": round(float(tr_s[k]), 4) if k in tr_s else "",
                "footprint_r": round(float(fp_s[k]), 4) if k in fp_s else "",
                "accepted": bool(dd["acc"][k]), "snr": round(float(dd["snr"][k]), 3),
                "r_value": round(float(dd["rval"][k]), 3), "size_px": int(dd["size"][k]),
                "row": round(float(dd["row"][k]), 1), "col": round(float(dd["col"][k]), 1),
                "edge_px": round(float(dd["edge_px"][k]), 1), "dark": bool(dd["dark"][k]),
                "mean_rel": round(float(dd["mean_rel"][k]), 3),
                "cn_peak": round(float(dd["cn_peak"][k]), 3), "pnr_peak": round(float(dd["pnr_peak"][k]), 2),
                "seed_ok_full": bool(dd["seed_ok_full"][k])})
    return row, comp_rows, (i[ok], j[ok])


def kept_est(sid, name):
    """Components the autoinspection kept (delete == 0) in run `name`, or None when it was not run."""
    import scipy.sparse as sp
    d = WORK / sid / f"ai_{name}"
    if not (d / "kept_A.npz").exists():
        return None
    A = sp.load_npz(d / "kept_A.npz").tocsc().astype(np.float32)
    return {"A": A, "C": np.load(d / "kept_C.npy"), "n": A.shape[1]}


def compare_kept(k0, k1, dims):
    """Kept components of R1 found among the kept components of a variant (IoU >= main cut), both ways."""
    i, j, iou_p, _ = match(footprint_masks(k0["A"], dims), footprint_masks(k1["A"], dims))
    n = int((iou_p >= IOU_CUTS[0]).sum())
    return {"ai_n_kept": k1["n"], "ai_n_kept_matched": n, "ai_frac_ref_kept_matched": n / max(k0["n"], 1),
            "ai_frac_var_kept_matched": n / max(k1["n"], 1)}


def identical(e0, e1):
    """Bit-identical CNMF-E output (A, C, accepted set), else the largest differences."""
    same_shape = e0["A"].shape == e1["A"].shape and e0["C"].shape == e1["C"].shape
    if not same_shape:
        return False, f"shapes A {e0['A'].shape} vs {e1['A'].shape}"
    dA = abs(e0["A"] - e1["A"]).max()
    dC = float(np.abs(e0["C"] - e1["C"]).max())
    same = dA == 0 and dC == 0 and np.array_equal(e0["acc"], e1["acc"])
    return bool(same), f"max |dA| {dA:.3g}, max |dC| {dC:.3g}, accepted equal {np.array_equal(e0['acc'], e1['acc'])}"


def run_record(sid, name):
    p = WORK / sid / f"{name}_run.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def cmd_compare():
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows, comps, figdata = [], [], {}
    for sid in SESSIONS:
        r1p = WORK / sid / "R1_estimates.pickle"
        if not r1p.exists():
            say(f"{sid}: no R1 yet, skipped")
            continue
        specs = variant_specs(sid)
        maps = session_maps(sid)
        dims = maps["cn"].shape
        full = (specs["R1"]["min_corr"], specs["R1"]["min_pnr"])
        e0 = load_est(r1p)
        f0 = footprint_masks(e0["A"], dims)
        d0 = describe(e0, f0, maps, dims, full)
        stored = load_est(series_paths().estimates(sid))
        same, how = identical(e0, stored)
        say(f"{sid}: R1 vs stored production (pool {stored['pool']}): identical {same} ({how})")
        figdata[sid] = {"maps": maps, "dims": dims, "R1": (e0, f0)}
        others = [("stored", stored, {"spec": {"tsub": 1, "min_corr": full[0], "min_pnr": full[1],
                                               "pool": stored["pool"], "thr": "full maps"}})]
        for name in ("R1r", "R1b", "R1p31", "T2", "TK", "TKf", "TKi1"):
            p = WORK / sid / f"{name}_estimates.pickle"
            if p.exists():
                others.append((name, load_est(p), run_record(sid, name)))
        rec0 = run_record(sid, "R1")
        k0 = kept_est(sid, "R1")
        row0 = summary_row(sid, "R1", specs["R1"], rec0, e0, None, "")
        if k0 is not None:
            row0["ai_n_kept"] = k0["n"]
        reg_ai = (registry(sid)["stages"].get("autoinspect") or {}).get("metrics") or {}
        rows.append(row0)
        for name, e, rec in others:
            f = footprint_masks(e["A"], dims)
            dv = describe(e, f, maps, dims, full)
            row, cr, pairs = compare_pair(sid, name, e0, f0, d0, e, f, dv)
            ident = identical(e0, e) if name in ("stored", "R1r", "R1b", "R1p31") else (None, "")
            if name == "R1p31":  # the question of this run: does it reproduce the stored pool-31 result?
                same_st = identical(stored, e)
                ident = (ident[0], f"{ident[1]}; vs stored: identical={same_st[0]}; {same_st[1]}")
            k1 = kept_est(sid, name)
            if k0 is not None and k1 is not None:
                row.update(compare_kept(k0, k1, dims))
            elif name == "stored" and "n_kept" in reg_ai:
                row["ai_n_kept"] = reg_ai["n_kept"]  # the production autoinspection of the stored result
            rows.append(summary_row(sid, name, rec.get("spec") or specs.get(name), rec, e, row,
                                    "" if ident[0] is None else f"identical={ident[0]}; {ident[1]}"))
            comps.extend(cr)
            figdata[sid][name] = (e, f, pairs)
    write_csv(ASSETS / "cnmf_tsub_variants.csv", rows)
    write_csv(ASSETS / "cnmf_tsub_components.csv", comps)
    say(f"[OK] {len(rows)} variant rows, {len(comps)} component rows -> {ASSETS}")
    make_figures(figdata, rows)


def phase_peaks(sid, name, rec):
    """Peak private memory of the run's tree in the patch phase and after it, from the 2 s samples.

    The child does not report when the fit started relative to the launch; it is taken as the end of the run
    minus fit and evaluation (stopping the pool and pickling, a few seconds, are counted to the fit), so the
    phase boundary is good to about 10 s.
    """
    p = WORK / sid / f"{name}_mem.csv"
    if not p.exists() or not rec.get("patches_s"):
        return None, None
    with open(p, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["attempt_start"] == rec.get("started")]
    if not rows:
        return None, None
    t = np.array([float(r["t_s"]) for r in rows])
    m = np.array([float(r["tree_private_gb"]) for r in rows])
    fit0 = float(rec["wall_s"]) - float(rec["fit_s"]) - float(rec["evaluate_s"])
    cut = fit0 + float(rec["patches_s"])
    a, b = m[t <= cut], m[t > cut]
    return (float(a.max()) if len(a) else None), (float(b.max()) if len(b) else None)


def summary_row(sid, name, spec, rec, e, cmp_row, note):
    spec = spec or {}
    child = rec or {}
    pk_patch, pk_after = phase_peaks(sid, name, child) if child else (None, None)
    out = {"sid": sid, "variant": name, "tsub": spec.get("tsub"), "min_corr": spec.get("min_corr"),
           "min_pnr": spec.get("min_pnr"), "thr_source": spec.get("thr"), "pool": spec.get("pool"),
           "n_components": e["n"], "n_accepted": int(e["acc"].sum()),
           "fit_s": child.get("fit_s"), "patches_s": child.get("patches_s"),
           "after_patches_s": child.get("after_patches_s"), "evaluate_s": child.get("evaluate_s"),
           "run_wall_s": child.get("wall_s"), "peak_commit_gb": child.get("peak_commit_gb"),
           "peak_sampled_gb": child.get("peak_sampled_gb"), "peak_patch_phase_gb": pk_patch,
           "peak_after_patches_gb": pk_after, "min_free_gb": child.get("min_free_gb"),
           "batch_stage_start": (child.get("batch_start") or {}).get("stage"),
           "batch_sid_start": (child.get("batch_start") or {}).get("sid"),
           "batch_stage_end": (child.get("batch_end") or {}).get("stage"),
           "batch_sid_end": (child.get("batch_end") or {}).get("sid"),
           "ram_kills": child.get("ram_kills"), "worker_deaths": child.get("worker_deaths"), "note": note}
    if cmp_row:
        out.update({k: v for k, v in cmp_row.items() if k not in ("sid", "variant")})
    return out


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


# colours: categorical slots of the dataviz reference palette, in fixed order
C_REF, C_VAR, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


CONTOUR_FRAC = 0.5  # figure contours only: the 0.2 masks of neighbouring cells overlap too much to read


def _contours(ax, A, dims, ks, color, lw, ls="-"):
    h, w = dims
    for k in ks:
        a = A[:, k].toarray().ravel().reshape((h, w), order="F")
        if a.max() <= 0:
            continue
        ax.contour(a, levels=[CONTOUR_FRAC * a.max()], colors=[color], linewidths=lw, linestyles=ls)


def make_figures(figdata, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    sids = [s for s in SESSIONS if s in figdata]

    # 1. contours of R1 and of a variant over the correlation image
    for vname in ("TK", "T2"):
        if not all(vname in figdata[s] for s in sids):
            continue
        fig, axs = plt.subplots(2, len(sids), figsize=(6.2 * len(sids), 10.6), squeeze=False)
        for col_i, sid in enumerate(sids):
            fd = figdata[sid]
            e0, f0 = fd["R1"]
            e, f, (i_all, j_all) = fd[vname]
            cn = fd["maps"]["cn"]
            lo, hi = np.percentile(cn, [30, 99.9])  # J06_2D: cn is above 0.8 almost everywhere
            k = next((r["tsub"] for r in rows if r["sid"] == sid and r["variant"] == vname), "?")
            for row_i, acc_only in enumerate((False, True)):
                ax = axs[row_i][col_i]
                if acc_only:  # the same one-to-one pairs, kept where both sides are accepted
                    both = e0["acc"][i_all] & e["acc"][j_all]
                    i, j = i_all[both], j_all[both]
                    s0, s1 = np.flatnonzero(e0["acc"]), np.flatnonzero(e["acc"])
                    what = "accepted components"
                else:
                    i, j = i_all, j_all
                    s0, s1 = np.arange(e0["n"]), np.arange(e["n"])
                    what = "all components"
                # dimmed to grey 0.1-0.6 so that the white contours stay visible over bright cells
                ax.imshow(0.1 + 0.5 * np.clip((cn - lo) / (hi - lo), 0, 1), cmap="gray", vmin=0, vmax=1)
                _contours(ax, e0["A"], fd["dims"], i, "#ffffff", 0.5)
                _contours(ax, e0["A"], fd["dims"], np.setdiff1d(s0, i), C_REF, 1.3)
                _contours(ax, e["A"], fd["dims"], np.setdiff1d(s1, j), C_VAR, 1.3)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{sid}, {what}: R1 (tsub 1, {len(s0)}) vs {vname} (tsub {k}, {len(s1)})\n"
                             f"matched {len(i)}, only R1 {len(s0) - len(i)}, only {vname} {len(s1) - len(j)}",
                             color=INK, fontsize=9)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color="#bbbbbb", lw=1, label=f"matched (R1 contour, IoU >= {IOU_CUTS[0]})"),
                   Line2D([], [], color=C_REF, lw=1.6, label="only in R1"),
                   Line2D([], [], color=C_VAR, lw=1.6, label=f"only in {vname}")]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.018))
        fig.text(0.01, 0.003, f"background: correlation image of the full threshold maps, gSig 4 (30th-99.9th "
                              f"percentile, dimmed); "
                              f"contour at {CONTOUR_FRAC} of each footprint's maximum; matching uses masks at "
                              f"{MASK_FRAC}; accepted = evaluate_components", color=MUTED, fontsize=7)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(ASSETS / f"contours_R1_vs_{vname}.png", dpi=130)
        plt.close(fig)

    # 2. matched fractions per variant (noise floor first)
    order = ["R1r", "R1b", "R1p31", "stored", "T2", "TK", "TKf", "TKi1"]
    labels = {"R1r": f"R1r\n(repeat,\npool {POOL})", "stored": "stored\n(pool 31)", "R1b": f"R1b\n(pool {POOL_B})",
              "R1p31": "R1p31\n(pool 31)",
              "T2": "T2\n(tsub 2)",
              "TK": "TK\n(tsub k)", "TKf": "TKf\n(tsub k,\nfull thr.)", "TKi1": "TKi1\n(TK,\ninit_iter 1)"}
    cut = f"{IOU_CUTS[0]:.1f}"
    levels = [("all CNMF-E components", f"frac_ref_matched_iou{cut}", f"frac_var_matched_iou{cut}"),
              ("accepted by evaluate_components (accepted vs accepted)", f"frac_ref_acc_in_var_acc_iou{cut}",
               f"frac_var_acc_in_ref_acc_iou{cut}")]
    if any(r.get("ai_frac_ref_kept_matched") not in (None, "") for r in rows):
        levels.append(("kept by autoinspection (kept vs kept)", "ai_frac_ref_kept_matched",
                       "ai_frac_var_kept_matched"))
    fig, axs = plt.subplots(len(levels), len(sids), figsize=(5.6 * len(sids), 3.0 * len(levels) + 0.6),
                            squeeze=False, sharey=True)
    for li, (title, ka, kb) in enumerate(levels):
        for ax, sid in zip(axs[li], sids):
            rr = [next((r for r in rows if r["sid"] == sid and r["variant"] == v and r.get(ka) not in (None, "")),
                       None) for v in order]
            names = [v for v, r in zip(order, rr) if r is not None]
            rr = [r for r in rr if r is not None]
            x = np.arange(len(rr))
            a = [float(r[ka]) for r in rr]
            b = [float(r[kb]) for r in rr]
            ax.bar(x - 0.19, a, 0.36, color=C_REF, label="R1 found in the variant")
            ax.bar(x + 0.19, b, 0.36, color=C_VAR, label="variant found in R1")
            for xi, (va, vb) in enumerate(zip(a, b)):
                ax.text(xi - 0.19, va + 0.01, f"{va:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
                ax.text(xi + 0.19, vb + 0.01, f"{vb:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
            ax.set_xticks(x, [labels[v] for v in names], fontsize=7)
            ax.set_ylim(0, 1.15)
            ax.yaxis.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.set_title(f"{sid}: {title}", color=INK, fontsize=8.5)
        axs[li][0].set_ylabel(f"fraction matched\n(one-to-one, IoU >= {IOU_CUTS[0]})")
    handles, lab = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ASSETS / "matched_fractions.png", dpi=130)
    plt.close(fig)

    # 3. fit time and peak commit per variant (two panels, one axis each)
    run_order = [v for v in ("R1", "R1r", "R1b", "R1p31", "T2", "TK", "TKf", "TKi1")
                 if any(r["variant"] == v and r.get("fit_s") not in (None, "") for r in rows)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
    width = 0.8 / max(len(sids), 1)
    for si, (sid, col) in enumerate(zip(sids, (C_REF, C_VAR))):
        rr = [next((r for r in rows if r["sid"] == sid and r["variant"] == v), None) for v in run_order]
        x = np.arange(len(run_order)) + (si - (len(sids) - 1) / 2) * width

        def val(r, key):
            return float(r[key]) if r is not None and r.get(key) not in (None, "") else np.nan

        # fit time: patch phase (solid) + the rest of the fit (light), same session colour
        pt = np.array([val(r, "patches_s") / 60 for r in rr])
        at = np.array([val(r, "after_patches_s") / 60 for r in rr])
        axs[0].bar(x, pt, width * 0.9, color=col, label=f"{sid}: patches")
        axs[0].bar(x, at, width * 0.9, bottom=pt, color=col, alpha=0.4, label=f"{sid}: after patches")
        for xi, v in zip(x, pt + at):
            if np.isfinite(v):
                axs[0].text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
        pk = [val(r, "peak_commit_gb") for r in rr]
        axs[1].bar(x, pk, width * 0.9, color=col, label=sid)
        for xi, v in zip(x, pk):
            if np.isfinite(v):
                axs[1].text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
    pools = {r["variant"]: r.get("pool") for r in rows}
    for ax, lab in zip(axs, ("CNMF-E fit, min (patches + the rest)", "peak commit of the run's process tree, GB")):
        ax.set_xticks(np.arange(len(run_order)), [f"{v}\n(pool {pools.get(v, '?')})" for v in run_order],
                      fontsize=8)
        ax.set_ylabel(lab)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axs[0].legend(frameon=False, fontsize=7)
    axs[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(ASSETS / "time_memory.png", dpi=130)
    plt.close(fig)

    # 4. SNR and PNR at the peak pixel: matched vs unmatched, R1 vs TK
    if all("TK" in figdata[s] for s in sids):
        fig, axs = plt.subplots(2, len(sids), figsize=(5.2 * len(sids), 6.0), squeeze=False)
        rng = np.random.default_rng(0)
        for col_i, sid in enumerate(sids):
            fd = figdata[sid]
            e0, f0 = fd["R1"]
            e, f, (i, j) = fd["TK"]
            maps, dims = fd["maps"], fd["dims"]
            d0 = describe(e0, f0, maps, dims, (0, 0))
            dv = describe(e, f, maps, dims, (0, 0))
            m0 = np.isin(np.arange(e0["n"]), i)
            m1 = np.isin(np.arange(e["n"]), j)
            groups = [("R1\nmatched", d0, m0, C_REF), ("R1\nonly", d0, ~m0, C_REF),
                      ("TK\nonly", dv, ~m1, C_VAR), ("TK\nmatched", dv, m1, C_VAR)]
            for row_i, key in enumerate(("snr", "pnr_peak")):
                ax = axs[row_i][col_i]
                for gi, (lab, dd, mask, colr) in enumerate(groups):
                    y = dd[key][mask]
                    accm = dd["acc"][mask]
                    x = gi + rng.uniform(-0.18, 0.18, len(y))
                    ax.scatter(x[accm], y[accm], s=9, color=colr, alpha=0.8, linewidths=0)
                    ax.scatter(x[~accm], y[~accm], s=9, facecolors="none", edgecolors=colr, alpha=0.8,
                               linewidths=0.6)
                    if len(y):
                        ax.hlines(np.median(y), gi - 0.3, gi + 0.3, color=INK, lw=1)
                ax.set_xticks(range(4), [g[0] for g in groups], fontsize=8)
                ax.set_yscale("log")
                ax.yaxis.grid(True, color=GRID, lw=0.6)
                ax.set_axisbelow(True)
                ax.set_ylabel("SNR_comp (CaImAn)" if key == "snr" else "PNR of the full map at the peak pixel")
                if row_i == 0:
                    ax.set_title(f"{sid}: R1 vs TK", color=INK, fontsize=9)
        fig.text(0.01, 0.005, "filled: accepted by evaluate_components; open: rejected; bar: median",
                 color=MUTED, fontsize=7)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(ASSETS / "unmatched_snr_pnr_TK.png", dpi=130)
        plt.close(fig)


# ------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sid", required=True, choices=SESSIONS)
    r.add_argument("--variants", nargs="+", default=list(DEFAULT_ORDER), choices=list(DEFAULT_ORDER))
    c = sub.add_parser("cleanup")
    c.add_argument("--sid", required=True, choices=SESSIONS)
    sub.add_parser("compare")
    ai = sub.add_parser("autoinspect")
    ai.add_argument("--sid", required=True, choices=SESSIONS)
    ai.add_argument("--variants", nargs="+", default=list(DEFAULT_ORDER), choices=list(DEFAULT_ORDER))
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.sid, a.variants)
    elif a.cmd == "cleanup":
        cleanup(a.sid)
    elif a.cmd == "compare":
        cmd_compare()
    elif a.cmd == "autoinspect":
        cmd_autoinspect(a.sid, a.variants)


if __name__ == "__main__":
    main()
