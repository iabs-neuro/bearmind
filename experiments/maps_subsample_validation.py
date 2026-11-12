"""Threshold maps from a subset of frames vs maps from all frames (LT1 feedback, item 14).

The thresholds stage builds the CNMF-E correlation (cn) and PNR maps for every gSig from ALL frames of the MC
movie, as CNMF-E initialization sees them (tsub 1). A person picks min_corr / min_pnr on those maps, and CNMF-E
then applies the numbers to maps it computes itself from all frames. This script measures what changes when the
maps are built from a subset of frames:

  dec      every k-th frame of the whole session (k = 5 below 25 fps, 8 at 25 fps and above)
  first10  the first 600 s, all frames (sessions longer than 600 s only)
  mid10    600 s from the middle of the session, all frames (sessions longer than 600 s only)

and when they are built from the temporally downsampled movie (a different question: CaImAn's own tsub is a block
mean of tsub frames, and its initialization compares PNR with min_pnr * sqrt(tsub); compute_init_maps(tsub=k)
reproduces both, so these maps are what CNMF-E initialization would see if CNMF-E itself ran with tsub=k):

  tsub2    compute_init_maps(tsub=2) on all frames
  tsubK    compute_init_maps(tsub=k) on all frames, k as in dec (dec vs tsubK: every k-th frame vs mean of k)

Each variant is compared with the full-frame maps of the same session: the stage's suggestion on both, map
agreement, seeds on the FULL maps at both suggestions, seeds of the variant maps at their own suggestion vs seeds
of the full maps at the full suggestion, the GUI counters, and the wall time of the variant maps.

Reference (full-frame maps): the stage's stored npz when it was made with the current map code and parameters
(stage done, gsig_range 3..7, tsub 1, mirror maps present, finished after the last commit of bm_qc.py and
bearmind/stages/thresholds.py, and neither file modified in the working copy). Otherwise the full maps are
recomputed here for sessions under 20 000 frames, and the session is skipped for longer ones.

One CSV row per session x variant x gSig is appended to <out>/maps_subsample_results.csv. A session x variant
already in the CSV is skipped unless --force, so the script can be rerun on sessions the batch finishes later.

Usage (env python of the bearmind conda env):
  python experiments/maps_subsample_validation.py --root H:/Projects/LT1/CalciumData --sid LT1_J01_1D
  python experiments/maps_subsample_validation.py --root <series root> [--root ...] --work <scratch dir>
  python experiments/maps_subsample_validation.py --figures-only --figure-sid LT1_J01_1D --work <scratch dir>

Safety (the batch runs on the same machine): every heavy step checks that at least 40 GB of physical memory is
available, the process tree of this script is kept under 25 GB (worker count and strip memory are chosen from
the strip model in bm_qc, and the tree is watched while the pool runs), at most 4 worker processes, every wait
has a timeout, and the decoded movie is deleted when the session is done. The input roots are only read.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import inspect
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Before numpy is imported (here and in the spawned workers, which inherit the environment): OpenBLAS commits a
# buffer per thread at import, about 4 GB of commit on this 32-thread machine, which would count against the
# 25 GB limit for nothing; the maps use no multithreaded BLAS (the stage's pool runs one BLAS thread too).
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np  # noqa: E402
import psutil  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import bm_qc  # noqa: E402
from bearmind import config  # noqa: E402
from bearmind.stages.thresholds import _suggest as stage_suggest  # noqa: E402
from bearmind.threshold_model import ThresholdMaps, counters  # noqa: E402
from bearmind.video import _iter_avi, decode_to_memmap, video_info  # noqa: E402

GSIGS = [3, 4, 5, 6, 7]          # the stage's gsig_range; a stored npz with another range is not a reference
TSUB = 1                          # CNMF-E runs with tsub 1 on all frames (the stored reference)
# the stage calls compute_init_maps without thresh_init, so the function default is what CNMF-E is mirrored with
THRESH_INIT = inspect.signature(bm_qc.compute_init_maps).parameters["thresh_init"].default
WINDOW_S = 600.0
MIN_BLOB_PX = 2                   # the GUI's blob size for its counters
OVERLAP_PX = 2.0
MIN_FREE_GB = 40.0
MAX_OWN_GB = 25.0
MAX_WORKERS = 4
RECOMPUTE_MAX_FRAMES = 20000
WORKER_BASE_GB = 0.8              # private memory of a spawned worker before its strips (numpy, cv2, caiman imports)
MAIN_RESERVE_GB = 3.0             # this process (maps of one session, CSV rows) and a margin under the 25 GB
POLL_S = 5.0
CSV_NAME = "maps_subsample_results.csv"
REFCHECK_NAME = "reference_check.csv"
MAP_KEYS = ("cn", "pnr", "cn_neg", "pnr_neg")
MAP_NAMES = {"cn": "cn", "pnr": "pnr", "cn_neg": "cn_mirror", "pnr_neg": "pnr_mirror"}
SUBSET_VARIANTS = ("dec", "first10", "mid10")
TSUB_VARIANTS = ("tsub2", "tsubK")
VARIANTS = SUBSET_VARIANTS + TSUB_VARIANTS
G_SUG = config.fallback_gsig(GSIGS)  # the stage suggests at this gSig only
# Peak of compute_init_maps measured with tracemalloc / peak commit on LT1_J03_1D: 0.84-0.86 of
# bm_qc.init_maps_peak_bytes (the 6-copies model), so the model is used as an upper bound as it is.
PEAK_FACTOR = 1.0


class GuardStop(RuntimeError):
    """Memory or time guard: stop the run (the batch on this machine has priority)."""


# ----------------------------------------------------------------------------------------------------------------
# sessions and references
# ----------------------------------------------------------------------------------------------------------------

def dec_step(fps: float) -> int:
    return 5 if fps < 25 else 8


def variant_specs(n_frames: int, fps: float, variants) -> dict:
    """{variant: (start, stop, step, tsub)} for the variants that apply to the session."""
    n10 = int(round(WINDOW_S * fps))
    out = {}
    for v in variants:
        if v == "dec":
            out[v] = (0, n_frames, dec_step(fps), 1)
        elif v == "tsub2":
            out[v] = (0, n_frames, 1, 2)
        elif v == "tsubK":
            out[v] = (0, n_frames, 1, dec_step(fps))
        elif n_frames > n10:  # a session of 10 min or less: the window would be the whole session
            if v == "first10":
                out[v] = (0, n10, 1, 1)
            elif v == "mid10":
                s = (n_frames - n10) // 2
                out[v] = (s, s + n10, 1, 1)
    return out


def n_used(spec) -> int:
    start, stop, step, _ = spec
    return len(range(start, stop, step))


def load_session(root: Path, sid: str) -> dict:
    rec = json.loads((root / "_bearmind" / "registry" / f"{sid}.json").read_text(encoding="utf-8"))
    th = rec["stages"].get("thresholds", {}) or {}
    mc = rec["stages"].get("mc", {}) or {}
    maps_rel = (th.get("outputs") or {}).get("threshold_maps", f"_bearmind/thresholds/{sid}_maps.npz")
    return {"root": root, "sid": sid, "fps": float(rec["fps"]), "n_frames": int(rec["n_frames"]),
            "mc_video": root / mc["outputs"]["mc_video"], "maps_npz": root / maps_rel, "th": th}


def git(*args) -> str:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


MAP_CODE = ("bm_qc.py", "bearmind/stages/thresholds.py")


def map_code_state() -> dict:
    """Commit time of the last change of the map code (local naive time, as the registry writes), HEAD, dirty."""
    t = dt.datetime.fromisoformat(git("log", "-1", "--format=%cI", "--", *MAP_CODE)).astimezone().replace(tzinfo=None)
    return {"time": t, "head": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", *MAP_CODE))}


def reference_status(s: dict, code: dict) -> tuple[bool, str]:
    """(stored npz usable as the full reference, reason when not)."""
    th = s["th"]
    if th.get("state") != "done":
        return False, f"thresholds stage {th.get('state')}"
    if not s["maps_npz"].is_file():
        return False, "maps npz missing"
    p = th.get("params") or {}
    if [int(g) for g in p.get("gsig_range", [])] != GSIGS or int(p.get("tsub", -1)) != TSUB:
        return False, f"stage params {p}"
    if code["dirty"]:
        return False, "map code modified in the working copy"
    fin = th.get("finished")
    if not fin or dt.datetime.fromisoformat(fin) < code["time"]:
        return False, f"made {fin}, before the last map-code commit {code['time']:%Y-%m-%dT%H:%M}"
    with np.load(s["maps_npz"]) as z:
        need = [f"{k}_{g}" for g in GSIGS for k in ("cn", "pnr", "cn_mirror", "pnr_mirror")] + ["mean"]
        missing = [k for k in need if k not in z.files]
    if missing:
        return False, f"npz lacks {missing[:3]}"
    return True, ""


def stored_maps(npz: Path) -> dict:
    """{g: maps dict as compute_init_maps returns it} from the stage's npz. 'mean' is the stage's projection mean;
    the stage's own suggestion used compute_init_maps' mean, which differs only by float rounding (checked against
    the registry suggestion in the CSV)."""
    out = {}
    with np.load(npz) as z:
        for g in GSIGS:
            out[g] = {"cn": z[f"cn_{g}"], "pnr": z[f"pnr_{g}"], "cn_neg": z[f"cn_mirror_{g}"],
                      "pnr_neg": z[f"pnr_mirror_{g}"], "mean": z["mean"], "gSig": g}
    return out


# ----------------------------------------------------------------------------------------------------------------
# memory guard and worker pool
# ----------------------------------------------------------------------------------------------------------------

def free_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def require_free(what: str) -> None:
    f = free_gb()
    if f < MIN_FREE_GB:
        raise GuardStop(f"{what}: {f:.1f} GB physical memory available, below {MIN_FREE_GB:.0f} GB")


def own_tree_gb() -> float:
    """Private (committed) memory of this process and its workers; the memmapped movie is file cache, not private."""
    me = psutil.Process()
    total = 0
    for p in [me] + me.children(recursive=True):
        try:
            total += p.memory_info().private
        except psutil.Error:
            pass
    return total / 1e9


def _worker_init():
    import cv2
    cv2.setNumThreads(1)  # get_noise_fft switches cv2 threading off anyway; one thread per worker from the start


def run_pool(fn, jobs: list, workers: int, timeout_s: float, label: str) -> tuple[list, float, float]:
    """fn(job) for every job in a spawn pool, one fresh process per job (so its peak memory is its own).
    Returns (results in job order, elapsed s, peak private GB of this process tree)."""
    if not jobs:
        return [], 0.0, 0.0
    require_free(label)
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    peak = 0.0
    pool = ctx.Pool(max(1, min(workers, len(jobs))), initializer=_worker_init, maxtasksperchild=1)
    try:
        handles = [pool.apply_async(fn, (j,)) for j in jobs]
        while not all(h.ready() for h in handles):
            if time.perf_counter() - t0 > timeout_s:
                raise GuardStop(f"{label}: timeout after {timeout_s:.0f} s")
            require_free(label)
            tree = own_tree_gb()
            peak = max(peak, tree)
            if tree > MAX_OWN_GB:
                raise GuardStop(f"{label}: this run uses {tree:.1f} GB, above {MAX_OWN_GB:.0f} GB")
            time.sleep(POLL_S)
        results = [h.get(timeout=60) for h in handles]
        pool.close()
        pool.join()
        return results, time.perf_counter() - t0, peak
    finally:
        pool.terminate()
        pool.join()


def _maps_job(job):
    """Pool worker: the maps of one variant and gSig from the shared uint8 memmap (opened read-only here)."""
    u8_path, shape, offset, variant, g, (start, stop, step, tsub), mem_gb = job
    base = psutil.Process().memory_info().private / 1e9
    movie = np.memmap(u8_path, dtype=np.uint8, mode="r", shape=tuple(shape))
    try:
        t0 = time.perf_counter()
        m = bm_qc.compute_init_maps(movie, gSig=g, tsub=tsub, mirror=True, thresh_init=THRESH_INIT,
                                    frame_slice=slice(start - offset, stop - offset, step), max_mem_gb=mem_gb,
                                    verbose=False)
        sec = time.perf_counter() - t0
    finally:
        del movie
    out = {k: np.asarray(m[k], np.float32) for k in MAP_KEYS + ("mean",)}
    out.update(gSig=g, n_frames=int(m["n_frames"]), tsub=int(m["tsub"]), seconds=sec, base_private_gb=base,
               peak_private_gb=psutil.Process().memory_info().peak_pagefile / 1e9)
    return variant, g, out


def _decode_job(job):
    """Pool worker: decoding in its own process, so the decoder's buffers never stay in the main process."""
    video, out, lo, hi = job
    t0 = time.perf_counter()
    mov = decode_range(Path(video), Path(out), lo, hi)
    shape = tuple(int(x) for x in mov.shape)
    del mov
    return shape, time.perf_counter() - t0


def _suggest_job(job):
    """Pool worker: the stage's suggestion on one maps set, and its seeds / null seeds / FDR on the same maps."""
    tag, g, maps = job
    t0 = time.perf_counter()
    try:
        s = stage_suggest(maps, g)
    except ValueError as e:
        return tag, g, {"error": str(e), "seconds": time.perf_counter() - t0}
    ev = bm_qc.evaluate_thresholds(maps, s["min_corr"], s["min_pnr"], 4 * g + 1)
    s.update(n_seeds=ev["n_seeds"], n_null=ev.get("n_null_seeds"), fdr=ev.get("fdr"),
             seconds=time.perf_counter() - t0)
    return tag, g, s


def job_peak_gb(T_used: int, tsub: int, g: int, mem_gb: float, h: int, w: int) -> float:
    """Peak memory of one compute_init_maps call: the strip model of bm_qc (an upper bound, PEAK_FACTOR), or, for
    tsub > 1, the float32 copy of all T_used frames of a strip that _tsub_mean makes before averaging, whichever
    is larger (the model counts only the downsampled frames)."""
    T_ds = math.ceil(T_used / tsub)
    margin = bm_qc._psf_kernel(g).shape[0] // 2 + 1
    strip = min(h, bm_qc.init_maps_strip_rows(mem_gb, T_ds, h, w, g) + 2 * margin)
    model = PEAK_FACTOR * bm_qc.init_maps_peak_bytes(mem_gb, T_ds, h, w, g)
    # _tsub_mean: the float32 strip of all frames, its block mean, and (T % tsub != 0) the concatenated copy
    pre = (4 * T_used + 8 * T_ds) * w * strip if tsub > 1 else 0
    return max(model, pre) / 1e9


def plan_memory(jobs: list[tuple[int, int, int]], h: int, w: int, workers_max: int) -> tuple[int, float]:
    """(workers, max_mem_gb per call) for jobs [(T_used, tsub, gSig)] so that workers x (job peak + worker base)
    + main reserve stays under the 25 GB limit. Larger strips recompute fewer filter-margin rows, so for every
    worker count the strips get as much memory as fits, and the count with the best workers / margin overhead
    wins."""
    budget = MAX_OWN_GB - MAIN_RESERVE_GB
    best = None
    for n in range(min(workers_max, MAX_WORKERS, len(jobs)), 0, -1):
        per = budget / n - WORKER_BASE_GB
        # the largest max_mem_gb whose heaviest job still fits in `per` (job peaks grow with max_mem_gb)
        lo_, hi_ = 0.05, per
        if max(job_peak_gb(T, k, g, lo_, h, w) for T, k, g in jobs) > per:
            continue  # even 8-row strips do not fit with n workers
        for _ in range(40):
            mid = (lo_ + hi_) / 2
            if max(job_peak_gb(T, k, g, mid, h, w) for T, k, g in jobs) <= per:
                lo_ = mid
            else:
                hi_ = mid
        mem = lo_

        def overhead(T, k, g):
            margin = bm_qc._psf_kernel(g).shape[0] // 2 + 1
            rows = bm_qc.init_maps_strip_rows(mem, math.ceil(T / k), h, w, g)
            return min(h, rows + 2 * margin) / min(h, rows)
        speed = n / np.mean([overhead(T, k, g) for T, k, g in jobs])
        if best is None or speed > best[2]:
            best = (n, mem, speed)
    if best is None:
        raise GuardStop(f"no worker count keeps the strips under {MAX_OWN_GB} GB")
    return best[0], math.floor(best[1] * 100) / 100


# ----------------------------------------------------------------------------------------------------------------
# decoding
# ----------------------------------------------------------------------------------------------------------------

def decode_range(video: Path, out: Path, lo: int, hi: int) -> np.memmap:
    """Frames [lo, hi) of the MC movie as a uint8 (T, h, w) file; the whole file is decoded only when needed."""
    n, h, w, _ = video_info(video)
    need_gb = (hi - lo) * h * w / 1e9
    free_disk = psutil.disk_usage(str(out.parent)).free / 1e9
    if free_disk < need_gb + 20:
        raise GuardStop(f"decode: {free_disk:.0f} GB free on the work disk, {need_gb:.0f} GB needed")
    if lo == 0 and hi == n:
        return decode_to_memmap(video, out)
    k = 0
    try:
        with open(out, "wb") as f:
            buf = []
            for i, fr in enumerate(_iter_avi(video)):
                if i >= hi:
                    break
                if i >= lo:
                    buf.append(fr)
                    if len(buf) == 1000:
                        f.write(np.stack(buf).tobytes())
                        k += len(buf)
                        buf = []
            if buf:
                f.write(np.stack(buf).tobytes())
                k += len(buf)
        if k != hi - lo:
            raise ValueError(f"{video}: decoded {k} of {hi - lo} frames")
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    return np.memmap(out, dtype=np.uint8, mode="r", shape=(hi - lo, h, w))


# ----------------------------------------------------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------------------------------------------------

def top5(x: np.ndarray) -> float:
    """Median of the top 5 % values: the scale of the bright tail, where the thresholds sit."""
    x = x[np.isfinite(x)]
    return float(np.median(x[x >= np.percentile(x, 95)]))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok].astype(np.float64), b[ok].astype(np.float64))[0, 1])


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of the points of a within OVERLAP_PX of a point of b."""
    if len(a) == 0:
        return float("nan")
    if len(b) == 0:
        return 0.0
    from scipy.spatial import cKDTree
    d, _ = cKDTree(b).query(a, k=1)
    return float(np.mean(d <= OVERLAP_PX))


def tmaps(m: dict, g: int) -> ThresholdMaps:
    return ThresholdMaps([g], m["mean"], m["mean"], {g: m["cn"]}, {g: m["pnr"]}, {g: m["cn_neg"]}, {g: m["pnr_neg"]})


def compare(full: dict, var: dict, g: int, sf: dict, sv: dict) -> dict:
    row = {}
    valid = bm_qc.valid_mask(full)  # the pixels seeds are searched in (border 5 px and dark corners excluded)
    for k in MAP_KEYS:
        a, b = full[k][valid], var[k][valid]
        row[f"r_{MAP_NAMES[k]}"] = pearson(a, b)
        row[f"top5_{MAP_NAMES[k]}_ratio"] = top5(b) / top5(a)
    ok_f, ok_v = "error" not in sf, "error" not in sv
    gsiz = 4 * g + 1
    if ok_f:
        si = bm_qc.count_seeds(full["cn"], full["pnr"], sf["min_corr"], sf["min_pnr"], gsiz, valid=valid,
                               min_dist=2 * g)
        row["seeds_full_at_full"] = len(si)
        c = counters(tmaps(full, g), g, sf["min_corr"], sf["min_pnr"], MIN_BLOB_PX)
        row.update(gui_blobs_full=c.n_blobs, gui_single_full=c.single_frac, gui_mirror_full=c.n_mirror_blobs)
    if ok_v:
        sii = bm_qc.count_seeds(full["cn"], full["pnr"], sv["min_corr"], sv["min_pnr"], gsiz, valid=valid,
                                min_dist=2 * g)
        row["seeds_full_at_var"] = len(sii)
        ev = bm_qc.evaluate_thresholds(full, sv["min_corr"], sv["min_pnr"], gsiz)
        row.update(null_full_at_var=ev.get("n_null_seeds"), fdr_full_at_var=ev.get("fdr"))
        c = counters(tmaps(var, g), g, sv["min_corr"], sv["min_pnr"], MIN_BLOB_PX)
        row.update(gui_blobs_var=c.n_blobs, gui_single_var=c.single_frac, gui_mirror_var=c.n_mirror_blobs)
        c = counters(tmaps(full, g), g, sv["min_corr"], sv["min_pnr"], MIN_BLOB_PX)
        row.update(gui_blobs_fullmaps_at_var=c.n_blobs, gui_mirror_fullmaps_at_var=c.n_mirror_blobs)
    if ok_f and ok_v:
        row["seed_overlap_var_in_full"] = overlap(sii, si)
        row["seed_overlap_full_in_var"] = overlap(si, sii)
        # the variant maps at their own suggestion vs the full maps at the full suggestion: for tsub variants this
        # approximates whether CNMF-E initialization with tsub=k starts from the same cells as with tsub 1
        sv_own = bm_qc.count_seeds(var["cn"], var["pnr"], sv["min_corr"], sv["min_pnr"], gsiz,
                                   valid=bm_qc.valid_mask(var), min_dist=2 * g)
        row["seeds_var_at_var"] = len(sv_own)
        row["own_overlap_var_in_full"] = overlap(sv_own, si)
        row["own_overlap_full_in_var"] = overlap(si, sv_own)
    return row


def sugg_cols(prefix: str, s: dict) -> dict:
    if "error" in s:
        return {f"{prefix}_method": "none", f"{prefix}_error": s["error"]}
    return {f"{prefix}_gsig": s["gSig"], f"{prefix}_min_corr": s["min_corr"], f"{prefix}_min_pnr": s["min_pnr"],
            f"{prefix}_method": s["method"], f"{prefix}_nseeds_own": s["n_seeds"],
            f"{prefix}_null_own": s["n_null"], f"{prefix}_fdr_own": s["fdr"]}


# ----------------------------------------------------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_csv(path: Path, rows: list[dict]) -> None:
    """Appends rows; a new column set rewrites the file with the union of the columns (old rows keep blanks)."""
    if not rows:
        return
    old = read_csv(path)
    cols = list(old[0].keys()) if old else []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    if old and cols != list(old[0].keys()):
        rows = old + rows
        mode = "w"
    else:
        mode = "a" if old else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if mode == "w":
            wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


# ----------------------------------------------------------------------------------------------------------------
# one session
# ----------------------------------------------------------------------------------------------------------------

def _cache_path(work: Path, rootname: str, sid: str, v: str) -> Path:
    return work / f"mss_maps_{rootname}_{sid}_{v}.npz"


def save_cached(work, rootname, sid, v, spec, gs, vmaps, vplan, decode) -> None:
    """The maps of one variant with their job timings, so the analysis can be rerun without recomputing."""
    meta = {"spec": list(spec), "gs": list(gs), "plan": vplan, "decode": decode,
            "jobs": {str(g): {k: vmaps[g][k] for k in ("seconds", "peak_private_gb", "base_private_gb", "n_frames",
                                                         "tsub")} for g in gs}}
    np.savez_compressed(_cache_path(work, rootname, sid, v), meta=json.dumps(meta),
                        **{f"{k}_{g}": vmaps[g][k] for g in gs for k in MAP_KEYS + ("mean",)})


def load_cached(work, rootname, sid, v, spec, gs, maps, plan, decode) -> bool:
    p = _cache_path(work, rootname, sid, v)
    if not p.is_file():
        return False
    with np.load(p) as z:
        if "meta" not in z.files:
            return False
        meta = json.loads(str(z["meta"]))
        if meta["spec"] != list(spec) or not set(gs) <= set(meta["gs"]):
            return False
        for g in gs:
            m = {k: z[f"{k}_{g}"] for k in MAP_KEYS + ("mean",)}
            m.update(meta["jobs"][str(g)], gSig=g)
            maps[v][g] = m
    plan[v] = meta["plan"]
    decode.update(meta["decode"])
    return True


def run_session(s: dict, variants, args, code: dict, done_keys: set) -> None:
    sid, root = s["sid"], s["root"]
    tag = f"{root.name}/{sid}"
    ok, why = reference_status(s, code)
    if ok and not args.recompute_full:
        ref = "stored"
    elif s["n_frames"] < RECOMPUTE_MAX_FRAMES:
        ref = "recomputed"
        why = why or "recompute forced (--recompute-full)"
    else:
        print(f"[SKIP] {tag}: stored maps are not a valid reference ({why}) and {s['n_frames']} frames >= "
              f"{RECOMPUTE_MAX_FRAMES}")
        return
    specs = variant_specs(s["n_frames"], s["fps"], variants)
    todo = {v: sp for v, sp in specs.items() if args.force or (root.name, sid, v) not in done_keys}
    if not todo:
        print(f"[SKIP] {tag}: all variants already in the CSV")
        return
    print(f"[..] {tag}: {s['n_frames']} frames at {s['fps']:.2f} fps, reference {ref}"
          f"{' (' + why + ')' if why else ''}, variants {list(todo)}", flush=True)

    need = ({"full": (0, s["n_frames"], 1, 1)} if ref == "recomputed" else {}) | todo

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    u8 = work / f"mss_{root.name}_{sid}_u8.dat"
    gs = args.gsigs
    maps = {v: {} for v in need}
    plan = {}
    decode = {}
    # maps computed by an earlier run (with their timings) are reused, so a failure after the maps costs no recompute
    compute = {v: sp for v, sp in need.items()
               if not (args.reuse_maps and load_cached(work, root.name, sid, v, sp, gs, maps, plan, decode))}
    if len(compute) < len(need):
        print(f"[..] {tag}: maps reused from {work} for {sorted(set(need) - set(compute))}", flush=True)
    if compute:
        lo = min(sp[0] for sp in compute.values())
        hi = max(sp[1] for sp in compute.values())
        try:
            (shape, decode_s), = run_pool(_decode_job, [(str(s["mc_video"]), str(u8), lo, hi)], 1, 3600,
                                          f"{tag} decode")[0]
            decode = {"decode_s": decode_s, "decoded_frames": hi - lo}
            h, w = shape[1], shape[2]
            print(f"[..] {tag}: decoded frames {lo}..{hi} in {decode_s:.0f} s", flush=True)
            # one pool per variant: its wall time is the variant's time, and each variant gets the worker count and
            # strip memory that fit its own frame count (a long tsub2 needs one big worker, a decimated movie
            # four). With one gSig, all variants share one pool (one job each; the job time is the variant time).
            groups = [[v] for v in compute] if len(gs) > 1 else [list(compute)]
            for group in groups:
                jl = [(n_used(need[v]), need[v][3], g) for v in group for g in gs]
                workers, mem_gb = plan_memory(jl, h, w, args.workers)
                jobs = [(str(u8), shape, lo, v, g, need[v], mem_gb) for v in group for g in sorted(gs, reverse=True)]
                res, pool_s, pool_peak = run_pool(_maps_job, jobs, workers, args.timeout_h * 3600,
                                                  f"{tag} {'+'.join(group)} maps")
                for v, g, m in res:
                    maps[v][g] = m
                for v in group:
                    T_used, tsub = n_used(need[v]), need[v][3]
                    plan[v] = {"workers": workers, "max_mem_gb": mem_gb, "pool_wall_s": pool_s,
                               "pool_peak_gb": pool_peak, "shared_pool": len(group) > 1,
                               "pred_peak_gb": max(job_peak_gb(T_used, tsub, g, mem_gb, h, w) for g in gs)}
                    save_cached(work, root.name, sid, v, need[v], gs, maps[v], plan[v], decode)
                    when = (f"jobs {[round(maps[v][g]['seconds']) for g in gs]} s" if len(group) > 1
                            else f"in {pool_s:.0f} s")
                    print(f"[OK] {tag}: {v} maps ({T_used} frames, tsub {tsub}) {when} with {workers} workers x "
                          f"max_mem_gb {mem_gb}; run peak {pool_peak:.1f} GB, worker peaks "
                          f"{max(m['peak_private_gb'] for m in maps[v].values()):.1f} GB", flush=True)
        finally:
            gc.collect()
            try:
                u8.unlink(missing_ok=True)
            except OSError as e:
                print(f"[WARN] {tag}: could not delete {u8}: {e}")
    decode_s = decode.get("decode_s", float("nan"))
    decoded_frames = decode.get("decoded_frames", 0)

    full = stored_maps(s["maps_npz"]) if ref == "stored" else maps["full"]
    if ref == "stored":  # for the figures
        np.savez_compressed(work / f"mss_maps_{root.name}_{sid}_full.npz",
                            **{f"{k}_{g}": full[g][k] for g in gs for k in MAP_KEYS + ("mean",)})

    # the stage's suggestion on every maps set and gSig (the stage itself suggests at G_SUG only)
    sets = {"full": full, **{v: maps[v] for v in todo}}
    sjobs = [(name, g, {**{k: m[g][k] for k in MAP_KEYS + ("mean",)}, "gSig": g})
             for name, m in sets.items() for g in gs]
    sres, sug_s, _ = run_pool(_suggest_job, sjobs, MAX_WORKERS, 3 * 3600, f"{tag} suggestions")
    sug = {(name, g): r for name, g, r in sres}
    print(f"[OK] {tag}: suggestions in {sug_s:.0f} s", flush=True)

    reg = (s["th"].get("metrics") or {}).get("suggested") or {}
    sf0 = sug[("full", G_SUG)]
    reg_match = (bool(reg) and "error" not in sf0 and int(reg.get("gSig", -1)) == G_SUG
                 and abs(float(reg["min_corr"]) - sf0["min_corr"]) < 1e-9
                 and abs(float(reg["min_pnr"]) - sf0["min_pnr"]) < 1e-9 and reg.get("method") == sf0["method"])

    if ref == "recomputed" and s["maps_npz"].is_file():
        write_refcheck(s, maps["full"], sug, reg, args, gs)

    stage_s = None
    if s["th"].get("started") and s["th"].get("finished"):
        stage_s = (dt.datetime.fromisoformat(s["th"]["finished"])
                   - dt.datetime.fromisoformat(s["th"]["started"])).total_seconds()
    full_job_s = sum(full[g]["seconds"] for g in gs) if ref == "recomputed" else None
    rows = []
    now = dt.datetime.now().isoformat(timespec="seconds")
    for v, sp in todo.items():
        a, b, step, tsub = sp
        var_job_s = [maps[v][g]["seconds"] for g in gs]
        for g in gs:
            sf, sv = sug[("full", g)], sug[(v, g)]
            row = {"root": root.name, "sid": sid, "fps": round(s["fps"], 3), "n_frames": s["n_frames"],
                   "duration_min": round(s["n_frames"] / s["fps"] / 60, 2), "variant": v,
                   "kind": "tsub" if v in TSUB_VARIANTS else "subset",
                   "frame_start": a, "frame_stop": b, "frame_step": step, "tsub": tsub,
                   "n_frames_used": maps[v][g]["n_frames"], "n_samples": math.ceil(maps[v][g]["n_frames"] / tsub),
                   "frac_frames": round(maps[v][g]["n_frames"] / s["n_frames"], 4),
                   "gsig": g, "is_stage_gsig": int(g == G_SUG), "reference": ref, "reference_note": why,
                   "registry_suggestion": json.dumps(reg) if g == G_SUG else "",
                   "full_sugg_eq_registry": int(reg_match) if g == G_SUG else ""}
            row.update(sugg_cols("full", sf))
            row.update(sugg_cols("var", sv))
            if "error" not in sf and "error" not in sv:
                row.update(d_min_corr=sv["min_corr"] - sf["min_corr"], d_min_pnr=sv["min_pnr"] - sf["min_pnr"],
                           same_method=int(sf["method"] == sv["method"]))
            row.update(compare(full[g], maps[v][g], g, sf, sv))
            pv = plan[v]
            row.update(var_job_s=round(maps[v][g]["seconds"], 1),
                       var_job_peak_gb=round(maps[v][g]["peak_private_gb"], 2),
                       var_job_base_gb=round(maps[v][g]["base_private_gb"], 2),
                       var_maps_s_sum=round(sum(var_job_s), 1), var_maps_s_max=round(max(var_job_s), 1),
                       var_pool_wall_s=round(pv["pool_wall_s"], 1) if not pv["shared_pool"] else "",
                       var_workers=pv["workers"], n_gsigs_run=len(gs),
                       var_max_mem_gb=pv["max_mem_gb"], var_pred_peak_gb=round(pv["pred_peak_gb"], 2),
                       var_pool_peak_gb=round(pv["pool_peak_gb"], 2),
                       full_maps_s_sum=round(full_job_s, 1) if full_job_s else "",
                       full_job_s=round(full[g]["seconds"], 1) if ref == "recomputed" else "",
                       full_pool_wall_s=(round(plan["full"]["pool_wall_s"], 1) if ref == "recomputed"
                                        and not plan["full"]["shared_pool"] else ""),
                       stage_full_wall_s=stage_s, decode_s=round(decode_s, 1), decoded_frames=decoded_frames,
                       thresh_init=THRESH_INIT, head=code["head"], run_time=now)
            rows.append(row)
    append_csv(Path(args.out) / CSV_NAME, rows)
    print(f"[OK] {tag}: {len(rows)} rows appended", flush=True)


def write_refcheck(s: dict, full: dict, sug: dict, reg: dict, args, gs) -> None:
    """Recomputed full maps vs the stage's stored npz (tells whether the stored maps were current after all)."""
    st = stored_maps(s["maps_npz"])
    rows = []
    for g in gs:
        row = {"root": s["root"].name, "sid": s["sid"], "gsig": g}
        for k in MAP_KEYS:
            d = np.abs(full[g][k].astype(np.float64) - st[g][k].astype(np.float64))
            row[f"maxabs_{MAP_NAMES[k]}"] = float(np.nanmax(d))
            row[f"equal_{MAP_NAMES[k]}"] = int(np.array_equal(full[g][k], st[g][k]))
        sf = sug[("full", g)]
        row["recomputed_sugg"] = json.dumps({k: sf.get(k) for k in ("min_corr", "min_pnr", "method")})
        if g == G_SUG:
            row["registry_sugg"] = json.dumps(reg)
        rows.append(row)
    append_csv(Path(args.out) / REFCHECK_NAME, rows)


# ----------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------

# dataviz categorical slots in fixed order: subset variants 1-3, tsub variants 4-5
COLORS = {"dec": "#2a78d6", "first10": "#eb6834", "mid10": "#1baf7a", "tsub2": "#eda100", "tsubK": "#e87ba4"}
LABELS = {"full": "full", "dec": "V_dec", "first10": "V_first10", "mid10": "V_mid10", "tsub2": "V_tsub2",
          "tsubK": "V_tsubK"}


def _scatter(rows, variants, title, fn, plt):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    for ax, key in ((axes[0], "min_corr"), (axes[1], "min_pnr")):
        xs = []
        allx = [float(r[f"full_{key}"]) for r in rows if r["variant"] in variants and r.get(f"full_{key}")]
        # the full suggestion is the same for all variants of a session: variants are shifted sideways by a fixed
        # small step so that equal points stay visible (the caption of the report says so)
        step = 0.008 * (max(allx) - min(allx) + 1e-9) if allx else 0.0
        for iv, v in enumerate(variants):
            rv = [r for r in rows if r["variant"] == v and r.get(f"var_{key}") not in (None, "")
                  and r.get(f"full_{key}") not in (None, "")]
            for long_, mk in ((False, "o"), (True, "s")):
                sel = [r for r in rv if (float(r["duration_min"]) > 10.5) == long_]
                if not sel:
                    continue
                x = [float(r[f"full_{key}"]) + (iv - (len(variants) - 1) / 2) * step for r in sel]
                y = [float(r[f"var_{key}"]) for r in sel]
                xs += x + y
                ax.scatter(x, y, s=30 if long_ else 18, marker=mk, color=COLORS[v], edgecolor="white", linewidth=0.6,
                           label=f"{LABELS[v]}, {'40 min' if long_ else '5-10 min'} (n={len(sel)})", zorder=3)
        if not xs:
            continue
        lo, hi = min(xs), max(xs)
        pad = 0.05 * (hi - lo) + 1e-3
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#52514e", lw=0.8, ls="--", zorder=1)
        ax.set_xlabel(f"{key}, full-frame maps")
        ax.set_ylabel(f"{key}, variant maps")
        ax.grid(color="#e6e5e0", lw=0.5)
        ax.set_title(f"{title}: {key} (gSig {G_SUG})", fontsize=8)
    axes[1].legend(fontsize=6.5, frameon=False, loc="best")
    fig.savefig(fn, dpi=200)
    plt.close(fig)
    print(f"[OK] {fn}")


def _maps_figure(sid, rootname, rows, variants, fn, work, plt):
    import matplotlib
    from matplotlib.colors import ListedColormap
    g = G_SUG
    sets = {}
    for v in ("full",) + tuple(variants):
        p = work / f"mss_maps_{rootname}_{sid}_{v}.npz"
        if p.is_file():
            with np.load(p) as z:
                sets[v] = {k: z[f"{k}_{g}"] for k in MAP_KEYS + ("mean",)}
    sr = [r for r in rows if r["sid"] == sid and r["root"] == rootname]
    if "full" not in sets or not sr:
        print(f"[WARN] maps of {sid} not found in {work}")
        return
    sug = {"full": (float(sr[0]["full_min_corr"]), float(sr[0]["full_min_pnr"]))}
    for r in sr:
        if r.get("var_min_corr"):
            sug[r["variant"]] = (float(r["var_min_corr"]), float(r["var_min_pnr"]))
    names = [v for v in ("full",) + tuple(variants) if v in sets and v in sug]
    if len(names) < 2:
        return
    full = sets["full"]
    valid = bm_qc.valid_mask({**full, "gSig": g})
    lims = {k: np.percentile(full[k][valid], [1, 99.5]) for k in ("cn", "pnr")}  # one grey scale for all rows
    fig, axes = plt.subplots(len(names), 3, figsize=(7.2, 1.95 * len(names)), constrained_layout=True)
    for i, v in enumerate(names):
        m = sets[v]
        c, p = sug[v]
        own = (m["cn"] >= c) & (m["pnr"] >= p)
        for j, k in enumerate(("cn", "pnr")):
            ax = axes[i, j]
            ax.imshow(m[k], cmap="gray", vmin=lims[k][0], vmax=lims[k][1], interpolation="nearest")
            ax.imshow(np.ma.masked_where(~own, own), cmap=ListedColormap(["#eda100"]), alpha=0.85,
                      interpolation="nearest")
            ax.set_title(f"{LABELS[v]} {k}; its suggestion {c:.2f} / {p:.2f}", fontsize=6.5)
            ax.set_xticks([])
            ax.set_yticks([])
        # passing pixels on the FULL maps (what CNMF-E sees): full suggestion vs this variant's suggestion
        cf, pf = sug["full"]
        a = (full["cn"] >= cf) & (full["pnr"] >= pf)
        b = (full["cn"] >= c) & (full["pnr"] >= p)
        rgb = np.zeros(a.shape + (3,))
        rgb[a & b] = (0.8, 0.8, 0.78)
        rgb[a & ~b] = matplotlib.colors.to_rgb("#2a78d6")
        rgb[~a & b] = matplotlib.colors.to_rgb("#eb6834")
        ax = axes[i, 2]
        ax.imshow(rgb, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("full maps, passing at the full suggestion" if v == "full" else
                     f"full maps, passing: grey at both suggestions,\nblue full only, orange {LABELS[v]} only",
                     fontsize=6.5)
    fig.suptitle(f"{sid}, gSig {g}. Yellow: pixels passing min_corr and min_pnr of the row's own suggestion.\n"
                 f"One grey scale per column (full-map percentiles 1-99.5).", fontsize=7)
    fig.savefig(fn, dpi=200)
    plt.close(fig)
    print(f"[OK] {fn}")


def make_figures(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    rows = [r for r in read_csv(out / CSV_NAME) if r.get("is_stage_gsig") == "1"]
    if not rows:
        print("[WARN] no rows for figures")
        return
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    _scatter(rows, SUBSET_VARIANTS, "Frame subsets", out / "suggestions_subset_variants.png", plt)
    _scatter(rows, TSUB_VARIANTS, "Temporal downsampling", out / "suggestions_tsub_variants.png", plt)
    if args.figure_sid:
        sr = [r for r in rows if r["sid"] == args.figure_sid]
        if not sr:
            print(f"[WARN] {args.figure_sid} not in the CSV")
            return
        work = Path(args.work)
        _maps_figure(args.figure_sid, sr[0]["root"], rows, SUBSET_VARIANTS,
                     out / f"maps_{args.figure_sid}_subset_g{G_SUG}.png", work, plt)
        _maps_figure(args.figure_sid, sr[0]["root"], rows, TSUB_VARIANTS,
                     out / f"maps_{args.figure_sid}_tsub_g{G_SUG}.png", work, plt)


# ----------------------------------------------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------------------------------------------

def _f(r, k):
    v = r.get(k)
    return float(v) if v not in (None, "") else float("nan")


def make_summary(args) -> None:
    """Per variant over sessions at the stage's gSig: summary_by_variant.csv (and the same as text)."""
    out = Path(args.out)
    rows = [r for r in read_csv(out / CSV_NAME) if r.get("is_stage_gsig") == "1"]
    res = []
    for v in VARIANTS:
        rv = [r for r in rows if r["variant"] == v]
        if not rv:
            continue
        a = lambda k: np.array([_f(r, k) for r in rv])  # noqa: E731
        dc, dp = np.abs(a("d_min_corr")), np.abs(a("d_min_pnr"))
        seeds_ratio = a("seeds_full_at_var") / a("seeds_full_at_full")
        own_ratio = a("seeds_var_at_var") / a("seeds_full_at_full")
        blobs_ratio = a("gui_blobs_var") / a("gui_blobs_full")
        time_ratio = a("var_maps_s_sum") / a("full_maps_s_sum")  # summed job seconds, same pool settings
        q = lambda x, fn: float(fn(x[np.isfinite(x)])) if np.isfinite(x).any() else float("nan")  # noqa: E731
        res.append({
            "variant": v, "kind": rv[0].get("kind", ""), "n_sessions": len(rv),
            "sessions": " ".join(sorted(r["sid"] for r in rv)),
            "median_abs_d_min_corr": q(dc, np.median), "max_abs_d_min_corr": q(dc, np.max),
            "median_abs_d_min_pnr": q(dp, np.median), "max_abs_d_min_pnr": q(dp, np.max),
            "median_d_min_pnr": q(a("d_min_pnr"), np.median), "median_d_min_corr": q(a("d_min_corr"), np.median),
            "frac_same_suggestion": q((dc < 1e-9) & (dp < 1e-9), np.mean),
            "frac_same_method": q(a("same_method"), np.mean),
            "gsig_agreement": q(a("full_gsig") == a("var_gsig"), np.mean),
            "median_seeds_ratio_fullmaps": q(seeds_ratio, np.median), "min_seeds_ratio_fullmaps": q(seeds_ratio, np.min),
            "max_seeds_ratio_fullmaps": q(seeds_ratio, np.max),
            "median_overlap_var_in_full": q(a("seed_overlap_var_in_full"), np.median),
            "min_overlap_var_in_full": q(a("seed_overlap_var_in_full"), np.min),
            "median_overlap_full_in_var": q(a("seed_overlap_full_in_var"), np.median),
            "min_overlap_full_in_var": q(a("seed_overlap_full_in_var"), np.min),
            "median_fdr_full_at_var": q(a("fdr_full_at_var"), np.median),
            "max_fdr_full_at_var": q(a("fdr_full_at_var"), np.max),
            "median_fdr_full_own": q(a("full_fdr_own"), np.median),
            "median_own_seeds_ratio": q(own_ratio, np.median), "min_own_seeds_ratio": q(own_ratio, np.min),
            "max_own_seeds_ratio": q(own_ratio, np.max),
            "median_own_overlap_var_in_full": q(a("own_overlap_var_in_full"), np.median),
            "min_own_overlap_var_in_full": q(a("own_overlap_var_in_full"), np.min),
            "median_own_overlap_full_in_var": q(a("own_overlap_full_in_var"), np.median),
            "min_own_overlap_full_in_var": q(a("own_overlap_full_in_var"), np.min),
            "median_r_cn": q(a("r_cn"), np.median), "min_r_cn": q(a("r_cn"), np.min),
            "median_r_pnr": q(a("r_pnr"), np.median), "min_r_pnr": q(a("r_pnr"), np.min),
            "median_top5_pnr_ratio": q(a("top5_pnr_ratio"), np.median),
            "min_top5_pnr_ratio": q(a("top5_pnr_ratio"), np.min), "max_top5_pnr_ratio": q(a("top5_pnr_ratio"), np.max),
            "median_top5_cn_ratio": q(a("top5_cn_ratio"), np.median),
            "median_top5_pnr_mirror_ratio": q(a("top5_pnr_mirror_ratio"), np.median),
            "median_gui_blobs_ratio": q(blobs_ratio, np.median),
            "median_var_pool_wall_s": q(a("var_pool_wall_s"), np.median),
            "median_var_maps_s_sum": q(a("var_maps_s_sum"), np.median),
            "median_time_ratio_vs_full_recomputed": q(time_ratio, np.median),
        })
    if not res:
        return
    path = out / "summary_by_variant.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        wr.writeheader()
        for r in res:
            wr.writerow({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()})
    print(f"[OK] {path}")
    for r in res:
        print(f"  {r['variant']:8s} n={r['n_sessions']:2d} |d corr| med {r['median_abs_d_min_corr']:.3f} "
              f"max {r['max_abs_d_min_corr']:.3f}; |d pnr| med {r['median_abs_d_min_pnr']:.2f} "
              f"max {r['max_abs_d_min_pnr']:.2f}; seeds on full x{r['median_seeds_ratio_fullmaps']:.2f} "
              f"overlap {r['median_overlap_var_in_full']:.2f}/{r['median_overlap_full_in_var']:.2f}")


def _fmt(x, nd=2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(x) else f"{x:.{nd}f}"


def make_tables(args) -> None:
    """Markdown tables of the CSV for the report (tables.md in --out)."""
    out = Path(args.out)
    rows = read_csv(out / CSV_NAME)
    if not rows:
        return
    order = {v: i for i, v in enumerate(VARIANTS)}
    key = lambda r: (float(r["duration_min"]) < 20, r["root"], r["sid"], order[r["variant"]], int(r["gsig"]))  # noqa: E731
    rows.sort(key=key)
    st = [r for r in rows if r["is_stage_gsig"] == "1"]
    L = []
    for kind, title in (("subset", "Frame subsets"), ("tsub", "Temporal downsampling (tsub)")):
        rk = [r for r in st if r["kind"] == kind]
        if not rk:
            continue
        L += [f"### {title}: suggestion and maps (gSig {G_SUG})", "",
              "| session | min | variant | frames (samples) | full corr / pnr | variant corr / pnr | d corr | d pnr "
              "| PNR top5 var/full | cn top5 var/full | r cn | r pnr |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rk:
            L.append(f"| {r['sid']} | {_fmt(r['duration_min'], 0)} | {r['variant']} | {r['n_frames_used']} "
                     f"({r['n_samples']}) | {_fmt(r['full_min_corr'])} / {_fmt(r['full_min_pnr'], 1)} "
                     f"({r['full_method']}) | {_fmt(r['var_min_corr'])} / {_fmt(r['var_min_pnr'], 1)} "
                     f"({r['var_method']}) | {_fmt(r.get('d_min_corr'))} | {_fmt(r.get('d_min_pnr'), 1)} "
                     f"| {_fmt(r['top5_pnr_ratio'])} | {_fmt(r['top5_cn_ratio'], 3)} | {_fmt(r['r_cn'], 3)} "
                     f"| {_fmt(r['r_pnr'], 3)} |")
        L += ["", f"### {title}: seeds and GUI counters (gSig {G_SUG})", "",
              "| session | variant | seeds on FULL maps: full sugg. / variant sugg. | overlap var->full / full->var "
              "| FDR on FULL at variant sugg. (full sugg.) | own: variant maps at own sugg. | own overlap "
              "var->full / full->var | GUI blobs: variant / full / full maps at variant sugg. | mirror blobs same |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rk:
            L.append(f"| {r['sid']} | {r['variant']} | {r.get('seeds_full_at_full', '-')} / "
                     f"{r.get('seeds_full_at_var', '-')} | {_fmt(r.get('seed_overlap_var_in_full'))} / "
                     f"{_fmt(r.get('seed_overlap_full_in_var'))} | {_fmt(r.get('fdr_full_at_var'))} "
                     f"({_fmt(r.get('full_fdr_own'))}) | {r.get('seeds_var_at_var', '-')} | "
                     f"{_fmt(r.get('own_overlap_var_in_full'))} / {_fmt(r.get('own_overlap_full_in_var'))} | "
                     f"{r.get('gui_blobs_var', '-')} / {r.get('gui_blobs_full', '-')} / "
                     f"{r.get('gui_blobs_fullmaps_at_var', '-')} | {r.get('gui_mirror_var', '-')} / "
                     f"{r.get('gui_mirror_full', '-')} / {r.get('gui_mirror_fullmaps_at_var', '-')} |")
        L.append("")
    multi = [r for r in rows if r.get("n_gsigs_run") not in (None, "", "1")]
    if multi:
        L += ["### Per gSig (sessions run with all gSigs)", "",
              "| session | variant | gSig | full corr / pnr | variant corr / pnr | PNR top5 var/full | r cn | r pnr "
              "| seeds on FULL: full / var | overlap var->full / full->var |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in multi:
            L.append(f"| {r['sid']} | {r['variant']} | {r['gsig']} | {_fmt(r['full_min_corr'])} / "
                     f"{_fmt(r['full_min_pnr'], 1)} | {_fmt(r['var_min_corr'])} / {_fmt(r['var_min_pnr'], 1)} | "
                     f"{_fmt(r['top5_pnr_ratio'])} | {_fmt(r['r_cn'], 3)} | {_fmt(r['r_pnr'], 3)} | "
                     f"{r.get('seeds_full_at_full', '-')} / {r.get('seeds_full_at_var', '-')} | "
                     f"{_fmt(r.get('seed_overlap_var_in_full'))} / {_fmt(r.get('seed_overlap_full_in_var'))} |")
        L.append("")
    L += ["### Time of the maps", "",
          "| session | variant | gSigs | job s (sum over gSigs) | pool wall s | workers x max_mem_gb | worker peak GB "
          "| full maps job s (recomputed) | stage wall s (registry, full maps, 5 gSigs) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in st:
        L.append(f"| {r['sid']} | {r['variant']} | {r.get('n_gsigs_run', '')} | {_fmt(r['var_maps_s_sum'], 0)} | "
                 f"{_fmt(r.get('var_pool_wall_s'), 0)} | {r.get('var_workers', '')} x {r.get('var_max_mem_gb', '')} | "
                 f"{_fmt(r.get('var_job_peak_gb'), 1)} | {_fmt(r.get('full_maps_s_sum'), 0)} | "
                 f"{_fmt(r.get('stage_full_wall_s'), 0)} |")
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[OK] {out / 'tables.md'}")


# ----------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", action="append", default=[], help="series root (repeatable)")
    ap.add_argument("--sid", action="append", default=[], help="session id (repeatable); default: every session "
                    "of the roots with a done thresholds stage")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"upper bound, at most {MAX_WORKERS}")
    ap.add_argument("--out", default=str(REPO / "docs" / "superpowers" / "specs" / "assets" / "maps-validation"))
    ap.add_argument("--work", required=True, help="scratch folder for the decoded movie and the maps")
    ap.add_argument("--timeout-h", type=float, default=4.0, help="per maps pool")
    ap.add_argument("--force", action="store_true", help="recompute session x variant already in the CSV")
    ap.add_argument("--recompute-full", action="store_true",
                    help="recompute the full reference even when the stored npz is valid (sessions < 20 000 frames)")
    ap.add_argument("--gsigs", default=",".join(map(str, GSIGS)),
                    help=f"gSigs to compute (must include the stage gSig {G_SUG}); fewer gSigs = faster")
    ap.add_argument("--reuse-maps", action=argparse.BooleanOptionalAction, default=True,
                    help="reuse variant maps (and their timings) saved in --work by an earlier run")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--figure-sid", default="")
    args = ap.parse_args()
    args.workers = max(1, min(args.workers, MAX_WORKERS))
    args.gsigs = sorted({int(g) for g in args.gsigs.split(",")})
    if G_SUG not in args.gsigs or not set(args.gsigs) <= set(GSIGS):
        raise SystemExit(f"--gsigs must be within {GSIGS} and include {G_SUG}")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if not args.figures_only:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        bad = set(variants) - set(VARIANTS)
        if bad:
            raise SystemExit(f"unknown variants {bad}")
        code = map_code_state()
        print(f"[..] map code: HEAD {code['head']}, last change {code['time']:%Y-%m-%d %H:%M}, "
              f"dirty {code['dirty']}; thresh_init {THRESH_INIT}; free {free_gb():.0f} GB", flush=True)
        done = {(r["root"], r["sid"], r["variant"]) for r in read_csv(Path(args.out) / CSV_NAME)}
        for root in map(Path, args.root):
            sids = args.sid or sorted(p.stem for p in (root / "_bearmind" / "registry").glob("*.json"))
            for sid in sids:
                reg = root / "_bearmind" / "registry" / f"{sid}.json"
                if not reg.is_file():
                    continue
                s = load_session(root, sid)
                if not args.sid and s["th"].get("state") != "done":
                    continue
                try:
                    run_session(s, variants, args, code, done)
                except GuardStop as e:
                    print(f"[FAILED] {root.name}/{sid}: {e}", flush=True)
                    raise SystemExit(2)
                except Exception:
                    print(f"[FAILED] {root.name}/{sid}:", flush=True)
                    traceback.print_exc()
    make_summary(args)
    make_tables(args)
    make_figures(args)


if __name__ == "__main__":
    main()
