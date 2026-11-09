"""Expensive halves of the equivalence checks with the old code (test_equivalence.py), cached on disk.

Each run starts in a fresh process with the thread settings of BLAS_ENV, so that the old and the new side
get the same BLAS and pool settings, and OPENBLAS_CORETYPE is in place before numpy loads. The tests call
`ensure_*`; a run is repeated only when its cache key changes. The key of every cached run (MC old side,
CNMF-E old and new side) = the input clip (file sizes and mtimes), the resolved crop/mc/cnmf config, the
CaImAn and NumPy versions, the pool and BLAS settings, and a hash of all code it may run: every *.py under
bearmind/, ml/ and history/, every root-level *.py and this file (`code_files`). So the cache is invalidated by
any change to the package, ml/, the old code in history/ or the root modules; over-invalidation is accepted.
The autoinspection run (new env on the input of an old-env run) is keyed the same way on its two files, the
stored autoinspection_config, the model file, the code and the versions of what it runs (`autoinspect_versions`).
By hand, to rebuild one side:

    python tests/bearmind/equivalence_runs.py mc|cnmf-old|cnmf-new SERIES SID OUT_DIR N_PROCESSES
    python tests/bearmind/equivalence_runs.py autoinspect INPUT_ESTIMATES OLD_PROCESSED OUT_DIR 0

The cache root is BEARMIND_EQUIV_CACHE, or <system temp>/bearmind_equiv; delete a folder to recompute it.
"""
import os

# spawned pool workers re-import this file: the BLAS core type must be set before numpy loads there too
os.environ.setdefault("OPENBLAS_CORETYPE", "Sandybridge")

import filecmp  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Same BLAS settings on both sides of the CNMF-E comparison (the pool size and the BLAS thread count
# can change CNMF-E output). One thread everywhere: the CNMF stage gives its pool workers one thread anyway.
BLAS_ENV = {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            "OPENBLAS_CORETYPE": "Sandybridge"}
CACHE_VERSION = 1
RUN_TIMEOUT_S = 2 * 3600


# ----------------------------------------------------------------------------------------------- cache

def cache_root() -> Path:
    return Path(os.environ.get("BEARMIND_EQUIV_CACHE") or Path(tempfile.gettempdir()) / "bearmind_equiv")


def cache_dir(series_root, sid) -> Path:
    return cache_root() / f"{Path(series_root).name}_{sid}"


def _file_sig(path) -> list:
    st = Path(path).stat()
    return [st.st_size, st.st_mtime_ns]


def code_files(root: Path = REPO) -> list[Path]:
    """The code a cached run may execute: every module of the package, every root-level module, the ml
    package (auto_inspector imports ml.data_utils), the old code under history/ (the old side runs
    history.legacy_pipeline), this helper.

    Deliberately wide: the stages import bm_qc, registry, config and more, and the old functions import half of
    the root modules; any change to one of them recomputes the cache (a recompute is ~10 min on the clip).
    """
    root = Path(root)
    files = (set(root.glob("bearmind/**/*.py")) | set(root.glob("*.py")) | set(root.glob("ml/**/*.py"))
             | set(root.glob("history/**/*.py")))
    files.add(root / "tests" / "bearmind" / "equivalence_runs.py")
    return sorted(f for f in files if f.is_file())


def code_sig(root: Path = REPO) -> str:
    h = hashlib.sha256()
    for f in code_files(root):
        h.update(f.relative_to(root).as_posix().encode() + b"\0")
        h.update(f.read_bytes() + b"\0")
    return h.hexdigest()


def _versions() -> dict:
    from importlib.metadata import version
    return {"caiman": version("caiman"), "numpy": version("numpy")}


def _config_sig(cfg) -> dict:
    return {k: cfg.get(k) for k in ("crop", "mc", "cnmf")}


def _load(series_root, sid):
    from bearmind import config
    from bearmind.paths import SeriesPaths
    from bearmind.registry import load_record

    paths = SeriesPaths(Path(series_root))
    return paths, load_record(paths, sid), config.resolve(paths, sid=sid)


def mc_dict(cfg, n_frames) -> dict:
    """The old DoMotionCorrection dict with the parameters the MC stage passes to MotionCorrect.

    The stage overrides splits_rig (chunks of ~500 frames) and shifts_opencv (bearmind/stages/mc.py).
    write_mp4 is read by DoMotionCorrection itself.
    """
    return {**cfg["mc"], "splits_rig": math.ceil(n_frames / 500), "shifts_opencv": True, "write_mp4": False}


def mc_key(series_root, sid) -> dict:
    paths, rec, cfg = _load(series_root, sid)
    raw = [[f"{p.folder}/{c}", *_file_sig(paths.root / p.folder / c)] for p in rec.parts for c in p.chunks]
    return {"version": CACHE_VERSION, "kind": "mc", "sid": sid, "n_frames": rec.n_frames, "raw": raw,
            "config": _config_sig(cfg), "mc_dict": mc_dict(cfg, rec.n_frames),
            "code": code_sig(), **_versions()}


def cnmf_key(series_root, sid, side, n_processes) -> dict:
    from bearmind.stages.cnmf import cnmf_params

    paths, rec, cfg = _load(series_root, sid)
    return {"version": CACHE_VERSION, "kind": f"cnmf-{side}", "sid": sid,
            "mc_video": _file_sig(paths.mc_video(sid)), "config": _config_sig(cfg),
            "params": cnmf_params(cfg, rec.fps), "n_processes": int(n_processes), "blas": BLAS_ENV,
            "code": code_sig(), **_versions()}


# Arguments of run_auto_inspection that the old run stored in est.autoinspection_config (ae_launch).
AI_ARGS = ("brain", "ml_threshold", "event_method", "n_iter", "hybrid_kinetics", "cthr", "corr_thr",
           "include_event_based", "include_heavy", "detect_corner_artifacts", "correlation_method",
           "wavelet_backend", "enable_merge", "pxlthr_distance_boundary", "d_snr_thr")


def load_pickle(path):
    with open(path, "rb") as fh:  # read only: nothing is written next to the source data
        return pickle.load(fh)


def model_file(stored) -> Path:
    """The repo's copy of the model an old run used (it stored a path on its own machine)."""
    from pathlib import PureWindowsPath

    return REPO / "production_models" / PureWindowsPath(str(stored)).name


def autoinspect_args(config: dict) -> tuple[dict, int]:
    """run_auto_inspection keyword arguments and fps from an old run's autoinspection_config."""
    kw = {k: config[k] for k in AI_ARGS if k in config}
    rules = config.get("deletion_rules")
    if rules not in (None, "DEFAULT_DELETION_RULES"):
        kw["deletion_rules"] = rules
    kw["ml_model_path"] = str(model_file(config["ml_model_path"]))
    return kw, int(config["fps"])


def autoinspect_versions() -> dict:
    """Versions of what autoinspection runs: driada computes the event features and kinetics_opt, with numba
    and ssqueezepy (wavelets) under it; interpret scores the model."""
    from importlib.metadata import version

    return {**_versions(), **{p: version(p) for p in ("pandas", "scipy", "interpret", "joblib", "driada",
                                                       "numba", "ssqueezepy")}}


def autoinspect_key(input_pickle, old_processed) -> dict:
    config = load_pickle(old_processed).autoinspection_config
    kw, fps = autoinspect_args(config)
    model = Path(kw["ml_model_path"])
    return {"version": CACHE_VERSION, "kind": "autoinspect",
            "input": [str(input_pickle), *_file_sig(input_pickle)],
            "old": [str(old_processed), *_file_sig(old_processed)],
            "args": kw, "fps": fps, "model": [model.name, hashlib.sha256(model.read_bytes()).hexdigest()],
            "code": code_sig(), **autoinspect_versions()}


def stage_pool(series_root, sid, n_processes) -> int:
    """Pool size the CNMF stage picks with n_cpus = n_processes + 1; smaller when the RAM budget limits it."""
    from bearmind.stages.cnmf import cnmf_params, patch_workers
    from bearmind.video import video_info

    paths, rec, cfg = _load(series_root, sid)
    cfg["machine"] = {**cfg["machine"], "n_cpus": n_processes + 1}
    return patch_workers(video_info(paths.mc_video(sid))[0], cnmf_params(cfg, rec.fps)["tsub"], cfg)


def _norm(x):
    return json.loads(json.dumps(x, default=str))


def _cached(out: Path, key: dict) -> bool:
    meta = out / "meta.json"
    if not meta.is_file():
        return False
    return json.loads(meta.read_text(encoding="utf-8")).get("key") == _norm(key)


def _launch(kind, series_root, sid, out: Path, n_processes) -> None:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    # faulthandler: a native crash in a pool worker leaves its Python traceback in run.log
    env = {**os.environ, **BLAS_ENV, "PYTHONPATH": str(REPO), "PYTHONFAULTHANDLER": "1"}
    log = out / "run.log"
    cmd = [sys.executable, str(Path(__file__).resolve()), kind, str(series_root), sid, str(out), str(n_processes)]
    with open(log, "w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT_S)
    if r.returncode != 0 or not (out / "meta.json").is_file():
        tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"{kind} run failed (exit {r.returncode}), log {log}:\n{tail}")


def ensure_mc(series_root, sid) -> Path:
    out = cache_dir(series_root, sid) / "mc"
    if not _cached(out, mc_key(series_root, sid)):
        _launch("mc", series_root, sid, out, 0)
    return out


def ensure_autoinspect(input_pickle, old_processed) -> Path:
    out = cache_root() / f"autoinspect_{Path(old_processed).parent.name}"
    if not _cached(out, autoinspect_key(input_pickle, old_processed)):
        _launch("autoinspect", input_pickle, str(old_processed), out, 0)
    return out


def ensure_cnmf(series_root, sid, n_processes) -> dict:
    outs = {}
    for side in ("old", "new"):
        out = cache_dir(series_root, sid) / f"cnmf_{side}"
        if not _cached(out, cnmf_key(series_root, sid, side, n_processes)):
            _launch(f"cnmf-{side}", series_root, sid, out, n_processes)
        outs[side] = out
    return outs


def read_meta(out: Path) -> dict:
    return json.loads((out / "meta.json").read_text(encoding="utf-8"))


def _write_meta(out: Path, key: dict, **fields) -> None:
    (out / "meta.json").write_text(json.dumps({"key": _norm(key), **_norm(fields)}, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------------------------- comparisons

def frame_diff_hist(old_tif, new_avi) -> dict:
    """Histogram of (new - old) grey levels over all frames: {difference: pixel count}."""
    import tifffile

    from bearmind.video import _iter_avi

    old = tifffile.memmap(str(old_tif), mode="r")
    hist = np.zeros(511, np.int64)
    n = 0
    try:
        for i, fr in enumerate(_iter_avi(Path(new_avi))):
            d = fr.astype(np.int16) - np.asarray(old[i], dtype=np.int16)
            hist += np.bincount((d + 255).ravel(), minlength=511)
            n += 1
        if n != old.shape[0]:
            raise ValueError(f"{new_avi}: {n} frames, old output has {old.shape[0]}")
    finally:
        base = getattr(old, "_mmap", None)
        del old
        if base is not None:
            base.close()
    return {int(v) - 255: int(hist[v]) for v in np.nonzero(hist)[0]}


def save_ac(path, est) -> None:
    from scipy import sparse

    A = sparse.csc_matrix(est.A)
    idx = getattr(est, "idx_components", None)
    np.savez(path, A_data=A.data, A_indices=A.indices, A_indptr=A.indptr, A_shape=np.array(A.shape),
             C=np.asarray(est.C), idx_components=np.array([] if idx is None else idx, dtype=int),
             dims=np.array(est.dims))


def load_ac(path) -> dict:
    from scipy import sparse

    with np.load(path) as d:
        A = sparse.csc_matrix((d["A_data"], d["A_indices"], d["A_indptr"]), shape=tuple(d["A_shape"]))
        return {"A": A, "C": d["C"], "idx_components": d["idx_components"]}


def footprint_corr(A1, A2) -> np.ndarray:
    """Pearson correlation of every footprint column of A1 with every column of A2 (sparse, d pixels)."""
    d = A1.shape[0]
    s1, s2 = (np.asarray(a.sum(axis=0)).ravel() for a in (A1, A2))
    q1, q2 = (np.asarray(a.multiply(a).sum(axis=0)).ravel() for a in (A1, A2))
    cov = (A1.T @ A2).toarray() - np.outer(s1, s2) / d
    sd = np.outer(np.sqrt(np.maximum(q1 - s1 ** 2 / d, 0)), np.sqrt(np.maximum(q2 - s2 ** 2 / d, 0)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, cov / sd, np.nan)


def trace_corr(C1, C2) -> np.ndarray:
    z = []
    for C in (C1, C2):
        c = np.asarray(C, dtype=np.float64)
        c = c - c.mean(axis=1, keepdims=True)
        nrm = np.linalg.norm(c, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            z.append(np.where(nrm > 0, c / nrm, np.nan))
    return z[0] @ z[1].T


def match_components(old: dict, new: dict, thr: float = 0.95) -> dict:
    """One-to-one matching (Hungarian on 1 - footprint corr); a pair counts if both correlations exceed thr."""
    from scipy.optimize import linear_sum_assignment

    fp = footprint_corr(old["A"], new["A"])
    tc = trace_corr(old["C"], new["C"])
    i, j = linear_sum_assignment(1 - np.nan_to_num(fp, nan=-1.0))
    fpm, tcm = fp[i, j], tc[i, j]
    ok = (fpm > thr) & (tcm > thr)
    k_old, k_new = old["A"].shape[1], new["A"].shape[1]

    def pct(x):
        x = x[np.isfinite(x)]
        return {q: float(np.percentile(x, q)) for q in (0, 1, 5, 25, 50)} if len(x) else {}

    acc_old = set(old["idx_components"].tolist())
    acc_new = set(new["idx_components"].tolist())
    ok_pairs = [(int(a), int(b)) for a, b, g in zip(i, j, ok) if g]
    return {
        "n_old": int(k_old), "n_new": int(k_new), "n_pairs": int(len(i)), "n_matched": int(ok.sum()),
        "frac_old": float(ok.sum() / k_old) if k_old else 0.0,
        "frac_new": float(ok.sum() / k_new) if k_new else 0.0,
        "n_fail_footprint": int((~(fpm > thr)).sum()), "n_fail_trace": int(((fpm > thr) & ~(tcm > thr)).sum()),
        "footprint_corr_pct": pct(fpm), "trace_corr_pct": pct(tcm),
        "n_accepted_old": len(acc_old), "n_accepted_new": len(acc_new),
        "n_matched_both_accepted": sum(1 for a, b in ok_pairs if a in acc_old and b in acc_new),
        "unmatched_old": sorted(set(range(k_old)) - {a for a, _ in ok_pairs}),
        "unmatched_new": sorted(set(range(k_new)) - {b for _, b in ok_pairs}),
    }


# ----------------------------------------------------------------------------------------------- the runs

class PoolWatch:
    """Tells a dead pool worker from successive sets of live worker pids.

    multiprocessing.Pool replaces a worker that dies (e.g. a segfault in native code), and the task it held
    never returns, so Pool.map waits forever. The current pool = the workers seen while at least one of them
    is alive. A member that disappears while the pool runs, with a new worker joining the same pool (in the
    same or a later sample), is a death: Pool replaces dead workers only (CaImAn pools have no
    maxtasksperchild). Members vanishing with no replacement (terminate) or a new pool after the old one is
    gone are not. Each of our runs starts one pool (DoMotionCorrection, DoCNMF and the CNMF stage each call
    setup_cluster once).
    """

    def __init__(self):
        self.pool: set[int] = set()
        self.lost: set[int] = set()

    def update(self, alive) -> set[int]:
        """Feed the live worker pids; returns the dead members once a replacement shows up, else an empty set."""
        alive = set(alive)
        if not alive & self.pool:  # no pool yet, or the previous one is gone: start over
            self.pool, self.lost = alive, set()
            return set()
        self.lost |= self.pool - alive
        joined = alive - self.pool
        self.pool |= joined
        return set(self.lost) if joined and self.lost else set()


class PeakMemory:
    """Peak RSS and private bytes of this process and all its children, sampled every second.

    Also a watchdog (PoolWatch): when a pool worker dies, the run is aborted instead of hanging until the
    timeout.
    """

    def __init__(self, every_s: float = 1.0, watch: bool = True):
        self.every_s, self.watch = every_s, watch
        self.peak_rss_gb = self.peak_private_gb = 0.0
        self._watch = PoolWatch()
        self._cmd: dict[int, bool] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _is_worker(self, p) -> bool:
        import psutil

        if p.pid not in self._cmd:
            try:
                self._cmd[p.pid] = "spawn_main" in " ".join(p.cmdline())
            except psutil.Error:
                return False
        return self._cmd[p.pid]

    def _sample(self):
        import psutil

        me = psutil.Process()
        kids = me.children(recursive=True)
        rss = priv = 0
        for p in [me] + kids:
            try:
                mi = p.memory_info()
            except psutil.Error:
                continue
            rss += mi.rss
            priv += getattr(mi, "private", 0)
        self.peak_rss_gb = max(self.peak_rss_gb, rss / 2 ** 30)
        self.peak_private_gb = max(self.peak_private_gb, priv / 2 ** 30)
        dead = self._watch.update(p.pid for p in kids if self._is_worker(p)) if self.watch else set()
        if dead:
            print(f"[FAILED] pool worker(s) died: {sorted(dead)}; aborting", file=sys.stderr, flush=True)
            for p in kids:
                try:
                    p.kill()
                except psutil.Error:
                    pass
            os._exit(3)

    def _loop(self):
        while not self._stop.wait(self.every_s):
            self._sample()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        return False


def _blas_now() -> dict:
    return {k: os.environ.get(k) for k in BLAS_ENV}


def _remove(*patterns_in) -> None:
    for folder, pattern in patterns_in:
        for f in Path(folder).rglob(pattern):
            try:
                f.unlink()
            except OSError:
                pass


def run_mc_old(series_root, sid, out: Path) -> None:
    """Old path: cropped temp TIF (same frames and box as the stage) -> DoMotionCorrection."""
    from bearmind.stages import mc as mc_stage
    from bearmind.video import SessionReader

    paths, rec, cfg = _load(series_root, sid)
    key = mc_key(series_root, sid)
    crop = mc_stage._crop(cfg)
    tmp = out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["CAIMAN_TEMP"] = str(tmp)
    tif = out / f"{sid}_CR.tif"
    # the stage's own TIF writer and reader: the comparison isolates motion correction from decoding
    mc_stage._write_tif(SessionReader(paths.root, rec, crop=crop), tif)

    from history import legacy_pipeline as old

    made = []

    class Recording(old.MotionCorrect):  # DoMotionCorrection keeps its MotionCorrect object to itself
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    old.MotionCorrect = Recording
    d = mc_dict(cfg, rec.n_frames)
    t0 = time.time()
    with PeakMemory() as pm:
        old.DoMotionCorrection(str(tif), dict(d))
    wall = time.time() - t0
    np.save(out / "old_shifts.npy", np.asarray(made[0].shifts_rig, dtype=np.float64))
    os.replace(str(tif)[:-4] + "_MC.tif", out / "old_MC.tif")
    made.clear()
    gc.collect()  # Windows cannot delete a file that is still mapped
    tif.unlink()
    _remove((out, "*.mmap"))
    shutil.rmtree(tmp, ignore_errors=True)
    _write_meta(out, key, wall_s=wall, peak_rss_gb=pm.peak_rss_gb, peak_private_gb=pm.peak_private_gb,
                mc_dict=d, crop=None if crop is None else [crop.top, crop.left, crop.height, crop.width],
                old_pool=14)  # DoMotionCorrection hard-codes 14 workers; MC shifts do not depend on it


def run_cnmf_old(series_root, sid, out: Path, n_processes: int) -> None:
    """Old DoCNMF on a TIF of the stage's MC video, with the stage's parameters and a pinned pool size."""
    import caiman
    import caiman.cluster
    import tifffile

    from bearmind.stages import cnmf as cnmf_stage
    from bearmind.video import decode_to_memmap

    paths, rec, cfg = _load(series_root, sid)
    key = cnmf_key(series_root, sid, "old", n_processes)
    params = cnmf_stage.cnmf_params(cfg, rec.fps)
    tmp = out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["CAIMAN_TEMP"] = str(tmp)
    u8_path = tmp / "mc_u8.dat"
    u8 = decode_to_memmap(paths.mc_video(sid), u8_path)
    tif = out / f"{sid}_MC.tif"
    tifffile.imwrite(str(tif), u8, photometric="minisblack")

    pools = []
    setup = caiman.cluster.setup_cluster

    def pinned(*a, **k):  # DoCNMF asks for n_processes=None (all cores - 1)
        k["n_processes"] = n_processes
        res = setup(*a, **k)
        pools.append(res[2])
        return res

    caiman.cluster.setup_cluster = pinned

    from history import legacy_pipeline as old

    t0 = time.time()
    with PeakMemory() as pm:
        # noise_ampl = the CaImAn save_memmap default the stage's memmap writer reproduces (DoCNMF default: 0)
        est = old.DoCNMF(str(tif), dict(params), out_name=str(out / "old_estimates.pickle"),
                         noise_ampl=float(cnmf_stage.ADD_TO_MOVIE), parallel_mode=True)
    wall = time.time() - t0
    save_ac(out / "old.npz", est)

    # same CNMF input on both sides: the old memmap against the stage's memmap writer on the same frames
    old_mm = sorted(out.rglob("*.mmap"))  # save_memmap puts it next to the TIF or under CAIMAN_TEMP
    (out / "chk").mkdir()
    new_mm = cnmf_stage.write_memmap(u8, out / "chk", cnmf_stage.mem_budget_gb(cfg), base="chk")
    same = len(old_mm) == 1 and filecmp.cmp(old_mm[0], new_mm, shallow=False)
    est = u8 = None
    gc.collect()  # Windows cannot delete a file that is still mapped
    _remove((out, "*.mmap"), (tmp, "*"))
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(out / "chk", ignore_errors=True)
    tif.unlink()
    (out / "old_estimates.pickle").unlink()
    _write_meta(out, key, wall_s=wall, peak_rss_gb=pm.peak_rss_gb, peak_private_gb=pm.peak_private_gb,
                pool=pools, cnmf_n_processes=[p - 1 for p in pools],  # DoCNMF passes pool - 1 to CNMF
                blas=_blas_now(), memmap_identical=bool(same), params=params)


def run_cnmf_new(series_root, sid, out: Path, n_processes: int) -> None:
    """The CNMF stage on a copy of the series' MC video (the series itself is not touched)."""
    from bearmind.paths import SeriesPaths
    from bearmind.stages import cnmf as cnmf_stage
    from bearmind.video import video_info

    src, rec, cfg = _load(series_root, sid)
    key = cnmf_key(series_root, sid, "new", n_processes)
    paths = SeriesPaths(out / "series")
    for a, b in ((src.mc_video(sid), paths.mc_video(sid)), (src.threshold_maps(sid), paths.threshold_maps(sid))):
        if a.exists():
            b.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(a, b)
    cfg["machine"] = {**cfg["machine"], "n_cpus": n_processes + 1, "temp_dir": str(out / "tmp")}
    params = cnmf_stage.cnmf_params(cfg, rec.fps)
    T = video_info(paths.mc_video(sid))[0]
    n_workers = cnmf_stage.patch_workers(T, params["tsub"], cfg)
    if n_workers != n_processes:
        raise RuntimeError(f"stage pool would be {n_workers}, not {n_processes} (RAM budget)")
    t0 = time.time()
    with PeakMemory() as pm:
        res = cnmf_stage.run(paths, rec, cfg)
    wall = time.time() - t0
    import pickle

    with open(paths.estimates(sid), "rb") as f:
        save_ac(out / "new.npz", pickle.load(f))
    shutil.rmtree(paths.root, ignore_errors=True)
    _write_meta(out, key, wall_s=wall, peak_rss_gb=pm.peak_rss_gb, peak_private_gb=pm.peak_private_gb,
                pool=[res.metrics["patch_workers"]], cnmf_n_processes=[res.metrics["patch_workers"]],
                blas=_blas_now(), metrics=res.metrics, params=params)


def run_autoinspect_new(input_pickle, old_processed, out: Path) -> None:
    """ae_launch.run_auto_inspection in this env on an old run's input, with the arguments it stored.

    Artifacts go to the cache folder; the source data are only read.
    """
    import ae_launch

    key = autoinspect_key(input_pickle, old_processed)
    kw, fps = autoinspect_args(load_pickle(old_processed).autoinspection_config)
    art = out / "artifacts"
    t0 = time.time()
    # watch=False: joblib's worker processes come and go between calls; that is not a dead pool worker
    with PeakMemory(watch=False) as pm:
        res = ae_launch.run_auto_inspection(str(input_pickle), fps=fps, artifacts_dir=str(art), **kw)
    wall = time.time() - t0
    res["metrics_df"].to_pickle(out / "decision_df.pkl")  # before merging: one row per input component
    res["estimates"].metrics_df.to_pickle(out / "processed_df.pkl")  # what the processed pickle carries
    shutil.rmtree(art, ignore_errors=True)
    _write_meta(out, key, wall_s=wall, peak_rss_gb=pm.peak_rss_gb, peak_private_gb=pm.peak_private_gb,
                summary={k: int(v) for k, v in res["summary"].items()}, args=kw, fps=fps)


def compare_metrics(old, new, features) -> dict:
    """Row-by-row comparison of two metrics DataFrames (rows matched by component_idx).

    Per model feature: max |old - new| over rows where both are finite, rows where only one is NaN, and
    whether np.allclose(rtol=1e-6, atol=1e-9, equal_nan=True) holds; ml_keep_probability within 1e-6; the
    delete / merge / decision columns equal.
    """
    o = old.sort_values("component_idx").reset_index(drop=True)
    n = new.sort_values("component_idx").reset_index(drop=True)
    out = {"rows_old": len(o), "rows_new": len(n),
           "same_rows": o["component_idx"].tolist() == n["component_idx"].tolist(), "features": {},
           "missing": sorted(f for f in features if f not in o.columns or f not in n.columns)}
    if not out["same_rows"]:
        return out

    def num(df, c):
        return df[c].to_numpy(dtype=float)

    for f in features:
        if f in out["missing"]:
            continue
        a, b = num(o, f), num(n, f)
        both = np.isfinite(a) & np.isfinite(b)
        out["features"][f] = {
            "max_abs_diff": float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0,
            "nan_mismatch": int((np.isnan(a) != np.isnan(b)).sum()),
            "allclose": bool(np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True))}
    if "ml_keep_probability" in o.columns and "ml_keep_probability" in n.columns:
        a, b = num(o, "ml_keep_probability"), num(n, "ml_keep_probability")
        both = np.isfinite(a) & np.isfinite(b)
        out["ml_keep_probability"] = {
            "max_abs_diff": float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0,
            "allclose": bool(np.allclose(a, b, rtol=0, atol=1e-6, equal_nan=True))}
    for c in ("delete", "merge", "decision"):
        if c in o.columns and c in n.columns:
            out[f"{c}_equal"] = o[c].tolist() == n[c].tolist()
            out[f"{c}_n_differ"] = int(sum(a != b for a, b in zip(o[c].tolist(), n[c].tolist())))
    return out


def main(argv) -> int:
    kind, series_root, sid, out = argv[1], Path(argv[2]), argv[3], Path(argv[4])
    n = int(argv[5]) if len(argv) > 5 else 0
    out.mkdir(parents=True, exist_ok=True)
    if kind == "mc":
        run_mc_old(series_root, sid, out)
    elif kind == "cnmf-old":
        run_cnmf_old(series_root, sid, out, n)
    elif kind == "cnmf-new":
        run_cnmf_new(series_root, sid, out, n)
    elif kind == "autoinspect":  # here SERIES is the input estimates and SID the old processed pickle
        run_autoinspect_new(series_root, Path(sid), out)
    else:
        raise SystemExit(f"unknown run {kind}")
    print("[OK]", kind, sid)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
