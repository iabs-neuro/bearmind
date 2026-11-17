"""Auto thresholds vs the author's thresholds on LT1: every number of auto-vs-author.md and of the explainer page.

Run from the repo root with the bearmind env and PYTHONDONTWRITEBYTECODE=1:

  python docs/superpowers/specs/2026-11-16-lt1-thresholds/auto_vs_author.py traces SID...
      whole-frame mean (over the seed-valid area) and 32-px block means of each MC frame, streamed in chunks
  python .../auto_vs_author.py frames SID...
      the gSig 4 seed maps recomputed by streaming the MC movie (exactly as bm_qc.compute_init_maps at tsub 1),
      once with all frames and once per set of left-out frames (black frames, start of the session, the 2 %
      brightest / dimmest frames, a time-shifted control); suggestion and seed counts on each version
  python .../auto_vs_author.py widen [SID...]
      mirror_fdr on a wider grid (min_corr 0.50-0.98 step 0.02 plus 0.99, min_pnr 2-40 step 0.5): decision R2
  python .../auto_vs_author.py report
      prints every number used in auto-vs-author.md (and writes the small tables next to this script)

Reads only: H:/Projects/LT1/CalciumData/_bearmind (maps, registry), the MC movies, the snapshot files in this
folder. Heavy per-frame caches go to LT1_CACHE (default: %TEMP%/lt1_thresholds_cache), never to the series.
"""
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(r"H:\Projects\LT1\CalciumData")
SERIES = ROOT / "_bearmind"
CACHE = Path(os.environ.get("LT1_CACHE", Path(tempfile.gettempdir()) / "lt1_thresholds_cache"))
G = 4  # gSig of every author choice and of every batch suggestion on LT1
GSIZ = 4 * G + 1
WIDE_CORR = np.round(np.r_[np.arange(0.5, 0.981, 0.02), 0.99], 3)
WIDE_PNR = np.round(np.arange(2, 40.01, 0.5), 2)
BLOCK = 32  # px, block means of the trace pass
START_S = 15.0  # s, "start of the session" frame set
EXTREME_FRAC = 0.02  # brightest / dimmest share of frames
BASE_WIN_S = 30.0  # s, running-median baseline of the whole-frame trace


def maps_for(z, g=G):
    return {"cn": z[f"cn_{g}"], "pnr": z[f"pnr_{g}"], "cn_neg": z[f"cn_mirror_{g}"],
            "pnr_neg": z[f"pnr_mirror_{g}"], "mean": z["mean"], "max": z["max"], "gSig": g}


def load_maps(sid):
    with np.load(SERIES / "thresholds" / f"{sid}_maps.npz") as z:
        return {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in maps_for(z).items()}


def registry(sid):
    return json.loads((SERIES / "registry" / f"{sid}.json").read_text(encoding="utf-8"))


def mc_path(sid):
    return ROOT / registry(sid)["stages"]["mc"]["outputs"]["mc_video"]


def all_sids():
    return sorted(p.stem for p in (SERIES / "registry").glob("*.json"))


def read_csv(name):
    with open(HERE / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    return None if x in (None, "", "None") else float(x)


def author_values(sid):
    """(min_corr, min_pnr) the author set for CNMF-E (session config), or None."""
    import yaml
    p = SERIES / "config" / "sessions" / f"{sid}.yaml"
    if not p.exists():
        return None
    cn = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("cnmf") or {}
    if any(cn.get(k) is None for k in ("gSig", "min_corr", "min_pnr")) or int(cn["gSig"]) != G:
        return None
    return float(cn["min_corr"]), float(cn["min_pnr"])


# ----------------------------------------------------------------------------------------------------------------------
# traces: whole-frame mean of the MC movie
# ----------------------------------------------------------------------------------------------------------------------

def cmd_traces(sids):
    from bearmind.seed_mask import valid_mask
    from bearmind.video import _iter_avi
    CACHE.mkdir(parents=True, exist_ok=True)
    for sid in sids:
        out = CACHE / f"{sid}_trace.npz"
        if out.exists():
            print(f"[OK] {sid} trace cached")
            continue
        m = load_maps(sid)
        valid = valid_mask(m)
        h, w = valid.shape
        hb, wb = h // BLOCK, w // BLOCK
        mask = valid.astype(np.uint8)
        vb = valid[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).mean((1, 3))
        g, blocks, t0 = [], [], time.time()
        for fr in _iter_avi(mc_path(sid)):
            g.append(cv2.mean(fr, mask)[0])
            blocks.append(cv2.resize(fr[:hb * BLOCK, :wb * BLOCK].astype(np.float32), (wb, hb),
                                     interpolation=cv2.INTER_AREA).astype(np.float16))
        np.savez_compressed(out, g=np.array(g, np.float32), blocks=np.array(blocks), block_valid=vb,
                            fps=float(registry(sid)["fps"]))
        print(f"[OK] {sid} trace: {len(g)} frames, {time.time() - t0:.0f} s", flush=True)


def trace_dff(g, fps):
    """Whole-frame dF/F against a running-median baseline (BASE_WIN_S), black frames bridged."""
    from scipy.ndimage import median_filter
    g = np.asarray(g, np.float64)
    med = np.median(g)
    black = g < 0.5 * med
    gg = g.copy()
    gg[black] = med
    win = int(round(BASE_WIN_S * fps)) | 1
    base = median_filter(gg, size=win, mode="nearest")
    return (gg - base) / base, black


def frame_sets(sid):
    """Elementary frame sets (bool over frames) from the whole-frame trace; disjoint from black/start by design."""
    z = np.load(CACHE / f"{sid}_trace.npz")
    g, fps = z["g"], float(z["fps"])
    T = len(g)
    dff, black = trace_dff(g, fps)
    start = np.zeros(T, bool)
    start[:int(round(START_S * fps))] = True
    start &= ~black
    rest = ~(black | start)
    hi = np.quantile(dff[rest], 1 - EXTREME_FRAC)
    lo = np.quantile(dff[rest], EXTREME_FRAC)
    bright = rest & (dff >= hi)
    dim = rest & (dff <= lo)
    # control: the bright set shifted in time, by half the session or more in 1-s steps until no shifted frame falls on
    # the black / start frames, so the control has exactly as many frames as the set it controls
    step = max(1, int(round(fps)))
    ctrl = np.roll(bright, T // 2) & rest
    for k in range(int(T // step)):
        c = np.roll(bright, T // 2 + k * step) & rest
        if c.sum() == bright.sum():
            ctrl = c
            break
    return {"black": black, "start": start, "bright": bright, "dim": dim, "ctrl": ctrl}, dff, fps


VARIANTS = {  # version of the maps -> elementary frame sets left out
    "all": (),
    "no_black": ("black",),
    "no_start": ("black", "start"),
    "no_bright": ("black", "start", "bright"),
    "no_dim": ("black", "start", "dim"),
    "no_ctrl": ("black", "start", "ctrl"),
}


# ----------------------------------------------------------------------------------------------------------------------
# frames: seed maps recomputed by streaming, with frame sets left out
# ----------------------------------------------------------------------------------------------------------------------

def _noise_exact(va, psf, T):
    """Per-pixel noise exactly as bm_qc._noise (get_noise_fft 'mean' on its three 1024-frame windows)."""
    from caiman.source_extraction.cnmf.pre_processing import get_noise_fft
    wins = [(1, 1025), (int(T // 2 - 512), int(T // 2 + 512)), (T - 1024, T)]
    h, w = va.shape[1:]
    filt = np.empty((3072, h, w), np.float32)
    k = 0
    for a, b in wins:
        blk = va[a:b]
        for fr in blk:
            filt[k] = cv2.filter2D(fr.astype(np.float32), -1, psf, borderType=1)
            k += 1
    sn = np.empty((h, w), np.float64)
    for r0 in range(0, h, 64):
        r1 = min(h, r0 + 64)
        y = np.ascontiguousarray(filt[:, r0:r1].reshape(3072, -1).T)
        sn[r0:r1] = get_noise_fft(y, noise_method="mean")[0].reshape(r1 - r0, w)
    del filt
    sn[sn <= 0] = np.inf
    return sn.astype(np.float32)


_OFFS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _pair_slices(dy, dx, h, w):
    """(a, b) index slices so that a[y, x] and b[y, x] are the pixel pair (y, x) - (y + dy, x + dx)."""
    if dx >= 0:
        return (slice(0, h - dy), slice(0, w - dx)), (slice(dy, h), slice(dx, w))
    return (slice(0, h - dy), slice(-dx, w)), (slice(dy, h), slice(0, w + dx))


def _new_sums(h, w):
    return {"n": 0, "S": np.zeros((h, w)), "S2": np.zeros((h, w)),
            "C": [np.zeros((h - dy, w - abs(dx))) for dy, dx in _OFFS]}


def _add_sums(acc, Y):
    # float32 sums within a chunk (<= 200 frames), float64 across chunks
    acc["n"] += Y.shape[0]
    acc["S"] += Y.sum(0)
    acc["S2"] += np.einsum("tij,tij->ij", Y, Y)
    h, w = Y.shape[1:]
    for k, (dy, dx) in enumerate(_OFFS):
        a, b = _pair_slices(dy, dx, h, w)
        acc["C"][k] += np.einsum("tij,tij->ij", Y[:, a[0], a[1]], Y[:, b[0], b[1]])


def _corr_image(acc):
    """Mean Pearson correlation with the 8 neighbours (zero border), as caiman local_correlations_fft."""
    n = acc["n"]
    mu = acc["S"] / n
    sd = np.sqrt(np.maximum(acc["S2"] / n - mu ** 2, 0))
    h, w = mu.shape
    tot = np.zeros((h, w))
    cnt = np.zeros((h, w))
    for k, (dy, dx) in enumerate(_OFFS):
        a, b = _pair_slices(dy, dx, h, w)
        cov = acc["C"][k] / n - mu[a] * mu[b]
        den = sd[a] * sd[b]
        rho = np.where(den > 0, cov / np.where(den > 0, den, 1), 0)
        tot[a] += rho
        tot[b] += rho
        cnt[a] += 1
        cnt[b] += 1
    return (tot / cnt).astype(np.float32)


def _combine(groups, sigs, drop_bits):
    out = None
    for s in sigs:
        if s & drop_bits:
            continue
        a = groups[s]
        if out is None:
            out = {"n": a["n"], "S": a["S"].copy(), "S2": a["S2"].copy(), "C": [c.copy() for c in a["C"]]}
        else:
            out["n"] += a["n"]
            out["S"] += a["S"]
            out["S2"] += a["S2"]
            for k in range(len(out["C"])):
                out["C"][k] += a["C"][k]
    return out


def stream_maps(sid, chunk=200):
    """Maps of every VARIANTS version from one pass over the MC movie; also argmax / argmin frame per pixel."""
    import bm_qc
    from bearmind.video import VideoArray
    cv2.setNumThreads(4)
    sets, _, _ = frame_sets(sid)
    names = list(sets)
    bit = {nm: 1 << i for i, nm in enumerate(names)}
    sig = np.zeros(len(sets["black"]), np.int64)
    for nm in names:
        sig[sets[nm]] |= bit[nm]
    m0 = load_maps(sid)
    va = VideoArray(mc_path(sid))
    T, h, w = va.shape
    assert T == len(sig), (T, len(sig))
    psf = bm_qc._psf_kernel(G)
    mean_f = cv2.filter2D(m0["mean"].astype(np.float32), -1, psf, borderType=1)
    t0 = time.time()
    sn = _noise_exact(va, psf, T)
    thr = (2 * sn).astype(np.float32)
    print(f"  noise {time.time() - t0:.0f} s", flush=True)
    groups = {"r": {}, "m": {}}
    ext = {v: {"max": np.full((h, w), -np.inf, np.float32), "amax": np.zeros((h, w), np.int32),
               "min": np.full((h, w), np.inf, np.float32), "amin": np.zeros((h, w), np.int32)} for v in VARIANTS}
    vbits = {v: sum(bit[nm] for nm in drop) for v, drop in VARIANTS.items()}
    for s in range(0, T, chunk):
        blk = va[s:s + chunk]
        n = blk.shape[0]
        filt = np.empty((n, h, w), np.float32)
        for t in range(n):
            filt[t] = cv2.filter2D(blk[t].astype(np.float32), -1, psf, borderType=1)
        filt -= mean_f
        sg = sig[s:s + n]
        for sv in np.unique(sg):
            idx = np.flatnonzero(sg == sv)
            sub = filt[idx]
            mx, amx = sub.max(0), idx[sub.argmax(0)] + s
            mn, amn = sub.min(0), idx[sub.argmin(0)] + s
            for v, vb in vbits.items():
                if sv & vb:
                    continue
                e = ext[v]
                up = mx > e["max"]
                e["max"][up], e["amax"][up] = mx[up], amx[up]
                dn = mn < e["min"]
                e["min"][dn], e["amin"][dn] = mn[dn], amn[dn]
            for key, sign in (("r", 1), ("m", -1)):
                y = sub if sign == 1 else -sub
                y = np.where(y >= thr, y, 0).astype(np.float32)
                if sv not in groups[key]:
                    groups[key][sv] = _new_sums(h, w)
                _add_sums(groups[key][sv], y)
        if (s // chunk) % 20 == 0:
            print(f"  {s + n}/{T} frames, {time.time() - t0:.0f} s", flush=True)
    sigs = sorted(groups["r"])
    out = {}
    for v, vb in vbits.items():
        e = ext[v]
        out[v] = {"cn": _corr_image(_combine(groups["r"], sigs, vb)),
                  "pnr": np.nan_to_num(e["max"] / sn).astype(np.float32),
                  "cn_neg": _corr_image(_combine(groups["m"], sigs, vb)),
                  "pnr_neg": np.nan_to_num(-e["min"] / sn).astype(np.float32),
                  "mean": m0["mean"], "max": m0["max"], "gSig": G,
                  "n_frames": int(sum(groups["r"][s_]["n"] for s_ in sigs if not s_ & vb))}
    return out, ext["all"], sets, m0


def evaluate_version(m, author):
    import bm_qc
    from bearmind.seed_mask import valid_mask
    sug, grid = bm_qc.suggest_init_thresholds(m, gSiz=GSIZ)
    res = {"n_frames": m.get("n_frames")}
    for meth in ("mirror_fdr", "robust_bg", "pct_margin"):
        r = sug.get(meth) or {}
        res[meth] = {k: r.get(k) for k in ("min_corr", "min_pnr", "n_seeds", "n_null_seeds", "fdr")}
    b = sug["mirror_fdr"] if sug["mirror_fdr"].get("min_corr") is not None else sug["robust_bg"]
    res["batch"] = {"min_corr": b["min_corr"], "min_pnr": b["min_pnr"],
                    "method": "mirror_fdr" if b is sug["mirror_fdr"] else "robust_bg"}
    valid = valid_mask(m)
    res["median_pnr_valid"] = float(np.median(m["pnr"][valid]))
    res["median_pnr_neg_valid"] = float(np.median(m["pnr_neg"][valid]))
    if author:
        res["at_author"] = bm_qc.evaluate_thresholds(m, author[0], author[1], GSIZ)
    res["fdr_min_in_grid"] = float(grid["fdr"][grid["n_real"] > 0].min())
    return res, grid


def cmd_frames(sids):
    from bearmind.seed_mask import valid_mask
    (HERE / "frames_test").mkdir(exist_ok=True)
    for sid in sids:
        t0 = time.time()
        cmd_traces([sid])
        versions, ext, sets, m0 = stream_maps(sid)
        valid = valid_mask(m0)
        a = versions["all"]
        check = {k: float(np.abs(a[k][valid] - m0[k][valid]).max()) for k in ("cn", "pnr", "cn_neg", "pnr_neg")}
        print(f"  check vs batch maps (max abs diff in valid area): {check}", flush=True)
        au = author_values(sid)
        res = {"sid": sid, "check_max_abs_diff": check, "author": au,
               "set_sizes": {k: int(v.sum()) for k, v in sets.items()}, "versions": {}}
        grids = {}
        for v, m in versions.items():
            res["versions"][v], grids[v] = evaluate_version(m, au)
            print(f"  {v}: {json.dumps(res['versions'][v]['batch'])} at_author "
                  f"{json.dumps(res['versions'][v].get('at_author'))}", flush=True)
        # where and when the null seeds of the original maps peak
        import bm_qc as q
        halo = q.halo_mask(m0, valid, GSIZ)
        pt = au if au else (0.94, 17.0)
        s_real, s_null, _ = q.threshold_seeds(m0, pt[0], pt[1], GSIZ, valid, halo)
        res["seed_peak_frames"] = {"point": pt, "real_argmax": ext["amax"][tuple(s_real.T)].tolist(),
                                   "null_argmin": ext["amin"][tuple(s_null.T)].tolist(),
                                   "real_yx": s_real.tolist(), "null_yx": s_null.tolist()}
        np.savez_compressed(CACHE / f"{sid}_frames.npz", amax=ext["amax"], amin=ext["amin"],
                            **{f"{v}_{k}": versions[v][k] for v in versions for k in ("cn", "pnr", "cn_neg", "pnr_neg")},
                            **{f"set_{k}": s for k, s in sets.items()})
        for v, gr in grids.items():
            np.savez_compressed(CACHE / f"{sid}_frames_grid_{v}.npz", **{k: np.asarray(x) for k, x in gr.items()})
        (HERE / "frames_test" / f"{sid}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"[OK] {sid} frames test {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# widen: decision R2
# ----------------------------------------------------------------------------------------------------------------------

def halo_capped(maps, valid, excl_radius=GSIZ, k_pnr=5.0, k_corr=3.0):
    """What-if halo: bm_qc.halo_mask with its correlation cut capped at the 99th percentile of the correlation map.

    bm_qc.halo_mask asks for cn > median + 3 robust SD; once the robust SD of cn is above ~0.058 that cut is above
    1 and the mask is empty. With the cap the mask equals bm_qc.halo_mask wherever the original cut is below the
    99th percentile.
    """
    import bm_qc
    mc, sc = bm_qc._robust(maps["cn"][valid])
    mp, sp = bm_qc._robust(maps["pnr"][valid])
    cut = min(mc + k_corr * sc, float(np.percentile(maps["cn"][valid], 99)))
    strong = valid & (maps["pnr"] > mp + k_pnr * sp) & (maps["cn"] > cut)
    r = int(excl_radius)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(strong.astype(np.uint8), disk).astype(bool)


def grid_counts(m, corr_grid, pnr_grid, halos):
    """n_real and n_null (one per halo mask) on a grid, with the calls of bm_qc.threshold_seeds."""
    import bm_qc
    from bearmind.seed_mask import valid_mask
    valid = valid_mask(m, border_pix=5)
    min_dist = 2 * m["gSig"]
    n_real = np.zeros((len(corr_grid), len(pnr_grid)), int)
    n_null = {k: np.zeros(n_real.shape) for k in halos}
    area = {k: valid.sum() / max((valid & ~h).sum(), 1) for k, h in halos.items()}
    for i, c in enumerate(corr_grid):
        for j, p in enumerate(pnr_grid):
            n_real[i, j] = len(bm_qc.count_seeds(m["cn"], m["pnr"], c, p, GSIZ, valid=valid, min_dist=min_dist))
            for k, h in halos.items():
                s0 = bm_qc.count_seeds(m["cn_neg"], m["pnr_neg"], c, p, GSIZ, valid=valid & ~h, min_dist=min_dist)
                n_null[k][i, j] = len(s0) * area[k]
    return n_real, n_null, area


def select_mirror(corr_grid, pnr_grid, n_real, n_null, target_fdr=0.1, marginal_fdr=0.5, path=True):
    """The mirror_fdr choice of bm_qc.suggest_init_thresholds (lines 383-409), on precomputed counts.

    Returns (min_corr, min_pnr) or None; plus the admissible mask. path=False drops the 'admissible all the way
    from the strict corner' rule (a what-if only).
    """
    fdr = n_null / np.maximum(n_real, 1)
    mfdr = np.zeros_like(fdr)
    for ax_ in (0, 1):
        d_real = n_real - np.roll(n_real, -1, axis=ax_)
        d_null = n_null - np.roll(n_null, -1, axis=ax_)
        mm = np.where(d_real > 0, d_null / np.maximum(d_real, 1), 0)
        if ax_ == 0:
            mm[-1, :] = 0
        else:
            mm[:, -1] = 0
        mfdr = np.maximum(mfdr, mm)
    ok_path = ((fdr <= target_fdr) | (n_real == 0)) & (mfdr <= marginal_fdr)
    if path:
        for i in range(len(corr_grid) - 2, -1, -1):
            ok_path[i, :] &= ok_path[i + 1, :]
        for j in range(len(pnr_grid) - 2, -1, -1):
            ok_path[:, j] &= ok_path[:, j + 1]
    ok = ok_path & (fdr <= target_fdr) & (n_real > 0)
    if not ok.any():
        return None, ok
    ii, jj = np.meshgrid(np.arange(len(corr_grid)), np.arange(len(pnr_grid)), indexing="ij")
    score = np.where(ok, n_real - fdr + 1e-6 * (ii + jj), -np.inf)
    i, j = np.unravel_index(np.argmax(score), score.shape)
    return (float(corr_grid[i]), float(pnr_grid[j])), ok


def robust_bg(m):
    import bm_qc
    from bearmind.seed_mask import valid_mask
    valid = valid_mask(m, border_pix=5)
    mc, sc = bm_qc._robust(m["cn"][valid])
    mp, sp = bm_qc._robust(m["pnr"][valid])
    return float(np.clip(mc + 3 * sc, 0, 0.99)), float(max(mp + 3 * sp, 0))


def _widen_one(sid):
    from bearmind.seed_mask import valid_mask
    import bm_qc
    cv2.setNumThreads(1)
    out = CACHE / "wide" / f"{sid}.npz"
    if out.exists():
        return sid, "cached"
    t0 = time.time()
    m = load_maps(sid)
    valid = valid_mask(m, border_pix=5)
    halos = {"cur": bm_qc.halo_mask(m, valid, GSIZ), "capped": halo_capped(m, valid)}
    same = bool(np.array_equal(halos["cur"], halos["capped"]))
    # The old grid points (0.50-0.98 x 2-20) are a sub-grid of the wide grid; their n_real / n_null with the
    # current halo are the snapshot grid's (checked in the report: n_wide_subgrid_mismatch). Only the new points are
    # counted here, plus every point for the capped halo where it differs from the current one.
    g0 = np.load(HERE / "grids" / f"{sid}_g4.npz")
    nc, npn = len(g0["corr_grid"]), len(g0["pnr_grid"])
    assert np.allclose(WIDE_CORR[:nc], g0["corr_grid"]) and np.allclose(WIDE_PNR[:npn], g0["pnr_grid"])
    new_pts = np.ones((len(WIDE_CORR), len(WIDE_PNR)), bool)
    new_pts[:nc, :npn] = False
    n_real = np.zeros(new_pts.shape, int)
    n_null = {"cur": np.zeros(new_pts.shape), "capped": np.zeros(new_pts.shape)}
    area = {k: valid.sum() / max((valid & ~h).sum(), 1) for k, h in halos.items()}
    n_real[:nc, :npn] = g0["n_real"]
    n_null["cur"][:nc, :npn] = g0["n_null"]
    md = 2 * m["gSig"]
    for i, c in enumerate(WIDE_CORR):
        for j, p in enumerate(WIDE_PNR):
            if new_pts[i, j]:
                n_real[i, j] = len(bm_qc.count_seeds(m["cn"], m["pnr"], c, p, GSIZ, valid=valid, min_dist=md))
            for k in (("cur", "capped") if new_pts[i, j] else ("capped",)):
                if k == "capped" and same:
                    continue
                s0 = bm_qc.count_seeds(m["cn_neg"], m["pnr_neg"], c, p, GSIZ, valid=valid & ~halos[k], min_dist=md)
                n_null[k][i, j] = len(s0) * area[k]
    if same:
        n_null["capped"] = n_null["cur"].copy()
    np.savez_compressed(out, corr_grid=WIDE_CORR, pnr_grid=WIDE_PNR, n_real=n_real, n_null_cur=n_null["cur"],
                        n_null_capped=n_null["capped"], area_cur=area["cur"], area_capped=area["capped"],
                        halo_frac_cur=halos["cur"][valid].mean(), halo_frac_capped=halos["capped"][valid].mean(),
                        robust=np.array(robust_bg(m)))
    return sid, f"done {time.time() - t0:.0f} s"


def cmd_widen(sids, workers=4):
    from concurrent.futures import ProcessPoolExecutor
    (CACHE / "wide").mkdir(parents=True, exist_ok=True)
    if not sids:
        chosen = [s for s in all_sids() if author_values(s)]
        sids = chosen + [s for s in all_sids() if s not in chosen]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for sid, st in ex.map(_widen_one, sids):
            print(f"[OK] {sid} wide grid {st} ({time.time() - t0:.0f} s)", flush=True)


def cmd_check_widen(sids):
    """The counts on the wide grid against bm_qc.suggest_init_thresholds (same grid) and the snapshot grids."""
    import bm_qc
    for sid in sids:
        z = np.load(CACHE / "wide" / f"{sid}.npz")
        m = load_maps(sid)
        sug, grid = bm_qc.suggest_init_thresholds(m, gSiz=GSIZ, corr_grid=WIDE_CORR, pnr_grid=WIDE_PNR)
        mine, _ = select_mirror(WIDE_CORR, WIDE_PNR, z["n_real"], z["n_null_cur"])
        r = sug["mirror_fdr"]
        same = (np.array_equal(grid["n_real"], z["n_real"]) and np.allclose(grid["n_null"], z["n_null_cur"]))
        print(f"[{'OK' if same else 'FAILED'}] {sid} counts equal bm_qc on the wide grid; "
              f"bm_qc pick {r.get('min_corr')}/{r.get('min_pnr')}, replica {mine}")


# ----------------------------------------------------------------------------------------------------------------------
# report: every number of auto-vs-author.md
# ----------------------------------------------------------------------------------------------------------------------

ORIG_CORR = np.round(np.arange(0.5, 0.981, 0.02), 3)
ORIG_PNR = np.round(np.arange(2, 20.01, 0.5), 2)
HIGH_MICE = ("LT1_J06", "LT1_J08", "LT1_J10", "LT1_J18")
CLEAN_MICE = ("LT1_J01", "LT1_J03", "LT1_J13")
STRIPE_SID = "LT1_J10_2D"  # stripe artifact, investigated separately (feedback item 25)
NUM = {}


def P(key, value, text):
    """Print one number with what it counts, and keep it for the explainer page."""
    NUM[key] = value
    print(f"{key} = {value}    # {text}")


def sug_type(c, p, method):
    if method == "robust_bg":
        return "fallback"
    if c >= 0.98 - 1e-9:
        return "corr_edge"
    if p >= 20 - 1e-9:
        return "pnr_edge"
    return "interior"


def batch_pick(pick, robust):
    return (pick, "mirror_fdr") if pick is not None else ((round(float(robust[0]), 3), round(float(robust[1]), 2)), "robust_bg")


def eval_point(m, c, p, halo):
    """n_real, n_null (rescaled), fdr at one pair, as bm_qc.threshold_seeds / evaluate_thresholds, any halo."""
    import bm_qc
    from bearmind.seed_mask import valid_mask
    valid = valid_mask(m, border_pix=5)
    md = 2 * m["gSig"]
    s = bm_qc.count_seeds(m["cn"], m["pnr"], c, p, GSIZ, valid=valid, min_dist=md)
    s0 = bm_qc.count_seeds(m["cn_neg"], m["pnr_neg"], c, p, GSIZ, valid=valid & ~halo, min_dist=md)
    n0 = len(s0) * valid.sum() / max((valid & ~halo).sum(), 1)
    return len(s), n0, n0 / max(len(s), 1)


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return round(float(np.median(x)), 3) if x else None


def binned_baseline(x, fps, win_s=BASE_WIN_S):
    """Running median of 1-s bin medians over win_s, interpolated to frames (cheap baseline for long traces)."""
    from scipy.ndimage import median_filter
    nb = max(1, int(round(fps)))
    n = len(x) // nb
    bm = np.median(x[:n * nb].reshape(n, nb, *x.shape[1:]), axis=1)
    k = int(round(win_s)) | 1
    bmf = median_filter(bm, size=(k,) + (1,) * (bm.ndim - 1), mode="nearest")
    t_b = (np.arange(n) + 0.5) * nb
    t = np.arange(len(x))
    if bm.ndim == 1:
        return np.interp(t, t_b, bmf)
    return np.stack([np.interp(t, t_b, bmf[:, i]) for i in range(bm.shape[1])], 1)


def trace_metrics(sid):
    z = np.load(CACHE / f"{sid}_trace.npz")
    g, fps = z["g"].astype(np.float64), float(z["fps"])
    T = len(g)
    medg = np.median(g)
    black = g < 0.5 * medg
    rest = ~black
    rest[:int(round(START_S * fps))] = False
    gg = g.copy()
    gg[black] = medg
    d = gg / binned_baseline(gg, fps) - 1
    sd = 1.4826 * np.median(np.abs(d[rest] - np.median(d[rest])))
    blocks = z["blocks"].astype(np.float32)
    vb = z["block_valid"] > 0.9
    bb = blocks[:, vb].astype(np.float64)
    bb[black] = np.median(bb[rest], axis=0)
    bd = bb / binned_baseline(bb, fps) - 1
    bsd = 1.4826 * np.median(np.abs(bd[rest] - np.median(bd[rest], axis=0)), axis=0)
    ext = (bd > 5 * bsd).mean(1)
    wide = rest & (ext >= 0.25)
    return {"sid": sid, "fps": fps, "minutes": T / fps / 60, "n_black": int(black.sum()),
            "start_excess": float(g[1:int(2 * fps)].mean() / np.median(g[rest]) - 1),
            "global_sd": float(sd), "global_max": float(d[rest].max()), "global_min": float(d[rest].min()),
            "n_global_5sd": int((rest & (d > 5 * sd)).sum()), "n_global_m5sd": int((rest & (d < -5 * sd)).sum()),
            "n_wide_events": int(wide.sum()), "wide_per_min": float(wide.sum() / (rest.sum() / fps / 60)),
            "ext_p999": float(np.percentile(ext[rest], 99.9)), "n_blocks": int(vb.sum())}


def cmd_report():
    import bm_qc
    from bearmind.seed_mask import valid_mask
    rows = read_csv("lt1_thresholds.csv")
    S = {r["sid"]: r for r in rows}
    sids = [r["sid"] for r in rows]
    mouse = {s: S[s]["mouse"] for s in sids}
    au = {s: (fnum(S[s]["author_min_corr"]), fnum(S[s]["author_min_pnr"])) for s in sids
          if fnum(S[s]["author_min_corr"]) is not None}
    src = {s: S[s]["author_source"] for s in au}
    # Own-session choice: made in the window while looking at that session. "Accept for all sessions of this mouse"
    # (bearmind/actions.py apply_to_mouse) logs every session it writes as window_mouse, the shown one included; only
    # the shown one carries the window counters (n_blobs) in the log. The last log row per session is the one in force.
    lastrow = {}
    for r in read_csv("threshold_choices_log.csv"):
        if r["when"] >= lastrow.get(r["sid"], {}).get("when", ""):
            lastrow[r["sid"]] = r
    ind = [s for s in au if src[s] == "window" or (src[s] == "window_mouse" and lastrow.get(s, {}).get("n_blobs"))]
    bat = {s: (fnum(S[s]["batch_sug_min_corr"]), fnum(S[s]["batch_sug_min_pnr"]), S[s]["batch_sug_method"])
           for s in sids}
    typ = {s: sug_type(*bat[s]) for s in sids}

    print("\n## 1. Sessions, choices, independence")
    P("n_sessions", len(sids), "sessions in lt1_thresholds.csv")
    P("n_author", len(au), "sessions whose session config has author min_corr/min_pnr (gSig 4)")
    P("n_own", len(ind), "of them chosen in the window while looking at that session (source window, or window_mouse "
      "with the window counters = the session shown when 'Accept for all sessions of this mouse' was pressed)")
    P("n_own_shown", sum(1 for s in ind if src[s] == "window_mouse"),
      "of the own-session choices: shown session of 'Accept for all sessions of this mouse': " +
      " ".join(s for s in ind if src[s] == "window_mouse"))
    P("n_copies", sum(1 for s in au if s not in ind), "copies written by 'Accept for all sessions of this mouse'")
    mice_au = sorted({mouse[s] for s in au})
    P("n_mice_author", len(mice_au), "mice with at least one author choice")
    P("n_mice_own", len({mouse[s] for s in ind}), "mice with at least one own-session choice")
    per_mouse = {mm: (sum(1 for s in ind if mouse[s] == mm), sum(1 for s in au if mouse[s] == mm and s not in ind))
                 for mm in mice_au}
    print("   per mouse (own, copies):", per_mouse)
    P("n_mice_one_own", sum(1 for v in per_mouse.values() if v[0] == 1),
      "mice whose choices come from ONE own-session choice (+ copies)")
    notown = {s: list(au[s]) for s in au if s not in ind
              and all(abs(au[s][0] - au[o][0]) > 1e-9 or abs(au[s][1] - au[o][1]) > 1e-9 for o in ind if mouse[o] == mouse[s])}
    P("button_values_not_own", notown, "sessions written by 'Accept for all sessions of this mouse' whose values match no "
      "own-session choice of that mouse (no log row with counters carries them)")
    P("n_mice_all_own", sum(1 for v in per_mouse.values() if v[1] == 0),
      "mice where every session was chosen separately")
    eq = [s for s in au if abs(au[s][0] - bat[s][0]) < 1e-9 and abs(au[s][1] - bat[s][1]) < 1e-9]
    P("n_author_equals_batch", len(eq), "author values identical to the batch suggestion: " + " ".join(eq))
    au_c = np.array([au[s][0] for s in au])
    au_p = np.array([au[s][1] for s in au])
    P("author_corr_range", [float(au_c.min()), float(au_c.max())], "min/max author min_corr (all author choices)")
    P("author_pnr_range", [float(au_p.min()), float(au_p.max())], "min/max author min_pnr (all author choices)")
    P("author_corr_median", med(au_c), "median author min_corr")
    P("author_pnr_median", med(au_p), "median author min_pnr")
    out_grid = [s for s in au if au[s][1] > 20 + 1e-9 or au[s][0] > 0.98 + 1e-9]
    P("n_author_outside_grid", len(out_grid), "author points outside the 0.50-0.98 x 2-20 grid: " + " ".join(out_grid))

    print("\n## 2. Batch suggestion types (all 84)")
    for t in ("interior", "pnr_edge", "corr_edge", "fallback"):
        P(f"n_type_{t}", sum(1 for s in sids if typ[s] == t), f"batch suggestions of type {t}")
    P("n_type_edge_or_fallback", sum(1 for s in sids if typ[s] != "interior"), "pnr_edge + corr_edge + fallback")
    ce = [s for s in sids if typ[s] == "corr_edge"]
    flat = [s for s in ce if len(set(np.load(HERE / "grids" / f"{s}_g4.npz")["n_real"][-1].tolist())) == 1]
    P("corr_edge_flat", [len(flat), len(ce)],
      "corr_edge suggestions whose min_corr 0.98 row has the same seed count at every min_pnr 2-20: n, of all corr_edge")
    fdr_dec = []
    for s in flat:
        g0 = np.load(HERE / "grids" / f"{s}_g4.npz")
        f_row = g0["fdr"][-1]
        if f_row[-1] < f_row[0] - 1e-12:
            fdr_dec.append(s)
    # bm_qc.suggest_init_thresholds score = n_real - fdr + 1e-6 (i + j): equal seeds -> lower FDR wins, then the
    # stricter point breaks exact ties
    P("corr_edge_flat_fdr", [len(fdr_dec), len(flat) - len(fdr_dec)],
      "flat rows: FDR lower at min_pnr 20 than at 2 (lower FDR wins), FDR equal along the row (exact tie, the stricter "
      "point wins): " + " ".join(fdr_dec))
    fb = [s for s in sids if typ[s] == "fallback"]
    P("fallback_pnr_range", [min(bat[s][1] for s in fb), max(bat[s][1] for s in fb)],
      "batch min_pnr of the robust_bg fallback suggestions, min/max (max: " + max(fb, key=lambda s: bat[s][1]) + ")")

    print("\n## 3. Author vs batch suggestion and vs each method")
    for grp, ss in (("all", list(au)), ("own", ind)):
        dc = np.array([au[s][0] - bat[s][0] for s in ss])
        dp = np.array([au[s][1] - bat[s][1] for s in ss])
        P(f"{grp}_corr_lower", int((dc < -1e-9).sum()), f"[{grp}] author min_corr below the batch suggestion")
        P(f"{grp}_corr_same", int((np.abs(dc) < 1e-9).sum()), f"[{grp}] author min_corr equal to it")
        P(f"{grp}_corr_higher", int((dc > 1e-9).sum()), f"[{grp}] author min_corr above it")
        P(f"{grp}_pnr_lower", int((dp < -1e-9).sum()), f"[{grp}] author min_pnr below the batch suggestion")
        P(f"{grp}_pnr_same", int((np.abs(dp) < 1e-9).sum()), f"[{grp}] author min_pnr equal to it")
        P(f"{grp}_pnr_higher", int((dp > 1e-9).sum()), f"[{grp}] author min_pnr above it")
        P(f"{grp}_dcorr_median", med(dc), f"[{grp}] median author - batch, min_corr")
        P(f"{grp}_dpnr_median", med(dp), f"[{grp}] median author - batch, min_pnr")
        for t in ("interior", "pnr_edge", "corr_edge", "fallback"):
            st = [s for s in ss if typ[s] == t]
            if not st:
                continue
            P(f"{grp}_{t}_n", len(st), f"[{grp}] sessions with batch type {t}")
            P(f"{grp}_{t}_dcorr", [med([au[s][0] - bat[s][0] for s in st]),
                                   round(min(au[s][0] - bat[s][0] for s in st), 3),
                                   round(max(au[s][0] - bat[s][0] for s in st), 3)],
              f"[{grp}] {t}: author - batch min_corr, median [min, max]")
            P(f"{grp}_{t}_dpnr", [med([au[s][1] - bat[s][1] for s in st]),
                                  round(min(au[s][1] - bat[s][1] for s in st), 2),
                                  round(max(au[s][1] - bat[s][1] for s in st), 2)],
              f"[{grp}] {t}: author - batch min_pnr, median [min, max]")
    for meth in ("mirror_fdr", "robust_bg", "pct_margin"):
        ss = [s for s in au if fnum(S[s][f"{meth}_min_corr"]) is not None]
        dc = [au[s][0] - fnum(S[s][f"{meth}_min_corr"]) for s in ss]
        dp = [au[s][1] - fnum(S[s][f"{meth}_min_pnr"]) for s in ss]
        P(f"vs_{meth}", {"n": len(ss), "dcorr_median": med(dc), "dpnr_median": med(dp),
                         "corr_lower": int(sum(d < -1e-9 for d in dc)), "pnr_lower": int(sum(d < -1e-9 for d in dp))},
          f"author - {meth} (recomputed in the snapshot), sessions where {meth} gives a value")

    print("\n## 4. Halo mask (null seeds are not counted near clearly active cells)")
    halo, cut, sdcn, halo_cap = {}, {}, {}, {}
    maps_cache = {}
    for s in sids:
        m = load_maps(s)
        valid = valid_mask(m, border_pix=5)
        mc, sc = bm_qc._robust(m["cn"][valid])
        cut[s], sdcn[s] = float(mc + 3 * sc), float(sc)
        h = bm_qc.halo_mask(m, valid, GSIZ)
        hc = halo_capped(m, valid)
        halo[s], halo_cap[s] = float(h[valid].mean()), float(hc[valid].mean())
        if s in au:
            maps_cache[s] = (m, h, hc)
    empty = [s for s in sids if halo[s] == 0]
    P("n_halo_empty", len(empty), "sessions whose bm_qc.halo_mask is empty")
    P("halo_empty_mice", sorted({mouse[s] for s in empty}), "mice with an empty halo mask")
    P("n_halo_empty_cut_above_1", sum(1 for s in empty if cut[s] > 1), "of them: corr cut median + 3 SD above 1")
    P("cut_range_empty", [round(float(min(cut[s] for s in empty)), 3), round(float(max(cut[s] for s in empty)), 3)],
      "corr cut (median + 3 robust SD of cn) in empty-halo sessions, min/max")
    ne = [s for s in sids if halo[s] > 0]
    P("n_halo_present", len(ne), "sessions whose bm_qc.halo_mask is not empty")
    nb = {}
    for s in au:
        k = "mirror_fdr" if bat[s][2] == "mirror_fdr" else "robust_bg"
        nb[s] = (fnum(S[s][f"{k}_n_seeds"]), fnum(S[s]["author_n_seeds"]))
    for grp, ss in (("halo_empty", [s for s in au if s in empty]), ("halo_present", [s for s in au if s in ne])):
        if grp == "halo_empty":
            nb_all = [fnum(S[s][("mirror_fdr" if bat[s][2] == "mirror_fdr" else "robust_bg") + "_n_seeds"]) for s in empty]
            P("seeds_sug_empty_all", [med(nb_all), len(nb_all)],
              "all empty-halo sessions: median seeds at the batch suggestion, n sessions")
        P(f"seeds_sug_vs_author_{grp}", [med([nb[s][0] for s in ss]), med([nb[s][1] for s in ss]),
                                         med([nb[s][0] / nb[s][1] for s in ss if nb[s][1]]), len(ss)],
          f"{grp}: seeds at the batch suggestion, seeds at the author's point (batch maps), median ratio, n sessions")
    mp_all = {}
    for s in sids:
        m = maps_cache[s][0] if s in maps_cache else load_maps(s)
        mp_all[s] = float(np.median(m["pnr"][valid_mask(m, border_pix=5)]))
    fbs = [s for s in sids if typ[s] == "fallback" and s != STRIPE_SID]
    P("fallback_median_pnr", [round(min(mp_all[s] for s in fbs), 1), round(max(mp_all[s] for s in fbs), 1)],
      "median PNR over the valid area in robust_bg-fallback sessions, min/max")
    P("halo_present_median_pnr", [round(min(mp_all[s] for s in ne), 1), round(max(mp_all[s] for s in ne), 1)],
      "median PNR over the valid area in sessions with a halo, min/max")
    P("stripe_session_median_pnr", round(mp_all[STRIPE_SID], 1),
      f"{STRIPE_SID} (stripe artifact, investigated separately): median PNR over the valid area")
    P("cut_range_nonempty", [round(float(min(cut[s] for s in ne)), 3), round(float(max(cut[s] for s in ne)), 3)],
      "corr cut in sessions with a halo, min/max")
    P("halo_frac_nonempty", [round(min(halo[s] for s in ne), 2), med([halo[s] for s in ne]),
                             round(max(halo[s] for s in ne), 2)],
      "share of the valid area under the halo where it exists: min, median, max")
    for t in ("interior", "pnr_edge", "corr_edge", "fallback"):
        P(f"halo_x_{t}", [sum(1 for s in empty if typ[s] == t), sum(1 for s in ne if typ[s] == t)],
          f"batch type {t}: [empty halo, halo present]")
    P("halo_capped_frac_empty", [round(min(halo_cap[s] for s in empty), 2), med([halo_cap[s] for s in empty]),
                                 round(max(halo_cap[s] for s in empty), 2)],
      "capped halo (what-if) in the empty-halo sessions: share of valid area min, median, max")
    P("n_capped_equals_cur", sum(1 for s in ne if abs(halo_cap[s] - halo[s]) < 1e-12),
      f"of {len(ne)} sessions with a halo, the capped halo is identical")
    afd = {s: fnum(S[s]["author_fdr"]) for s in au}
    P("author_fdr_halo_empty", [med([afd[s] for s in au if s in empty]),
                                round(min(afd[s] for s in au if s in empty), 3),
                                round(max(afd[s] for s in au if s in empty), 3), sum(1 for s in au if s in empty)],
      "mirror FDR at the author's point, empty-halo sessions: median, min, max, n")
    P("author_fdr_halo_present", [med([afd[s] for s in au if s in ne]),
                                  round(min(afd[s] for s in au if s in ne), 3),
                                  round(max(afd[s] for s in au if s in ne), 3), sum(1 for s in au if s in ne)],
      "mirror FDR at the author's point, sessions with a halo: median, min, max, n")
    cap_at_au = {}
    for s in au:
        m, h, hc = maps_cache[s]
        cap_at_au[s] = tuple(float(x) for x in eval_point(m, au[s][0], au[s][1], hc))
    P("author_fdr_capped_halo_empty", [med([cap_at_au[s][2] for s in au if s in empty]),
                                       round(min(cap_at_au[s][2] for s in au if s in empty), 3),
                                       round(max(cap_at_au[s][2] for s in au if s in empty), 3)],
      "FDR at the author's point with the capped halo, empty-halo sessions: median, min, max")
    P("author_fdr_capped_halo_present", med([cap_at_au[s][2] for s in au if s in ne]),
      "FDR at the author's point with the capped halo, sessions with a halo (median; halo unchanged in most)")
    # the window's own counters at the author's Accept (threshold_choices_log.csv, choices made on that session)
    logr = read_csv("threshold_choices_log.csv")
    last = {}
    for r in logr:
        if r["n_blobs"] and r["when"] >= last.get(r["sid"], {}).get("when", ""):
            last[r["sid"]] = r
    rat = {s: float(r["n_mirror_blobs"]) / max(float(r["n_blobs"]), 1) for s, r in last.items()}
    P("window_mirror_ratio", [med([rat[s] for s in rat if s in empty]), sum(1 for s in rat if s in empty),
                              med([rat[s] for s in rat if s in ne]), sum(1 for s in rat if s in ne)],
      "window counters at the last Accept on the session: mirrored blobs / blobs, median and n for empty-halo "
      "sessions, then for sessions with a halo (the window counts blobs without a halo mask)")
    # where are the null seeds? distance to the nearest real seed, against random pixels of the same null area
    from scipy.spatial import cKDTree
    near, no_null = {}, []
    for s in au:
        m, h, hc = maps_cache[s]
        valid = valid_mask(m, border_pix=5)
        sr, s0, _ = bm_qc.threshold_seeds(m, au[s][0], au[s][1], GSIZ, valid, h)
        if len(sr) == 0 or len(s0) == 0:
            no_null.append(s)
            continue
        tree = cKDTree(sr)
        dn, dr = tree.query(s0)[0], tree.query(np.argwhere(valid & ~h))[0]
        near[s] = (float(np.median(dn)), float(np.median(dr)), float((dn <= GSIZ).mean()), float((dr <= GSIZ).mean()))
    for grp, ss in (("halo_empty", [s for s in near if s in empty]), ("halo_present", [s for s in near if s in ne])):
        P(f"null_near_real_{grp}", [med([near[s][0] for s in ss]), med([near[s][1] for s in ss]),
                                    med([near[s][2] for s in ss]), med([near[s][3] for s in ss]), len(ss)],
          f"{grp}, author's point: median distance (px) null seed -> nearest real seed, every pixel of the null area "
          f"-> nearest real seed; share at <= {GSIZ} px for null seeds, for those pixels; n sessions (medians)")
    P("no_null_at_author", no_null, "sessions with an author choice and no null seed at the author's point")

    print("\n## 5. R2: wider grid (min_corr ... 0.98 + 0.99, min_pnr 2-40); and the capped-halo what-if")
    W = {}
    for s in sids:
        f = CACHE / "wide" / f"{s}.npz"
        if f.exists():
            W[s] = np.load(f)
    P("n_wide", len(W), "sessions with the wide grid computed")
    nc, npn = len(ORIG_CORR), len(ORIG_PNR)
    mism = []
    for s, z in W.items():
        g0 = np.load(HERE / "grids" / f"{s}_g4.npz")
        if not (np.array_equal(z["n_real"][:nc, :npn], g0["n_real"]) and
                np.allclose(z["n_null_cur"][:nc, :npn], g0["n_null"])):
            mism.append(s)
    P("n_wide_subgrid_mismatch", len(mism), "sessions whose wide counts differ from the snapshot grid on the old points")
    res = {}
    for s, z in W.items():
        rb = tuple(z["robust"])
        o, _ = select_mirror(ORIG_CORR, ORIG_PNR, z["n_real"][:nc, :npn], z["n_null_cur"][:nc, :npn])
        w_, _ = select_mirror(WIDE_CORR, WIDE_PNR, z["n_real"], z["n_null_cur"])
        oc, _ = select_mirror(ORIG_CORR, ORIG_PNR, z["n_real"][:nc, :npn], z["n_null_capped"][:nc, :npn])
        wc, _ = select_mirror(WIDE_CORR, WIDE_PNR, z["n_real"], z["n_null_capped"])
        res[s] = {"orig": batch_pick(o, rb), "wide": batch_pick(w_, rb), "orig_cap": batch_pick(oc, rb),
                  "wide_cap": batch_pick(wc, rb)}
    chg = {}
    for s in ne:
        if abs(halo_cap[s] - halo[s]) < 1e-12:
            continue
        chg[s] = {"halo": [round(halo[s], 3), round(halo_cap[s], 3)],
                  "author_fdr": [afd.get(s), round(cap_at_au[s][2], 3) if s in cap_at_au else None],
                  "suggestion": [list(res[s]["orig"][0]), list(res[s]["orig_cap"][0])]}
    P("capped_changes_with_halo", chg, "sessions with a halo where the capped halo differs: halo share, FDR at the "
      "author's point, suggestion on the old grid (current -> capped)")
    bad = [s for s in res if abs(res[s]["orig"][0][0] - bat[s][0]) > 1e-6 or abs(res[s]["orig"][0][1] - bat[s][1]) > 1e-6]
    P("n_orig_replica_mismatch", len(bad), "sessions where the replica on the old grid differs from the batch suggestion: "
      + " ".join(bad))

    def direction(a, b):
        return "up" if b > a + 1e-9 else ("down" if b < a - 1e-9 else "same")
    for key in ("wide", "orig_cap", "wide_cap"):
        dirs = {"corr": {}, "pnr": {}}
        for s in res:
            for k, ax in (("corr", 0), ("pnr", 1)):
                d = direction(res[s]["orig"][0][ax], res[s][key][0][ax])
                dirs[k][d] = dirs[k].get(d, 0) + 1
        P(f"r2_{key}_direction_all", dirs, f"{key} vs the current suggestion, all sessions with the wide grid")
        mth = {}
        for s in res:
            mth[res[s][key][1]] = mth.get(res[s][key][1], 0) + 1
        P(f"r2_{key}_methods", mth, f"{key}: sessions where mirror_fdr finds a point / falls back to robust_bg")
        sa = [s for s in au if s in res]
        P(f"r2_{key}_vs_author", {
            "n": len(sa),
            "corr_above_author": sum(1 for s in sa if res[s][key][0][0] > au[s][0] + 1e-9),
            "pnr_above_author": sum(1 for s in sa if res[s][key][0][1] > au[s][1] + 1e-9),
            "closer_corr": sum(1 for s in sa if abs(res[s][key][0][0] - au[s][0]) < abs(res[s]["orig"][0][0] - au[s][0]) - 1e-9),
            "farther_corr": sum(1 for s in sa if abs(res[s][key][0][0] - au[s][0]) > abs(res[s]["orig"][0][0] - au[s][0]) + 1e-9),
            "closer_pnr": sum(1 for s in sa if abs(res[s][key][0][1] - au[s][1]) < abs(res[s]["orig"][0][1] - au[s][1]) - 1e-9),
            "farther_pnr": sum(1 for s in sa if abs(res[s][key][0][1] - au[s][1]) > abs(res[s]["orig"][0][1] - au[s][1]) + 1e-9),
            "median_abs_dcorr": med([abs(res[s][key][0][0] - au[s][0]) for s in sa]),
            "median_abs_dpnr": med([abs(res[s][key][0][1] - au[s][1]) for s in sa])},
            f"{key} against the author's values (sessions with an author choice)")
    sa = [s for s in au if s in res]
    P("current_vs_author", {"median_abs_dcorr": med([abs(res[s]["orig"][0][0] - au[s][0]) for s in sa]),
                            "median_abs_dpnr": med([abs(res[s]["orig"][0][1] - au[s][1]) for s in sa])},
      "current suggestion against the author's values")
    # is the author's point admissible under the current rule? (nearest point of the old grid, author points inside it)
    inside = [s for s in au if s in W and au[s][0] <= 0.98 + 1e-9 and au[s][1] <= 20 + 1e-9]
    adm = 0
    for s in inside:
        z = W[s]
        _, ok0 = select_mirror(ORIG_CORR, ORIG_PNR, z["n_real"][:nc, :npn], z["n_null_cur"][:nc, :npn])
        i = int(np.argmin(np.abs(ORIG_CORR - au[s][0])))
        j = int(np.argmin(np.abs(ORIG_PNR - au[s][1])))
        adm += int(ok0[i, j])
    P("author_admissible", [adm, len(inside)],
      "author points inside the old grid that are admissible under the current rule (nearest grid point): n, of n inside")
    # why the wide grid changes the pick, per session
    why, small = {}, []
    for s, z in W.items():
        nr, nn = z["n_real"], z["n_null_cur"]
        o, _ = select_mirror(ORIG_CORR, ORIG_PNR, nr[:nc, :npn], nn[:nc, :npn])
        w_, okw = select_mirror(WIDE_CORR, WIDE_PNR, nr, nn)
        fdr = nn / np.maximum(nr, 1)
        mf = np.zeros_like(fdr)
        for ax_ in (0, 1):
            d_r = nr - np.roll(nr, -1, axis=ax_)
            d_n = nn - np.roll(nn, -1, axis=ax_)
            mm = np.where(d_r > 0, d_n / np.maximum(d_r, 1), 0)
            if ax_ == 0:
                mm[-1, :] = 0
            else:
                mm[:, -1] = 0
            mf = np.maximum(mf, mm)
        fdr_ok = (fdr <= 0.1) | (nr == 0)
        base = fdr_ok & (mf <= 0.5)
        if o == w_:
            k = "same"
        elif o is None:
            k = "fallback_now_found"
        else:
            i = int(np.argmin(np.abs(WIDE_CORR - o[0])))
            j = int(np.argmin(np.abs(WIDE_PNR - o[1])))
            if okw[i, j]:
                k = "old_admissible_new_more_seeds"
            else:
                rect = np.zeros(base.shape, bool)
                rect[i:, j:] = True
                bad_pts = rect & ~base
                k = "old_out_marginal_only" if not (bad_pts & ~fdr_ok).any() else "old_out_fdr"
                # null seeds added by one loosening step at the blocking points (raw count, before the area rescale)
                for ax_ in (0, 1):
                    d_n = nn - np.roll(nn, -1, axis=ax_)
                    sel_ = bad_pts & fdr_ok & (mf > 0.5)
                    small += [(float(np.round(v / float(z["area_cur"]))), float(v)) for v in d_n[sel_]]
        why[k] = why.get(k, 0) + 1
    P("r2_why", why, "per session, why the wide-grid pick differs: same; fallback_now_found (no admissible point on the "
      "old grid); old_admissible_new_more_seeds (a new point has more seeds); old_out_marginal_only (the old pick "
      "drops out because a stricter new point fails only the marginal FDR <= 0.5 while FDR <= 0.1); old_out_fdr")
    small = [x for x in small if x[0] > 0]
    P("r2_marginal_null_added", [med([x[0] for x in small]), float(np.percentile([x[0] for x in small], 90)),
                                 med([x[1] for x in small]), round(float(np.percentile([x[1] for x in small], 90)), 2)],
      "at the blocking points of old_out_marginal_only: null seeds added by one loosening step, raw count median and "
      "90th percentile, then the same in the area-scaled units of bm_qc (count x valid / (valid & ~halo))")
    edge = [s for s in res if typ[s] != "interior"]
    print("   per session (current -> wide -> capped halo old grid -> capped halo wide grid | author):")
    for s in sorted(res):
        r = res[s]
        print(f"   {s} {typ[s]:9s} {r['orig'][0]} -> {r['wide'][0]} {r['wide'][1][:6]} -> {r['orig_cap'][0]} "
              f"{r['orig_cap'][1][:6]} -> {r['wide_cap'][0]} {r['wide_cap'][1][:6]} | {au.get(s)}")
    P("n_edge_sessions", len(edge), "sessions with a non-interior current suggestion (in the wide set)")

    print("\n## 6. Which rule reproduces the author? (rule family sweep; leave-one-mouse-out)")
    fam = [(h_, g_, t, mg, pa) for h_ in ("cur", "capped") for g_ in ("orig", "wide") for t in (0.1, 0.2, 0.3, 0.5)
           for mg in (0.5, np.inf) for pa in (True, False)]

    def pick_for(s, rule):
        h_, g_, t, mg, pa = rule
        z = W[s]
        nr, nn = z["n_real"], z[f"n_null_{h_}"]
        cg, pg = WIDE_CORR, WIDE_PNR
        if g_ == "orig":
            nr, nn, cg, pg = nr[:nc, :npn], nn[:nc, :npn], ORIG_CORR, ORIG_PNR
        p, _ = select_mirror(cg, pg, nr, nn, target_fdr=t, marginal_fdr=mg, path=pa)
        return batch_pick(p, tuple(z["robust"]))[0]

    def err(a, b):  # distance in grid steps
        return abs(a[0] - b[0]) / 0.02 + abs(a[1] - b[1]) / 0.5
    ind_w = [s for s in ind if s in W]
    mice_w = sorted({mouse[s] for s in ind_w})
    picks = {r: {s: pick_for(s, r) for s in ind_w} for r in fam}

    def mouse_err(r, ss):  # mean over mice of the mean error within a mouse (mice weigh equally)
        return float(np.mean([np.mean([err(picks[r][s], au[s]) for s in ss if mouse[s] == m2])
                              for m2 in sorted({mouse[x] for x in ss})]))
    rank = sorted(fam, key=lambda r: mouse_err(r, ind_w))
    cur_rule = ("cur", "orig", 0.1, 0.5, True)

    def rname(r):
        return (f"halo={r[0]} grid={r[1]} target={r[2]} marginal={'off' if np.isinf(r[3]) else r[3]} "
                f"path={'on' if r[4] else 'off'}")

    def summ(r):
        pk = picks[r]
        return {"err": round(mouse_err(r, ind_w), 3),
                "median_abs_dcorr": med([abs(pk[s][0] - au[s][0]) for s in ind_w]),
                "median_abs_dpnr": med([abs(pk[s][1] - au[s][1]) for s in ind_w]),
                "mean_dpnr": round(float(np.mean([pk[s][1] - au[s][1] for s in ind_w])), 2),
                "mean_dcorr": round(float(np.mean([pk[s][0] - au[s][0] for s in ind_w])), 3)}
    print(f"   {len(fam)} rules, {len(ind_w)} own-session choices of {len(mice_w)} mice; error = mean over mice of "
          "|dcorr|/0.02 + |dpnr|/0.5 (grid steps)")
    for r in rank[:10] + [cur_rule]:
        print(f"   {summ(r)}  {rname(r)}")
    P("rule_best", dict(summ(rank[0]), rule=rname(rank[0])), "best rule of the family on all own-session choices (in-sample)")
    P("rule_best_ties", [rname(r) for r in fam if r != rank[0] and all(picks[r][s] == picks[rank[0]][s] for s in ind_w)],
      "rules with exactly the same picks as the best rule on every own-session choice")
    P("rule_current", summ(cur_rule), "current rule (halo cur, old grid, target 0.1, marginal 0.5, path on)")
    for key, r in (("rule_capped_only", ("capped", "orig", 0.1, 0.5, True)),
                   ("rule_wide_only", ("cur", "wide", 0.1, 0.5, True)),
                   ("rule_capped_t03", ("capped", "orig", 0.3, 0.5, True)),
                   ("rule_cur_t03", ("cur", "orig", 0.3, 0.5, True))):
        P(key, summ(r), rname(r))
    lomo, lomo_const = [], []
    for mm in mice_w:
        test = [s for s in ind_w if mouse[s] == mm]
        train = [s for s in ind_w if mouse[s] != mm]
        best = min(fam, key=lambda r: mouse_err(r, train))
        lomo.append((mm, best, float(np.mean([err(picks[best][s], au[s]) for s in test]))))
        cc = (float(np.median([au[s][0] for s in train])), float(np.median([au[s][1] for s in train])))
        lomo_const.append(float(np.mean([err(cc, au[s]) for s in test])))
    P("lomo_mice", len(mice_w), "mice in the leave-one-mouse-out test (own-session choices only)")
    P("lomo_rule", {"mean_err_steps": round(float(np.mean([e for _, _, e in lomo])), 3),
                    "rules_chosen": sorted({rname(b) for _, b, _ in lomo})},
      "rule picked on the other mice, error on the held-out mouse")
    P("lomo_current_rule", round(mouse_err(cur_rule, ind_w), 3), "current rule, same per-mouse error")
    P("lomo_constant", round(float(np.mean(lomo_const)), 3),
      "constant = median author values of the other mice, error on the held-out mouse")
    P("constant_all", [med([au[s][0] for s in ind]), med([au[s][1] for s in ind])],
      "median author min_corr, min_pnr over the own-session choices")

    print("\n## 7. Frames test: maps recomputed without frame sets")
    FT = {}
    for f in sorted((HERE / "frames_test").glob("*.json")):
        FT[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    P("frames_sessions", sorted(FT), "sessions with the frames test")
    vb = {}
    for s_, r_ in FT.items():
        a_ = r_["versions"]["all"].get("at_author") or {}
        vb[s_] = [fnum(S[s_]["author_fdr"]), a_.get("fdr"), fnum(S[s_]["author_n_seeds"]), a_.get("n_seeds")]
    P("ft_vs_batch", vb, "author's point: [FDR batch maps, FDR maps recomputed from the movie, seeds batch maps, "
      "seeds recomputed]")
    P("ft_vs_batch_max", [round(max(abs(v[0] - v[1]) for v in vb.values()), 3),
                          int(max(abs(v[2] - v[3]) for v in vb.values())),
                          round(max(max(r_["check_max_abs_diff"].values()) for r_ in FT.values()), 4)],
      "max |FDR batch - recomputed|, max |seeds batch - recomputed| at the author's point; max |map difference|")
    for s, r in FT.items():
        P(f"ft_{s}_check", r["check_max_abs_diff"], f"{s}: max |recomputed - batch map| in the valid area")
        P(f"ft_{s}_sets", r["set_sizes"], f"{s}: frames in each left-out set")
        for v, x in r["versions"].items():
            a = x.get("at_author") or {}
            P(f"ft_{s}_{v}", {"batch": [x["batch"]["min_corr"], x["batch"]["min_pnr"], x["batch"]["method"]],
                              "robust_bg": [x["robust_bg"]["min_corr"], x["robust_bg"]["min_pnr"]],
                              "author_n": a.get("n_seeds"), "author_null": a.get("n_null_seeds"),
                              "author_fdr": a.get("fdr"), "median_pnr": round(x["median_pnr_valid"], 2),
                              "median_pnr_neg": round(x["median_pnr_neg_valid"], 2)},
              f"{s}, maps {v}: suggestion, robust_bg, seeds / null / FDR at the author's point")
        z = np.load(CACHE / f"{s}_frames.npz")
        m0 = load_maps(s)
        vcut = {}
        for v in r["versions"]:
            mv = dict(m0, cn=z[f"{v}_cn"], pnr=z[f"{v}_pnr"], cn_neg=z[f"{v}_cn_neg"], pnr_neg=z[f"{v}_pnr_neg"])
            vv = valid_mask(mv, border_pix=5)
            mc_, sc_ = bm_qc._robust(mv["cn"][vv])
            vcut[v] = [round(float(mc_ + 3 * sc_), 3), round(float(bm_qc.halo_mask(mv, vv, GSIZ)[vv].mean()), 3)]
        P(f"ft_{s}_halo", vcut, f"{s}: per maps version [corr cut of the halo mask, share of the valid area in the mask]")
        sp = r["seed_peak_frames"]
        T = len(z["set_black"])
        nul = np.array(sp["null_argmin"])
        rea = np.array(sp["real_argmax"])
        shares = {}
        for k in ("black", "start", "bright", "dim"):
            sk = z[f"set_{k}"]
            shares[k] = [round(float(sk.mean()), 4), round(float(sk[nul].mean()), 3) if len(nul) else None,
                         round(float(sk[rea].mean()), 3) if len(rea) else None]
        P(f"ft_{s}_peak_sets", shares, f"{s}: [share of frames, share of null-seed minima, share of real-seed maxima] "
          "falling in each set")
        if len(nul):
            u, cnts = np.unique(nul, return_counts=True)
            P(f"ft_{s}_null_frames", [len(nul), len(u), int(cnts.max())],
              f"{s}: null seeds, distinct frames of their minima, most null seeds sharing one frame")
            # does a real transient next to the null seed peak in the same frame? (negative halo of a cell)
            m0 = load_maps(s)
            red = (m0["pnr"] >= sp["point"][1]) & (m0["cn"] >= sp["point"][0])
            amax = z["amax"]
            hh, ww = amax.shape
            yy, xx = np.mgrid[-GSIZ:GSIZ + 1, -GSIZ:GSIZ + 1]
            disk = yy ** 2 + xx ** 2 <= GSIZ ** 2

            def coincide(t_of):
                hit = 0
                for (y, x), t in zip(sp["null_yx"], t_of):
                    y0, y1, x0, x1 = max(0, y - GSIZ), min(hh, y + GSIZ + 1), max(0, x - GSIZ), min(ww, x + GSIZ + 1)
                    dsk = disk[y0 - y + GSIZ:y1 - y + GSIZ, x0 - x + GSIZ:x1 - x + GSIZ]
                    win = red[y0:y1, x0:x1] & dsk
                    if (np.abs(amax[y0:y1, x0:x1][win] - t) <= 2).any():
                        hit += 1
                return hit / len(t_of)
            co, ch = coincide(nul), coincide((nul + T // 2) % T)
            P(f"ft_{s}_halo_coincidence", [round(co, 3), round(ch, 3)],
              f"{s}: share of null seeds with a real red pixel within {GSIZ} px peaking within 2 frames of the null "
              "minimum; same for a frame shifted by half the session (chance)")
            a = r["versions"]["all"].get("at_author") or {}
            if a.get("n_seeds"):
                P(f"ft_{s}_fdr_unexplained", round(a["fdr"] * (1 - (co - ch) / (1 - ch)), 3),
                  f"{s}: rough FDR at the author's point after removing the null seeds explained by a coincident "
                  "neighbouring transient beyond chance: fdr * (1 - (share - chance) / (1 - chance))")

    print("\n## 8. Whole-frame traces (calcium-wave check)")
    TM = {}
    for s in sids:
        if (CACHE / f"{s}_trace.npz").exists() and mouse[s] in HIGH_MICE + CLEAN_MICE:
            TM[s] = trace_metrics(s)
    for s, t in TM.items():
        print(f"   {s} {'HIGH ' if mouse[s] in HIGH_MICE else 'clean'} halo {halo[s]:.2f} sd {t['global_sd']:.4f} "
              f"max {t['global_max']:.3f} min {t['global_min']:.3f} n>5sd {t['n_global_5sd']} n<-5sd {t['n_global_m5sd']} "
              f"wide {t['n_wide_events']} ({t['wide_per_min']:.2f}/min) ext99.9 {t['ext_p999']:.3f} "
              f"start {t['start_excess']:+.3f} black {t['n_black']}" + ("  [stripe artifact]" if s == STRIPE_SID else ""))
    for grp, mice in (("high", HIGH_MICE), ("clean", CLEAN_MICE)):
        ss = [s for s in TM if mouse[s] in mice]
        P(f"trace_{grp}", {"n": len(ss), "global_sd_median": med([TM[s]["global_sd"] for s in ss]),
                           "n_5sd_per_min_median": med([TM[s]["n_global_5sd"] / TM[s]["minutes"] for s in ss]),
                           "wide_per_min_median": med([TM[s]["wide_per_min"] for s in ss]),
                           "start_excess_median": med([TM[s]["start_excess"] for s in ss]),
                           "black_frames_total": sum(TM[s]["n_black"] for s in ss)},
          f"{grp} mice: whole-frame dF/F robust SD, frames > 5 SD per min, wide-field event frames per min "
          "(>= 25 % of 32-px blocks above 5 SD at once), start excess, black frames")
    P("halo_by_group", {"high_empty": sum(1 for s in sids if mouse[s] in HIGH_MICE and halo[s] == 0),
                        "high_n": sum(1 for s in sids if mouse[s] in HIGH_MICE),
                        "clean_empty": sum(1 for s in sids if mouse[s] in CLEAN_MICE and halo[s] == 0),
                        "clean_n": sum(1 for s in sids if mouse[s] in CLEAN_MICE)}, "empty halo by group")
    P("sdcn_by_halo", [med([sdcn[s] for s in empty]), med([sdcn[s] for s in ne])],
      "robust SD of the corr map: empty-halo sessions, sessions with a halo (medians)")

    print("\n## 9. Example sessions of the explainer")
    ex = {"interior": "LT1_J24_1D", "pnr_edge": "LT1_J03_1D", "corr_edge": "LT1_J18_1D", "fallback": "LT1_J06_1D"}
    P("example_cases", ex, "example sessions of the grid section")
    for s in ex.values():
        P(f"ex_{s}", {"batch": [bat[s][0], bat[s][1], bat[s][2]], "author": list(au[s]),
                      "source": "own" if s in ind else "copy",
                      "author_fdr": afd[s], "author_fdr_capped": round(cap_at_au[s][2], 3), "halo": round(halo[s], 3),
                      "halo_capped": round(halo_cap[s], 3), "cut": round(cut[s], 3),
                      "wide": res[s]["wide"] if s in res else None, "orig_cap": res[s]["orig_cap"] if s in res else None},
          f"{s}: batch suggestion, author, FDR at the author's point (current / capped halo), halo shares, R2 pick")

    # per-session table for the explainer
    table = []
    for s in sids:
        r = res.get(s)
        table.append({"sid": s, "mouse": mouse[s], "type": typ[s], "batch": bat[s], "author": au.get(s),
                      "source": ("own" if s in ind else "copy") if s in au else None, "halo": halo[s], "halo_capped": halo_cap[s], "cut": cut[s],
                      "author_fdr": afd.get(s), "author_fdr_capped": cap_at_au[s][2] if s in cap_at_au else None,
                      "wide": r and r["wide"], "orig_cap": r and r["orig_cap"], "wide_cap": r and r["wide_cap"],
                      "trace": TM.get(s)})
    (HERE / "report_numbers.json").write_text(json.dumps({"numbers": NUM, "sessions": table}, indent=1,
                                                          default=float), encoding="utf-8")
    print("\n[OK] report_numbers.json written")


def _fmt(v, d=None):
    """Russian decimal comma; lists and dicts separated by '; '."""
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v)).replace("-", "−")
    if isinstance(v, (float, np.floating)):
        if d is not None:  # half away from zero on the decimal value, as the page does (f-strings round the binary one)
            from decimal import Decimal, ROUND_HALF_UP
            s = str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
        else:
            s = f"{v:g}"
        return s.replace(".", ",").replace("-", "−")
    if isinstance(v, (list, tuple)):
        return "[" + "; ".join(_fmt(x, d) for x in v) + "]"
    if isinstance(v, dict):
        return "; ".join(f"{_ru_key(k)} {_fmt(x, d)}" for k, x in v.items())
    return str(v).replace("LT1_", "")


def _ru_key(k):
    return {"up": "выше", "down": "ниже", "same": "так же", "mirror_fdr": "mirror_fdr", "robust_bg": "robust_bg",
            "fallback_now_found": "вместо запасного метода нашлась точка за старым краем",
            "old_out_marginal_only": "старая точка выпала из-за предельного FDR",
            "old_admissible_new_more_seeds": "новая точка с большим числом затравок",
            "old_out_fdr": "старая точка выпала из-за FDR"}.get(k, str(k).replace("LT1_", ""))


def cmd_render_md():
    """auto-vs-author.md from auto-vs-author.template.md: {{key.path:digits}} -> report_numbers.json."""
    import re
    num = json.loads((HERE / "report_numbers.json").read_text(encoding="utf-8"))["numbers"]

    def get(expr):
        key, _, d = expr.partition(":")
        parts = key.split(".")
        k0 = parts[0]
        v = num[k0]
        for p in parts[1:]:
            v = v[int(p)] if isinstance(v, list) else v[p]
        return _fmt(v, int(d) if d else None)
    tpl = (HERE / "auto-vs-author.template.md").read_text(encoding="utf-8")
    out = re.sub(r"\{\{([^}]+)\}\}", lambda mm: get(mm.group(1)), tpl)
    sids = sorted({k[3:-6] for k in num if k.startswith("ft_") and k.endswith("_check")})
    names = {"all": "все кадры", "no_black": "без чёрных", "no_start": "без первых 15 с", "no_bright": "без 2 % ярких",
             "no_dim": "без 2 % тёмных", "no_ctrl": "контроль (столько же кадров, сдвинутых)"}
    lines = ["| карты | " + " | ".join(s.replace("LT1_", "") for s in sids) + " |", "|---" * (len(sids) + 1) + "|"]
    for v, nm in names.items():
        cells = []
        for s in sids:
            x = num.get(f"ft_{s}_{v}")
            hv = num.get(f"ft_{s}_halo", {}).get(v)
            cells.append("—" if not x else f"{_fmt(x['batch'][0])} / {_fmt(x['batch'][1])} ({x['batch'][2]}); у автора "
                                           f"FDR {_fmt(x['author_fdr'])}, затравок {x['author_n']}; маска "
                                           f"{_fmt(hv[1]) if hv else '—'}")
        lines.append(f"| {nm} | " + " | ".join(cells) + " |")
    lines.append("| сверка «все кадры» с картами батча, макс. разница corr / pnr | " + " | ".join(
        f"{_fmt(num[f'ft_{s}_check']['cn'], 4)} / {_fmt(num[f'ft_{s}_check']['pnr'], 4)}" for s in sids) + " |")
    lines.append("| кадров в группах: чёрные / первые 15 с / яркие / тёмные / контроль | " + " | ".join(
        " / ".join(str(num[f"ft_{s}_sets"][k]) for k in ("black", "start", "bright", "dim", "ctrl")) for s in sids) + " |")
    timing = []
    for s in sids:
        c = num.get(f"ft_{s}_halo_coincidence")
        nf = num.get(f"ft_{s}_null_frames")
        ps = num.get(f"ft_{s}_peak_sets")
        if c:
            timing.append(f"- {s.replace('LT1_', '')}: {_fmt(c[0], 2)} против {_fmt(c[1], 2)} случайно; {nf[0]} ложных "
                          f"затравок, минимумы в {nf[1]} разных кадрах, не больше {nf[2]} в одном кадре; в 2 % ярких "
                          f"кадров доля ложных минимумов {_fmt(ps['bright'][1])} (доля кадров {_fmt(ps['bright'][0])}), "
                          f"в первых 15 с — {_fmt(ps['start'][1])} (доля кадров {_fmt(ps['start'][0])}).")
    unexpl = "; ".join(f"{s.replace('LT1_', '')} {_fmt(num[f'ft_{s}_all']['author_fdr'])} -> "
                       f"{_fmt(num[f'ft_{s}_fdr_unexplained'])}" for s in sids if f"ft_{s}_fdr_unexplained" in num)
    out = (out.replace("FRAMES_TABLE", "\n".join(lines)).replace("FRAMES_TIMING", "\n".join(timing))
           .replace("FRAMES_UNEXPLAINED", unexpl))
    (HERE / "auto-vs-author.md").write_text(out, encoding="utf-8")
    left = re.findall(r"\{\{|\}\}", out)
    print(f"[{'OK' if not left else 'FAILED'}] auto-vs-author.md rendered" + (f", {len(left)} tokens left" if left else ""))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "render_md":
        cmd_render_md()
        raise SystemExit(0)
    if cmd == "traces":
        cmd_traces(args)
    elif cmd == "frames":
        cmd_frames(args)
    elif cmd == "widen":
        cmd_widen(args)
    elif cmd == "check_widen":
        cmd_check_widen(args)
    elif cmd == "report":
        cmd_report()
    else:
        raise SystemExit(f"unknown command {cmd}")
