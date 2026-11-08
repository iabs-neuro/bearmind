"""Corpus scan behind inspection.wave.warn_frac (phase 5, task 6, rulings W1-W3 of fix round 1).

The sessions the author considers normal are the sessions of the May 2026 paper corpus that were kept after
inspection: every session with an inspection_artifacts_<SID> folder (autoinspected and exported in May 2026) in
CC\\7_output, RFC\\4_Estimates_inspections and FOF\\4_Estimates_inspections under ROOT.

The share is measured on what the inspection page shows: the CNMF-E estimates 4_Estimates/<SID>_estimates.pickle
with their idx_components (columns without a prefix). Side columns:
  nofloor_*  the same estimates with the rule before the noise floor (threshold median + k_mad x 1.4826 x MAD);
  manual_*   the manual-inspection estimates that the production autoinspection read (4_Estimates_refined, the
             file whose save date the session's *_processed.pickle name carries): the components the person kept.
Noise sources of the kept components (bearmind.wave.component_noise, which skips a value below 1e-6 x max|C|):
n_sn_caiman (estimates.neurons_sn), n_sn_resid (1.4826 x MAD of YrA), n_sn_diff (none: wave_frac's
first-difference floor). The server gives no hint below 10 kept components; no corpus session has so few.

wave_frac is bearmind.wave.wave_frac with k_mad and active_frac of bearmind/defaults.yaml (inspection.wave).
Reads only: every file is opened "rb" and nothing is written under ROOT. One session at a time, freed before
the next. The CSV is written row by row, so an interrupted scan keeps what it measured.

python docs/superpowers/specs/wave-frac-2026-11-08/scan_wave_frac.py [ROOT] [--out CSV]
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from bearmind import config  # noqa: E402
from bearmind.wave import component_noise, wave_frac  # noqa: E402

DATASETS = {"CC": "7_output", "RFC": "4_Estimates_inspections", "FOF": "4_Estimates_inspections"}
PROCESSED = re.compile(r"^(?P<sid>.+?)_(?P<date>\d\d-\d\d-\d{4} \d\d-\d\d-\d\d)_processed\.pickle$")
FIELDS = ["dataset", "sid", "file", "n_frames", "n_kept", "n_mad0", "n_sn_caiman", "n_sn_resid", "n_sn_diff",
          "wave_frac", "nofloor_wave_frac", "seconds", "manual_file", "manual_n_kept", "manual_wave_frac", "error"]
READABLE = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)


def sessions(root: Path):
    """(dataset, sid, CNMF-E estimates, manual estimates or None) of every inspected session."""
    for ds, insp in DATASETS.items():
        folder = root / ds / insp
        refined = next((root / ds / n for n in ("4_Estimates_refined", "4_Estimates_Refined")
                        if (root / ds / n).is_dir()), None)
        for d in sorted(p for p in folder.iterdir() if p.is_dir() and p.name.startswith("inspection_artifacts_")):
            sid = d.name[len("inspection_artifacts_"):]
            manual = None
            for p in sorted(d.glob("*_processed.pickle")):
                m = PROCESSED.match(p.name)
                if m and m.group("sid") == sid and refined is not None:
                    f = refined / f"{sid}{m.group('date')}_estimates.pickle"
                    manual = f if f.is_file() else None
            yield ds, sid, root / ds / "4_Estimates" / f"{sid}_estimates.pickle", manual


def _load(path: Path):
    with open(path, "rb") as fh:  # read-only
        return pickle.load(fh)


def measure(path: Path, k_mad: float, active_frac: float, side: bool) -> dict:
    est = _load(path)
    C, idx = est.C, np.asarray(est.idx_components, dtype=np.intp).ravel()
    t0 = time.perf_counter()
    sn = component_noise(est, idx)  # as the inspection server calls it
    frac = wave_frac(C, idx, k_mad=k_mad, active_frac=active_frac, sn=sn)
    seconds = time.perf_counter() - t0
    out = dict(n_frames=int(C.shape[1]), n_kept=int(idx.size), wave_frac=frac, seconds=seconds)
    if side:
        caiman = getattr(est, "neurons_sn", None)
        caiman = np.asarray(caiman, dtype=np.float64).ravel() if caiman is not None else np.array([])
        valid = (np.isfinite(sn) & (sn > 0))[idx]
        from_caiman = np.zeros(idx.size, dtype=bool)
        if caiman.size == C.shape[0]:
            from_caiman = valid & (sn[idx] == caiman[idx])  # component_noise kept CaImAn's value
        X = np.asarray(C[idx], dtype=np.float64)
        med = np.median(X, axis=1, keepdims=True)
        out.update(
            # MAD 0: the trace sits at one value in more than half of its frames
            n_mad0=int(np.count_nonzero(np.median(np.abs(X - med), axis=1) == 0)),
            n_sn_caiman=int(from_caiman.sum()), n_sn_resid=int((valid & ~from_caiman).sum()),
            n_sn_diff=int((~valid).sum()),
            # the rule without the floor: a noise SD too small to matter is a valid sn, so the floor is the MAD
            nofloor_wave_frac=wave_frac(C, idx, k_mad=k_mad, active_frac=active_frac,
                                        sn=np.full(C.shape[0], 1e-300)))
        del X
    del est, C
    gc.collect()
    return out


def readable_above(x: float) -> float:
    """The smallest value of READABLE x 10^k strictly above x."""
    if x <= 0:
        return 0.001
    k = math.floor(math.log10(x))
    for e in (k, k + 1):
        for m in READABLE:
            v = round(m * 10.0 ** e, 10)
            if v > x:
                return v
    raise AssertionError("unreachable")


def summary(values, label: str) -> None:
    v = np.asarray(values, dtype=float)
    p = np.percentile(v, [50, 95, 99])  # numpy's default (linear) interpolation
    print(f"{label}: n={v.size} min={v.min():.5f} median={p[0]:.5f} p95={p[1]:.5f} p99={p[2]:.5f} "
          f"max={v.max():.5f} readable_above_p99={readable_above(p[2]):g}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=r"H:\Dataset\CalciumData")
    ap.add_argument("--out", default=str(Path(__file__).with_name("wave_frac_corpus.csv")))
    a = ap.parse_args(argv)
    root, out = Path(a.root), Path(a.out)
    if out.resolve().is_relative_to(root.resolve()):
        print(f"[FAILED] the output must not be under {root}")
        return 2
    w = config.load_defaults()["inspection"]["wave"]
    k_mad, active_frac = float(w["k_mad"]), float(w["active_frac"])
    print(f"[OK] k_mad={k_mad} active_frac={active_frac} root={root}")
    rows = []
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, FIELDS)
        wr.writeheader()
        for ds, sid, cnmf, manual in sessions(root):
            row = dict(dataset=ds, sid=sid, error="")
            t0 = time.perf_counter()
            try:
                if cnmf.is_file():
                    row.update(file=cnmf.name, **measure(cnmf, k_mad, active_frac, side=True))
                else:
                    row["error"] = "no CNMF-E estimates"
                if manual is not None:
                    m = measure(manual, k_mad, active_frac, side=False)
                    row.update(manual_file=manual.name, manual_n_kept=m["n_kept"], manual_wave_frac=m["wave_frac"])
            except Exception as e:  # one unreadable session must not end the scan; the CSV names it
                row["error"] = f"{type(e).__name__}: {e}"
            wr.writerow(row)
            fh.flush()
            rows.append(row)
            print(f"{ds} {sid} wave_frac={row.get('wave_frac', '-')} nofloor={row.get('nofloor_wave_frac', '-')} "
                  f"manual={row.get('manual_wave_frac', '-')} {time.perf_counter() - t0:.1f}s {row['error']}"
                  .encode("ascii", "backslashreplace").decode("ascii"), flush=True)
    good = [r for r in rows if r.get("wave_frac") is not None and not r["error"]]
    if good:
        chosen = [r["wave_frac"] for r in good]
        summary(chosen, "CNMF-E estimates, noise floor (what the page shows)")
        summary([r["nofloor_wave_frac"] for r in good], "CNMF-E estimates, no floor")
        man = [r["manual_wave_frac"] for r in rows if r.get("manual_wave_frac") is not None]
        if man:
            summary(man, "manual-inspection estimates, noise floor")
        warn = readable_above(float(np.percentile(chosen, 99)))
        above = [r["sid"] for r in good if r["wave_frac"] > warn]
        above_m = [r["sid"] for r in rows if r.get("manual_wave_frac") is not None and r["manual_wave_frac"] > warn]
        print(f"above warn_frac {warn:g}: CNMF-E {len(above)} {above}; manual {len(above_m)} {above_m}")
        print("noise sources of kept components: caiman", sum(r["n_sn_caiman"] for r in good), "residual",
              sum(r["n_sn_resid"] for r in good), "first difference", sum(r["n_sn_diff"] for r in good))
    print(f"[OK] {len(rows)} sessions, {len(good)} measured, CSV: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
