"""Corpus scan behind inspection.wave.warn_frac (phase 5, task 6, ruling P5-2).

The sessions the author considers normal are the sessions of the May 2026 paper corpus that were kept after
inspection: every session with an inspection_artifacts_<SID> folder (autoinspected and exported in May 2026) in
CC\\7_output, RFC\\4_Estimates_inspections and FOF\\4_Estimates_inspections under ROOT.

Which estimates per session (column kind):
  manual  the manual-inspection estimates that the production autoinspection read: in 4_Estimates_refined, the
          file whose save date the session's *_processed.pickle name carries; idx_components as the person
          left them;
  cnmf    when that file is missing: the CNMF-E estimates 4_Estimates/<SID>_estimates.pickle with their
          idx_components.
The CNMF-E estimates are measured for every session as well (columns cnmf_*): they are what the inspection page
shows, before the person's choice.

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
from bearmind.wave import wave_frac  # noqa: E402

DATASETS = {"CC": "7_output", "RFC": "4_Estimates_inspections", "FOF": "4_Estimates_inspections"}
PROCESSED = re.compile(r"^(?P<sid>.+?)_(?P<date>\d\d-\d\d-\d{4} \d\d-\d\d-\d\d)_processed\.pickle$")
FIELDS = ["dataset", "sid", "kind", "file", "n_frames", "n_kept", "n_mad0", "wave_frac", "seconds",
          "cnmf_file", "cnmf_n_kept", "cnmf_wave_frac", "error"]
READABLE = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)


def sessions(root: Path):
    """(dataset, sid, manual estimates or None, CNMF-E estimates) of every inspected session."""
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
            yield ds, sid, manual, root / ds / "4_Estimates" / f"{sid}_estimates.pickle"


def measure(path: Path, k_mad: float, active_frac: float) -> dict:
    with open(path, "rb") as fh:  # read-only
        est = pickle.load(fh)
    C, idx = est.C, np.asarray(est.idx_components, dtype=np.intp).ravel()
    del est
    t0 = time.perf_counter()
    frac = wave_frac(C, idx, k_mad=k_mad, active_frac=active_frac)
    seconds = time.perf_counter() - t0
    X = np.asarray(C[idx], dtype=np.float64)
    med = np.median(X, axis=1, keepdims=True)
    # MAD 0: the trace sits at one value in more than half of its frames; it is active in every frame above it
    n_mad0 = int(np.count_nonzero(np.median(np.abs(X - med), axis=1) == 0))
    out = dict(n_frames=int(C.shape[1]), n_kept=int(idx.size), n_mad0=n_mad0, wave_frac=frac, seconds=seconds)
    del C, X
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
        for ds, sid, manual, cnmf in sessions(root):
            row = dict(dataset=ds, sid=sid, error="")
            t0 = time.perf_counter()
            try:
                c = measure(cnmf, k_mad, active_frac) if cnmf.is_file() else None
                if c is not None:
                    row.update(cnmf_file=cnmf.name, cnmf_n_kept=c["n_kept"], cnmf_wave_frac=c["wave_frac"])
                if manual is not None:
                    row.update(kind="manual", file=manual.name, **measure(manual, k_mad, active_frac))
                elif c is not None:
                    row.update(kind="cnmf", file=cnmf.name, **c)
                else:
                    row.update(error="no estimates file")
            except Exception as e:  # one unreadable session must not end the scan; the CSV names it
                row["error"] = f"{type(e).__name__}: {e}"
            wr.writerow(row)
            fh.flush()
            rows.append(row)
            frac = row.get("wave_frac")
            print(f"{ds} {sid} {row.get('kind', '-')} wave_frac={frac if frac is not None else '-'} "
                  f"cnmf={row.get('cnmf_wave_frac', '-')} {time.perf_counter() - t0:.1f}s {row['error']}"
                  .encode("ascii", "backslashreplace").decode("ascii"), flush=True)
    good = [r for r in rows if r.get("wave_frac") is not None and not r["error"]]
    for kind in ("manual", "cnmf"):
        print(f"kind {kind}: {sum(r['kind'] == kind for r in good)} sessions")
    if good:
        chosen = [r["wave_frac"] for r in good]
        summary(chosen, "chosen kind per session")
        cn = [r["cnmf_wave_frac"] for r in rows if r.get("cnmf_wave_frac") is not None]
        if cn:
            summary(cn, "CNMF-E estimates (what the page shows)")
        warn = readable_above(float(np.percentile(chosen, 99)))
        above = [r["sid"] for r in good if r["wave_frac"] > warn]
        above_cn = [r["sid"] for r in rows if r.get("cnmf_wave_frac") is not None and r["cnmf_wave_frac"] > warn]
        print(f"above warn_frac {warn:g}: chosen kind {len(above)} {above}; CNMF-E {len(above_cn)} {above_cn}")
    print(f"[OK] {len(rows)} sessions, {len(good)} measured, CSV: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
