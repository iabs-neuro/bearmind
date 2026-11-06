import torch  # noqa: F401
"""Re-evaluate merge grouping on saved autoinspection sessions: old groups (saved 'merge') against
the neighbourhood groups of auto_inspector.assign_merge_groups. Only regroups from the saved csv,
FBD and FCD; autoinspection is not run. Read-only on the data; writes merge_regroup_out.txt next to itself.

usage: merge_regroup.py <corpus root> <series_short root>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from auto_inspector import assign_merge_groups  # noqa: E402

DEFAULTS = dict(ml_threshold=0.72, pxlthr_distance_boundary=15, d_snr_thr=10)
LINES = []


def say(text):
    print(text)
    LINES.append(text)


def thresholds(art, sid):
    th = dict(DEFAULTS)
    meta = art / f"{sid}_metadata.json"
    if meta.exists():
        cfg = json.loads(meta.read_text(encoding="utf-8")).get("autoinspection_config", {})
        for k in th:
            if cfg.get(k) is not None:
                th[k] = cfg[k]
    return th


def group_stats(dec, merge, FBD, FCD, pxl):
    """Groups = delete==0 rows sharing one merge id, size >= 2 (as merge_spread_corpus.groups_of)."""
    keep = dec["delete"].to_numpy() == 0
    out = []
    for gid in np.unique(merge[keep & (merge > 0)]):
        p = np.nonzero(keep & (merge == gid))[0]
        if len(p) < 2:
            continue
        adj = (FBD[np.ix_(p, p)] <= pxl) & ~np.eye(len(p), dtype=bool)
        seen, stack = {0}, [0]
        while stack:
            k = stack.pop()
            for j in np.nonzero(adj[k])[0]:
                if int(j) not in seen:
                    seen.add(int(j))
                    stack.append(int(j))
        sub = FCD[np.ix_(p, p)][~np.eye(len(p), dtype=bool)]
        out.append(dict(n=len(p), connected=len(seen) == len(p), max_fcd=float(sub.max())))
    return out


def session(art, sid):
    dec = pd.read_csv(art / f"{sid}_metrics_with_decisions.csv")
    FBD = np.load(art / f"{sid}_FBD.npy")
    FCD = np.load(art / f"{sid}_FCD.npy")
    th = thresholds(art, sid)
    p = pd.to_numeric(dec["ml_keep_probability"], errors="coerce")
    failed = [c for c in dec.columns if c.startswith("failed_")]
    brain = (p < th["ml_threshold"]) | (dec[failed].eq(1).any(axis=1) if failed else False)
    inp = dec[["component_idx", "corr_groups", "caiman_snr"]].copy()
    inp["delete"] = brain.astype(int).to_numpy()
    new = assign_merge_groups(inp, FBD, th["pxlthr_distance_boundary"], th["d_snr_thr"])
    same = new["delete"].to_numpy().tolist() == dec["delete"].tolist()
    nbad = int((new["delete"].to_numpy() != dec["delete"].to_numpy()).sum())
    pxl = th["pxlthr_distance_boundary"]
    before = group_stats(dec, dec["merge"].to_numpy(), FBD, FCD, pxl)
    after = group_stats(dec.assign(delete=new["delete"].to_numpy()), new["merge"].to_numpy(), FBD, FCD, pxl)
    return same, nbad, before, after


def sessions(corpus, series):
    for ds in ("CC", "RFC", "FOF"):
        arts = list((corpus / ds / "7_output").glob("inspection_artifacts_*")) + \
               list((corpus / ds / "4_Estimates_inspections").glob("inspection_artifacts_*"))
        for art in sorted(arts):
            yield ds, art, art.name.replace("inspection_artifacts_", "")
    for art in sorted((series / "5_Inspection").glob("*")):
        if art.is_dir():
            yield "series_short", art, art.name


def summary(name, nsess, nsame, before, after):
    def fmt(rows):
        n = np.array([r["n"] for r in rows]) if rows else np.array([0])
        mf = np.array([r["max_fcd"] for r in rows]) if rows else np.array([0.0])
        return dict(groups=len(rows), members=int(n.sum()), biggest=int(n.max()),
                    unconn=sum(not r["connected"] for r in rows), maxfcd=float(mf.max()),
                    g30=int((mf > 30).sum()) if rows else 0, g100=int((mf > 100).sum()) if rows else 0)
    b, a = fmt(before), fmt(after)
    say(f"{name}: sessions {nsess}, delete identical {nsame}, groups {b['groups']} -> {a['groups']}, "
        f"members in groups {b['members']} -> {a['members']}, biggest group {b['biggest']} -> {a['biggest']}, "
        f"not FBD-connected {b['unconn']} -> {a['unconn']}, max centre distance {b['maxfcd']:.0f} -> {a['maxfcd']:.0f} px, "
        f"> 30 px {b['g30']} -> {a['g30']}, > 100 px {b['g100']} -> {a['g100']}")


def main():
    corpus, series = Path(sys.argv[1]), Path(sys.argv[2])
    acc = {}
    for ds, art, sid in sessions(corpus, series):
        missing = [f for f, ok in [("metrics_with_decisions.csv", (art / f"{sid}_metrics_with_decisions.csv").exists()),
                                   ("FBD.npy", (art / f"{sid}_FBD.npy").exists()),
                                   ("FCD.npy", (art / f"{sid}_FCD.npy").exists())] if not ok]
        if missing:
            say(f"[SKIP] {art}: no {', '.join(missing)}")
            continue
        same, nbad, before, after = session(art, sid)
        say(f"[OK] {ds} {sid} delete identical" if same else f"[FAILED] {ds} {sid} delete differs: {nbad} rows differ")
        d = acc.setdefault(ds, dict(n=0, same=0, b=[], a=[]))
        d["n"] += 1; d["same"] += int(same); d["b"] += before; d["a"] += after
    for ds in ("CC", "FOF", "RFC"):
        if ds in acc:
            summary(ds, acc[ds]["n"], acc[ds]["same"], acc[ds]["b"], acc[ds]["a"])
    corp = [acc[k] for k in ("CC", "FOF", "RFC") if k in acc]
    summary("ALL", sum(d["n"] for d in corp), sum(d["same"] for d in corp),
            sum((d["b"] for d in corp), []), sum((d["a"] for d in corp), []))
    if "series_short" in acc:
        d = acc["series_short"]
        summary("series_short", d["n"], d["same"], d["b"], d["a"])
    (Path(__file__).parent / "merge_regroup_out.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


main()
