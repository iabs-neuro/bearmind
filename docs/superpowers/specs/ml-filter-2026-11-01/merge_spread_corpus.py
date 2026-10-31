"""Side check over old production corpus + new short series: are autoinspection merge groups spatially
coherent? A group is the set of delete=0 rows sharing one 'merge' id (= one correlation cluster id).
Reports groups whose members are not connected by FBD<=pxl edges, and max centre distance. Read-only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def groups_of(art, sid, pxl):
    dec = pd.read_csv(art / f"{sid}_metrics_with_decisions.csv")
    FBD = np.load(art / f"{sid}_FBD.npy")
    FCD = np.load(art / f"{sid}_FCD.npy")
    pos = {int(c): i for i, c in enumerate(dec["component_idx"])}
    out = []
    for gid, g in dec[(dec["merge"] > 0) & (dec["delete"] == 0)].groupby("merge"):
        idx = [int(c) for c in g["component_idx"]]
        if len(idx) < 2:
            continue
        p = [pos[c] for c in idx]
        b, c = FBD[np.ix_(p, p)], FCD[np.ix_(p, p)]
        off = ~np.eye(len(p), dtype=bool)
        adj = (b <= pxl) & off
        seen, stack = {0}, [0]
        while stack:
            k = stack.pop()
            for j in np.nonzero(adj[k])[0]:
                if int(j) not in seen:
                    seen.add(int(j)); stack.append(int(j))
        out.append(dict(n=len(p), connected=len(seen) == len(p), max_fcd=float(c[off].max())))
    return out


base = Path(sys.argv[1])
allrows = []
for ds in ("CC", "RFC", "FOF"):
    arts = list((base / ds / "7_output").glob("inspection_artifacts_*")) + \
           list((base / ds / "4_Estimates_inspections").glob("inspection_artifacts_*"))
    for art in sorted(arts):
        sid = art.name.replace("inspection_artifacts_", "")
        if not (art / f"{sid}_FCD.npy").exists():
            continue
        for r in groups_of(art, sid, 15):
            allrows.append(dict(ds=ds, sid=sid, **r))
df = pd.DataFrame(allrows)
for ds, g in df.groupby("ds"):
    print(f"{ds}: groups {len(g)}, members median {g['n'].median():.0f} max {g['n'].max()}, "
          f"not FBD-connected {int((~g['connected']).sum())} ({(~g['connected']).mean():.0%}), "
          f"max centre distance > 30 px: {int((g['max_fcd'] > 30).sum())}, > 100 px: {int((g['max_fcd'] > 100).sum())}")
print(f"ALL: groups {len(df)}, not connected {int((~df['connected']).sum())} ({(~df['connected']).mean():.0%}), "
      f"members in non-connected groups {int(df.loc[~df['connected'], 'n'].sum())} of {int(df['n'].sum())}")
