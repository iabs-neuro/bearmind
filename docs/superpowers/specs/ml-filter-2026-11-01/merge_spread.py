"""Side check: spatial spread of autoinspection merge groups (pairwise FBD among sources). Read-only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, sid = Path(sys.argv[1]), sys.argv[2]
insp = root / "5_Inspection" / sid
dec = pd.read_csv(insp / f"{sid}_metrics_with_decisions.csv")
FBD = np.load(insp / f"{sid}_FBD.npy")
FCD = np.load(insp / f"{sid}_FCD.npy")
pos = {int(c): i for i, c in enumerate(dec["component_idx"])}
for gid, g in dec[dec["merge"] > 0].groupby("merge"):
    idx = [int(c) for c in g["component_idx"]]
    if len(idx) < 2:
        continue
    p = [pos[c] for c in idx]
    sub_b = FBD[np.ix_(p, p)]
    sub_c = FCD[np.ix_(p, p)]
    off = ~np.eye(len(p), dtype=bool)
    # connected via FBD<=15 edges?
    adj = (sub_b <= 15) & off
    seen, stack = {0}, [0]
    while stack:
        k = stack.pop()
        for j in np.nonzero(adj[k])[0]:
            if j not in seen:
                seen.add(int(j)); stack.append(int(j))
    print(f"group {gid}: n={len(idx)} max FCD {sub_c[off].max():.1f} px, max FBD {sub_b[off].max():.1f}, "
          f"connected by FBD<=15: {len(seen) == len(p)}")
