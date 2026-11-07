"""thresholds.run on a real series_short clip, serial (n_cpus 2) and parallel (machine n_cpus), on scratch roots
whose 3_MC is a junction to series_short\\3_MC. Writes only into the scratch roots and the machine temp folder.

usage: maps_par.py SID
"""
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
import numpy as np

from bearmind import config
from bearmind.paths import SeriesPaths
from bearmind.registry import load_record
from bearmind.stages import thresholds

SERIES = SeriesPaths(Path(r"H:\Projects\_bearmind_test\series_short"))
HERE = Path(__file__).resolve().parent


def main():
    sid = sys.argv[1]
    rec = load_record(SERIES, sid)
    base = config.resolve(SERIES, sid)
    out = {}
    for name, n_cpus in (("serial", 2), ("parallel", int(base["machine"]["n_cpus"]))):
        root = SeriesPaths(HERE / f"maps_{name}")
        assert (root.root / "3_MC").is_dir(), "make the 3_MC junction first"
        cfg = copy.deepcopy(base)
        cfg["machine"]["n_cpus"] = n_cpus
        t0 = time.time()
        res = thresholds.run(root, rec, cfg)
        dt = time.time() - t0
        out[name] = dict(seconds=round(dt, 1), map_workers=res.metrics["map_workers"],
                         suggested=res.metrics["suggested"], npz=str(root.threshold_maps(sid)))
        print(f"[OK] {name}: {dt:.1f} s, map_workers {res.metrics['map_workers']}, suggested {res.metrics['suggested']}",
              flush=True)
    a, b = np.load(out["serial"]["npz"]), np.load(out["parallel"]["npz"])
    same = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    out["arrays"] = sorted(a.files)
    out["all_arrays_equal"] = bool(same)
    out["suggested_equal"] = out["serial"]["suggested"] == out["parallel"]["suggested"]
    print(("[OK]" if same else "[FAILED]") + f" arrays equal: {same}; {len(a.files)} arrays")
    (HERE / f"maps_par_{sid}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
