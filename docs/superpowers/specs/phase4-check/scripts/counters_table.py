"""Read-only: the tab's counters (threshold_model.counters) at the suggested thresholds and at the thresholds of
core Task 17, for both sessions of series_short. Task 17 values are read from series_short's session configs
(the configs Task 17 wrote), the suggestion from series_short's registry, the maps from series_short.

usage: python counters_table.py OUT.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
import yaml  # noqa: E402

from bearmind import config, registry  # noqa: E402
from bearmind import threshold_model as tm  # noqa: E402
from bearmind.paths import SeriesPaths  # noqa: E402

SRC = SeriesPaths(Path(r"H:\Projects\_bearmind_test\series_short"))
out = {}
for sid in ("NOF_H01_1D", "NOF_H03_3D"):
    rec = registry.load_record(SRC, sid)
    sug = rec.stages["thresholds"].metrics["suggested"]
    t17 = yaml.safe_load(SRC.session_config(sid).read_text(encoding="utf-8"))["cnmf"]
    cfg = config.resolve(SRC, sid=sid, mouse=rec.mouse)
    blob = int(cfg["thresholds"]["min_blob_px"])
    maps = tm.load_maps(SRC.threshold_maps(sid))
    row = {"min_blob_px": blob, "usable_gsigs": list(maps.usable_gsigs), "map_shape": list(maps.cn[4].shape),
           "cnmf_metrics_series_short": {k: rec.stages["cnmf"].metrics.get(k)
                                         for k in ("n_components", "n_accepted", "gSig", "min_corr", "min_pnr")}}
    for name, v in (("suggested", sug), ("task17", t17)):
        g, c, p = int(v["gSig"]), float(v["min_corr"]), float(v["min_pnr"])
        k = tm.counters(maps, g, c, p, blob)
        strip = {g2: tm.counters(maps, g2, c, p, blob).n_blobs for g2 in maps.usable_gsigs}
        row[name] = {"gSig": g, "min_corr": c, "min_pnr": p, "n_blobs": k.n_blobs,
                     "single_pct": round(100 * k.single_frac, 1), "n_mirror_blobs": k.n_mirror_blobs,
                     "strip_blobs": strip}
    out[sid] = row
    print(sid, json.dumps(row))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
