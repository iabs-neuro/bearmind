"""Per-session facts from the LT1 registry (read-only) plus the gap statistics recomputed from the timestamp files.

Output: dfr_sessions.json (list of dicts) next to this script. Console output ASCII only.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
from bearmind.paths import SeriesPaths  # noqa: E402
from bearmind.registry import read_all  # noqa: E402
from bearmind.timestamps import robust_fps, session_timestamps  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"H:\Projects\LT1\CalciumData")
OUT = Path(__file__).with_name(sys.argv[2] if len(sys.argv) > 2 else "dfr_sessions.json")
GAP_FACTOR = 2.0

paths = SeriesPaths(ROOT)
recs, broken = read_all(paths)
rows = []
for r in recs:
    row = {"sid": r.sid, "mouse": r.mouse, "mode": r.mode, "n_parts": len(r.parts), "n_frames": r.n_frames,
           "n_video_frames": sum(sum(p.chunk_frames) for p in r.parts), "fps": r.fps, "fps_info": r.fps_info,
           "excluded": r.excluded,
           "anomalies": [{"code": a.code, "level": a.level, "auto_fixed": a.auto_fixed, "message": a.message,
                          "data": a.data} for a in r.anomalies],
           "decisions": [{"what": d.what, "value": d.value} for d in r.decisions],
           "stages": {k: v.state.value for k, v in r.stages.items()},
           "mc_max_shift_px": (r.stages.get("mc").metrics or {}).get("max_shift_px") if r.stages.get("mc") else None,
           "mc_template_corr": (r.stages.get("mc").metrics or {}).get("mean_template_corr") if r.stages.get("mc") else None,
           "crop": (r.stages.get("crop").metrics or {}) if r.stages.get("crop") else None}
    try:
        ts = session_timestamps(paths, r)
        t = np.asarray(ts.t_ms, dtype=np.float64)
        d = np.diff(t)
        d = d[d > 0]
        m = float(np.median(d))
        gaps = d[d > GAP_FACTOR * m]
        est = robust_fps(t, GAP_FACTOR)
        row.update({
            "ts_rows": int(len(t)),
            "span_s": float((t[-1] - t[0]) / 1000.0),
            "mean_interval_ms": float(d.mean()),
            "median_interval_ms": m,
            "mean_interval_no_gaps_ms": float(d[d <= GAP_FACTOR * m].mean()),
            "n_gaps": int(len(gaps)),
            "gap_total_s": float(gaps.sum() / 1000.0),
            "gap_excess_s": float((gaps - m).sum() / 1000.0),
            "gap_frames_lost": int(np.maximum(np.rint(gaps / m) - 1, 1).sum()) if len(gaps) else 0,
            "max_gap_ms": float(d.max()),
            "fps_recomputed": est.fps,
            "fps_naive_span": float((len(t) - 1) / ((t[-1] - t[0]) / 1000.0)),
            "fps_mean_of_inverse": float(np.mean(1000.0 / d)),
            "n_outliers": est.n_outliers,
            "gaps_ms": [round(float(g), 1) for g in gaps[:50]],
        })
    except Exception as e:  # recorded, not hidden: the page says which sessions could not be read
        row["ts_error"] = f"{type(e).__name__}: {e}"
    rows.append(row)
    print(f"[OK] {r.sid} fps={r.fps} n={r.n_frames} gaps={row.get('n_gaps')}")

OUT.write_text(json.dumps({"root": str(ROOT), "broken": [(str(p), w) for p, w in broken], "rows": rows},
                          indent=1, default=float), encoding="utf-8")
print(f"[OK] wrote {OUT} ({len(rows)} sessions, {len(broken)} broken)")
