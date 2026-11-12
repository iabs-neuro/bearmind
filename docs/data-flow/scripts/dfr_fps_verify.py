"""Where the frame rate of each session of the finished test root arrives (read-only).

For every session of H:/Projects/_bearmind_test/lt1_J03_J06: the registry fps, the MC stage params, the MC video
container rate (header only), the CNMF-E params in the registry and in the estimates pickle (cnmf_dict['fr']), the
autoinspect params / metrics, the processed pickle (autoinspection_config['fps'], est.time), the export json (fps,
autoinspection_fps, cnmf_params.fr, autoinspection_config.fps, provenance.fps), the npz fps, the exported timestamps.
Pickles are loaded one at a time and released. Output: dfr_fps_verify.json; console ASCII.
"""
import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
import av  # noqa: E402

from bearmind.paths import SeriesPaths  # noqa: E402
from bearmind.registry import read_all  # noqa: E402
from bearmind.timestamps import read_timestamps, robust_fps  # noqa: E402

ROOT = Path(r"H:\Projects\_bearmind_test\lt1_J03_J06")
paths = SeriesPaths(ROOT)
recs, _ = read_all(paths)
out = []


def container_rate(p):
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        return {"average_rate": str(s.average_rate), "average_rate_f": float(s.average_rate) if s.average_rate else None,
                "time_base": str(s.time_base), "frames": int(s.frames), "h": s.codec_context.height,
                "w": s.codec_context.width}


for r in recs:
    row = {"sid": r.sid, "registry_fps": r.fps, "n_frames": r.n_frames}
    st = r.stages
    row["mc_params_fps"] = st["mc"].params.get("fps")
    row["mc_video"] = container_rate(paths.mc_video(r.sid))
    row["cnmf_params_fr"] = st["cnmf"].params.get("fr")
    row["cnmf_metrics"] = st["cnmf"].metrics
    row["thresholds_metrics"] = st["thresholds"].metrics
    row["mc_metrics"] = st["mc"].metrics
    row["crop_metrics"] = st["crop"].metrics
    row["gate_metrics"] = st["gate"].metrics
    row["autoinspect_params_fps"] = st["autoinspect"].params.get("fps")
    row["autoinspect_metrics"] = st["autoinspect"].metrics
    row["export_metrics"] = st["export"].metrics
    row["checks"] = {k: [(c.name, c.level, c.value) for c in v.checks] for k, v in st.items() if v.checks}
    row["stage_anomalies"] = [(a.code, a.level, a.data.get("stage"), a.message) for a in r.anomalies]
    row["decisions"] = [(d.what, d.value, d.when) for d in r.decisions]

    with open(paths.estimates(r.sid), "rb") as f:
        est = pickle.load(f)
    cd = getattr(est, "cnmf_dict", None) or {}
    row["est_cnmf_dict_fr"] = cd.get("fr")
    row["est_A_shape"] = list(est.A.shape)
    row["est_C_shape"] = list(est.C.shape)
    row["est_n_idx_components"] = len(est.idx_components) if est.idx_components is not None else None
    row["est_patch_workers"] = getattr(est, "patch_workers", None)
    del est
    gc.collect()

    manual = paths.manual_result(r.sid)
    if manual.is_file():
        with open(manual, "rb") as f:
            m = pickle.load(f)
        row["manual_cnmf_dict_fr"] = (getattr(m, "cnmf_dict", None) or {}).get("fr")
        row["manual_n_idx_components"] = len(m.idx_components)
        row["manual_C_shape"] = list(m.C.shape)
        del m
        gc.collect()

    proc = ROOT / st["autoinspect"].outputs["processed"]
    with open(proc, "rb") as f:
        p = pickle.load(f)
    ac = getattr(p, "autoinspection_config", None) or {}
    row["proc_autoinspection_config_fps"] = ac.get("fps")
    t = getattr(p, "time", None)
    if t is not None:
        t = np.asarray(t, dtype=float)
        row["proc_time"] = {"len": int(len(t)), "first": float(t[0]), "last": float(t[-1]),
                            "step": float(t[1] - t[0]) if len(t) > 1 else None}
    row["proc_cnmf_dict_fr"] = (getattr(p, "cnmf_dict", None) or {}).get("fr")
    row["proc_C_shape"] = list(p.C.shape)
    del p
    gc.collect()

    meta = json.loads((paths.export_json(r.sid)).read_text(encoding="utf-8"))
    row["json_fps"] = meta.get("fps")
    row["json_autoinspection_fps"] = meta.get("autoinspection_fps")
    row["json_cnmf_params_fr"] = (meta.get("cnmf_params") or {}).get("fr")
    row["json_autoinspection_config_fps"] = (meta.get("autoinspection_config") or {}).get("fps")
    row["json_provenance_fps"] = (meta.get("provenance") or {}).get("fps")
    row["json_git"] = (meta.get("provenance") or {}).get("code")
    row["json_stats"] = meta.get("autoinspection_stats")
    with np.load(paths.export_npz(r.sid)) as z:
        row["npz_keys"] = {k: list(z[k].shape) for k in z.files}
        row["npz_fps"] = float(z["fps"])
    ts = read_timestamps(paths.export_timestamps(r.sid))
    est_ts = robust_fps(ts.t_ms)
    row["export_ts_rows"] = int(len(ts.raw))
    row["export_ts_fps"] = est_ts.fps
    row["export_ts_span_s"] = float((ts.t_ms[-1] - ts.t_ms[0]) / 1000.0)
    out.append(row)
    print(f"[OK] {r.sid}: registry {r.fps:.4f} mc {row['mc_params_fps']:.4f} video {row['mc_video']['average_rate']} "
          f"cnmf {row['cnmf_params_fr']:.4f} est {row['est_cnmf_dict_fr']} ai {row['autoinspect_params_fps']} "
          f"proc {row['proc_autoinspection_config_fps']} json {row['json_fps']} npz {row['npz_fps']} "
          f"ts {row['export_ts_fps']:.4f}")

Path(__file__).with_name("dfr_fps_verify.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print("[OK] written")
