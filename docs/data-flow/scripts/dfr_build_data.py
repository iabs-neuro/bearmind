"""Collect the page data (LT1 sessions, test-root verification) from the helper outputs into one JSON.

Inputs, next to this script: dfr_sessions.json, dfr_gapwindow.json, dfr_avirate.json, dfr_firstframe.json,
dfr_fps_verify.json. Output: dfr_page_data.json.
"""
import json
from pathlib import Path

here = Path(__file__).parent
S = json.loads((here / "dfr_sessions.json").read_text(encoding="utf-8"))["rows"]
GW = json.loads((here / "dfr_gapwindow.json").read_text(encoding="utf-8"))
AR = json.loads((here / "dfr_avirate.json").read_text(encoding="utf-8"))
FF = {o["sid"]: o for o in json.loads((here / "dfr_firstframe.json").read_text(encoding="utf-8"))}
V = json.loads((here / "dfr_fps_verify.json").read_text(encoding="utf-8"))

win = {}
for grp in ("30", "20"):
    for sid, gm, ex in GW["res"][grp]:
        win.setdefault(sid, 0)
        if ex >= 0.5:
            win[sid] += int(round(ex))

sessions = []
for r in S:
    ok_ts = "ts_error" not in r
    sessions.append({
        "sid": r["sid"], "mouse": r["mouse"], "fps": round(r["fps"], 4), "n": r["n_frames"],
        "nv": r["n_video_frames"], "dur": round(r["n_frames"] / r["fps"] / 60, 2),
        "mean": round(r["mean_interval_ms"], 3) if ok_ts else None,
        "med": round(r["median_interval_ms"], 3) if ok_ts else None,
        "gaps": r.get("n_gaps"), "gap_s": round(r.get("gap_total_s", 0), 3),
        "lost": r.get("gap_frames_lost"),

        "maxgap": round(r.get("max_gap_ms", 0), 1), "outl": r["fps_info"].get("n_outliers"),
        "span_fps": round(r["fps_naive_span"], 4) if r["sid"] not in GW["skipped"] else None,
        "avi": AR[r["sid"]]["avg"],
        "mc_shift": round(r["mc_max_shift_px"], 1) if r["mc_max_shift_px"] is not None else None,
        "mc_shift_rest": round(FF[r["sid"]]["max_rest"], 1),
        "f0": round(FF[r["sid"]]["m0"] / FF[r["sid"]]["m2"], 2) if FF[r["sid"]]["m2"] else None,
        "mc_corr": round(r["mc_template_corr"], 3) if r["mc_template_corr"] is not None else None,
        "crop": (r["crop"] or {}).get("fov_method"), "box": (r["crop"] or {}).get("box"),
        "anom": ", ".join(f"{a['code']}" + (" (auto)" if a["auto_fixed"] else "") for a in r["anomalies"]),
        "anom_text": "; ".join(a["message"] for a in r["anomalies"]),
    })

verify = []
for r in V:
    verify.append({
        "sid": r["sid"], "reg": r["registry_fps"], "mc": r["mc_params_fps"],
        "avi": r["mc_video"]["average_rate"], "avi_f": r["mc_video"]["average_rate_f"],
        "cnmf": r["cnmf_params_fr"], "est": r["est_cnmf_dict_fr"], "ai": r["autoinspect_params_fps"],
        "proc": r["proc_autoinspection_config_fps"], "proc_t_last": r["proc_time"]["last"],
        "proc_step": r["proc_time"]["step"], "ts_span": round(r["export_ts_span_s"], 2),
        "json": r["json_fps"], "json_ai": r["json_autoinspection_fps"], "json_cnmf": r["json_cnmf_params_fr"],
        "npz": r["npz_fps"], "ts_rows": r["export_ts_rows"], "n": r["n_frames"], "C_T": r["npz_keys"]["C"][1],
        "ts_fps": round(r["export_ts_fps"], 6),
        "thr": r["thresholds_metrics"]["suggested"], "crop": r["crop_metrics"]["fov_method"],
        "pos_neg": r["crop_metrics"].get("activity_pos_neg"), "box": r["crop_metrics"].get("box"),
        "mc_shift": round(r["mc_metrics"]["max_shift_px"], 1), "mc_corr": round(r["mc_metrics"]["mean_template_corr"], 3),
        "n_comp": r["cnmf_metrics"]["n_components"], "n_acc": r["cnmf_metrics"]["n_accepted"],
        "missed": r["cnmf_metrics"]["missed_cells"], "snr": round(r["cnmf_metrics"]["median_snr"], 2),
        "gate": r["gate_metrics"]["gate"], "ai_n0": r["autoinspect_metrics"]["n_initial"],
        "ai_merge": r["autoinspect_metrics"]["n_merge_groups"], "ai_after": r["autoinspect_metrics"]["n_after_merge"],
        "ai_del": r["autoinspect_metrics"]["n_deleted"], "ai_kept": r["autoinspect_metrics"]["n_kept"],
        "ai_frac": round(r["autoinspect_metrics"]["deleted_frac"], 3),
        "exp": r["export_metrics"]["n_neurons"], "exp_snr": round(r["export_metrics"]["median_snr"], 2),
        "checks": "; ".join(f"{s}.{n}={lv}" for s, cs in r["checks"].items() for n, lv, v in cs if lv != "ok"),
        "need_gib": r["cnmf_metrics"]["need_gib"], "ssub_B": r["cnmf_metrics"]["ssub_B"],
        "workers": r["cnmf_metrics"]["patch_workers"],
    })

(here / "dfr_page_data.json").write_text(json.dumps({"sessions": sessions, "verify": verify}, ensure_ascii=True),
                                         encoding="utf-8")
print("[OK] sessions", len(sessions), "verify", len(verify))

