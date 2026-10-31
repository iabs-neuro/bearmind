"""Old production exports (May 2026 MINI3 run, copied under H:\\Dataset\\CalciumData\\<DS>\\7_output):
compare the exported set with the autoinspection delete column. Reads only csv/json, never the pickles.

For each inspection_artifacts_<SID> folder with <SID>_metadata.json and <SID>_metrics_with_decisions.csv:
  exported rows (metadata metrics_df) by decision/delete;
  from the pre-transform decision csv: delete=0 count, merge groups actually performed (>=2 members with
  delete=0), their sources, SNR-loser deletes (delete=1, P>=thr);
  predicted delete==0 export count = delete0 - sources + groups; predicted old-rule count = P>=thr - sources.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

base = Path(sys.argv[1])
rows = []
for ds in ("CC", "RFC", "FOF"):
    for art in sorted(list((base / ds / "7_output").glob("inspection_artifacts_*")) + list((base / ds / "4_Estimates_inspections").glob("inspection_artifacts_*"))):
        sid = art.name.replace("inspection_artifacts_", "")
        mj, mc = art / f"{sid}_metadata.json", art / f"{sid}_metrics_with_decisions.csv"
        if not (mj.exists() and mc.exists()):
            continue
        m = json.load(open(mj))
        st, cfg = m["autoinspection_stats"], m["autoinspection_config"]
        thr = cfg.get("ml_threshold", 0.72)
        md = pd.DataFrame(m["metrics_df"])
        dec = pd.read_csv(mc)
        P = pd.to_numeric(dec["ml_keep_probability"], errors="coerce")
        d0 = dec["delete"] == 0
        groups = [g for _, g in dec[(dec["merge"] > 0) & d0].groupby("merge") if len(g) >= 2]
        n_src = sum(len(g) for g in groups)
        src_P_ok = all((pd.to_numeric(g["ml_keep_probability"], errors="coerce") >= thr).all() for g in groups)
        losers = int(((dec["delete"] == 1) & (P >= thr)).sum())
        exp_P = pd.to_numeric(md["ml_keep_probability"], errors="coerce")
        rows.append(dict(
            ds=ds, sid=sid, fb=m["feedback_applied"], n_init=len(dec), n_proc=st["n_before_ml_filter"],
            exported=st["n_after_ml_filter"], exp_rows=len(md),
            exp_del1=int((md["delete"] == 1).sum()), exp_from_merge=int((md["decision"] == "from_merge").sum()),
            exp_P_nan=int(exp_P.isna().sum()), exp_P_lt=int((exp_P < thr).sum()),
            delete0=int(d0.sum()), groups=len(groups), sources=n_src, src_all_P_ok=src_P_ok, losers=losers,
            pred_old=int((P >= thr).sum()) - n_src,
            pred_del0=int(d0.sum()) - n_src + len(groups),
        ))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.to_string(index=False))
print()
print("sessions:", len(df), "| feedback applied:", int(df["fb"].sum()))
print("old rule prediction == actual exported:", int((df["pred_old"] == df["exported"]).sum()), "/", len(df))
print("exported rows with delete=1 (SNR losers):", int(df["exp_del1"].sum()),
      "| == losers in csv:", int((df["exp_del1"] == df["losers"]).sum()), "/", len(df))
print("exported from_merge rows:", int(df["exp_from_merge"].sum()), "| exported NaN P:", int(df["exp_P_nan"].sum()),
      "| exported P<thr:", int(df["exp_P_lt"].sum()))
print("merge groups performed (= merge products dropped by old rule):", int(df["groups"].sum()),
      "| their sources:", int(df["sources"].sum()), "| all sources P>=thr:", bool(df["src_all_P_ok"].all()))
for ds, g in df.groupby("ds"):
    print(f"{ds}: n={len(g)} exported(old) {int(g['exported'].sum())}  delete==0 rule {int(g['pred_del0'].sum())}  "
          f"losers leaked {int(g['exp_del1'].sum())}  merge products dropped {int(g['groups'].sum())} "
          f"(from {int(g['sources'].sum())} sources)")
print(f"ALL: exported(old) {int(df['exported'].sum())}  delete==0 rule {int(df['pred_del0'].sum())}  "
      f"losers leaked {int(df['exp_del1'].sum())} ({df['exp_del1'].sum() / df['exported'].sum():.1%} of exported)  "
      f"merge products dropped {int(df['groups'].sum())}")
