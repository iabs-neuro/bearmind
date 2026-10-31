"""Classify rows of a processed pickle's metrics_df: why delete/keep, why NaN P, what the export rule picks.

Usage: python analyze.py <series_root> <SID>   (read-only)
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

root, sid = Path(sys.argv[1]), sys.argv[2]
insp = root / "5_Inspection" / sid
est = pickle.load(open(insp / f"{sid}_processed.pickle", "rb"))
raw = pickle.load(open(root / "4_Estimates" / f"{sid}_estimates.pickle", "rb"))
dec = pd.read_csv(insp / f"{sid}_metrics_with_decisions.csv")  # decision_df BEFORE transform (old indices)
FBD = np.load(insp / f"{sid}_FBD.npy")
cfg = est.autoinspection_config
thr = cfg["ml_threshold"]
print(f"=== {sid}  brain={cfg['brain']} thr={thr} enable_merge={cfg['enable_merge']} "
      f"pxl={cfg['pxlthr_distance_boundary']} dsnr={cfg['d_snr_thr']} corner={cfg['detect_corner_artifacts']}")

# --- raw CNMF estimates
print(f"raw: C rows {raw.C.shape[0]}, idx_components {len(raw.idx_components)}, "
      f"idx_components_bad {len(raw.idx_components_bad)}")
# --- processed estimates
bad = getattr(est, "idx_components_bad", [])
print(f"processed: A {est.A.shape}, C rows {est.C.shape[0]}, idx_components {len(est.idx_components)} "
      f"(min {min(est.idx_components)}, max {max(est.idx_components)}), idx_components_bad {len(bad)}, "
      f"idx_components_bad_original {len(getattr(est, 'idx_components_bad_original', []))}")

# --- decision_df (pre-transform): merges
print(f"decision_df rows {len(dec)}; delete=1 {int((dec['delete'] == 1).sum())}; "
      f"merge>0 {int((dec['merge'] > 0).sum())}")
groups = dec[dec["merge"] > 0].groupby("merge")["component_idx"].apply(list)
valid = [g for g in groups if len(g) >= 2]
n_src = sum(len(g) for g in valid)
print(f"merge groups with >=2 members: {len(valid)}, sources {n_src}, net reduction {n_src - len(valid)} "
      f"(expected {len(raw.idx_components)} - {len(est.idx_components)} = "
      f"{len(raw.idx_components) - len(est.idx_components)})")
singletons = [g for g in groups if len(g) < 2]
print(f"merge groups with 1 member (merge col set, no merge performed): {len(singletons)}")

df = est.metrics_df.copy()
P = df["ml_keep_probability"]
print(f"metrics_df rows {len(df)}, component_idx {df['component_idx'].min()}..{df['component_idx'].max()}, "
      f"unique {df['component_idx'].nunique()}, == idx_components: "
      f"{set(df['component_idx']) == set(est.idx_components)}")
print("decision value counts:", df["decision"].value_counts(dropna=False).to_dict())

failed_cols = [c for c in df.columns if c.startswith("failed_") and c != "failed_corner_artifact"]
any_fail = (df[failed_cols].fillna(0) > 0).any(axis=1)
d1 = df["delete"] == 1
fm = df["decision"] == "from_merge"
corner = df.get("is_corner_artifact", pd.Series(0, index=df.index)) == 1

cats = pd.Series("?", index=df.index)
cats[~d1 & P.notna() & (P >= thr)] = "keep: P>=thr"
cats[~d1 & fm] = "keep: merge product (from_merge), P=NaN"
cats[d1 & corner] = "delete: corner artifact"
cats[d1 & ~corner & any_fail & P.isna()] = "delete: threshold rule, P=NaN"
cats[d1 & ~corner & ~any_fail & P.notna() & (P < thr)] = "delete: ML P<thr"
cats[d1 & ~corner & ~any_fail & P.notna() & (P >= thr)] = "delete: SNR-loser in merge cluster, P>=thr"
df["cat"] = cats
tab = df.groupby("cat").agg(n=("component_idx", "size"))
print(tab.to_string())

old_rule = set(df.loc[P >= thr, "component_idx"]) & set(est.idx_components)
del0 = set(df.loc[~d1, "component_idx"]) & set(est.idx_components)
print(f"OLD rule (P>=thr within idx_components): {len(old_rule)}")
print(f"delete==0 within idx_components: {len(del0)}")
print(f"  in old rule but delete=1: {len(old_rule - del0)} -> cats "
      f"{df[df['component_idx'].isin(old_rule - del0)]['cat'].value_counts().to_dict()}")
print(f"  delete=0 but not in old rule: {len(del0 - old_rule)} -> cats "
      f"{df[df['component_idx'].isin(del0 - old_rule)]['cat'].value_counts().to_dict()}")

# SNR-loser check against the pre-transform decision_df: each such row must have a cluster partner,
# kept, within pxl FBD, with SNR higher by > d_snr_thr
losers = dec[(dec["delete"] == 1) & dec["ml_keep_probability"].notna() & (dec["ml_keep_probability"] >= thr)]
fcols = [c for c in dec.columns if c.startswith("failed_") and c != "failed_corner_artifact"]
for _, r in losers.iterrows():
    i = int(r["component_idx"])
    pos = dec.index[dec["component_idx"] == i][0]
    same = dec[(dec["corr_groups"] == r["corr_groups"]) & (dec["component_idx"] != i)]
    part = []
    for _, q in same.iterrows():
        qpos = dec.index[dec["component_idx"] == q["component_idx"]][0]
        part.append((int(q["component_idx"]), round(float(FBD[pos, qpos]), 1), round(float(q["caiman_snr"]), 1),
                     int(q["delete"])))
    print(f"  SNR-loser old idx {i}: P={r['ml_keep_probability']:.3f} snr={r['caiman_snr']:.1f} "
          f"corr_group={r['corr_groups']} anyfail={int((r[fcols].fillna(0) > 0).any())} "
          f"partners(idx,FBD,snr,delete)={part}")

# merge products: P NaN, delete 0. ML was never applied to them (transform_metrics_df_indices sets NaN).
mp = df[fm]
print(f"merge products: {len(mp)}, P NaN {int(mp['ml_keep_probability'].isna().sum())}, delete=0 "
      f"{int((mp['delete'] == 0).sum())}")
# Sources of each merge: were they all P>=thr? (delete=0 is required to merge; any source with P<thr would
# have been delete=1 and excluded from the merge)
for g in valid:
    sub = dec[dec["component_idx"].isin(g)]
    print(f"  group sources {g}: P={[round(x, 3) for x in sub['ml_keep_probability']]} "
          f"delete={sub['delete'].tolist()}")
# rows with delete=0 and P<thr? (should be none) ; delete=0 & NaN not from_merge? (should be none)
print("delete=0 & P<thr:", int((~d1 & (P < thr)).sum()),
      "| delete=0 & NaN & not from_merge:", int((~d1 & P.isna() & ~fm).sum()))
# SNR of exported sets
snr = np.asarray(est.SNR_comp)
for name, s in [("old rule", old_rule), ("delete==0", del0)]:
    ix = np.array(sorted(s), dtype=int)
    print(f"median SNR_comp {name}: {np.median(snr[ix]):.3f} (n={len(ix)})")
