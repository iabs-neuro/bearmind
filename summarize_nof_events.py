"""Summaries and distribution figures for NOF per-event wavelet stats. ASCII-only.

Run:
  conda run -n bearmind python summarize_nof_events.py \
      --events nof_wavelet_events.csv \
      --out-fig figures/nof_event_distributions.png \
      --out-csv nof_event_summary.csv
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="nof_wavelet_events.csv")
    ap.add_argument("--out-fig", default="figures/nof_event_distributions.png")
    ap.add_argument("--out-csv", default="nof_event_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.events)
    print("[INFO] %d events, %d sessions" % (len(df), df["session_name"].nunique()))

    # Per-neuron aggregates (join to EBM dataset by session_name + component_idx).
    def _iqr(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(np.subtract(*np.percentile(x, [75, 25]))) if len(x) else np.nan

    def _median_pos(x):  # median over valid (>=0) t_off only
        x = np.asarray(x, dtype=float)
        x = x[x >= 0]
        return float(np.median(x)) if len(x) else -1.0

    g = df.groupby(["session_name", "component_idx"])
    summary = g.agg(
        n_events=("amplitude_rel", "size"),
        amp_rel_median=("amplitude_rel", "median"),
        amp_rel_iqr=("amplitude_rel", _iqr),
        t_rise_median=("t_rise_s", "median"),
        t_off_median=("t_off_s", _median_pos),
        snr_median=("event_snr", "median"),
        dur_median=("duration_s", "median"),
    ).reset_index()
    summary.to_csv(args.out_csv, index=False)
    print("[OK] wrote per-neuron summary -> %s" % args.out_csv)

    out_dir = os.path.dirname(os.path.abspath(args.out_fig))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fields = [
        ("t_rise_s", "t_rise (s)", False),
        ("t_off_s", "t_off (s)", True),          # drop -1 sentinels
        ("amplitude_rel", "amplitude (dF/F0)", False),
        ("event_snr", "event SNR (proxy)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, label, drop_neg) in zip(axes.ravel(), fields):
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if drop_neg:
            vals = vals[vals >= 0]
        ax.hist(vals, bins=60)
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("count")
    fig.suptitle("NOF wavelet per-event distributions (n=%d events)" % len(df))
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=130)
    print("[OK] wrote figure -> %s" % args.out_fig)


if __name__ == "__main__":
    main()
