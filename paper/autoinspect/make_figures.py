"""
Regenerate the two figures for the DCNA 2026 autoinspect paper.

Both figures are rebuilt from the NEWEST production model (EBM v9_iter8):
  - Figure 1: feature importance  -> figs/figure1_feature_importance.png
  - Figure 2: threshold analysis   -> figs/figure2_threshold_analysis.png

Design constraints (paper figures, captions live in LaTeX):
  * NO titles / suptitles baked into the PNGs.
  * Large axis labels, ticks and legends.
  * Figure 2 is recomputed on the model's own held-out test split over a wide
    threshold sweep (no stale/hard-coded operating-point lines).

Run from anywhere:
    python paper/autoinspect/make_figures.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------- paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
ML_DIR = REPO_ROOT / "ml"
FIG_DIR = SCRIPT_DIR / "figs"
FIG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ML_DIR))
from data_utils import get_feature_cols, FBETA_BETA  # noqa: E402

MODEL_PATH = ML_DIR / "ebm_v9_iter8" / "model.pkl"
DATASET_PATH = ML_DIR / "results" / "training_dataset_v9_corrected_iter7.csv"
IMPORTANCE_CSV = ML_DIR / "results" / "v9_iter8_feature_importance.csv"
SUMMARY_PATH = ML_DIR / "ebm_v9_iter8" / "summary.json"

RANDOM_SEED = 46          # matches summary.json for v9_iter8
TEST_SIZE = 0.25
OPERATING_POINT = 0.75    # deployed default threshold (summary.json)
N_TOP_FEATURES = 20

# ---------------------------------------------------------------- global style: BIG fonts
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "xtick.labelsize": 17,
    "ytick.labelsize": 16,
    "legend.fontsize": 17,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "figure.dpi": 110,
})


def _pretty(name: str) -> str:
    return name.replace("_", " ")


# ================================================================ Figure 1
def make_feature_importance():
    df = pd.read_csv(IMPORTANCE_CSV).sort_values("importance", ascending=False)
    df = df.head(N_TOP_FEATURES).iloc[::-1]  # ascending -> largest on top

    labels = [_pretty(f) for f in df["feature"]]
    vals = df["importance"].to_numpy()
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(vals)))

    fig, ax = plt.subplots(figsize=(9.5, 10))
    ax.barh(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean absolute score contribution")
    ax.margins(y=0.01)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(length=6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "figure1_feature_importance.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] wrote {out}  ({len(vals)} features)")


# ================================================================ Figure 2
def _reproduce_test_split():
    df = pd.read_csv(DATASET_PATH)
    session_col = "session_name" if "session_name" in df.columns else "session"
    feature_cols = get_feature_cols(df)

    sessions = df[session_col].unique()
    session_to_exp = df.groupby(session_col)["experiment"].first().to_dict()
    experiments = [session_to_exp[s] for s in sessions]

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    _, test_idx = next(splitter.split(sessions, experiments))
    test_sessions = set(sessions[test_idx])
    test_mask = df[session_col].isin(test_sessions)

    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, "ground_truth"].to_numpy()
    return X_test, y_test


def make_threshold_analysis():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    X_test, y_test = _reproduce_test_split()
    proba = model.predict_proba(X_test)[:, 1]
    pos = y_test == 1

    thr = np.arange(0.30, 0.951, 0.01)
    prec, rec, fbeta, fp, fn = [], [], [], [], []
    beta2 = FBETA_BETA ** 2
    for t in thr:
        pred = proba >= t
        tp = int(np.sum(pred & pos))
        fp_ = int(np.sum(pred & ~pos))
        fn_ = int(np.sum(~pred & pos))
        p = tp / (tp + fp_) if (tp + fp_) else 0.0
        r = tp / (tp + fn_) if (tp + fn_) else 0.0
        fb = ((1 + beta2) * p * r / (beta2 * p + r)) if (beta2 * p + r) else 0.0
        prec.append(p); rec.append(r); fbeta.append(fb); fp.append(fp_); fn.append(fn_)
    fbeta = np.array(fbeta); fp = np.array(fp); fn = np.array(fn)
    total = fp + fn
    best_t = float(thr[int(np.argmax(fbeta))])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 6.4))

    # --- left: F-beta vs threshold
    axL.plot(thr, fbeta, color="#1f4e9c", linewidth=3.0)
    axL.axvline(OPERATING_POINT, color="#2ca02c", linestyle="--", linewidth=2.2,
                label=f"operating point ({OPERATING_POINT:.2f})")
    axL.axvline(best_t, color="#d62728", linestyle=":", linewidth=2.2,
                label=f"optimal ({best_t:.2f})")
    axL.set_xlabel("Decision threshold")
    axL.set_ylabel(r"$F_\beta$ score ($\beta=0.577$)")
    axL.grid(alpha=0.3)
    axL.tick_params(length=6)
    axL.legend(frameon=False, loc="lower center")
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)

    # --- right: error counts vs threshold
    axR.plot(thr, fp, color="#d62728", linewidth=3.0, label="false positives")
    axR.plot(thr, fn, color="#ff7f0e", linewidth=3.0, label="false negatives")
    axR.plot(thr, total, color="black", linestyle="--", linewidth=2.4, label="total errors")
    axR.axvline(OPERATING_POINT, color="#2ca02c", linestyle="--", linewidth=2.0)
    axR.set_xlabel("Decision threshold")
    axR.set_ylabel("Misclassified neurons (count)")
    axR.grid(alpha=0.3)
    axR.tick_params(length=6)
    axR.legend(frameon=False, loc="upper center")
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / "figure2_threshold_analysis.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # sanity check against summary.json
    summary = json.loads(Path(SUMMARY_PATH).read_text())
    cm = summary["confusion_matrix"]
    i075 = int(np.argmin(np.abs(thr - OPERATING_POINT)))
    print(f"[fig2] wrote {out}")
    print(f"[fig2] check @0.75: fp={fp[i075]} (summary {cm['fp']}), "
          f"fn={fn[i075]} (summary {cm['fn']}); optimal Fbeta @ {best_t:.2f}")


if __name__ == "__main__":
    make_feature_importance()
    make_threshold_analysis()
    print("done.")
