"""Analyze model errors to understand weaknesses and propose improvements."""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Feature columns
feature_cols = [
    'area', 'circularity', 'max_edge', 'convexity', 'caiman_snr', 'caiman_r_score',
    'events_per_min', 'events_fraction', 't_rise', 't_off', 'wavelet_snr',
    'r2_score', 'event_r2_score', 'nmae', 'nrmse', 'snr_recon', 'noise_level',
    'baseline', 'tau_decay', 'trace_skewness', 'footprint_compactness',
    'trace_kurtosis', 'aspect_ratio', 'eccentricity', 'edge_distance', 'nn_distance_center'
]


def load_dataset(artifacts_dir, experiments=None):
    """Load dataset from capcan_artifacts directories."""
    artifacts_path = Path(artifacts_dir)
    session_dirs = sorted([d for d in artifacts_path.iterdir()
                          if d.is_dir() and d.name.startswith('capcan_artifacts_')])

    if experiments:
        filtered_dirs = []
        for d in session_dirs:
            session_name = d.name.replace('capcan_artifacts_', '')
            exp_id = session_name.split('_')[0]
            if exp_id in experiments:
                filtered_dirs.append(d)
        session_dirs = filtered_dirs

    all_features = []
    all_labels = []
    all_sessions = []

    for session_dir in session_dirs:
        try:
            df_raw = pd.read_csv(session_dir / "metrics_init.csv")
            df_gt = pd.read_csv(session_dir / "metrics_gt.csv")

            if df_raw['center'].dtype == 'object':
                df_raw['center'] = df_raw['center'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
            if df_gt['center'].dtype == 'object':
                df_gt['center'] = df_gt['center'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

            if 'is_corner_artifact' in df_raw.columns:
                df_raw = df_raw[df_raw['is_corner_artifact'] == 0].copy()

            raw_centers = np.array(df_raw['center'].tolist())
            gt_centers = np.array(df_gt['center'].tolist())

            labels = np.zeros(len(df_raw), dtype=int)
            for i, raw_center in enumerate(raw_centers):
                distances = np.linalg.norm(gt_centers - raw_center, axis=1)
                if distances.min() <= 3:
                    labels[i] = 1

            features = df_raw[feature_cols].copy()
            features = features.replace([np.inf, -np.inf], np.nan)

            session_name = session_dir.name.replace('capcan_artifacts_', '')
            sessions = [session_name] * len(features)

            all_features.append(features)
            all_labels.append(labels)
            all_sessions.extend(sessions)
        except Exception as e:
            continue

    features_df = pd.concat(all_features, ignore_index=True)
    labels = np.concatenate(all_labels)

    return features_df, labels, all_sessions


def analyze_errors(model_path, artifacts_dir, experiments, threshold=0.5):
    """Analyze model errors."""
    print("=" * 80)
    print("MODEL ERROR ANALYSIS")
    print("=" * 80)

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load data
    exp_list = experiments.split(',') if experiments else None
    X, y, sessions = load_dataset(artifacts_dir, exp_list)
    print(f"Loaded {len(X)} samples ({y.sum()} KEEP, {len(y) - y.sum()} DELETE)")

    # Get predictions
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Classification groups
    tp_mask = (y_pred == 1) & (y == 1)  # True positive: correctly kept
    fp_mask = (y_pred == 1) & (y == 0)  # False positive: wrongly kept (BAD!)
    fn_mask = (y_pred == 0) & (y == 1)  # False negative: wrongly deleted
    tn_mask = (y_pred == 0) & (y == 0)  # True negative: correctly deleted

    n_tp, n_fp, n_fn, n_tn = tp_mask.sum(), fp_mask.sum(), fn_mask.sum(), tn_mask.sum()
    print(f"\nConfusion Matrix (threshold={threshold}):")
    print(f"  TP (correct KEEP): {n_tp}")
    print(f"  FP (wrong KEEP):   {n_fp}  <-- FOCUS: precision errors")
    print(f"  FN (wrong DELETE): {n_fn}")
    print(f"  TN (correct DEL):  {n_tn}")

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}")

    # Analyze FALSE POSITIVES (precision errors)
    print("\n" + "=" * 80)
    print("FALSE POSITIVE ANALYSIS (model says KEEP but should DELETE)")
    print("=" * 80)

    X_fp = X[fp_mask].copy()
    X_tp = X[tp_mask].copy()
    X_tn = X[tn_mask].copy()

    print(f"\nFalse positives: {len(X_fp)} samples")
    print(f"True positives: {len(X_tp)} samples")
    print(f"True negatives (correctly deleted): {len(X_tn)} samples")

    # Compare FP vs TP feature distributions
    print("\n--- Feature Comparison: FP vs TP ---")
    print(f"{'Feature':<25} {'FP_mean':>10} {'TP_mean':>10} {'TN_mean':>10} {'FP-TP':>10} {'Separable?':>12}")
    print("-" * 80)

    problematic_features = []
    for col in feature_cols:
        fp_vals = X_fp[col].dropna()
        tp_vals = X_tp[col].dropna()
        tn_vals = X_tn[col].dropna()

        if len(fp_vals) > 10 and len(tp_vals) > 10:
            fp_mean = fp_vals.mean()
            tp_mean = tp_vals.mean()
            tn_mean = tn_vals.mean() if len(tn_vals) > 0 else np.nan
            diff = fp_mean - tp_mean

            # t-test to see if distributions are separable
            t_stat, p_val = stats.ttest_ind(fp_vals, tp_vals, equal_var=False)
            separable = "YES" if p_val < 0.001 and abs(t_stat) > 3 else "weak" if p_val < 0.05 else "NO"

            # Track problematic features (FP looks like TP)
            if separable == "NO":
                problematic_features.append(col)

            print(f"{col:<25} {fp_mean:>10.3f} {tp_mean:>10.3f} {tn_mean:>10.3f} {diff:>+10.3f} {separable:>12}")

    print("\n--- Features where FP looks like TP (not separable) ---")
    print(f"Problematic features: {problematic_features}")

    # Analyze FP probability distribution
    print("\n--- FP Confidence Distribution ---")
    fp_probs = y_proba[fp_mask]
    print(f"FP probabilities: min={fp_probs.min():.3f}, max={fp_probs.max():.3f}, mean={fp_probs.mean():.3f}")
    print(f"FP near threshold (0.5-0.6): {((fp_probs >= 0.5) & (fp_probs < 0.6)).sum()}")
    print(f"FP confident (>0.7): {(fp_probs >= 0.7).sum()}")
    print(f"FP very confident (>0.8): {(fp_probs >= 0.8).sum()}")

    # High-confidence FP analysis
    print("\n--- High-Confidence FP (prob > 0.7) ---")
    high_conf_fp_mask = fp_mask & (y_proba > 0.7)
    X_high_fp = X[high_conf_fp_mask]
    print(f"Count: {len(X_high_fp)}")

    if len(X_high_fp) > 5:
        print("\nMean features of high-confidence FP vs TP:")
        for col in ['trace_kurtosis', 'trace_skewness', 'r2_score', 'snr_recon', 'area', 'edge_distance']:
            fp_mean = X_high_fp[col].mean()
            tp_mean = X_tp[col].mean()
            print(f"  {col}: FP={fp_mean:.3f}, TP={tp_mean:.3f}, diff={fp_mean-tp_mean:+.3f}")

    # Session analysis
    print("\n--- FP by Session/Experiment ---")
    fp_sessions = [sessions[i] for i in range(len(sessions)) if fp_mask.iloc[i] if i < len(sessions)]
    if fp_sessions:
        session_counts = pd.Series(fp_sessions).value_counts()
        exp_counts = pd.Series([s.split('_')[0] for s in fp_sessions]).value_counts()
        print("FP by experiment:")
        for exp, count in exp_counts.items():
            total_exp = sum(1 for s in sessions if s.split('_')[0] == exp)
            print(f"  {exp}: {count} FP ({100*count/total_exp:.1f}%)")

    # Improvement suggestions
    print("\n" + "=" * 80)
    print("IMPROVEMENT SUGGESTIONS")
    print("=" * 80)

    print("""
1. FEATURES WITH LOW DISCRIMINATIVE POWER (FP looks like TP):
   - These features don't help separate bad neurons that look good:""")
    for f in problematic_features[:5]:
        print(f"     - {f}")

    print("""
2. POTENTIAL NEW METRICS TO ADD:
   - Spatial coherence: correlation between footprint and neighboring pixels
   - Temporal stability: variance of baseline over time
   - Event shape consistency: std of event shapes within neuron
   - Overlap with other neurons: fraction of footprint overlapping others
   - Signal-to-background: ratio vs local background, not global
   - Rise/decay ratio: t_rise / t_off ratio (biological constraint)
   - Footprint fragmentation: number of disconnected components
   - Neuropil contamination: correlation with surrounding ring

3. HIGH-CONFIDENCE FALSE POSITIVES:
   - These are neurons the model is VERY confident about but wrong
   - Count: {hc_count}
   - These likely need NEW features not currently captured

4. MODEL IMPROVEMENTS:
   - Use class weights to penalize FP more than FN
   - Ensemble with different thresholds per experiment type
   - Add experiment-type as a feature (one-hot encoded)
""".format(hc_count=(fp_mask & (y_proba > 0.7)).sum()))

    return X_fp, X_tp, X_tn, y_proba, fp_mask


if __name__ == "__main__":
    # Analyze the all-experiments model
    analyze_errors(
        model_path="ml/ebm_grid_search_v3_all/ebm_best_NOF_RFC_3DM_FOF.pkl",
        artifacts_dir="data/capcan_validation_127_v3",
        experiments="NOF,RFC,3DM,FOF",
        threshold=0.5
    )
