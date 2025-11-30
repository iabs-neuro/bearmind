"""
Evaluate trained decision tree model on individual sessions.

This script helps identify which sessions have poor performance,
potentially indicating corrupted data or unsuitable characteristics.
"""
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support


def load_session_data(session_dir):
    """Load metrics for a session from capcan_artifacts directory."""
    try:
        raw_metrics = os.path.join(session_dir, "metrics_init.csv")
        gt_metrics = os.path.join(session_dir, "metrics_gt.csv")

        df_raw = pd.read_csv(raw_metrics)
        df_gt = pd.read_csv(gt_metrics)

        # Parse center column if needed
        if df_raw['center'].dtype == 'object':
            df_raw['center'] = df_raw['center'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
        if df_gt['center'].dtype == 'object':
            df_gt['center'] = df_gt['center'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

        return df_raw, df_gt
    except Exception as e:
        print(f"  ERROR loading {os.path.basename(session_dir)}: {e}")
        return None, None


def create_session_dataset(session_dir, max_distance=3):
    """Create features and labels for a single session."""
    df_raw, df_gt = load_session_data(session_dir)

    if df_raw is None:
        return None, None, None

    # Filter corner artifacts
    if 'is_corner_artifact' in df_raw.columns:
        non_corner_mask = df_raw['is_corner_artifact'] == 0
        df_raw_filtered = df_raw[non_corner_mask].copy()
    else:
        df_raw_filtered = df_raw.copy()

    # Create labels by matching to GT
    raw_centers = np.array(df_raw_filtered['center'].tolist())
    gt_centers = np.array(df_gt['center'].tolist())

    labels = np.zeros(len(df_raw_filtered), dtype=int)

    for i, raw_center in enumerate(raw_centers):
        distances = np.linalg.norm(gt_centers - raw_center, axis=1)
        min_dist = distances.min()

        if min_dist <= max_distance:
            labels[i] = 1
        else:
            labels[i] = 0

    # Extract features (21 features)
    feature_cols = [
        'area',
        'circularity',
        'max_edge',
        'convexity',
        'caiman_snr',
        'caiman_r_score',
        'events_per_min',
        'events_fraction',
        't_rise',
        't_off',
        'wavelet_snr',
        'r2_score',
        'event_r2_score',
        'nmae',
        'nrmse',
        'snr_recon',
        'noise_level',
        'baseline',
        'tau_decay',
        'trace_skewness',
        'footprint_compactness'
    ]

    features = df_raw_filtered[feature_cols].copy()
    features = features.replace([np.inf, -np.inf], np.nan)

    return features, labels, len(df_raw)


def evaluate_per_session(model_path="ml/models/decision_tree_capcan_model.pkl",
                         artifacts_dir="data/capcan_validation_127",
                         output_path="ml/results/per_session_performance.csv"):
    """
    Evaluate model on each session individually.

    Args:
        model_path: Path to trained model
        artifacts_dir: Directory with capcan_artifacts
        output_path: Where to save results CSV

    Returns:
        pd.DataFrame with per-session results
    """
    print("="*80)
    print("PER-SESSION MODEL EVALUATION")
    print("="*80)

    # Load model
    print(f"\nLoading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Find all session directories
    artifacts_path = Path(artifacts_dir)
    session_dirs = sorted([d for d in artifacts_path.iterdir()
                          if d.is_dir() and d.name.startswith('capcan_artifacts_')])

    print(f"Found {len(session_dirs)} sessions\n")
    print("-"*80)

    # Evaluate each session
    results = []

    for session_dir in session_dirs:
        session_name = session_dir.name.replace('capcan_artifacts_', '')
        exp_id = session_name.split('_')[0]

        # Load session data
        X, y, n_neurons_total = create_session_dataset(session_dir)

        if X is None:
            print(f"[FAILED] {session_name} - could not load data")
            continue

        # Predict
        try:
            y_pred = model.predict(X)

            # Calculate metrics
            prec, rec, f1, _ = precision_recall_fscore_support(
                y, y_pred, average='binary', zero_division=0
            )

            # Count neurons
            n_neurons_filtered = len(X)
            n_corners = n_neurons_total - n_neurons_filtered
            n_keep_true = y.sum()
            n_keep_pred = y_pred.sum()
            n_delete_true = len(y) - n_keep_true
            n_delete_pred = len(y_pred) - n_keep_pred

            results.append({
                'session': session_name,
                'experiment': exp_id,
                'n_neurons_total': n_neurons_total,
                'n_neurons_filtered': n_neurons_filtered,
                'n_corners': n_corners,
                'n_keep_true': n_keep_true,
                'n_delete_true': n_delete_true,
                'n_keep_pred': n_keep_pred,
                'n_delete_pred': n_delete_pred,
                'precision': prec,
                'recall': rec,
                'f1': f1
            })

            print(f"[OK] {session_name:30s} | P={prec:.2%} R={rec:.2%} F1={f1:.2%} | {n_neurons_filtered} neurons")

        except Exception as e:
            print(f"[ERROR] {session_name} - {e}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("-"*80)
    print(f"\nResults saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nOverall (all {len(results_df)} sessions):")
    print(f"  Precision: {results_df['precision'].mean():.2%} +/- {results_df['precision'].std():.2%}")
    print(f"  Recall:    {results_df['recall'].mean():.2%} +/- {results_df['recall'].std():.2%}")
    print(f"  F1 Score:  {results_df['f1'].mean():.2%} +/- {results_df['f1'].std():.2%}")

    # Group by experiment
    print("\n" + "-"*80)
    print("BY EXPERIMENT:")
    print("-"*80)

    for exp in sorted(results_df['experiment'].unique()):
        exp_data = results_df[results_df['experiment'] == exp]
        print(f"\n{exp} ({len(exp_data)} sessions):")
        print(f"  Precision: {exp_data['precision'].mean():.2%} +/- {exp_data['precision'].std():.2%}")
        print(f"  Recall:    {exp_data['recall'].mean():.2%} +/- {exp_data['recall'].std():.2%}")
        print(f"  F1 Score:  {exp_data['f1'].mean():.2%} +/- {exp_data['f1'].std():.2%}")

    # Identify worst sessions
    print("\n" + "-"*80)
    print("WORST 10 SESSIONS BY PRECISION:")
    print("-"*80)

    worst_prec = results_df.nsmallest(10, 'precision')
    print(worst_prec[['session', 'experiment', 'precision', 'recall', 'f1',
                      'n_neurons_filtered']].to_string(index=False))

    print("\n" + "-"*80)
    print("WORST 10 SESSIONS BY F1 SCORE:")
    print("-"*80)

    worst_f1 = results_df.nsmallest(10, 'f1')
    print(worst_f1[['session', 'experiment', 'precision', 'recall', 'f1',
                    'n_neurons_filtered']].to_string(index=False))

    # Identify best sessions
    print("\n" + "-"*80)
    print("BEST 10 SESSIONS BY F1 SCORE:")
    print("-"*80)

    best_f1 = results_df.nlargest(10, 'f1')
    print(best_f1[['session', 'experiment', 'precision', 'recall', 'f1',
                   'n_neurons_filtered']].to_string(index=False))

    print("\n" + "="*80)

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model on individual sessions")
    parser.add_argument("--model", default="ml/models/decision_tree_capcan_model.pkl",
                       help="Path to trained model (default: ml/models/decision_tree_capcan_model.pkl)")
    parser.add_argument("--artifacts-dir", default="data/capcan_validation_127",
                       help="Directory containing capcan_artifacts_* subdirectories")
    parser.add_argument("--output", default="ml/results/per_session_performance.csv",
                       help="Output CSV path (default: ml/results/per_session_performance.csv)")

    args = parser.parse_args()

    results = evaluate_per_session(model_path=args.model,
                                   artifacts_dir=args.artifacts_dir,
                                   output_path=args.output)
