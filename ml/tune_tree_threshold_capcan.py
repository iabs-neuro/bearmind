"""
Tune decision tree threshold for capcan_artifacts data.

Default sklearn predict() uses 0.5 threshold on predict_proba().
Lowering threshold -> more lenient -> higher recall, lower precision
Raising threshold -> more conservative -> lower recall, higher precision
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, fbeta_score
import matplotlib.pyplot as plt
from data_utils import FEATURE_COLS, FBETA_BETA, compute_fbeta
import random


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


def create_test_data(session_dirs, max_distance=3):
    """Create test dataset from capcan_artifacts directories."""
    all_features = []
    all_labels = []

    for session_dir in session_dirs:
        df_raw, df_gt = load_session_data(session_dir)

        if df_raw is None:
            continue

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

        # Use centralized feature columns from data_utils
        features = df_raw_filtered[FEATURE_COLS].copy()
        features = features.replace([np.inf, -np.inf], np.nan)

        all_features.append(features)
        all_labels.append(labels)

    features_df = pd.concat(all_features, ignore_index=True)
    labels = np.concatenate(all_labels)

    # Final cleaning
    features_df = features_df.replace([np.inf, -np.inf], np.nan)

    return features_df, labels


def tune_threshold(model_path="ml/models/decision_tree_capcan_model.pkl",
                   artifacts_dir="dev/validation_artifacts",
                   test_fraction=0.25,
                   random_seed=42,
                   aggregate_seeds=False,
                   n_seeds=10):
    """
    Tune decision threshold to find optimal recall/precision trade-off.

    Args:
        model_path: Path to single model (ignored if aggregate_seeds=True)
        artifacts_dir: Directory with capcan_artifacts
        test_fraction: Fraction of data for testing
        random_seed: Random seed for data split
        aggregate_seeds: If True, aggregate results from multiple models
        n_seeds: Number of seed models to aggregate (0 to n_seeds-1)
    """
    print("="*60)
    print("DECISION TREE THRESHOLD TUNING (CAPCAN DATA)")
    if aggregate_seeds:
        print(f"AGGREGATING RESULTS FROM {n_seeds} MODELS")
    print("="*60)

    # Load trained model(s)
    if aggregate_seeds:
        print(f"\nLoading {n_seeds} models (seed 0 to {n_seeds-1})...")
        models = []
        for seed in range(n_seeds):
            model_file = f"ml/models/decision_tree_capcan_seed{seed}.pkl"
            try:
                with open(model_file, 'rb') as f:
                    models.append(pickle.load(f))
                print(f"  Loaded: {model_file}")
            except FileNotFoundError:
                print(f"  WARNING: {model_file} not found, skipping")

        if not models:
            raise RuntimeError("No models loaded! Check that seed models exist.")
        print(f"Successfully loaded {len(models)} models")
    else:
        print(f"\nLoading single model from {model_path}...")
        with open(model_path, 'rb') as f:
            models = [pickle.load(f)]

    # If aggregating, we need to test each model with its own test set
    if aggregate_seeds:
        print("\nLoading test data for each seed...")
        artifacts_path = Path(artifacts_dir)
        session_dirs = sorted([d for d in artifacts_path.iterdir()
                              if d.is_dir() and d.name.startswith('capcan_artifacts_')])

        all_model_results = []

        for seed_idx, model in enumerate(models):
            print(f"\nProcessing model seed {seed_idx}...")

            # Use same split as this model's training
            n_train = int(len(session_dirs) * (1 - test_fraction))
            random.seed(seed_idx)
            random.shuffle(session_dirs)
            test_sessions = session_dirs[n_train:]

            X_test, y_test = create_test_data(test_sessions)

            # Get probability predictions for this model
            y_proba = model.predict_proba(X_test)[:, 1]

            # Test different thresholds
            thresholds = np.arange(0.1, 0.95, 0.05)

            model_results = []
            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)

                prec, rec, _, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='binary', zero_division=0
                )
                fb = compute_fbeta(prec, rec)

                model_results.append({
                    'seed': seed_idx,
                    'threshold': threshold,
                    'precision': prec,
                    'recall': rec,
                    'fbeta': fb
                })

            all_model_results.extend(model_results)

        # Aggregate results across all models
        all_results_df = pd.DataFrame(all_model_results)

        print("\n" + "="*60)
        print(f"AGGREGATED THRESHOLD TUNING RESULTS (F-beta β={FBETA_BETA:.3f})")
        print("="*60)
        print(f"\n{'Threshold':<12} {'Precision':<18} {'Recall':<18} {'F-beta':<18}")
        print("-"*80)

        results = []
        for threshold in thresholds:
            thresh_data = all_results_df[all_results_df['threshold'] == threshold]

            prec_mean = thresh_data['precision'].mean()
            prec_std = thresh_data['precision'].std()
            rec_mean = thresh_data['recall'].mean()
            rec_std = thresh_data['recall'].std()
            fbeta_mean = thresh_data['fbeta'].mean()
            fbeta_std = thresh_data['fbeta'].std()

            results.append({
                'threshold': threshold,
                'precision': prec_mean,
                'precision_std': prec_std,
                'recall': rec_mean,
                'recall_std': rec_std,
                'fbeta': fbeta_mean,
                'fbeta_std': fbeta_std
            })

            print(f"{threshold:<12.2f} {prec_mean:.2%} +/- {prec_std:.2%}   {rec_mean:.2%} +/- {rec_std:.2%}   {fbeta_mean:.2%} +/- {fbeta_std:.2%}")

    else:
        # Single model evaluation (original code)
        print("Loading test data...")
        artifacts_path = Path(artifacts_dir)
        session_dirs = sorted([d for d in artifacts_path.iterdir()
                              if d.is_dir() and d.name.startswith('capcan_artifacts_')])

        # Use same train/test split as original
        n_train = int(len(session_dirs) * (1 - test_fraction))
        random.seed(random_seed)
        random.shuffle(session_dirs)
        test_sessions = session_dirs[n_train:]

        print(f"Test sessions: {len(test_sessions)}")

        X_test, y_test = create_test_data(test_sessions)

        print(f"Test samples: {len(X_test)}")
        print(f"  Keep: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
        print(f"  Delete: {len(y_test) - y_test.sum()} ({(len(y_test) - y_test.sum())/len(y_test)*100:.1f}%)")

        # Get probability predictions
        print("\nComputing probability predictions...")
        y_proba = models[0].predict_proba(X_test)[:, 1]

        # Test different thresholds
        thresholds = np.arange(0.1, 0.95, 0.05)

        print("\n" + "="*60)
        print(f"THRESHOLD TUNING RESULTS (F-beta β={FBETA_BETA:.3f})")
        print("="*60)
        print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F-beta':<12}")
        print("-"*60)

        results = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            prec, rec, _, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary', zero_division=0
            )
            fb = compute_fbeta(prec, rec)

            results.append({
                'threshold': threshold,
                'precision': prec,
                'recall': rec,
                'fbeta': fb
            })

            print(f"{threshold:<12.2f} {prec:<12.2%} {rec:<12.2%} {fb:<12.2%}")

    results_df = pd.DataFrame(results)

    # Find optimal thresholds for different objectives
    print("\n" + "="*60)
    print("RECOMMENDED THRESHOLDS")
    print("="*60)

    # Best F-beta
    best_fbeta_idx = results_df['fbeta'].idxmax()
    best_fbeta_row = results_df.iloc[best_fbeta_idx]
    print(f"\nBest F-beta (β={FBETA_BETA:.3f}): threshold = {best_fbeta_row['threshold']:.2f}")
    print(f"  Precision: {best_fbeta_row['precision']:.2%}")
    print(f"  Recall:    {best_fbeta_row['recall']:.2%}")
    print(f"  F-beta:    {best_fbeta_row['fbeta']:.2%}")

    # Best recall (>= 90%)
    high_recall_df = results_df[results_df['recall'] >= 0.90]
    if not high_recall_df.empty:
        best_high_recall_idx = high_recall_df['fbeta'].idxmax()
        best_high_recall_row = results_df.iloc[best_high_recall_idx]
        print(f"\nBest F-beta with recall >= 90%: threshold = {best_high_recall_row['threshold']:.2f}")
        print(f"  Precision: {best_high_recall_row['precision']:.2%}")
        print(f"  Recall:    {best_high_recall_row['recall']:.2%}")
        print(f"  F-beta:    {best_high_recall_row['fbeta']:.2%}")

    # Current default (0.5)
    default_rows = results_df[results_df['threshold'] == 0.5]
    if not default_rows.empty:
        default_row = default_rows.iloc[0]
        print(f"\nCurrent default (0.5):")
        print(f"  Precision: {default_row['precision']:.2%}")
        print(f"  Recall:    {default_row['recall']:.2%}")
        print(f"  F-beta:    {default_row['fbeta']:.2%}")

    # Save results
    suffix = "_aggregated" if aggregate_seeds else ""
    output_path = f"ml/results/threshold_tuning_results_capcan{suffix}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Plot precision-recall curve
    print("\nGenerating precision-recall curve...")
    plt.figure(figsize=(10, 6))

    if aggregate_seeds:
        # Plot with error bars for aggregated results
        plt.errorbar(results_df['recall'], results_df['precision'],
                    xerr=results_df['recall_std'], yerr=results_df['precision_std'],
                    fmt='b-o', linewidth=2, markersize=6, capsize=3, alpha=0.7,
                    label='Mean +/- Std')
        title = f'Decision Tree (capcan, {len(models)} models): Precision-Recall Trade-off'
    else:
        plt.plot(results_df['recall'], results_df['precision'], 'b-o', linewidth=2, markersize=6)
        title = 'Decision Tree (capcan): Precision-Recall Trade-off'

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    # Mark special points
    plt.plot(best_fbeta_row['recall'], best_fbeta_row['precision'], 'go', markersize=12,
             label=f'Best F-beta (thr={best_fbeta_row["threshold"]:.2f})')

    if not default_rows.empty:
        plt.plot(default_row['recall'], default_row['precision'], 'ro', markersize=12,
                 label='Default (thr=0.5)')

    plt.legend(fontsize=10)
    plt.tight_layout()
    plot_path = f"ml/results/precision_recall_curve_capcan{suffix}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Plot saved to: {plot_path}")
    print("="*60)

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune decision tree threshold for capcan data")
    parser.add_argument("--model", default="ml/models/decision_tree_capcan_model.pkl",
                       help="Path to trained model (ignored if --aggregate is used)")
    parser.add_argument("--artifacts-dir", default="dev/validation_artifacts",
                       help="Directory containing capcan_artifacts_* subdirectories")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for train/test split (should match training)")
    parser.add_argument("--aggregate", action="store_true",
                       help="Aggregate results from all seed models (seed0-seed9)")
    parser.add_argument("--n-seeds", type=int, default=10,
                       help="Number of seed models to aggregate (default: 10)")

    args = parser.parse_args()

    results = tune_threshold(model_path=args.model,
                            artifacts_dir=args.artifacts_dir,
                            random_seed=args.seed,
                            aggregate_seeds=args.aggregate,
                            n_seeds=args.n_seeds)
