"""
Grid search to find optimal decision tree parameters for capcan data.

Tests combinations of:
- max_depth
- min_samples_split
- min_samples_leaf
- class_weight

Goal: Maximize recall while maintaining precision >= 85%
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
import random
from itertools import product


def load_session_data(session_dir):
    """Load metrics for a session from capcan_artifacts directory."""
    try:
        raw_metrics = os.path.join(session_dir, "metrics_init.csv")
        gt_metrics = os.path.join(session_dir, "metrics_gt.csv")

        df_raw = pd.read_csv(raw_metrics)
        df_gt = pd.read_csv(gt_metrics)

        if df_raw['center'].dtype == 'object':
            df_raw['center'] = df_raw['center'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
        if df_gt['center'].dtype == 'object':
            df_gt['center'] = df_gt['center'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

        return df_raw, df_gt
    except Exception as e:
        return None, None


def create_training_data(session_dirs, max_distance=3):
    """Create training dataset from capcan_artifacts directories."""
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

        # Match to GT
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

        # Extract ALL features
        feature_cols = [
            'area', 'circularity', 'max_edge', 'convexity',
            'caiman_snr', 'caiman_r_score', 'corr_groups',
            'events_per_min', 'events_fraction',
            't_rise', 't_off', 'wavelet_snr',
            'r2_score', 'event_r2_score',
            'nmae', 'nrmse', 'snr_recon'
        ]

        features = df_raw_filtered[feature_cols].copy()
        features = features.replace([np.inf, -np.inf], np.nan)

        all_features.append(features)
        all_labels.append(labels)

    features_df = pd.concat(all_features, ignore_index=True)
    labels = np.concatenate(all_labels)
    features_df = features_df.replace([np.inf, -np.inf], np.nan)

    return features_df, labels


def grid_search(artifacts_dir="dev/validation_artifacts",
                train_fraction=0.75,
                n_trials=3,
                output_dir="ml/results"):
    """
    Grid search over tree hyperparameters.

    Args:
        artifacts_dir: Directory with capcan_artifacts
        train_fraction: Fraction for training
        n_trials: Number of random train/test splits to average over
        output_dir: Where to save results
    """
    print("="*60)
    print("GRID SEARCH FOR OPTIMAL TREE PARAMETERS")
    print("="*60)

    # Parameter grid - exploring both directions
    param_grid = {
        'max_depth': [4, 5, 6, 7, 8],  # Both shallower (more conservative) and deeper
        'min_samples_split': [40, 50, 60, 80],  # Higher = more conservative
        'min_samples_leaf': [20, 25, 30, 40],  # Higher = more conservative
        'class_weight_factor': [0.8, 1.0, 1.2, 1.5, 2.0]  # <1.0 favors precision, >1.0 favors recall
    }

    # Load data
    artifacts_path = Path(artifacts_dir)
    session_dirs = sorted([d for d in artifacts_path.iterdir()
                          if d.is_dir() and d.name.startswith('capcan_artifacts_')])

    print(f"\nFound {len(session_dirs)} sessions")
    print(f"Will test {len(list(product(*param_grid.values())))} parameter combinations")
    print(f"Averaging over {n_trials} random train/test splits")
    print()

    all_results = []
    combo_count = 0
    total_combos = len(list(product(*param_grid.values())))

    # Grid search
    for max_depth in param_grid['max_depth']:
        for min_split in param_grid['min_samples_split']:
            for min_leaf in param_grid['min_samples_leaf']:
                for cw_factor in param_grid['class_weight_factor']:
                    combo_count += 1

                    # Average over multiple trials
                    trial_results = []

                    for trial in range(n_trials):
                        # Split data
                        n_train = int(len(session_dirs) * train_fraction)
                        random.seed(trial)
                        random.shuffle(session_dirs)

                        train_sessions = session_dirs[:n_train]
                        test_sessions = session_dirs[n_train:]

                        # Load data
                        X_train, y_train = create_training_data(train_sessions)
                        X_test, y_test = create_training_data(test_sessions)

                        # Train model
                        clf = DecisionTreeClassifier(
                            max_depth=max_depth,
                            min_samples_split=min_split,
                            min_samples_leaf=min_leaf,
                            class_weight={0: 1.0, 1: cw_factor},
                            random_state=42
                        )

                        clf.fit(X_train, y_train)

                        # Evaluate
                        y_pred = clf.predict(X_test)
                        prec, rec, f1, _ = precision_recall_fscore_support(
                            y_test, y_pred, average='binary'
                        )

                        trial_results.append({
                            'precision': prec,
                            'recall': rec,
                            'f1': f1
                        })

                    # Average across trials
                    avg_prec = np.mean([r['precision'] for r in trial_results])
                    avg_rec = np.mean([r['recall'] for r in trial_results])
                    avg_f1 = np.mean([r['f1'] for r in trial_results])

                    std_prec = np.std([r['precision'] for r in trial_results])
                    std_rec = np.std([r['recall'] for r in trial_results])
                    std_f1 = np.std([r['f1'] for r in trial_results])

                    all_results.append({
                        'max_depth': max_depth,
                        'min_samples_split': min_split,
                        'min_samples_leaf': min_leaf,
                        'class_weight_factor': cw_factor,
                        'precision_mean': avg_prec,
                        'precision_std': std_prec,
                        'recall_mean': avg_rec,
                        'recall_std': std_rec,
                        'f1_mean': avg_f1,
                        'f1_std': std_f1
                    })

                    print(f"[{combo_count}/{total_combos}] depth={max_depth} split={min_split} leaf={min_leaf} cw={cw_factor:.1f} "
                          f"-> P={avg_prec:.2%} R={avg_rec:.2%} F1={avg_f1:.2%}")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "grid_search_results.csv")
    results_df.to_csv(output_path, index=False)

    print("\n" + "="*60)
    print("TOP 10 CONFIGURATIONS BY F1 SCORE")
    print("="*60)

    top_f1 = results_df.nlargest(10, 'f1_mean')
    print(top_f1[['max_depth', 'min_samples_split', 'min_samples_leaf',
                  'class_weight_factor', 'precision_mean', 'recall_mean', 'f1_mean']].to_string(index=False))

    print("\n" + "="*60)
    print("TOP 10 BY PRECISION (Minimize False Positives)")
    print("="*60)

    top_precision = results_df.nlargest(10, 'precision_mean')
    print(top_precision[['max_depth', 'min_samples_split', 'min_samples_leaf',
                        'class_weight_factor', 'precision_mean', 'recall_mean', 'f1_mean']].to_string(index=False))

    print("\n" + "="*60)
    print("TOP 10 BY RECALL (Minimize False Negatives)")
    print("="*60)

    top_recall = results_df.nlargest(10, 'recall_mean')
    print(top_recall[['max_depth', 'min_samples_split', 'min_samples_leaf',
                     'class_weight_factor', 'precision_mean', 'recall_mean', 'f1_mean']].to_string(index=False))

    print("\n" + "="*60)
    print("BALANCED: High precision (>=90%) with best recall")
    print("="*60)

    high_prec_90 = results_df[results_df['precision_mean'] >= 0.90]
    if not high_prec_90.empty:
        balanced_90 = high_prec_90.nlargest(10, 'recall_mean')
        print(balanced_90[['max_depth', 'min_samples_split', 'min_samples_leaf',
                          'class_weight_factor', 'precision_mean', 'recall_mean', 'f1_mean']].to_string(index=False))
    else:
        print("No configurations achieved precision >= 90%")

    print("\n" + "="*60)
    print("COMPARISON WITH CURRENT BASELINE")
    print("="*60)
    print("Current params (depth=5, split=50, leaf=25, cw=balanced):")
    print("  Precision: 91.91% +/- 2.50%")
    print("  Recall:    78.86% +/- 2.67%")
    print("  F1:        84.85% +/- 1.84%")

    print(f"\nResults saved to: {output_path}")
    print("="*60)

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grid search for optimal tree parameters")
    parser.add_argument("--artifacts-dir", default="dev/validation_artifacts",
                       help="Directory with capcan_artifacts")
    parser.add_argument("--trials", type=int, default=3,
                       help="Number of random train/test splits (default: 3)")
    parser.add_argument("--output-dir", default="ml/results",
                       help="Output directory (default: ml/results/)")

    args = parser.parse_args()

    results = grid_search(
        artifacts_dir=args.artifacts_dir,
        n_trials=args.trials,
        output_dir=args.output_dir
    )
