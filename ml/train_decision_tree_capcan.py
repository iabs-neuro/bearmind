"""
Train a decision tree classifier using capcan_artifacts data.

This script adapts train_decision_tree.py to use metrics from
dev/validation_artifacts/capcan_artifacts_* directories.

Key differences from original:
- Reads from capcan_artifacts directory structure
- Uses snr_recon instead of SNR_diff (no matrices.pkl available)
- Different file naming: metrics_init.csv (raw) and metrics_gt.csv (GT)
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import random


def load_session_data(session_dir):
    """
    Load raw and GT metrics for a session from capcan_artifacts directory.

    Args:
        session_dir: Path to capcan_artifacts_* directory

    Returns:
        tuple: (df_raw, df_gt) or (None, None) if failed
    """
    try:
        # File paths in capcan_artifacts structure
        raw_metrics = os.path.join(session_dir, "metrics_init.csv")
        gt_metrics = os.path.join(session_dir, "metrics_gt.csv")

        # Load dataframes
        df_raw = pd.read_csv(raw_metrics)
        df_gt = pd.read_csv(gt_metrics)

        # Parse center column from string to numpy array if needed
        if df_raw['center'].dtype == 'object':
            df_raw['center'] = df_raw['center'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
        if df_gt['center'].dtype == 'object':
            df_gt['center'] = df_gt['center'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

        return df_raw, df_gt

    except Exception as e:
        print(f"  ERROR loading {os.path.basename(session_dir)}: {e}")
        return None, None


def create_training_data(session_dirs, max_distance=3):
    """
    Create training dataset with binary labels from multiple sessions.

    Args:
        session_dirs: List of paths to capcan_artifacts_* directories
        max_distance: Max distance (pixels) to match raw/GT neurons

    Returns:
        tuple: (features_df, labels, session_labels)
    """
    all_features = []
    all_labels = []
    all_sessions = []

    print(f"Loading {len(session_dirs)} sessions...")

    for session_dir in session_dirs:
        session_name = os.path.basename(session_dir)
        df_raw, df_gt = load_session_data(session_dir)

        if df_raw is None:
            continue

        # Exclude corner artifacts first (location-based, not quality-based)
        if 'is_corner_artifact' in df_raw.columns:
            non_corner_mask = df_raw['is_corner_artifact'] == 0
            df_raw_filtered = df_raw[non_corner_mask].copy()
            n_corners_filtered = (~non_corner_mask).sum()
        else:
            df_raw_filtered = df_raw.copy()
            n_corners_filtered = 0

        # Match raw neurons to GT by spatial distance
        raw_centers = np.array(df_raw_filtered['center'].tolist())
        gt_centers = np.array(df_gt['center'].tolist())

        labels = np.zeros(len(df_raw_filtered), dtype=int)

        for i, raw_center in enumerate(raw_centers):
            distances = np.linalg.norm(gt_centers - raw_center, axis=1)
            min_dist = distances.min()

            if min_dist <= max_distance:
                labels[i] = 1  # Keep (matched to GT)
            else:
                labels[i] = 0  # Delete (not in GT)

        # Select features for training - USE ALL AVAILABLE NUMERICAL FEATURES
        # Exclude: component_idx (ID), center (spatial position), is_corner_artifact (already filtered)
        # Exclude: corr_groups (merge group ID, not a quality metric)
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

        # Replace infinities with NaN (keep NaN as is)
        features = features.replace([np.inf, -np.inf], np.nan)

        all_features.append(features)
        all_labels.append(labels)
        all_sessions.extend([session_name] * len(features))

        n_keep = labels.sum()
        n_delete = len(labels) - n_keep
        print(f"  {session_name}: {len(df_raw)} neurons ({n_corners_filtered} corners filtered), {n_keep} keep, {n_delete} delete")

    # Combine all sessions
    features_df = pd.concat(all_features, ignore_index=True)
    labels = np.concatenate(all_labels)

    # Final cleaning: replace any remaining infinities with NaN (keep NaN as is)
    features_df = features_df.replace([np.inf, -np.inf], np.nan)

    print(f"\nTotal dataset: {len(features_df)} neurons")
    print(f"  Keep: {labels.sum()} ({labels.sum()/len(labels)*100:.1f}%)")
    print(f"  Delete: {len(labels) - labels.sum()} ({(len(labels) - labels.sum())/len(labels)*100:.1f}%)")

    return features_df, labels, all_sessions


def train_decision_tree(X_train, y_train, X_test, y_test,
                        max_depth=5, min_samples_split=100, min_samples_leaf=50):
    """
    Train decision tree classifier.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        max_depth: Maximum tree depth (controls complexity)
        min_samples_split: Minimum samples to split a node
        min_samples_leaf: Minimum samples in leaf node

    Returns:
        Trained DecisionTreeClassifier
    """
    print("\n" + "="*60)
    print("TRAINING DECISION TREE")
    print("="*60)
    print(f"Max depth: {max_depth}")
    print(f"Min samples split: {min_samples_split}")
    print(f"Min samples leaf: {min_samples_leaf}")

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )

    clf.fit(X_train, y_train)

    # Evaluate on training set
    y_train_pred = clf.predict(X_train)
    train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
        y_train, y_train_pred, average='binary'
    )

    print(f"\nTraining performance:")
    print(f"  Precision: {train_prec:.2%}")
    print(f"  Recall:    {train_rec:.2%}")
    print(f"  F1 Score:  {train_f1:.2%}")

    # Evaluate on test set
    y_test_pred = clf.predict(X_test)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='binary'
    )

    print(f"\nTest performance:")
    print(f"  Precision: {test_prec:.2%}")
    print(f"  Recall:    {test_rec:.2%}")
    print(f"  F1 Score:  {test_f1:.2%}")

    # Feature importance
    print(f"\nFeature importance:")
    for feat, imp in sorted(zip(X_train.columns, clf.feature_importances_),
                           key=lambda x: x[1], reverse=True):
        if imp > 0.01:
            print(f"  {feat:25s}: {imp:.3f}")

    return clf


def visualize_tree(clf, feature_names, output_dir="ml/models"):
    """Save tree visualization to file."""
    plt.figure(figsize=(20, 10))
    plot_tree(clf,
              feature_names=feature_names,
              class_names=['Delete', 'Keep'],
              filled=True,
              rounded=True,
              fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "decision_tree_capcan_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nTree visualization saved to: {output_path}")


def extract_rules(clf, feature_names, output_dir="ml/models"):
    """Extract and save decision rules in readable format."""
    rules_text = export_text(clf, feature_names=list(feature_names))

    output_path = os.path.join(output_dir, "decision_rules_capcan.txt")
    with open(output_path, 'w') as f:
        f.write("DECISION TREE RULES FOR NEURON QUALITY CLASSIFICATION\n")
        f.write("(Trained on capcan_artifacts data)\n")
        f.write("="*60 + "\n\n")
        f.write("Class 0 = DELETE neuron\n")
        f.write("Class 1 = KEEP neuron\n\n")
        f.write(rules_text)

    print(f"Decision rules saved to: {output_path}")


def main(train_fraction=0.75, max_depth=5, min_samples_split=100, min_samples_leaf=50,
         artifacts_dir="dev/validation_artifacts",
         output_dir="ml/models",
         experiments=None):
    """
    Main training pipeline for capcan_artifacts data.

    Args:
        train_fraction: Fraction of sessions for training (default 0.75)
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split node
        min_samples_leaf: Minimum samples in leaf
        artifacts_dir: Directory containing capcan_artifacts_* subdirectories
        output_dir: Where to save outputs
        experiments: List of experiment IDs to include (e.g., ['NOF', '3DM']). If None, use all.
    """
    print("="*60)
    print("DECISION TREE TRAINING ON CAPCAN_ARTIFACTS DATA")
    print("="*60)

    # Find all capcan_artifacts directories
    artifacts_path = Path(artifacts_dir)
    session_dirs = sorted([d for d in artifacts_path.iterdir()
                          if d.is_dir() and d.name.startswith('capcan_artifacts_')])

    # Filter by experiment IDs if specified
    if experiments is not None and len(experiments) > 0:
        filtered_dirs = []
        for d in session_dirs:
            session_name = d.name.replace('capcan_artifacts_', '')
            exp_id = session_name.split('_')[0]
            if exp_id in experiments:
                filtered_dirs.append(d)
        session_dirs = filtered_dirs
        print(f"Filtered to experiments: {experiments}")

    print(f"Found {len(session_dirs)} total sessions")

    # Split into train/test
    n_train = int(len(session_dirs) * train_fraction)
    random.seed(42)
    random.shuffle(session_dirs)

    train_sessions = session_dirs[:n_train]
    test_sessions = session_dirs[n_train:]

    print(f"Train sessions: {len(train_sessions)}")
    print(f"Test sessions:  {len(test_sessions)}")

    # Create training data
    print("\n" + "-"*60)
    print("LOADING TRAINING DATA")
    print("-"*60)
    X_train, y_train, train_session_labels = create_training_data(train_sessions)

    print("\n" + "-"*60)
    print("LOADING TEST DATA")
    print("-"*60)
    X_test, y_test, test_session_labels = create_training_data(test_sessions)

    # Train model
    clf = train_decision_tree(
        X_train, y_train, X_test, y_test,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

    # Save visualizations and rules
    print("\n" + "-"*60)
    print("SAVING OUTPUTS")
    print("-"*60)
    visualize_tree(clf, X_train.columns, output_dir)
    extract_rules(clf, X_train.columns, output_dir)

    # Save model
    import pickle as pkl
    model_path = os.path.join(output_dir, "decision_tree_capcan_model.pkl")
    with open(model_path, 'wb') as f:
        pkl.dump(clf, f)
    print(f"Model saved to: {model_path}")

    print("\n" + "="*60)
    print("DONE")
    print("="*60)


def run_multiple_seeds(n_seeds=10, train_fraction=0.75, max_depth=5,
                       min_samples_split=100, min_samples_leaf=50,
                       artifacts_dir="dev/validation_artifacts",
                       output_dir="ml/models",
                       experiments=None):
    """
    Run training with multiple random seeds to assess stability.

    Args:
        n_seeds: Number of different random seeds to try
        experiments: List of experiment IDs to include (e.g., ['NOF', '3DM']). If None, use all.
        Other args: Same as main()

    Returns:
        pd.DataFrame with results for each seed
    """
    print("="*60)
    print(f"MULTI-SEED TRAINING - {n_seeds} RANDOM SEEDS")
    print("="*60)

    results = []

    # Find all sessions
    artifacts_path = Path(artifacts_dir)
    session_dirs = sorted([d for d in artifacts_path.iterdir()
                          if d.is_dir() and d.name.startswith('capcan_artifacts_')])

    # Filter by experiment IDs if specified
    if experiments is not None and len(experiments) > 0:
        filtered_dirs = []
        for d in session_dirs:
            session_name = d.name.replace('capcan_artifacts_', '')
            exp_id = session_name.split('_')[0]
            if exp_id in experiments:
                filtered_dirs.append(d)
        session_dirs = filtered_dirs
        print(f"Filtered to experiments: {experiments}")

    print(f"Total sessions: {len(session_dirs)}")
    n_train = int(len(session_dirs) * train_fraction)

    for seed in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed + 1}/{n_seeds} (random_state={seed})")
        print('='*60)

        # Split with this seed
        random.seed(seed)
        random.shuffle(session_dirs)
        train_sessions = session_dirs[:n_train]
        test_sessions = session_dirs[n_train:]

        print(f"Train: {len(train_sessions)} sessions, Test: {len(test_sessions)} sessions")

        # Load data
        print("\nLoading training data...")
        X_train, y_train, _ = create_training_data(train_sessions)

        print("Loading test data...")
        X_test, y_test, _ = create_training_data(test_sessions)

        # Train
        print("Training...")
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            class_weight='balanced'
        )
        clf.fit(X_train, y_train)

        # Evaluate
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            y_train, y_train_pred, average='binary'
        )
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average='binary'
        )

        results.append({
            'seed': seed,
            'train_precision': train_prec,
            'train_recall': train_rec,
            'train_f1': train_f1,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1': test_f1,
            'n_train': len(X_train),
            'n_test': len(X_test)
        })

        # Save individual model
        model_path = os.path.join(output_dir, f"decision_tree_capcan_seed{seed}.pkl")
        with open(model_path, 'wb') as f:
            import pickle as pkl
            pkl.dump(clf, f)

        print(f"Test Performance: Precision={test_prec:.2%}, Recall={test_rec:.2%}, F1={test_f1:.2%}")
        print(f"Model saved: {model_path}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_path = os.path.join(output_dir, "multi_seed_results_capcan.csv")
    results_df.to_csv(results_path, index=False)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL SEEDS")
    print("="*60)
    print(f"\nTest Performance (mean +/- std):")
    print(f"  Precision: {results_df['test_precision'].mean():.2%} +/- {results_df['test_precision'].std():.2%}")
    print(f"  Recall:    {results_df['test_recall'].mean():.2%} +/- {results_df['test_recall'].std():.2%}")
    print(f"  F1 Score:  {results_df['test_f1'].mean():.2%} +/- {results_df['test_f1'].std():.2%}")

    print(f"\nTrain Performance (mean +/- std):")
    print(f"  Precision: {results_df['train_precision'].mean():.2%} +/- {results_df['train_precision'].std():.2%}")
    print(f"  Recall:    {results_df['train_recall'].mean():.2%} +/- {results_df['train_recall'].std():.2%}")
    print(f"  F1 Score:  {results_df['train_f1'].mean():.2%} +/- {results_df['train_f1'].std():.2%}")

    print(f"\nResults saved to: {results_path}")
    print("="*60)

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train decision tree on capcan_artifacts neuron quality data"
    )
    parser.add_argument(
        "--train-fraction", type=float, default=0.75,
        help="Fraction of sessions for training (default 0.75)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=5,
        help="Maximum tree depth (default 5)"
    )
    parser.add_argument(
        "--min-samples-split", type=int, default=100,
        help="Minimum samples to split node (default 100)"
    )
    parser.add_argument(
        "--min-samples-leaf", type=int, default=50,
        help="Minimum samples in leaf (default 50)"
    )
    parser.add_argument(
        "--artifacts-dir", default="dev/validation_artifacts",
        help="Directory containing capcan_artifacts_* subdirectories (default: dev/validation_artifacts)"
    )
    parser.add_argument(
        "--output-dir", default="ml/models",
        help="Output directory for results (default: ml/models/)"
    )
    parser.add_argument(
        "--multi-seed", type=int, default=0,
        help="Run with multiple random seeds (specify number, e.g., 10). Default: 0 (single run)"
    )
    parser.add_argument(
        "--experiments", type=str, nargs='+', default=None,
        help="Filter to specific experiments (e.g., --experiments NOF 3DM). If not specified, uses all experiments."
    )

    args = parser.parse_args()

    if args.multi_seed > 0:
        # Run with multiple seeds
        results_df = run_multiple_seeds(
            n_seeds=args.multi_seed,
            train_fraction=args.train_fraction,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            artifacts_dir=args.artifacts_dir,
            output_dir=args.output_dir,
            experiments=args.experiments
        )
    else:
        # Single run
        main(
            train_fraction=args.train_fraction,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            artifacts_dir=args.artifacts_dir,
            output_dir=args.output_dir,
            experiments=args.experiments
        )
