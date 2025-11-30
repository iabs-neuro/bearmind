"""
Comprehensive grid search for decision tree hyperparameters on capcan data.

Goal: Maximize precision (minimize false positives) in neuron quality classification.

This script:
1. Loads validation data from capcan_artifacts (21 features)
2. Tests extensive parameter combinations
3. Saves all trained models
4. Records detailed results for each configuration
"""
import os
import pickle
import numpy as np
import pandas as pd
import random
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from itertools import product
from tqdm import tqdm
from datetime import datetime


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


def create_dataset(session_dirs, max_distance=3):
    """Create dataset from capcan_artifacts directories."""
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

        # Extract ALL 21 features
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

        all_features.append(features)
        all_labels.append(labels)

    features_df = pd.concat(all_features, ignore_index=True)
    labels = np.concatenate(all_labels)

    # Final cleaning
    features_df = features_df.replace([np.inf, -np.inf], np.nan)

    return features_df, labels


def train_and_evaluate(X_train, y_train, X_test, y_test, params):
    """Train model with given parameters and evaluate."""
    # Convert class_weight_factor to sklearn format
    class_weight = {0: 1.0, 1: params['class_weight_factor']}

    model = DecisionTreeClassifier(
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        class_weight=class_weight,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate on train set
    y_train_pred = model.predict(X_train)
    train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
        y_train, y_train_pred, average='binary', zero_division=0
    )

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='binary', zero_division=0
    )

    return model, {
        'train_precision': train_prec,
        'train_recall': train_rec,
        'train_f1': train_f1,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1
    }


def grid_search(artifacts_dir="data/capcan_validation_127",
                test_fraction=0.25,
                random_seed=42,
                output_dir="ml/grid_search_models",
                experiments=None):
    """
    Comprehensive grid search for decision tree hyperparameters.

    Args:
        artifacts_dir: Directory with capcan_artifacts
        test_fraction: Fraction of data for testing
        random_seed: Random seed for reproducibility
        output_dir: Directory to save models and results
        experiments: List of experiment IDs to include (e.g., ['NOF', 'RFC']). If None, use all.
    """
    print("="*80)
    print("DECISION TREE GRID SEARCH (CAPCAN DATA - 21 FEATURES)")
    if experiments:
        exp_str = "_".join(experiments)
        print(f"EXPERIMENTS: {exp_str}")
    print("="*80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nModels will be saved to: {output_path}")

    # Define parameter grid
    param_grid = {
        'max_depth': [4, 5, 6, 7],
        'min_samples_split': [40, 60, 80, 100, 120],
        'min_samples_leaf': [20, 30, 40, 50, 60],
        'class_weight_factor': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }

    print("\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # Calculate total combinations
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\nTotal combinations to test: {total_combinations}")

    # Load data
    print("\nLoading data...")
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

    print(f"Found {len(session_dirs)} sessions")

    # Train/test split
    n_train = int(len(session_dirs) * (1 - test_fraction))
    random.seed(random_seed)
    random.shuffle(session_dirs)

    train_sessions = session_dirs[:n_train]
    test_sessions = session_dirs[n_train:]

    print(f"Train sessions: {len(train_sessions)}")
    print(f"Test sessions: {len(test_sessions)}")

    # Create datasets
    print("\nCreating train dataset...")
    X_train, y_train = create_dataset(train_sessions)
    print(f"Train samples: {len(X_train)}")
    print(f"  Keep: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
    print(f"  Delete: {len(y_train) - y_train.sum()} ({(len(y_train) - y_train.sum())/len(y_train)*100:.1f}%)")

    print("\nCreating test dataset...")
    X_test, y_test = create_dataset(test_sessions)
    print(f"Test samples: {len(X_test)}")
    print(f"  Keep: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
    print(f"  Delete: {len(y_test) - y_test.sum()} ({(len(y_test) - y_test.sum())/len(y_test)*100:.1f}%)")

    # Grid search
    print("\n" + "="*80)
    print("STARTING GRID SEARCH")
    print("="*80)

    results = []
    best_precision = 0
    best_params = None

    # Generate all parameter combinations
    param_combinations = list(product(
        param_grid['max_depth'],
        param_grid['min_samples_split'],
        param_grid['min_samples_leaf'],
        param_grid['class_weight_factor']
    ))

    start_time = datetime.now()

    # Create experiment suffix for filenames
    exp_suffix = ""
    if experiments is not None and len(experiments) > 0:
        exp_suffix = "_" + "_".join(experiments)

    for idx, (max_depth, min_split, min_leaf, class_weight) in enumerate(tqdm(param_combinations, desc="Grid search")):
        params = {
            'max_depth': max_depth,
            'min_samples_split': min_split,
            'min_samples_leaf': min_leaf,
            'class_weight_factor': class_weight
        }

        # Train and evaluate
        model, metrics = train_and_evaluate(X_train, y_train, X_test, y_test, params)

        # Save model with experiment suffix
        model_filename = f"dt{exp_suffix}_d{max_depth}_s{min_split}_l{min_leaf}_w{class_weight:.1f}.pkl"
        model_path = output_path / model_filename
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Record results
        result = {
            'model_id': idx,
            'model_file': model_filename,
            'max_depth': max_depth,
            'min_samples_split': min_split,
            'min_samples_leaf': min_leaf,
            'class_weight_factor': class_weight,
            **metrics
        }
        results.append(result)

        # Track best precision
        if metrics['test_precision'] > best_precision:
            best_precision = metrics['test_precision']
            best_params = params.copy()

    # Save results with experiment suffix
    results_df = pd.DataFrame(results)
    results_path = Path(f"ml/grid_search_results_21features{exp_suffix}.csv")
    results_df.to_csv(results_path, index=False)

    elapsed_time = datetime.now() - start_time

    # Print summary
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    print(f"\nElapsed time: {elapsed_time}")
    print(f"Results saved to: {results_path}")
    print(f"Models saved to: {output_path}/")

    # Best results by precision
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS BY TEST PRECISION")
    print("="*80)

    top_precision = results_df.nlargest(10, 'test_precision')
    print("\n" + top_precision[['max_depth', 'min_samples_split', 'min_samples_leaf',
                                 'class_weight_factor', 'test_precision', 'test_recall',
                                 'test_f1']].to_string(index=False))

    # Best results by F1
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS BY TEST F1 SCORE")
    print("="*80)

    top_f1 = results_df.nlargest(10, 'test_f1')
    print("\n" + top_f1[['max_depth', 'min_samples_split', 'min_samples_leaf',
                          'class_weight_factor', 'test_precision', 'test_recall',
                          'test_f1']].to_string(index=False))

    # Best balanced (precision >= 90%)
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS WITH PRECISION >= 90%")
    print("="*80)

    high_precision = results_df[results_df['test_precision'] >= 0.90]
    if not high_precision.empty:
        top_high_prec = high_precision.nlargest(10, 'test_f1')
        print("\n" + top_high_prec[['max_depth', 'min_samples_split', 'min_samples_leaf',
                                     'class_weight_factor', 'test_precision', 'test_recall',
                                     'test_f1']].to_string(index=False))
    else:
        print("\nNo configurations achieved precision >= 90%")

    # Parameter importance analysis
    print("\n" + "="*80)
    print("PARAMETER EFFECT ON PRECISION")
    print("="*80)

    print("\nBy max_depth:")
    depth_analysis = results_df.groupby('max_depth')['test_precision'].agg(['mean', 'std', 'max'])
    print(depth_analysis.to_string())

    print("\nBy class_weight_factor:")
    weight_analysis = results_df.groupby('class_weight_factor')['test_precision'].agg(['mean', 'std', 'max'])
    print(weight_analysis.to_string())

    print("\nBy min_samples_split:")
    split_analysis = results_df.groupby('min_samples_split')['test_precision'].agg(['mean', 'std', 'max'])
    print(split_analysis.to_string())

    print("\nBy min_samples_leaf:")
    leaf_analysis = results_df.groupby('min_samples_leaf')['test_precision'].agg(['mean', 'std', 'max'])
    print(leaf_analysis.to_string())

    print("\n" + "="*80)

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grid search for decision tree hyperparameters")
    parser.add_argument("--artifacts-dir", default="data/capcan_validation_127",
                       help="Directory containing capcan_artifacts_* subdirectories")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for train/test split")
    parser.add_argument("--output-dir", default="ml/grid_search_models",
                       help="Directory to save models")
    parser.add_argument("--experiments", type=str, nargs='+', default=None,
                       help="Filter to specific experiments (e.g., --experiments NOF RFC). If not specified, uses all experiments.")

    args = parser.parse_args()

    results = grid_search(artifacts_dir=args.artifacts_dir,
                         random_seed=args.seed,
                         output_dir=args.output_dir,
                         experiments=args.experiments)
