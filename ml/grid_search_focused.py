"""
Focused grid search for interpretable decision trees.

Searches over shallow depths (3-5) with conservative parameters
to find simple, robust, interpretable models.

Multi-seed training for robustness assessment.
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
        random_state=params.get('random_state', 42)
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

    # Count tree size
    n_nodes = model.tree_.node_count
    n_leaves = model.tree_.n_leaves

    return model, {
        'train_precision': train_prec,
        'train_recall': train_rec,
        'train_f1': train_f1,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1,
        'n_nodes': n_nodes,
        'n_leaves': n_leaves
    }


def grid_search_focused(artifacts_dir="data/capcan_validation_127",
                        test_fraction=0.25,
                        output_dir="ml/grid_search_focused",
                        experiments=None,
                        n_seeds=5):
    """
    Focused grid search for interpretable models with multi-seed training.

    Args:
        artifacts_dir: Directory with capcan_artifacts
        test_fraction: Fraction of data for testing
        output_dir: Directory to save models and results
        experiments: List of experiment IDs to include (e.g., ['NOF', 'RFC'])
        n_seeds: Number of random seeds to try
    """
    print("="*80)
    print("FOCUSED GRID SEARCH - INTERPRETABLE DECISION TREES")
    print("Multi-seed training for robustness")
    if experiments:
        exp_str = "_".join(experiments)
        print(f"EXPERIMENTS: {exp_str}")
    print("="*80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nModels will be saved to: {output_path}")

    # Define FOCUSED parameter grid (for interpretability)
    param_grid = {
        'max_depth': [3, 4, 5],  # Shallow trees only
        'min_samples_split': [60, 80, 100],  # Higher values (more conservative)
        'min_samples_leaf': [40, 50, 60, 70],  # Higher values (prevent overfitting)
        'class_weight_factor': [0.6, 0.7, 0.8, 0.9, 1.0]  # Balanced range
    }

    print("\nParameter grid (focused for interpretability):")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # Calculate total combinations
    combinations_per_seed = np.prod([len(v) for v in param_grid.values()])
    total_combinations = combinations_per_seed * n_seeds
    print(f"\nCombinations per seed: {combinations_per_seed}")
    print(f"Number of seeds: {n_seeds}")
    print(f"Total models to train: {total_combinations}")

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

    # Create experiment suffix for filenames
    exp_suffix = ""
    if experiments is not None and len(experiments) > 0:
        exp_suffix = "_" + "_".join(experiments)

    # Multi-seed grid search
    print("\n" + "="*80)
    print("STARTING MULTI-SEED GRID SEARCH")
    print("="*80)

    all_results = []
    start_time = datetime.now()

    # Generate all parameter combinations
    param_combinations = list(product(
        param_grid['max_depth'],
        param_grid['min_samples_split'],
        param_grid['min_samples_leaf'],
        param_grid['class_weight_factor']
    ))

    # Progress bar for all models
    pbar = tqdm(total=total_combinations, desc="Grid search (all seeds)")

    for seed in range(n_seeds):
        print(f"\n{'='*80}")
        print(f"SEED {seed}/{n_seeds-1}")
        print(f"{'='*80}")

        # Train/test split with this seed
        n_train = int(len(session_dirs) * (1 - test_fraction))
        random.seed(seed)
        shuffled_dirs = session_dirs.copy()
        random.shuffle(shuffled_dirs)

        train_sessions = shuffled_dirs[:n_train]
        test_sessions = shuffled_dirs[n_train:]

        print(f"Train sessions: {len(train_sessions)}")
        print(f"Test sessions: {len(test_sessions)}")

        # Create datasets
        X_train, y_train = create_dataset(train_sessions)
        X_test, y_test = create_dataset(test_sessions)

        print(f"Train samples: {len(X_train)} (Keep: {y_train.sum()}, {y_train.sum()/len(y_train)*100:.1f}%)")
        print(f"Test samples: {len(X_test)} (Keep: {y_test.sum()}, {y_test.sum()/len(y_test)*100:.1f}%)")

        # Train all parameter combinations with this seed
        for idx, (max_depth, min_split, min_leaf, class_weight) in enumerate(param_combinations):
            params = {
                'max_depth': max_depth,
                'min_samples_split': min_split,
                'min_samples_leaf': min_leaf,
                'class_weight_factor': class_weight,
                'random_state': seed
            }

            # Train and evaluate
            model, metrics = train_and_evaluate(X_train, y_train, X_test, y_test, params)

            # Save model with seed and experiment suffix
            model_filename = f"dt{exp_suffix}_seed{seed}_d{max_depth}_s{min_split}_l{min_leaf}_w{class_weight:.1f}.pkl"
            model_path = output_path / model_filename
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # Record results
            result = {
                'seed': seed,
                'model_id': seed * len(param_combinations) + idx,
                'model_file': model_filename,
                'max_depth': max_depth,
                'min_samples_split': min_split,
                'min_samples_leaf': min_leaf,
                'class_weight_factor': class_weight,
                **metrics
            }
            all_results.append(result)

            pbar.update(1)

    pbar.close()

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = Path(f"ml/grid_search_focused_results{exp_suffix}.csv")
    results_df.to_csv(results_path, index=False)

    elapsed_time = datetime.now() - start_time

    # Print summary
    print("\n" + "="*80)
    print("MULTI-SEED GRID SEARCH COMPLETE")
    print("="*80)
    print(f"\nElapsed time: {elapsed_time}")
    print(f"Total models trained: {len(results_df)}")
    print(f"Results saved to: {results_path}")
    print(f"Models saved to: {output_path}/")

    # Aggregate results across seeds
    print("\n" + "="*80)
    print("AGGREGATED RESULTS (averaged across seeds)")
    print("="*80)

    # Group by hyperparameters (excluding seed)
    grouped = results_df.groupby(['max_depth', 'min_samples_split',
                                   'min_samples_leaf', 'class_weight_factor'])

    agg_results = grouped.agg({
        'test_precision': ['mean', 'std'],
        'test_recall': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
        'n_leaves': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    agg_results.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                           for col in agg_results.columns.values]

    # Save aggregated results
    agg_path = Path(f"ml/grid_search_focused_aggregated{exp_suffix}.csv")
    agg_results.to_csv(agg_path, index=False)

    # Best by F1 (mean)
    print("\nTOP 10 CONFIGURATIONS BY MEAN TEST F1 SCORE")
    print("="*80)

    top_f1 = agg_results.nlargest(10, 'test_f1_mean')
    print("\n" + top_f1[['max_depth', 'min_samples_split', 'min_samples_leaf',
                          'class_weight_factor', 'test_precision_mean', 'test_precision_std',
                          'test_recall_mean', 'test_recall_std',
                          'test_f1_mean', 'test_f1_std',
                          'n_leaves_mean']].to_string(index=False))

    # Most interpretable (fewest leaves) with good performance
    print("\n" + "="*80)
    print("MOST INTERPRETABLE MODELS (fewest leaves, F1 >= 90%)")
    print("="*80)

    interpretable = agg_results[agg_results['test_f1_mean'] >= 0.90]
    if not interpretable.empty:
        interpretable_sorted = interpretable.nsmallest(10, 'n_leaves_mean')
        print("\n" + interpretable_sorted[['max_depth', 'min_samples_split', 'min_samples_leaf',
                                            'class_weight_factor', 'test_precision_mean',
                                            'test_recall_mean', 'test_f1_mean',
                                            'n_leaves_mean']].to_string(index=False))
    else:
        print("\nNo configurations achieved F1 >= 90%")

    # Best by precision
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS BY MEAN TEST PRECISION")
    print("="*80)

    top_prec = agg_results.nlargest(10, 'test_precision_mean')
    print("\n" + top_prec[['max_depth', 'min_samples_split', 'min_samples_leaf',
                            'class_weight_factor', 'test_precision_mean', 'test_precision_std',
                            'test_recall_mean', 'test_f1_mean',
                            'n_leaves_mean']].to_string(index=False))

    print(f"\n\nAggregated results saved to: {agg_path}")
    print("="*80)

    return results_df, agg_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Focused grid search for interpretable trees")
    parser.add_argument("--artifacts-dir", default="data/capcan_validation_127",
                       help="Directory containing capcan_artifacts_* subdirectories")
    parser.add_argument("--output-dir", default="ml/grid_search_focused",
                       help="Directory to save models")
    parser.add_argument("--experiments", type=str, nargs='+', default=None,
                       help="Filter to specific experiments (e.g., --experiments NOF RFC)")
    parser.add_argument("--n-seeds", type=int, default=5,
                       help="Number of random seeds for robustness (default: 5)")

    args = parser.parse_args()

    results, agg_results = grid_search_focused(
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        experiments=args.experiments,
        n_seeds=args.n_seeds
    )
