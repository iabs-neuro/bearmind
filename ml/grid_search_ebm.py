"""
Grid search for EBM (Explainable Boosting Machine) hyperparameters.

v2: Updated with modern InterpretML v0.5.1+ parameters:
    - greedy_ratio: Greedy boosting for better feature selection
    - smoothing_rounds: Post-training smoothing for generalization

Goals:
- Find max performance configurations
- Find simple interpretable models with high precision
- Explore precision-recall trade-offs via threshold tuning
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
import time
from pathlib import Path
from itertools import product
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_auc_score,
    fbeta_score
)
from joblib import Parallel, delayed
import argparse

# Import from ml/data_utils
sys.path.insert(0, str(Path(__file__).parent))
from data_utils import (
    load_session_data, create_dataset, FEATURE_COLS, load_all_sessions,
    stratified_session_split, print_split_info, FBETA_BETA
)

warnings.filterwarnings('ignore')


def train_and_evaluate_ebm(params, X_train, y_train, X_test, y_test, thresholds):
    """Train a single EBM model with modern parameters and evaluate at multiple thresholds."""
    from interpret.glassbox import ExplainableBoostingClassifier

    try:
        # Modern EBM parameters (v0.5.1+)
        ebm = ExplainableBoostingClassifier(
            feature_names=list(X_train.columns),
            max_bins=params['max_bins'],
            max_interaction_bins=min(64, params['max_bins'] // 4),
            interactions=params['interactions'],
            outer_bags=params.get('outer_bags', 8),
            inner_bags=0,
            learning_rate=params.get('learning_rate', 0.01),
            greedy_ratio=params.get('greedy_ratio', 0.0),
            smoothing_rounds=params.get('smoothing_rounds', 0),
            interaction_smoothing_rounds=params.get('smoothing_rounds', 0),
            validation_size=0.15,
            early_stopping_rounds=params.get('early_stopping_rounds', 50),
            early_stopping_tolerance=1e-4,
            max_rounds=params.get('max_rounds', 5000),
            min_samples_leaf=params['min_samples_leaf'],
            max_leaves=params['max_leaves'],
            random_state=params['random_state']
        )

        t_start = time.time()
        ebm.fit(X_train, y_train)
        train_time = time.time() - t_start

        # Get probabilities
        y_train_proba = ebm.predict_proba(X_train)[:, 1]
        y_test_proba = ebm.predict_proba(X_test)[:, 1]

        # Calculate ROC AUC
        train_auc = roc_auc_score(y_train, y_train_proba)
        test_auc = roc_auc_score(y_test, y_test_proba)

        # Evaluate at multiple thresholds
        results = []
        for thresh in thresholds:
            y_train_pred = (y_train_proba >= thresh).astype(int)
            y_test_pred = (y_test_proba >= thresh).astype(int)

            train_prec, train_rec, _, _ = precision_recall_fscore_support(
                y_train, y_train_pred, average='binary', zero_division=0
            )
            test_prec, test_rec, _, _ = precision_recall_fscore_support(
                y_test, y_test_pred, average='binary', zero_division=0
            )
            train_fbeta = fbeta_score(y_train, y_train_pred, beta=FBETA_BETA,
                                      average='binary', zero_division=0)
            test_fbeta = fbeta_score(y_test, y_test_pred, beta=FBETA_BETA,
                                     average='binary', zero_division=0)

            result = {
                **params,
                'threshold': thresh,
                'train_precision': train_prec,
                'train_recall': train_rec,
                'train_fbeta': train_fbeta,
                'train_auc': train_auc,
                'test_precision': test_prec,
                'test_recall': test_rec,
                'test_fbeta': test_fbeta,
                'test_auc': test_auc,
                'train_time_sec': train_time,
            }
            results.append(result)

        return results, ebm

    except Exception as e:
        print(f"  ERROR with params {params}: {e}")
        return None, None


def run_grid_search(
    artifacts_dir="data/capcan_validation_127_v2",
    output_dir="ml/ebm_grid_search",
    experiments=None,
    test_fraction=0.25,
    n_jobs=4,
    random_state=42
):
    """Run EBM grid search."""
    import sys
    print("=" * 80, flush=True)
    print("EBM GRID SEARCH", flush=True)
    print("=" * 80, flush=True)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load sessions
    artifacts_path = Path(artifacts_dir)
    session_dirs = sorted([d for d in artifacts_path.iterdir()
                          if d.is_dir() and d.name.startswith('capcan_artifacts_')])

    # Filter by experiments
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

    # Create experiment suffix
    exp_suffix = ""
    if experiments is not None and len(experiments) > 0:
        exp_suffix = "_" + "_".join(experiments)

    # Stratified train/test split by experiment
    train_sessions, test_sessions, split_info = stratified_session_split(
        session_dirs,
        test_fraction=test_fraction,
        random_state=random_state
    )

    print()
    print_split_info(split_info)

    # Create datasets
    print("\nLoading data...")
    X_train, y_train = create_dataset(train_sessions)
    X_test, y_test = create_dataset(test_sessions)

    print(f"Train samples: {len(X_train)} (KEEP: {y_train.sum()}, {100*y_train.mean():.1f}%)")
    print(f"Test samples: {len(X_test)} (KEEP: {y_test.sum()}, {100*y_test.mean():.1f}%)")

    # Define parameter grid - v2 with modern EBM parameters
    # Key parameters identified from improvement experiments:
    # - greedy_ratio: enables greedy boosting (major improvement)
    # - smoothing_rounds: post-training smoothing (major improvement)
    # - max_bins: shape function resolution
    # - min_samples_leaf, max_leaves: regularization
    param_grid = {
        # Shape function complexity
        'max_bins': [256, 1024],               # 2 - low vs high resolution
        # Interaction complexity
        'interactions': [0, 20],               # 2 - no interactions vs moderate
        # Modern EBM parameters (v0.5.1+)
        'greedy_ratio': [0.0, 10.0],           # 2 - disabled vs enabled
        'smoothing_rounds': [0, 75],           # 2 - disabled vs enabled
        # Regularization
        'min_samples_leaf': [2, 4],            # 2 - less vs more regularization
        'max_leaves': [2, 3],                  # 2 - simpler vs more complex
        # Fixed parameters
        'outer_bags': [8],                     # ensemble stability
        'learning_rate': [0.01],               # convergence speed
        'random_state': [42],                  # single seed
    }

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_params = [dict(zip(param_names, v)) for v in product(*param_values)]

    print(f"\nTotal model configurations: {len(all_params)}")

    # Thresholds to evaluate (for precision-recall trade-off)
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"Thresholds to evaluate: {thresholds}")
    print(f"Total evaluations: {len(all_params) * len(thresholds)}")

    # Run grid search in parallel
    print(f"\nRunning grid search with {n_jobs} parallel jobs...")
    estimated_time = len(all_params) * 25 / n_jobs / 60  # ~25s per model
    print(f"Estimated time: {estimated_time:.0f}-{estimated_time*2:.0f} minutes\n")

    def train_single(params, idx):
        print(f"  [{idx+1}/{len(all_params)}] bins={params['max_bins']}, "
              f"inter={params['interactions']}, greedy={params.get('greedy_ratio', 0)}, "
              f"smooth={params.get('smoothing_rounds', 0)}, leaf={params['min_samples_leaf']}, "
              f"leaves={params['max_leaves']}, seed={params['random_state']}")
        results, model = train_and_evaluate_ebm(
            params, X_train, y_train, X_test, y_test, thresholds
        )
        return results, model, params

    # Run in parallel
    parallel_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(train_single)(params, idx) for idx, params in enumerate(all_params)
    )

    # Collect results
    all_results = []
    best_models = {}

    for results, model, params in parallel_results:
        if results is not None:
            all_results.extend(results)

            # Track best model by F-beta at threshold 0.5
            for r in results:
                if r['threshold'] == 0.5:
                    key = f"bins{params['max_bins']}_inter{params['interactions']}"
                    if key not in best_models or r['test_fbeta'] > best_models[key]['fbeta']:
                        best_models[key] = {'fbeta': r['test_fbeta'], 'model': model, 'params': params}

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save full results
    results_path = output_path / f"ebm_grid_search{exp_suffix}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nFull results saved to: {results_path}")

    # Analysis
    print("\n" + "=" * 80)
    print("GRID SEARCH RESULTS ANALYSIS")
    print("=" * 80)

    # Aggregate across seeds for robust comparison
    df_t05 = results_df[results_df['threshold'] == 0.5].copy()

    # Group by hyperparameters (exclude random_state)
    group_cols = ['max_bins', 'interactions', 'greedy_ratio', 'smoothing_rounds',
                  'min_samples_leaf', 'max_leaves', 'threshold']
    agg_cols = ['test_fbeta', 'test_precision', 'test_recall', 'test_auc']

    df_agg = df_t05.groupby(
        [c for c in group_cols if c in df_t05.columns]
    ).agg({
        'test_fbeta': ['mean', 'std'],
        'test_precision': ['mean', 'std'],
        'test_recall': ['mean', 'std'],
        'test_auc': ['mean', 'std'],
    }).reset_index()
    df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns]

    # Best by F-beta
    print(f"\n--- TOP 10 BY F-BETA (β={FBETA_BETA:.3f}, threshold=0.5) ---")
    best_fbeta_agg = df_agg.nlargest(10, 'test_fbeta_mean')[
        ['max_bins', 'interactions', 'greedy_ratio', 'smoothing_rounds',
         'min_samples_leaf', 'max_leaves',
         'test_precision_mean', 'test_recall_mean', 'test_fbeta_mean', 'test_fbeta_std']
    ]
    print(best_fbeta_agg.to_string(index=False))

    # Effect of greedy_ratio
    print(f"\n--- EFFECT OF GREEDY_RATIO (mean F-beta across all configs) ---")
    greedy_effect = df_agg.groupby('greedy_ratio')['test_fbeta_mean'].agg(['mean', 'std', 'max'])
    print(greedy_effect.to_string())

    # Effect of smoothing_rounds
    print(f"\n--- EFFECT OF SMOOTHING_ROUNDS (mean F-beta across all configs) ---")
    smooth_effect = df_agg.groupby('smoothing_rounds')['test_fbeta_mean'].agg(['mean', 'std', 'max'])
    print(smooth_effect.to_string())

    # Best with greedy+smoothing enabled
    print("\n--- BEST CONFIGS WITH GREEDY+SMOOTHING ENABLED ---")
    modern = df_agg[(df_agg['greedy_ratio'] > 0) & (df_agg['smoothing_rounds'] > 0)]
    if len(modern) > 0:
        best_modern = modern.nlargest(5, 'test_fbeta_mean')[
            ['max_bins', 'interactions', 'min_samples_leaf', 'max_leaves',
             'test_fbeta_mean', 'test_fbeta_std']
        ]
        print(best_modern.to_string(index=False))

    # Best simple models (no interactions)
    print("\n--- BEST SIMPLE MODELS (interactions=0) ---")
    simple = df_agg[df_agg['interactions'] == 0]
    if len(simple) > 0:
        best_simple = simple.nlargest(5, 'test_fbeta_mean')[
            ['max_bins', 'greedy_ratio', 'smoothing_rounds',
             'min_samples_leaf', 'max_leaves', 'test_fbeta_mean', 'test_fbeta_std']
        ]
        print(best_simple.to_string(index=False))

    # Precision-recall trade-off for best config
    print(f"\n--- PRECISION-RECALL TRADE-OFF (best mean F-beta config, all thresholds) ---")
    best_config = df_agg.loc[df_agg['test_fbeta_mean'].idxmax()]
    mask = (
        (results_df['max_bins'] == best_config['max_bins']) &
        (results_df['interactions'] == best_config['interactions']) &
        (results_df['greedy_ratio'] == best_config['greedy_ratio']) &
        (results_df['smoothing_rounds'] == best_config['smoothing_rounds']) &
        (results_df['min_samples_leaf'] == best_config['min_samples_leaf']) &
        (results_df['max_leaves'] == best_config['max_leaves'])
    )
    tradeoff = results_df[mask].groupby('threshold').agg({
        'test_precision': 'mean',
        'test_recall': 'mean',
        'test_fbeta': 'mean'
    }).reset_index()
    print(tradeoff.to_string(index=False))

    # Save aggregated results
    agg_path = output_path / f"ebm_grid_search{exp_suffix}_aggregated.csv"
    df_agg.to_csv(agg_path, index=False)
    print(f"\nAggregated results saved to: {agg_path}")

    # Save best models
    print("\n" + "=" * 80)
    print("SAVING BEST MODELS")
    print("=" * 80)

    # Find best model based on mean F-beta across seeds
    # Group models by config (excluding seed)
    config_models = {}
    for results, model, params in parallel_results:
        if results is not None and model is not None:
            # Create config key (exclude random_state)
            config_key = (
                params['max_bins'],
                params['interactions'],
                params.get('greedy_ratio', 0),
                params.get('smoothing_rounds', 0),
                params['min_samples_leaf'],
                params['max_leaves']
            )
            if config_key not in config_models:
                config_models[config_key] = {'models': [], 'fbeta_scores': [], 'params': params}

            # Get F-beta at threshold 0.5
            for r in results:
                if r['threshold'] == 0.5:
                    config_models[config_key]['models'].append(model)
                    config_models[config_key]['fbeta_scores'].append(r['test_fbeta'])
                    break

    # Find config with best mean F-beta
    best_config_key = max(config_models.keys(),
                          key=lambda k: np.mean(config_models[k]['fbeta_scores']))
    best_config_data = config_models[best_config_key]
    best_mean_fbeta = np.mean(best_config_data['fbeta_scores'])
    best_std_fbeta = np.std(best_config_data['fbeta_scores'])

    # Save the model with median F-beta for this config (most representative)
    fbeta_scores = best_config_data['fbeta_scores']
    median_idx = np.argsort(fbeta_scores)[len(fbeta_scores) // 2]
    best_model = best_config_data['models'][median_idx]
    best_params = best_config_data['params']

    best_model_path = output_path / f"ebm_best{exp_suffix}.pkl"
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"Best overall model saved: {best_model_path}")
    print(f"  Config: bins={best_params['max_bins']}, inter={best_params['interactions']}, "
          f"greedy={best_params.get('greedy_ratio', 0)}, smooth={best_params.get('smoothing_rounds', 0)}, "
          f"leaf={best_params['min_samples_leaf']}, leaves={best_params['max_leaves']}")
    print(f"  Test F-beta (β={FBETA_BETA:.3f}): {best_mean_fbeta:.4f}")

    # Save simplest competitive model (no interactions)
    simple_configs = {k: v for k, v in config_models.items() if k[1] == 0}  # k[1] is interactions
    if simple_configs:
        best_simple_key = max(simple_configs.keys(),
                              key=lambda k: np.mean(simple_configs[k]['fbeta_scores']))
        best_simple_data = simple_configs[best_simple_key]
        simple_mean_fbeta = np.mean(best_simple_data['fbeta_scores'])
        simple_std_fbeta = np.std(best_simple_data['fbeta_scores'])

        # Get median model
        fbeta_scores = best_simple_data['fbeta_scores']
        median_idx = np.argsort(fbeta_scores)[len(fbeta_scores) // 2]
        simple_model = best_simple_data['models'][median_idx]
        simple_params = best_simple_data['params']

        simple_model_path = output_path / f"ebm_simple{exp_suffix}.pkl"
        with open(simple_model_path, 'wb') as f:
            pickle.dump(simple_model, f)

        print(f"\nSimplest good model saved: {simple_model_path}")
        print(f"  Config: bins={simple_params['max_bins']}, inter=0, "
              f"greedy={simple_params.get('greedy_ratio', 0)}, smooth={simple_params.get('smoothing_rounds', 0)}, "
              f"leaf={simple_params['min_samples_leaf']}, leaves={simple_params['max_leaves']}")
        print(f"  Test F-beta (β={FBETA_BETA:.3f}): {simple_mean_fbeta:.4f}")

    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EBM Grid Search")
    parser.add_argument("--artifacts-dir", default="data/capcan_validation_127_v2",
                       help="Directory containing capcan_artifacts_* subdirectories")
    parser.add_argument("--output-dir", default="ml/ebm_grid_search",
                       help="Directory to save results")
    parser.add_argument("--experiments", type=str, nargs='+', default=None,
                       help="Filter to specific experiments (e.g., --experiments NOF RFC)")
    parser.add_argument("--n-jobs", type=int, default=4,
                       help="Number of parallel jobs (default: 4)")
    parser.add_argument("--test-fraction", type=float, default=0.25,
                       help="Fraction of data for testing (default: 0.25)")

    args = parser.parse_args()

    results = run_grid_search(
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        experiments=args.experiments,
        n_jobs=args.n_jobs,
        test_fraction=args.test_fraction
    )
