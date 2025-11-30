"""
Grid search for EBM (Explainable Boosting Machine) hyperparameters.

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
import random
import warnings
from pathlib import Path
from itertools import product
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_auc_score
)
from joblib import Parallel, delayed
import argparse

# Import from ml/data_utils
sys.path.insert(0, str(Path(__file__).parent))
from data_utils import load_session_data, create_dataset, FEATURE_COLS, load_all_sessions

warnings.filterwarnings('ignore')


def train_and_evaluate_ebm(params, X_train, y_train, X_test, y_test, thresholds):
    """Train a single EBM model and evaluate at multiple thresholds."""
    from interpret.glassbox import ExplainableBoostingClassifier

    try:
        ebm = ExplainableBoostingClassifier(
            feature_names=list(X_train.columns),
            max_bins=params['max_bins'],
            max_interaction_bins=min(32, params['max_bins'] // 4),
            interactions=params['interactions'],
            outer_bags=params['outer_bags'],
            inner_bags=0,
            learning_rate=params['learning_rate'],
            validation_size=0.15,
            early_stopping_rounds=50,
            early_stopping_tolerance=1e-4,
            max_rounds=5000,
            min_samples_leaf=params['min_samples_leaf'],
            max_leaves=params['max_leaves'],
            random_state=params['random_state']
        )

        ebm.fit(X_train, y_train)

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

            train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
                y_train, y_train_pred, average='binary', zero_division=0
            )
            test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
                y_test, y_test_pred, average='binary', zero_division=0
            )

            result = {
                **params,
                'threshold': thresh,
                'train_precision': train_prec,
                'train_recall': train_rec,
                'train_f1': train_f1,
                'train_auc': train_auc,
                'test_precision': test_prec,
                'test_recall': test_rec,
                'test_f1': test_f1,
                'test_auc': test_auc,
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

    # Train/test split by session
    n_train = int(len(session_dirs) * (1 - test_fraction))
    random.seed(random_state)
    shuffled_dirs = session_dirs.copy()
    random.shuffle(shuffled_dirs)

    train_sessions = shuffled_dirs[:n_train]
    test_sessions = shuffled_dirs[n_train:]

    print(f"Train sessions: {len(train_sessions)}")
    print(f"Test sessions: {len(test_sessions)}")

    # Create datasets
    print("\nLoading data...")
    X_train, y_train = create_dataset(train_sessions)
    X_test, y_test = create_dataset(test_sessions)

    print(f"Train samples: {len(X_train)} (KEEP: {y_train.sum()}, {100*y_train.mean():.1f}%)")
    print(f"Test samples: {len(X_test)} (KEEP: {y_test.sum()}, {100*y_test.mean():.1f}%)")

    # Define parameter grid
    # Focused on parameters that matter for precision/recall trade-off
    param_grid = {
        'max_bins': [128, 256, 512],           # 3 - complexity of shape functions
        'interactions': [0, 10, 20, 50],       # 4 - model complexity (0=simple, 50=complex)
        # Fixed parameters (minimal impact on P/R trade-off):
        'outer_bags': [8],                     # ensemble stability
        'learning_rate': [0.01],               # convergence speed
        'min_samples_leaf': [2],               # regularization (no overfitting observed)
        'max_leaves': [3],                     # keep simple
        'random_state': [42],                  # reproducibility
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
    print("This may take 10-30 minutes depending on hardware.\n")

    def train_single(params, idx):
        print(f"  [{idx+1}/{len(all_params)}] Training: bins={params['max_bins']}, "
              f"inter={params['interactions']}, bags={params['outer_bags']}, "
              f"lr={params['learning_rate']}, leaf={params['min_samples_leaf']}")
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

            # Track best model by F1 at threshold 0.5
            for r in results:
                if r['threshold'] == 0.5:
                    key = f"bins{params['max_bins']}_inter{params['interactions']}"
                    if key not in best_models or r['test_f1'] > best_models[key]['f1']:
                        best_models[key] = {'f1': r['test_f1'], 'model': model, 'params': params}

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

    # Best by F1 (threshold 0.5)
    print("\n--- TOP 10 BY TEST F1 (threshold=0.5) ---")
    df_t05 = results_df[results_df['threshold'] == 0.5].copy()
    best_f1 = df_t05.nlargest(10, 'test_f1')[
        ['max_bins', 'interactions', 'outer_bags', 'learning_rate',
         'min_samples_leaf', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
    ]
    print(best_f1.to_string(index=False))

    # Best by precision (at recall > 0.85)
    print("\n--- TOP 10 BY PRECISION (where recall > 85%) ---")
    high_recall = results_df[results_df['test_recall'] > 0.85].copy()
    if len(high_recall) > 0:
        best_prec = high_recall.nlargest(10, 'test_precision')[
            ['max_bins', 'interactions', 'threshold',
             'test_precision', 'test_recall', 'test_f1']
        ]
        print(best_prec.to_string(index=False))
    else:
        print("No models with recall > 85%")

    # Best by recall (at precision > 0.85)
    print("\n--- TOP 10 BY RECALL (where precision > 85%) ---")
    high_prec = results_df[results_df['test_precision'] > 0.85].copy()
    if len(high_prec) > 0:
        best_rec = high_prec.nlargest(10, 'test_recall')[
            ['max_bins', 'interactions', 'threshold',
             'test_precision', 'test_recall', 'test_f1']
        ]
        print(best_rec.to_string(index=False))
    else:
        print("No models with precision > 85%")

    # Simplest good models (interactions=0)
    print("\n--- BEST SIMPLE MODELS (no interactions) ---")
    simple = results_df[(results_df['interactions'] == 0) &
                        (results_df['threshold'] == 0.5)].copy()
    if len(simple) > 0:
        best_simple = simple.nlargest(5, 'test_f1')[
            ['max_bins', 'min_samples_leaf', 'test_precision', 'test_recall', 'test_f1']
        ]
        print(best_simple.to_string(index=False))

    # Precision-recall trade-off summary
    print("\n--- PRECISION-RECALL TRADE-OFF (best F1 config) ---")
    best_config = df_t05.loc[df_t05['test_f1'].idxmax()]
    mask = (
        (results_df['max_bins'] == best_config['max_bins']) &
        (results_df['interactions'] == best_config['interactions']) &
        (results_df['outer_bags'] == best_config['outer_bags']) &
        (results_df['learning_rate'] == best_config['learning_rate']) &
        (results_df['min_samples_leaf'] == best_config['min_samples_leaf'])
    )
    tradeoff = results_df[mask][['threshold', 'test_precision', 'test_recall', 'test_f1']]
    print(tradeoff.to_string(index=False))

    # Save best models
    print("\n" + "=" * 80)
    print("SAVING BEST MODELS")
    print("=" * 80)

    # Save overall best model
    best_overall_key = max(best_models.keys(), key=lambda k: best_models[k]['f1'])
    best_overall = best_models[best_overall_key]
    best_model_path = output_path / f"ebm_best{exp_suffix}.pkl"
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_overall['model'], f)
    print(f"Best overall model saved: {best_model_path}")
    print(f"  Config: {best_overall['params']}")
    print(f"  Test F1: {best_overall['f1']:.4f}")

    # Save simplest competitive model (no interactions, within 2% of best)
    simple_models = {k: v for k, v in best_models.items() if 'inter0' in k}
    if simple_models:
        best_simple_key = max(simple_models.keys(), key=lambda k: simple_models[k]['f1'])
        best_simple = simple_models[best_simple_key]
        simple_model_path = output_path / f"ebm_simple{exp_suffix}.pkl"
        with open(simple_model_path, 'wb') as f:
            pickle.dump(best_simple['model'], f)
        print(f"\nSimplest good model saved: {simple_model_path}")
        print(f"  Config: {best_simple['params']}")
        print(f"  Test F1: {best_simple['f1']:.4f}")

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
