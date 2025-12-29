"""
EBM Improvement Experiments: Test modern InterpretML v0.5.1+ parameters.

Key untested parameters:
- greedy_ratio (10.0 default): Greedy boosting for better feature selection
- smoothing_rounds (75 default): Post-training smoothing for generalization
- max_bins=1024: Higher resolution shape functions
- outer_bags=14: More ensemble stability
- interactions='3x': Automatic interaction detection

Compares new configurations against current best model.
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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from joblib import Parallel, delayed
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import load_session_data, create_dataset, FEATURE_COLS, load_all_sessions

warnings.filterwarnings('ignore')


def train_and_evaluate_ebm(params, X_train, y_train, X_test, y_test, thresholds, config_name):
    """Train a single EBM model with modern parameters."""
    from interpret.glassbox import ExplainableBoostingClassifier

    try:
        # Build EBM with all parameters
        ebm_params = {
            'feature_names': list(X_train.columns),
            'max_bins': params.get('max_bins', 1024),
            'max_interaction_bins': params.get('max_interaction_bins', 64),
            'interactions': params.get('interactions', 0),
            'outer_bags': params.get('outer_bags', 14),
            'inner_bags': params.get('inner_bags', 0),
            'learning_rate': params.get('learning_rate', 0.015),
            'greedy_ratio': params.get('greedy_ratio', 10.0),
            'smoothing_rounds': params.get('smoothing_rounds', 75),
            'interaction_smoothing_rounds': params.get('interaction_smoothing_rounds', 75),
            'max_rounds': params.get('max_rounds', 50000),
            'early_stopping_rounds': params.get('early_stopping_rounds', 100),
            'early_stopping_tolerance': params.get('early_stopping_tolerance', 1e-5),
            'min_samples_leaf': params.get('min_samples_leaf', 4),
            'max_leaves': params.get('max_leaves', 2),
            'validation_size': params.get('validation_size', 0.15),
            'random_state': params.get('random_state', 42),
        }

        ebm = ExplainableBoostingClassifier(**ebm_params)

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

            train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
                y_train, y_train_pred, average='binary', zero_division=0
            )
            test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
                y_test, y_test_pred, average='binary', zero_division=0
            )

            result = {
                'config_name': config_name,
                **{k: v for k, v in params.items() if k != 'random_state'},
                'threshold': thresh,
                'train_precision': train_prec,
                'train_recall': train_rec,
                'train_f1': train_f1,
                'train_auc': train_auc,
                'test_precision': test_prec,
                'test_recall': test_rec,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'train_time_sec': train_time,
            }
            results.append(result)

        return results, ebm

    except Exception as e:
        print(f"  ERROR with {config_name}: {e}")
        return None, None


def run_improvement_experiments(
    artifacts_dir="data/capcan_validation_127_v5",
    output_dir="ml/ebm_improvement",
    experiments=None,
    test_fraction=0.25,
    n_jobs=4,
    random_state=42
):
    """Run EBM improvement experiments with modern parameters."""
    print("=" * 80)
    print("EBM IMPROVEMENT EXPERIMENTS (v0.5.1+ Parameters)")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load sessions
    artifacts_path = Path(artifacts_dir)
    session_dirs = sorted([d for d in artifacts_path.iterdir()
                          if d.is_dir() and d.name.startswith('capcan_artifacts_')])

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

    exp_suffix = ""
    if experiments is not None and len(experiments) > 0:
        exp_suffix = "_" + "_".join(experiments)

    # Train/test split
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

    # Define experiment configurations
    configs = {
        # Baseline: our current best settings
        'baseline_current': {
            'max_bins': 256,
            'interactions': 20,
            'outer_bags': 8,
            'learning_rate': 0.01,
            'greedy_ratio': 0.0,  # Disabled in current
            'smoothing_rounds': 0,  # Disabled in current
            'interaction_smoothing_rounds': 0,
            'max_rounds': 5000,
            'early_stopping_rounds': 50,
            'min_samples_leaf': 2,
            'max_leaves': 3,
        },

        # Experiment 1: v0.5.1 defaults (full modern settings)
        'modern_defaults': {
            'max_bins': 1024,
            'interactions': 20,  # Keep interactions comparable
            'outer_bags': 14,
            'learning_rate': 0.015,
            'greedy_ratio': 10.0,
            'smoothing_rounds': 75,
            'interaction_smoothing_rounds': 75,
            'max_rounds': 50000,
            'early_stopping_rounds': 100,
            'min_samples_leaf': 4,
            'max_leaves': 2,
        },

        # Experiment 2: Greedy only (isolate greedy impact)
        'greedy_only': {
            'max_bins': 256,
            'interactions': 20,
            'outer_bags': 8,
            'learning_rate': 0.01,
            'greedy_ratio': 10.0,  # NEW
            'smoothing_rounds': 0,
            'interaction_smoothing_rounds': 0,
            'max_rounds': 5000,
            'early_stopping_rounds': 50,
            'min_samples_leaf': 2,
            'max_leaves': 3,
        },

        # Experiment 3: Smoothing only (isolate smoothing impact)
        'smoothing_only': {
            'max_bins': 256,
            'interactions': 20,
            'outer_bags': 8,
            'learning_rate': 0.01,
            'greedy_ratio': 0.0,
            'smoothing_rounds': 75,  # NEW
            'interaction_smoothing_rounds': 75,  # NEW
            'max_rounds': 5000,
            'early_stopping_rounds': 50,
            'min_samples_leaf': 2,
            'max_leaves': 3,
        },

        # Experiment 4: Greedy + Smoothing combined
        'greedy_smoothing': {
            'max_bins': 256,
            'interactions': 20,
            'outer_bags': 8,
            'learning_rate': 0.01,
            'greedy_ratio': 10.0,  # NEW
            'smoothing_rounds': 75,  # NEW
            'interaction_smoothing_rounds': 75,  # NEW
            'max_rounds': 5000,
            'early_stopping_rounds': 50,
            'min_samples_leaf': 2,
            'max_leaves': 3,
        },

        # Experiment 5: Higher bins (1024)
        'high_bins': {
            'max_bins': 1024,  # NEW
            'interactions': 20,
            'outer_bags': 8,
            'learning_rate': 0.01,
            'greedy_ratio': 0.0,
            'smoothing_rounds': 0,
            'interaction_smoothing_rounds': 0,
            'max_rounds': 5000,
            'early_stopping_rounds': 50,
            'min_samples_leaf': 2,
            'max_leaves': 3,
        },

        # Experiment 6: More bags (14)
        'more_bags': {
            'max_bins': 256,
            'interactions': 20,
            'outer_bags': 14,  # NEW
            'learning_rate': 0.01,
            'greedy_ratio': 0.0,
            'smoothing_rounds': 0,
            'interaction_smoothing_rounds': 0,
            'max_rounds': 5000,
            'early_stopping_rounds': 50,
            'min_samples_leaf': 2,
            'max_leaves': 3,
        },

        # Experiment 7: Greedy + Smoothing + High bins (best combo hypothesis)
        'greedy_smooth_highbins': {
            'max_bins': 1024,
            'interactions': 20,
            'outer_bags': 14,
            'learning_rate': 0.015,
            'greedy_ratio': 10.0,
            'smoothing_rounds': 75,
            'interaction_smoothing_rounds': 75,
            'max_rounds': 50000,
            'early_stopping_rounds': 100,
            'min_samples_leaf': 2,  # Keep lower for less regularization
            'max_leaves': 3,
        },

        # Experiment 8: Higher greedy ratio
        'high_greedy': {
            'max_bins': 256,
            'interactions': 20,
            'outer_bags': 8,
            'learning_rate': 0.01,
            'greedy_ratio': 20.0,  # Higher than default
            'smoothing_rounds': 75,
            'interaction_smoothing_rounds': 75,
            'max_rounds': 5000,
            'early_stopping_rounds': 50,
            'min_samples_leaf': 2,
            'max_leaves': 3,
        },

        # Experiment 9: More smoothing
        'more_smoothing': {
            'max_bins': 256,
            'interactions': 20,
            'outer_bags': 8,
            'learning_rate': 0.01,
            'greedy_ratio': 10.0,
            'smoothing_rounds': 150,  # Double
            'interaction_smoothing_rounds': 150,
            'max_rounds': 5000,
            'early_stopping_rounds': 50,
            'min_samples_leaf': 2,
            'max_leaves': 3,
        },
    }

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"\nRunning {len(configs)} configurations...")
    print(f"Thresholds: {thresholds}")
    print("-" * 80)

    # Run experiments
    all_results = []
    all_models = {}

    for config_name, params in configs.items():
        print(f"\n[{config_name}]")
        print(f"  greedy_ratio={params.get('greedy_ratio', 0)}, "
              f"smoothing_rounds={params.get('smoothing_rounds', 0)}, "
              f"max_bins={params.get('max_bins', 256)}")

        results, model = train_and_evaluate_ebm(
            params, X_train, y_train, X_test, y_test, thresholds, config_name
        )

        if results is not None:
            all_results.extend(results)
            all_models[config_name] = model

            # Print threshold=0.5 result
            t05 = [r for r in results if r['threshold'] == 0.5][0]
            print(f"  -> F1={t05['test_f1']:.4f}, P={t05['test_precision']:.4f}, "
                  f"R={t05['test_recall']:.4f} (train: {t05['train_time_sec']:.1f}s)")

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    results_path = output_path / f"improvement_results{exp_suffix}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Analysis
    print("\n" + "=" * 80)
    print("IMPROVEMENT EXPERIMENT RESULTS (threshold=0.5)")
    print("=" * 80)

    df_t05 = results_df[results_df['threshold'] == 0.5].copy()
    df_t05 = df_t05.sort_values('test_f1', ascending=False)

    print("\nRanked by F1 Score:")
    print("-" * 80)
    print(f"{'Config':<25} {'F1':>8} {'Prec':>8} {'Rec':>8} {'AUC':>8} {'Time':>8}")
    print("-" * 80)

    baseline_f1 = df_t05[df_t05['config_name'] == 'baseline_current']['test_f1'].values[0]

    for _, row in df_t05.iterrows():
        delta = row['test_f1'] - baseline_f1
        delta_str = f"+{delta*100:.2f}%" if delta > 0 else f"{delta*100:.2f}%"
        print(f"{row['config_name']:<25} {row['test_f1']:>8.4f} {row['test_precision']:>8.4f} "
              f"{row['test_recall']:>8.4f} {row['test_auc']:>8.4f} {row['train_time_sec']:>7.1f}s")
        if row['config_name'] != 'baseline_current':
            print(f"{'':>25} {delta_str:>8}")

    # Save best models
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)

    best_config = df_t05.iloc[0]['config_name']
    best_model = all_models.get(best_config)
    if best_model:
        best_path = output_path / f"ebm_best_improved{exp_suffix}.pkl"
        with open(best_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Best model ({best_config}): {best_path}")

    # Save all models
    for config_name, model in all_models.items():
        model_path = output_path / f"ebm_{config_name}{exp_suffix}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved: {model_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return results_df, all_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EBM Improvement Experiments")
    parser.add_argument("--artifacts-dir", default="data/capcan_validation_127_v5",
                       help="Directory containing capcan_artifacts_* subdirectories")
    parser.add_argument("--output-dir", default="ml/ebm_improvement",
                       help="Directory to save results")
    parser.add_argument("--experiments", type=str, nargs='+', default=None,
                       help="Filter to specific experiments (e.g., --experiments NOF RFC FOF)")
    parser.add_argument("--n-jobs", type=int, default=4,
                       help="Number of parallel jobs (default: 4)")
    parser.add_argument("--test-fraction", type=float, default=0.25,
                       help="Fraction of data for testing (default: 0.25)")

    args = parser.parse_args()

    results, models = run_improvement_experiments(
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        experiments=args.experiments,
        n_jobs=args.n_jobs,
        test_fraction=args.test_fraction
    )
