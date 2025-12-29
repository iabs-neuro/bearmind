"""
Plot precision-recall tradeoff curves with cross-validation.

Shows mean PR curves with confidence bands across multiple CV splits.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
N_SPLITS = 10
TEST_SIZE = 0.25
FBETA_BETA = 0.577

def get_feature_cols(df):
    """Extract feature columns from dataframe."""
    NON_FEATURE_COLS = {
        'session_name', 'session', 'component_idx', 'ground_truth',
        'decision', 'delete', 'merge', 'experiment',
        'failed_area', 'failed_circularity', 'failed_corner_artifact',
        'is_corner_artifact', 'ml_keep_probability',
        'y_proba', 'y_pred'
    }
    return [col for col in df.columns if col not in NON_FEATURE_COLS]

def evaluate_at_thresholds(model, X, y, thresholds):
    """Evaluate model at multiple thresholds."""
    y_proba = model.predict_proba(X)[:, 1]

    precisions = []
    recalls = []
    fbetas = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        tp = ((y_pred == 1) & (y == 1)).sum()
        fp = ((y_pred == 1) & (y == 0)).sum()
        fn = ((y_pred == 0) & (y == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        fbeta_denom = (FBETA_BETA**2 * precision + recall)
        fbeta = (1 + FBETA_BETA**2) * precision * recall / fbeta_denom if fbeta_denom > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        fbetas.append(fbeta)

    return np.array(precisions), np.array(recalls), np.array(fbetas)

def plot_pr_curves_cv(
    model_paths,
    model_names,
    dataset_paths,
    output_path,
    n_splits=10,
    n_threshold_points=50,
    retrain=False
):
    """
    Plot PR curves with CV confidence bands.

    Parameters
    ----------
    model_paths : list of str
        Paths to model pickle files (or hyperparameters if retrain=True)
    model_names : list of str
        Names for each model
    dataset_paths : str or list of str
        Path(s) to dataset CSV. If single string, used for all models.
        If list, must match length of model_paths (one dataset per model).
    output_path : str
        Output path for plot
    n_splits : int
        Number of CV splits
    n_threshold_points : int
        Number of thresholds to evaluate
    retrain : bool
        If True, retrain model on each CV fold (proper CV, no data leakage)
        If False, use pre-trained models (faster, but has data leakage)
    """
    # Handle dataset_paths as single string or list
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths] * len(model_paths)
    elif len(dataset_paths) != len(model_paths):
        raise ValueError(f'Number of datasets ({len(dataset_paths)}) must match number of models ({len(model_paths)})')

    print('='*80)
    mode = 'PROPER CV (retrain each fold)' if retrain else 'PRE-TRAINED EVALUATION (data leakage)'
    print(f'PR TRADEOFF WITH CROSS-VALIDATION - {mode}')
    print('='*80)
    print(f'\nModels: {len(model_paths)}')
    for name, model_path, dataset_path in zip(model_names, model_paths, dataset_paths):
        print(f'  - {name}: {model_path}')
        print(f'    Dataset: {dataset_path}')
    print(f'\nCV splits: {n_splits}')
    print(f'Threshold points: {n_threshold_points}')
    print(f'Retrain mode: {retrain}')

    # Thresholds to evaluate
    thresholds = np.linspace(0.3, 0.9, n_threshold_points)

    # Store results for each model
    model_results = []

    # Process each model with its respective dataset
    for model_idx, (model_name, model_path, dataset_path) in enumerate(zip(model_names, model_paths, dataset_paths)):
        print(f'\n{"="*80}')
        print(f'Evaluating {model_name}')
        print(f'  Dataset: {dataset_path}')
        print('='*80)

        # Load dataset for this model
        df = pd.read_csv(dataset_path)
        print(f'Loaded: {len(df):,} neurons')

        # Determine session column
        if 'session_name' in df.columns:
            session_col = 'session_name'
        elif 'session' in df.columns:
            session_col = 'session'
        else:
            raise ValueError('No session column found')

        # Get features and labels
        feature_cols = get_feature_cols(df)
        y = df['ground_truth'].values
        sessions = df[session_col].values

        # Get unique sessions and experiments
        unique_sessions = df[session_col].unique()
        if 'experiment' in df.columns:
            session_to_exp = df.groupby(session_col)['experiment'].first().to_dict()
            experiments = np.array([session_to_exp[s] for s in unique_sessions])
        else:
            experiments = np.array([s.split('_')[0] for s in unique_sessions])

        # Load model and get hyperparameters/features
        with open(model_path, 'rb') as f:
            base_model = pickle.load(f)

        model_features = list(base_model.feature_names_in_)
        print(f'Using {len(model_features)} features')

        if retrain:
            # Extract EBM hyperparameters for retraining
            hyperparams = {
                'max_bins': base_model.max_bins,
                'interactions': base_model.interactions,
                'max_leaves': base_model.max_leaves,
                'min_samples_leaf': base_model.min_samples_leaf,
                'outer_bags': base_model.outer_bags,
                'learning_rate': base_model.learning_rate,
                'max_rounds': base_model.max_rounds,
                'early_stopping_rounds': base_model.early_stopping_rounds,
                'random_state': base_model.random_state
            }
            print('Hyperparameters:', hyperparams)

        X_model = df[model_features].values

        # Store metrics across all splits
        all_precisions = []
        all_recalls = []
        all_fbetas = []

        # Cross-validation loop
        for split_idx in range(n_splits):
            seed = 42 + split_idx

            # Stratified split
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
            train_sessions, test_sessions = next(splitter.split(unique_sessions, experiments))

            train_sessions_set = set(unique_sessions[train_sessions])
            test_sessions_set = set(unique_sessions[test_sessions])

            train_mask = np.array([s in train_sessions_set for s in sessions])
            test_mask = np.array([s in test_sessions_set for s in sessions])

            X_train = X_model[train_mask]
            y_train = y[train_mask]
            X_test = X_model[test_mask]
            y_test = y[test_mask]

            if retrain:
                # Train new model on this fold
                from interpret.glassbox import ExplainableBoostingClassifier
                fold_model = ExplainableBoostingClassifier(
                    feature_names=model_features,
                    **hyperparams
                )
                fold_model.fit(X_train, y_train)

                # Evaluate at all thresholds
                precisions, recalls, fbetas = evaluate_at_thresholds(fold_model, X_test, y_test, thresholds)
            else:
                # Use pre-trained model
                precisions, recalls, fbetas = evaluate_at_thresholds(base_model, X_test, y_test, thresholds)

            all_precisions.append(precisions)
            all_recalls.append(recalls)
            all_fbetas.append(fbetas)

            if split_idx % 2 == 0 or split_idx == n_splits - 1:
                print(f'  Split {split_idx+1}/{n_splits} complete')

        # Convert to arrays
        all_precisions = np.array(all_precisions)  # shape: (n_splits, n_thresholds)
        all_recalls = np.array(all_recalls)
        all_fbetas = np.array(all_fbetas)

        # Calculate mean and std
        mean_precision = np.mean(all_precisions, axis=0)
        std_precision = np.std(all_precisions, axis=0)
        mean_recall = np.mean(all_recalls, axis=0)
        std_recall = np.std(all_recalls, axis=0)
        mean_fbeta = np.mean(all_fbetas, axis=0)
        std_fbeta = np.std(all_fbetas, axis=0)

        # Find best F-beta
        best_idx = np.argmax(mean_fbeta)
        best_thresh = thresholds[best_idx]
        best_fbeta = mean_fbeta[best_idx]
        best_precision = mean_precision[best_idx]
        best_recall = mean_recall[best_idx]

        print(f'\nResults for {model_name}:')
        print(f'  Best F-beta: {best_fbeta:.4f} ± {std_fbeta[best_idx]:.4f} @ threshold={best_thresh:.2f}')
        print(f'  Precision:   {best_precision:.4f} ± {std_precision[best_idx]:.4f}')
        print(f'  Recall:      {best_recall:.4f} ± {std_recall[best_idx]:.4f}')

        model_results.append({
            'name': model_name,
            'thresholds': thresholds,
            'mean_precision': mean_precision,
            'std_precision': std_precision,
            'mean_recall': mean_recall,
            'std_recall': std_recall,
            'mean_fbeta': mean_fbeta,
            'std_fbeta': std_fbeta,
            'best_idx': best_idx,
            'best_thresh': best_thresh,
            'best_fbeta': best_fbeta,
            'best_precision': best_precision,
            'best_recall': best_recall
        })

    # Create plots
    print(f'\n{"="*80}')
    print('GENERATING PLOTS')
    print('='*80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot 1: PR curves with confidence bands (top-left)
    ax = axes[0, 0]
    for i, res in enumerate(model_results):
        color = colors[i % len(colors)]

        # Plot mean curve
        ax.plot(res['mean_recall'], res['mean_precision'],
                color=color, linewidth=2.5,
                label=f"{res['name']} (F-beta={res['best_fbeta']:.3f}±{res['std_fbeta'][res['best_idx']]:.3f})")

        # Plot confidence band (±1 std)
        ax.fill_between(
            res['mean_recall'],
            res['mean_precision'] - res['std_precision'],
            res['mean_precision'] + res['std_precision'],
            color=color, alpha=0.2
        )

        # Mark best point
        ax.scatter([res['best_recall']], [res['best_precision']],
                  color=color, s=150, zorder=5, marker='*',
                  edgecolors='black', linewidths=1.5)

    ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    mode_str = 'Retrained each fold' if retrain else 'Pre-trained (data leakage)'
    ax.set_title(f'Precision-Recall Curves ({mode_str}, {n_splits} splits, ±1 std)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.88, 1.01])
    ax.set_ylim([0.90, 1.01])

    # Plot 2: F-beta vs Threshold with bands (top-right)
    ax = axes[0, 1]
    for i, res in enumerate(model_results):
        color = colors[i % len(colors)]

        ax.plot(res['thresholds'], res['mean_fbeta'],
                color=color, linewidth=2.5, label=res['name'])

        ax.fill_between(
            res['thresholds'],
            res['mean_fbeta'] - res['std_fbeta'],
            res['mean_fbeta'] + res['std_fbeta'],
            color=color, alpha=0.2
        )

        # Mark best threshold
        ax.axvline(res['best_thresh'], color=color, linestyle='--',
                  alpha=0.5, linewidth=1)

    ax.set_xlabel('Threshold', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'F-beta (β={FBETA_BETA:.3f})', fontsize=13, fontweight='bold')
    ax.set_title(f'F-beta vs Threshold (mean ± std)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.3, 0.9])
    ax.set_ylim([0.86, 1.0])

    # Plot 3: Precision vs Threshold (bottom-left)
    ax = axes[1, 0]
    for i, res in enumerate(model_results):
        color = colors[i % len(colors)]

        ax.plot(res['thresholds'], res['mean_precision'],
                color=color, linewidth=2.5, label=res['name'])

        ax.fill_between(
            res['thresholds'],
            res['mean_precision'] - res['std_precision'],
            res['mean_precision'] + res['std_precision'],
            color=color, alpha=0.2
        )

    ax.set_xlabel('Threshold', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title('Precision vs Threshold (mean ± std)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.3, 0.9])
    ax.set_ylim([0.86, 1.01])

    # Plot 4: Recall vs Threshold (bottom-right)
    ax = axes[1, 1]
    for i, res in enumerate(model_results):
        color = colors[i % len(colors)]

        ax.plot(res['thresholds'], res['mean_recall'],
                color=color, linewidth=2.5, label=res['name'])

        ax.fill_between(
            res['thresholds'],
            res['mean_recall'] - res['std_recall'],
            res['mean_recall'] + res['std_recall'],
            color=color, alpha=0.2
        )

    ax.set_xlabel('Threshold', fontsize=13, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=13, fontweight='bold')
    ax.set_title('Recall vs Threshold (mean ± std)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.3, 0.9])
    ax.set_ylim([0.88, 1.01])

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f'\nPlot saved to: {output_path}')

    # Summary table
    print('\n' + '='*80)
    print('SUMMARY TABLE')
    print('='*80)
    print(f'\n{"Model":<20} {"Best F-beta":<20} {"@ Threshold":<15} {"Precision":<20} {"Recall":<20}')
    print('-'*100)
    for res in model_results:
        print(f"{res['name']:<20} "
              f"{res['best_fbeta']:.4f}±{res['std_fbeta'][res['best_idx']]:.4f}    "
              f"{res['best_thresh']:.2f}          "
              f"{res['best_precision']:.4f}±{res['std_precision'][res['best_idx']]:.4f}    "
              f"{res['best_recall']:.4f}±{res['std_recall'][res['best_idx']]:.4f}")

    print('\n' + '='*80)
    print('COMPLETE')
    print('='*80)

    return model_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot PR curves with CV')
    parser.add_argument('--models', nargs='+', required=True, help='Model paths')
    parser.add_argument('--names', nargs='+', required=True, help='Model names')
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Dataset paths (one per model, or single path for all)')
    parser.add_argument('--output', required=True, help='Output path')
    parser.add_argument('--n-splits', type=int, default=10, help='Number of CV splits')
    parser.add_argument('--n-thresholds', type=int, default=50, help='Number of thresholds')
    parser.add_argument('--retrain', action='store_true', help='Retrain model on each fold (proper CV)')

    args = parser.parse_args()

    plot_pr_curves_cv(
        model_paths=args.models,
        model_names=args.names,
        dataset_paths=args.datasets,
        output_path=args.output,
        n_splits=args.n_splits,
        n_threshold_points=args.n_thresholds,
        retrain=args.retrain
    )
