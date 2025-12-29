"""
Threshold optimization for EBM model.

Searches for optimal classification threshold in specified range.
"""
import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, fbeta_score
import sys

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import get_feature_cols, FBETA_BETA

def evaluate_threshold(y_true, y_proba, threshold, beta):
    """Evaluate metrics at specific threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    fbeta_denom = (beta**2 * precision + recall)
    fbeta = (1 + beta**2) * precision * recall / fbeta_denom if fbeta_denom > 0 else 0

    accuracy = (tp + tn) / (tp + fp + tn + fn)

    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'fbeta': fbeta,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def optimize_threshold(
    model_path,
    dataset_path,
    threshold_min=0.7,
    threshold_max=0.8,
    threshold_step=0.01,
    random_seed=42
):
    """
    Find optimal threshold for model.

    Parameters
    ----------
    model_path : str
        Path to trained model pickle
    dataset_path : str
        Path to dataset CSV
    threshold_min : float
        Minimum threshold to test
    threshold_max : float
        Maximum threshold to test
    threshold_step : float
        Step size between thresholds
    random_seed : int
        Random seed for train/test split
    """
    print('='*80)
    print('THRESHOLD OPTIMIZATION')
    print('='*80)
    print(f'Model: {model_path}')
    print(f'Dataset: {dataset_path}')
    print(f'Threshold range: {threshold_min:.2f} - {threshold_max:.2f} (step={threshold_step:.3f})')
    print(f'Random seed: {random_seed}')

    # Load model
    print('\nLoading model...')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load dataset
    print('Loading dataset...')
    df = pd.read_csv(dataset_path)
    print(f'Dataset: {len(df):,} neurons')

    # Determine session column
    if 'session_name' in df.columns:
        session_col = 'session_name'
    elif 'session' in df.columns:
        session_col = 'session'
    else:
        raise ValueError('Dataset must have "session_name" or "session" column')

    # Get features
    feature_cols = get_feature_cols(df)
    model_features = list(model.feature_names_in_)

    # Stratified train/test split
    print('\nCreating train/test split...')
    sessions = df[session_col].unique()
    if 'experiment' in df.columns:
        session_to_exp = df.groupby(session_col)['experiment'].first().to_dict()
        experiments = [session_to_exp[s] for s in sessions]
    else:
        experiments = [s.split('_')[0] for s in sessions]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_seed)
    train_idx, test_idx = next(splitter.split(sessions, experiments))
    train_sessions = set(sessions[train_idx])
    test_sessions = set(sessions[test_idx])

    train_mask = df[session_col].isin(train_sessions)
    test_mask = df[session_col].isin(test_sessions)

    X_test = df.loc[test_mask, model_features].values
    y_test = df.loc[test_mask, 'ground_truth'].values

    print(f'Test set: {len(X_test):,} neurons ({len(test_sessions)} sessions)')

    # Get predictions
    print('\nGenerating predictions...')
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate at different thresholds
    print('\n' + '='*80)
    print('THRESHOLD SEARCH')
    print('='*80)

    thresholds = np.arange(threshold_min, threshold_max + threshold_step/2, threshold_step)
    results = []

    for threshold in thresholds:
        metrics = evaluate_threshold(y_test, y_proba, threshold, FBETA_BETA)
        results.append(metrics)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Find optimal thresholds
    best_fbeta_idx = df_results['fbeta'].idxmax()
    best_accuracy_idx = df_results['accuracy'].idxmax()

    print(f'\nEvaluated {len(thresholds)} thresholds')
    print('\nTop 10 by F-beta:')
    print(df_results.nlargest(10, 'fbeta')[['threshold', 'fbeta', 'precision', 'recall', 'accuracy']].to_string(index=False))

    print('\n' + '='*80)
    print('OPTIMAL THRESHOLDS')
    print('='*80)

    best_fbeta = df_results.iloc[best_fbeta_idx]
    print(f'\nBest F-beta: {best_fbeta["fbeta"]:.4f} at threshold={best_fbeta["threshold"]:.3f}')
    print(f'  Precision: {best_fbeta["precision"]:.4f}')
    print(f'  Recall:    {best_fbeta["recall"]:.4f}')
    print(f'  Accuracy:  {best_fbeta["accuracy"]:.4f}')
    print(f'  TP={best_fbeta["tp"]:.0f}, FP={best_fbeta["fp"]:.0f}, FN={best_fbeta["fn"]:.0f}, TN={best_fbeta["tn"]:.0f}')

    # Compare to current threshold (0.75)
    current_threshold = 0.75
    if threshold_min <= current_threshold <= threshold_max:
        current_metrics = df_results[df_results['threshold'] == current_threshold].iloc[0]
        print(f'\nCurrent threshold (0.75): F-beta={current_metrics["fbeta"]:.4f}')
        fbeta_improvement = (best_fbeta["fbeta"] - current_metrics["fbeta"]) * 100
        print(f'Improvement: +{fbeta_improvement:.2f}% F-beta')

    # Save results
    output_dir = Path(model_path).parent
    output_file = output_dir / 'threshold_optimization.csv'
    df_results.to_csv(output_file, index=False)
    print(f'\nResults saved to: {output_file}')

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # F-beta vs threshold
        axes[0, 0].plot(df_results['threshold'], df_results['fbeta'], 'b-', linewidth=2)
        axes[0, 0].axvline(best_fbeta['threshold'], color='r', linestyle='--', alpha=0.7, label=f'Optimal: {best_fbeta["threshold"]:.3f}')
        if threshold_min <= current_threshold <= threshold_max:
            axes[0, 0].axvline(current_threshold, color='orange', linestyle='--', alpha=0.7, label='Current: 0.75')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('F-beta')
        axes[0, 0].set_title(f'F-beta vs Threshold (beta={FBETA_BETA:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Precision and Recall vs threshold
        axes[0, 1].plot(df_results['threshold'], df_results['precision'], 'g-', linewidth=2, label='Precision')
        axes[0, 1].plot(df_results['threshold'], df_results['recall'], 'b-', linewidth=2, label='Recall')
        axes[0, 1].axvline(best_fbeta['threshold'], color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Precision & Recall vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Accuracy vs threshold
        axes[1, 0].plot(df_results['threshold'], df_results['accuracy'], 'purple', linewidth=2)
        axes[1, 0].axvline(best_fbeta['threshold'], color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Accuracy vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)

        # Confusion matrix counts
        axes[1, 1].plot(df_results['threshold'], df_results['tp'], 'g-', linewidth=2, label='TP')
        axes[1, 1].plot(df_results['threshold'], df_results['fp'], 'r-', linewidth=2, label='FP')
        axes[1, 1].plot(df_results['threshold'], df_results['fn'], 'orange', linewidth=2, label='FN')
        axes[1, 1].axvline(best_fbeta['threshold'], color='r', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Error Counts vs Threshold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        plot_file = output_dir / 'threshold_optimization.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f'Plot saved to: {plot_file}')
        plt.close()

    except ImportError:
        print('\nMatplotlib not available, skipping plots')

    return df_results, best_fbeta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize classification threshold')
    parser.add_argument('--model', required=True, help='Path to model pickle')
    parser.add_argument('--dataset', required=True, help='Path to dataset CSV')
    parser.add_argument('--min', type=float, default=0.7, help='Minimum threshold')
    parser.add_argument('--max', type=float, default=0.8, help='Maximum threshold')
    parser.add_argument('--step', type=float, default=0.01, help='Threshold step size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    optimize_threshold(
        model_path=args.model,
        dataset_path=args.dataset,
        threshold_min=args.min,
        threshold_max=args.max,
        threshold_step=args.step,
        random_seed=args.seed
    )
