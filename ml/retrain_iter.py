"""
Universal iterative model retraining script.

Trains model on any corrected dataset, generates error reports for next iteration.
Works with any dataset version (v8, v9, v10, etc.).

Usage:
    python ml/retrain_iter.py --dataset ml/results/training_dataset_v9.csv --output ml/ebm_v9_iter1 --iter 1
    python ml/retrain_iter.py --dataset ml/results/training_dataset_v9_corrected_iter1.csv --output ml/ebm_v9_iter2 --iter 2 --base-model ml/ebm_v9_iter1/model.pkl
"""
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, fbeta_score

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import NON_FEATURE_COLS, get_feature_cols, FBETA_BETA


def compute_metrics(y_true, y_pred, y_proba, beta):
    """Compute classification metrics."""
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    fbeta = fbeta_score(y_true, y_pred, beta=beta, average='binary', zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    acc = (y_pred == y_true).mean()
    return {
        'precision': prec,
        'recall': rec,
        'fbeta': fbeta,
        'auc': auc,
        'accuracy': acc
    }


def retrain_iteration(
    dataset_path,
    output_dir,
    iteration,
    threshold=0.75,
    random_seed=42,
    base_model_path=None,
    n_top_errors=100
):
    """
    Retrain model on corrected dataset and generate error reports.

    Parameters
    ----------
    dataset_path : str
        Path to training dataset CSV
    output_dir : str
        Output directory for model and error reports
    iteration : int
        Iteration number (for naming/tracking)
    threshold : float
        Classification threshold
    random_seed : int
        Random seed for reproducibility
    base_model_path : str, optional
        Path to base model to copy hyperparameters from
    n_top_errors : int
        Number of top errors to export for review
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print('='*80)
    print(f'ITERATIVE RETRAINING - ITERATION {iteration}')
    print('='*80)
    print(f'Dataset: {dataset_path}')
    print(f'Output: {output_dir}')
    print(f'Threshold: {threshold}')
    print(f'Random seed: {random_seed}')

    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f'\nDataset: {len(df):,} neurons')

    # Determine session column
    if 'session_name' in df.columns:
        session_col = 'session_name'
    elif 'session' in df.columns:
        session_col = 'session'
    else:
        raise ValueError('Dataset must have "session_name" or "session" column')

    # Get feature columns automatically
    feature_cols = get_feature_cols(df)
    print(f'Features: {len(feature_cols)}')

    # Verify no leakage
    leaked = [col for col in feature_cols if col in NON_FEATURE_COLS]
    if leaked:
        raise ValueError(f'Feature leakage detected: {leaked}')

    # Load base model for hyperparameters if provided
    if base_model_path:
        print(f'\nLoading hyperparameters from: {base_model_path}')
        with open(base_model_path, 'rb') as f:
            base_model = pickle.load(f)

        hyperparams = {
            'max_bins': base_model.max_bins,
            'interactions': base_model.interactions,
            'max_leaves': base_model.max_leaves,
            'min_samples_leaf': base_model.min_samples_leaf,
            'outer_bags': getattr(base_model, 'outer_bags', 8),
            'learning_rate': getattr(base_model, 'learning_rate', 0.01),
            'max_rounds': getattr(base_model, 'max_rounds', 5000),
            'early_stopping_rounds': getattr(base_model, 'early_stopping_rounds', 50),
        }
        print('Hyperparameters:')
        for k, v in hyperparams.items():
            print(f'  {k}: {v}')
    else:
        # Default hyperparameters (v9 best)
        hyperparams = {
            'max_bins': 1024,
            'interactions': 20,
            'max_leaves': 3,
            'min_samples_leaf': 2,
            'outer_bags': 8,
            'learning_rate': 0.01,
            'max_rounds': 5000,
            'early_stopping_rounds': 50,
        }
        print('\nUsing default hyperparameters (v9 best)')

    # Stratified train/test split by session
    print('\n' + '-'*80)
    print('TRAIN/TEST SPLIT')
    print('-'*80)

    sessions = df[session_col].unique()
    if 'experiment' in df.columns:
        session_to_exp = df.groupby(session_col)['experiment'].first().to_dict()
        experiments = [session_to_exp[s] for s in sessions]
    else:
        # Infer from session name
        experiments = [s.split('_')[0] for s in sessions]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_seed)
    train_idx, test_idx = next(splitter.split(sessions, experiments))
    train_sessions = set(sessions[train_idx])
    test_sessions = set(sessions[test_idx])

    train_mask = df[session_col].isin(train_sessions)
    test_mask = df[session_col].isin(test_sessions)

    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, 'ground_truth'].values
    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, 'ground_truth'].values

    print(f'Train: {len(X_train):,} neurons ({len(train_sessions)} sessions, KEEP: {y_train.mean()*100:.2f}%)')
    print(f'Test:  {len(X_test):,} neurons ({len(test_sessions)} sessions, KEEP: {y_test.mean()*100:.2f}%)')

    # Train model
    print('\n' + '='*80)
    print(f'TRAINING MODEL (iteration {iteration})')
    print('='*80)

    from interpret.glassbox import ExplainableBoostingClassifier

    model = ExplainableBoostingClassifier(
        feature_names=feature_cols,
        max_bins=hyperparams['max_bins'],
        max_interaction_bins=min(64, hyperparams['max_bins'] // 4),
        interactions=hyperparams['interactions'],
        outer_bags=hyperparams['outer_bags'],
        inner_bags=0,
        learning_rate=hyperparams['learning_rate'],
        validation_size=0.15,
        early_stopping_rounds=hyperparams['early_stopping_rounds'],
        early_stopping_tolerance=1e-4,
        max_rounds=hyperparams['max_rounds'],
        min_samples_leaf=hyperparams['min_samples_leaf'],
        max_leaves=hyperparams['max_leaves'],
        random_state=random_seed
    )

    print('Training...')
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f'Training completed in {train_time:.1f}s')

    # Evaluate
    print('\n' + '='*80)
    print('EVALUATION')
    print('='*80)

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    y_train_pred = (y_train_proba >= threshold).astype(int)
    y_test_pred = (y_test_proba >= threshold).astype(int)

    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba, FBETA_BETA)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba, FBETA_BETA)

    print(f'\nTRAIN SET (threshold={threshold}):')
    for metric, value in train_metrics.items():
        print(f'  {metric.capitalize():<12} {value:.4f}')

    print(f'\nTEST SET (threshold={threshold}):')
    for metric, value in test_metrics.items():
        print(f'  {metric.capitalize():<12} {value:.4f}')

    # Save model
    model_path = output_path / 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'\nModel saved to: {model_path}')

    # Generate error reports
    print('\n' + '='*80)
    print('GENERATING ERROR REPORTS')
    print('='*80)

    df_test = df.loc[test_mask].copy()
    df_test['y_proba'] = y_test_proba
    df_test['y_pred'] = y_test_pred

    # Confusion matrix
    tp_mask = (y_test_pred == 1) & (y_test == 1)
    fp_mask = (y_test_pred == 1) & (y_test == 0)
    fn_mask = (y_test_pred == 0) & (y_test == 1)
    tn_mask = (y_test_pred == 0) & (y_test == 0)

    n_tp = tp_mask.sum()
    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()
    n_tn = tn_mask.sum()

    print(f'\nConfusion Matrix:')
    print(f'  True Positives:  {n_tp:,}')
    print(f'  False Positives: {n_fp:,}')
    print(f'  False Negatives: {n_fn:,}')
    print(f'  True Negatives:  {n_tn:,}')

    # Top errors for review
    df_fp = df_test[fp_mask].sort_values('y_proba', ascending=False).head(n_top_errors)
    df_fn = df_test[fn_mask].sort_values('y_proba', ascending=True).head(n_top_errors)

    fp_path = output_path / f'top{n_top_errors}_fp.csv'
    fn_path = output_path / f'top{n_top_errors}_fn.csv'

    df_fp.to_csv(fp_path, index=False)
    df_fn.to_csv(fn_path, index=False)

    print(f'\nError reports saved:')
    print(f'  FP: {fp_path} ({len(df_fp)} neurons)')
    print(f'  FN: {fn_path} ({len(df_fn)} neurons)')

    # Save summary
    summary = {
        'iteration': iteration,
        'dataset': str(dataset_path),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'threshold': threshold,
        'random_seed': random_seed,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'confusion_matrix': {
            'tp': int(n_tp),
            'fp': int(n_fp),
            'fn': int(n_fn),
            'tn': int(n_tn)
        },
        'hyperparameters': hyperparams
    }

    import json
    summary_path = output_path / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Summary: {summary_path}')

    print('\n' + '='*80)
    print('ITERATION COMPLETE')
    print('='*80)
    print(f'\nNext steps:')
    print(f'1. Visualize errors: python ml/visualize_errors.py --fp {fp_path} --fn {fn_path}')
    print(f'2. Review visualizations and classify errors')
    print(f'3. Apply corrections: python ml/apply_corrections.py --dataset {dataset_path} --output ...')

    return model, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Universal iterative model retraining')
    parser.add_argument('--dataset', required=True, help='Path to training dataset CSV')
    parser.add_argument('--output', required=True, help='Output directory for model and reports')
    parser.add_argument('--iter', type=int, default=1, help='Iteration number')
    parser.add_argument('--threshold', type=float, default=0.75, help='Classification threshold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--base-model', default=None, help='Base model to copy hyperparameters from')
    parser.add_argument('--n-errors', type=int, default=100, help='Number of top errors to export')

    args = parser.parse_args()

    retrain_iteration(
        dataset_path=args.dataset,
        output_dir=args.output,
        iteration=args.iter,
        threshold=args.threshold,
        random_seed=args.seed,
        base_model_path=args.base_model,
        n_top_errors=args.n_errors
    )
