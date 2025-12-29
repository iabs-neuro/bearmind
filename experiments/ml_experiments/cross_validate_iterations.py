"""
Cross-validation analysis across all iterations to assess seed-independent performance.

For each iteration, evaluates the model on multiple random splits to get
robust, seed-independent metrics.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')

# Configuration
N_SPLITS = 10  # Number of different train/test splits to evaluate
TEST_SIZE = 0.25
THRESHOLD = 0.75
FBETA_BETA = 0.577  # Same as training

# Iteration configurations
ITERATIONS = [
    {
        'name': 'Iter 1',
        'model': 'ml/ebm_v9_iter1/model.pkl',
        'dataset': 'ml/results/training_dataset_v9.csv',
        'corrections': 0
    },
    {
        'name': 'Iter 2',
        'model': 'ml/ebm_v9_iter2/model.pkl',
        'dataset': 'ml/results/training_dataset_v9_corrected_iter1.csv',
        'corrections': 158
    },
    {
        'name': 'Iter 3',
        'model': 'ml/ebm_v9_iter3/model.pkl',
        'dataset': 'ml/results/training_dataset_v9_corrected_iter2.csv',
        'corrections': 325
    },
    {
        'name': 'Iter 4',
        'model': 'ml/ebm_v9_iter4/model.pkl',
        'dataset': 'ml/results/training_dataset_v9_corrected_iter3.csv',
        'corrections': 483
    },
    {
        'name': 'Iter 5',
        'model': 'ml/ebm_v9_iter5/model.pkl',
        'dataset': 'ml/results/training_dataset_v9_corrected_iter4.csv',
        'corrections': 624
    }
]

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

def evaluate_model(model, X, y, threshold):
    """Evaluate model at specific threshold."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tp = ((y_pred == 1) & (y == 1)).sum()
    fp = ((y_pred == 1) & (y == 0)).sum()
    tn = ((y_pred == 0) & (y == 0)).sum()
    fn = ((y_pred == 0) & (y == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    fbeta_denom = (FBETA_BETA**2 * precision + recall)
    fbeta = (1 + FBETA_BETA**2) * precision * recall / fbeta_denom if fbeta_denom > 0 else 0

    accuracy = (tp + tn) / (tp + fp + tn + fn)

    # AUC from sklearn
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, y_proba)

    return {
        'precision': precision,
        'recall': recall,
        'fbeta': fbeta,
        'auc': auc,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

print('='*80)
print('CROSS-VALIDATION ANALYSIS ACROSS ITERATIONS')
print('='*80)
print(f'\nConfiguration:')
print(f'  N_SPLITS: {N_SPLITS}')
print(f'  TEST_SIZE: {TEST_SIZE}')
print(f'  THRESHOLD: {THRESHOLD}')
print(f'  Seeds: {list(range(42, 42 + N_SPLITS))}')

results = []

for iter_config in ITERATIONS:
    print(f'\n{"="*80}')
    print(f'{iter_config["name"].upper()} - {iter_config["corrections"]} cumulative corrections')
    print('='*80)

    # Load model and dataset
    print(f'Loading model: {iter_config["model"]}')
    with open(iter_config['model'], 'rb') as f:
        model = pickle.load(f)

    print(f'Loading dataset: {iter_config["dataset"]}')
    df = pd.read_csv(iter_config['dataset'])

    # Determine session column
    if 'session_name' in df.columns:
        session_col = 'session_name'
    elif 'session' in df.columns:
        session_col = 'session'
    else:
        raise ValueError('No session column found')

    # Get features
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values
    y = df['ground_truth'].values
    sessions = df[session_col].values

    # Get unique sessions and their experiments
    unique_sessions = df[session_col].unique()
    if 'experiment' in df.columns:
        session_to_exp = df.groupby(session_col)['experiment'].first().to_dict()
        experiments = np.array([session_to_exp[s] for s in unique_sessions])
    else:
        experiments = np.array([s.split('_')[0] for s in unique_sessions])

    # Get model features (in case different from dataset)
    model_features = list(model.feature_names_in_)
    X_model = df[model_features].values

    # Cross-validation loop
    fold_metrics = []

    for split_idx in range(N_SPLITS):
        seed = 42 + split_idx

        # Stratified split by session/experiment
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_sessions, test_sessions = next(splitter.split(unique_sessions, experiments))

        train_sessions_set = set(unique_sessions[train_sessions])
        test_mask = np.array([s in set(unique_sessions[test_sessions]) for s in sessions])

        X_test = X_model[test_mask]
        y_test = y[test_mask]

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, THRESHOLD)
        fold_metrics.append(metrics)

        if split_idx == 0:
            print(f'  Split 0 (seed {seed}): F-beta={metrics["fbeta"]:.4f}, AUC={metrics["auc"]:.4f}')

    # Calculate mean and std across folds
    mean_metrics = {
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'fbeta': np.mean([m['fbeta'] for m in fold_metrics]),
        'auc': np.mean([m['auc'] for m in fold_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics])
    }

    std_metrics = {
        'precision': np.std([m['precision'] for m in fold_metrics]),
        'recall': np.std([m['recall'] for m in fold_metrics]),
        'fbeta': np.std([m['fbeta'] for m in fold_metrics]),
        'auc': np.std([m['auc'] for m in fold_metrics]),
        'accuracy': np.std([m['accuracy'] for m in fold_metrics])
    }

    print(f'\nCross-Validation Results ({N_SPLITS} splits):')
    print(f'  Precision: {mean_metrics["precision"]:.4f} ± {std_metrics["precision"]:.4f}')
    print(f'  Recall:    {mean_metrics["recall"]:.4f} ± {std_metrics["recall"]:.4f}')
    print(f'  F-beta:    {mean_metrics["fbeta"]:.4f} ± {std_metrics["fbeta"]:.4f}')
    print(f'  AUC:       {mean_metrics["auc"]:.4f} ± {std_metrics["auc"]:.4f}')
    print(f'  Accuracy:  {mean_metrics["accuracy"]:.4f} ± {std_metrics["accuracy"]:.4f}')

    results.append({
        'iteration': iter_config['name'],
        'corrections': iter_config['corrections'],
        'mean': mean_metrics,
        'std': std_metrics,
        'fold_metrics': fold_metrics
    })

# Summary comparison
print(f'\n{"="*80}')
print('SUMMARY - SEED-INDEPENDENT IMPROVEMENT')
print('='*80)

print('\nMean F-beta across iterations:')
for r in results:
    print(f"  {r['iteration']}: {r['mean']['fbeta']:.4f} ± {r['std']['fbeta']:.4f} ({r['corrections']} corrections)")

print('\nMean AUC across iterations:')
for r in results:
    print(f"  {r['iteration']}: {r['mean']['auc']:.4f} ± {r['std']['auc']:.4f} ({r['corrections']} corrections)")

# Calculate improvements
print('\nImprovement from Iter 1:')
baseline_fbeta = results[0]['mean']['fbeta']
baseline_auc = results[0]['mean']['auc']

for r in results[1:]:
    fbeta_gain = (r['mean']['fbeta'] - baseline_fbeta) * 100
    auc_gain = (r['mean']['auc'] - baseline_auc) * 100
    print(f"  {r['iteration']}: F-beta +{fbeta_gain:.2f}%, AUC +{auc_gain:.2f}%")

# Save results
output_df = pd.DataFrame([{
    'iteration': r['iteration'],
    'corrections': r['corrections'],
    'fbeta_mean': r['mean']['fbeta'],
    'fbeta_std': r['std']['fbeta'],
    'auc_mean': r['mean']['auc'],
    'auc_std': r['std']['auc'],
    'precision_mean': r['mean']['precision'],
    'precision_std': r['std']['precision'],
    'recall_mean': r['mean']['recall'],
    'recall_std': r['std']['recall'],
    'accuracy_mean': r['mean']['accuracy'],
    'accuracy_std': r['std']['accuracy']
} for r in results])

output_df.to_csv('ml/results/cv_analysis_iter1_to_iter5.csv', index=False)
print(f'\nResults saved to: ml/results/cv_analysis_iter1_to_iter5.csv')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
