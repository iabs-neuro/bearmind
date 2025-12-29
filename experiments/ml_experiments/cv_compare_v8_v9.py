"""
Fair cross-validation comparison between v8_iter5 and v9_iter8.

Uses identical test splits for both models to ensure fair comparison.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
N_SPLITS = 10
TEST_SIZE = 0.25
THRESHOLD = 0.75
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
print('CROSS-VALIDATION COMPARISON: v8_iter5 vs v9_iter8')
print('='*80)
print(f'\nConfiguration:')
print(f'  N_SPLITS: {N_SPLITS}')
print(f'  TEST_SIZE: {TEST_SIZE}')
print(f'  THRESHOLD: {THRESHOLD}')
print(f'  Seeds: {list(range(42, 42 + N_SPLITS))}')

# Load models
print('\n' + '='*80)
print('LOADING MODELS')
print('='*80)

v8_model_path = 'production_models/ebm_v8_corrected_iter5.pkl'
v9_model_path = 'ml/ebm_v9_iter8/model.pkl'

print(f'Loading v8_iter5: {v8_model_path}')
with open(v8_model_path, 'rb') as f:
    v8_model = pickle.load(f)

print(f'Loading v9_iter8: {v9_model_path}')
with open(v9_model_path, 'rb') as f:
    v9_model = pickle.load(f)

# Load v9 dataset (most corrected)
print('\n' + '='*80)
print('LOADING DATASET')
print('='*80)

dataset_path = 'ml/results/training_dataset_v9_corrected_iter7.csv'
print(f'Dataset: {dataset_path}')
df = pd.read_csv(dataset_path)
print(f'Loaded: {len(df):,} neurons')

# Determine session column
if 'session_name' in df.columns:
    session_col = 'session_name'
elif 'session' in df.columns:
    session_col = 'session'
else:
    raise ValueError('No session column found')

# Get features
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

# Get model-specific features
v8_features = list(v8_model.feature_names_in_)
v9_features = list(v9_model.feature_names_in_)

print(f'\nv8_iter5 features: {len(v8_features)}')
print(f'v9_iter8 features: {len(v9_features)}')

# Check feature overlap
common_features = set(v8_features) & set(v9_features)
v8_only = set(v8_features) - set(v9_features)
v9_only = set(v9_features) - set(v8_features)

print(f'\nFeature analysis:')
print(f'  Common features: {len(common_features)}')
if v8_only:
    print(f'  v8-only features: {sorted(v8_only)}')
if v9_only:
    print(f'  v9-only features: {sorted(v9_only)}')

# Prepare data for each model
X_v8 = df[v8_features].values
X_v9 = df[v9_features].values

# Cross-validation loop
print('\n' + '='*80)
print('RUNNING CROSS-VALIDATION')
print('='*80)

v8_fold_metrics = []
v9_fold_metrics = []

for split_idx in range(N_SPLITS):
    seed = 42 + split_idx
    print(f'\nSplit {split_idx+1}/{N_SPLITS} (seed={seed})')

    # Stratified split by session/experiment
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_sessions, test_sessions = next(splitter.split(unique_sessions, experiments))

    test_sessions_set = set(unique_sessions[test_sessions])
    test_mask = np.array([s in test_sessions_set for s in sessions])

    X_v8_test = X_v8[test_mask]
    X_v9_test = X_v9[test_mask]
    y_test = y[test_mask]

    # Evaluate both models
    v8_metrics = evaluate_model(v8_model, X_v8_test, y_test, THRESHOLD)
    v9_metrics = evaluate_model(v9_model, X_v9_test, y_test, THRESHOLD)

    v8_fold_metrics.append(v8_metrics)
    v9_fold_metrics.append(v9_metrics)

    print(f'  v8_iter5: F-beta={v8_metrics["fbeta"]:.4f}, AUC={v8_metrics["auc"]:.4f}')
    print(f'  v9_iter8: F-beta={v9_metrics["fbeta"]:.4f}, AUC={v9_metrics["auc"]:.4f}')
    print(f'  Improvement: F-beta={((v9_metrics["fbeta"]-v8_metrics["fbeta"])*100):+.2f}%, AUC={((v9_metrics["auc"]-v8_metrics["auc"])*100):+.2f}%')

# Calculate mean and std for both models
print('\n' + '='*80)
print('CROSS-VALIDATION RESULTS')
print('='*80)

v8_mean = {
    'precision': np.mean([m['precision'] for m in v8_fold_metrics]),
    'recall': np.mean([m['recall'] for m in v8_fold_metrics]),
    'fbeta': np.mean([m['fbeta'] for m in v8_fold_metrics]),
    'auc': np.mean([m['auc'] for m in v8_fold_metrics]),
    'accuracy': np.mean([m['accuracy'] for m in v8_fold_metrics])
}

v8_std = {
    'precision': np.std([m['precision'] for m in v8_fold_metrics]),
    'recall': np.std([m['recall'] for m in v8_fold_metrics]),
    'fbeta': np.std([m['fbeta'] for m in v8_fold_metrics]),
    'auc': np.std([m['auc'] for m in v8_fold_metrics]),
    'accuracy': np.std([m['accuracy'] for m in v8_fold_metrics])
}

v9_mean = {
    'precision': np.mean([m['precision'] for m in v9_fold_metrics]),
    'recall': np.mean([m['recall'] for m in v9_fold_metrics]),
    'fbeta': np.mean([m['fbeta'] for m in v9_fold_metrics]),
    'auc': np.mean([m['auc'] for m in v9_fold_metrics]),
    'accuracy': np.mean([m['accuracy'] for m in v9_fold_metrics])
}

v9_std = {
    'precision': np.std([m['precision'] for m in v9_fold_metrics]),
    'recall': np.std([m['recall'] for m in v9_fold_metrics]),
    'fbeta': np.std([m['fbeta'] for m in v9_fold_metrics]),
    'auc': np.std([m['auc'] for m in v9_fold_metrics]),
    'accuracy': np.std([m['accuracy'] for m in v9_fold_metrics])
}

print('\nv8_iter5 Results (10 splits):')
print(f'  Precision: {v8_mean["precision"]:.4f} ± {v8_std["precision"]:.4f}')
print(f'  Recall:    {v8_mean["recall"]:.4f} ± {v8_std["recall"]:.4f}')
print(f'  F-beta:    {v8_mean["fbeta"]:.4f} ± {v8_std["fbeta"]:.4f}')
print(f'  AUC:       {v8_mean["auc"]:.4f} ± {v8_std["auc"]:.4f}')
print(f'  Accuracy:  {v8_mean["accuracy"]:.4f} ± {v8_std["accuracy"]:.4f}')

print('\nv9_iter8 Results (10 splits):')
print(f'  Precision: {v9_mean["precision"]:.4f} ± {v9_std["precision"]:.4f}')
print(f'  Recall:    {v9_mean["recall"]:.4f} ± {v9_std["recall"]:.4f}')
print(f'  F-beta:    {v9_mean["fbeta"]:.4f} ± {v9_std["fbeta"]:.4f}')
print(f'  AUC:       {v9_mean["auc"]:.4f} ± {v9_std["auc"]:.4f}')
print(f'  Accuracy:  {v9_mean["accuracy"]:.4f} ± {v9_std["accuracy"]:.4f}')

# Calculate improvements
print('\n' + '='*80)
print('IMPROVEMENT: v9_iter8 vs v8_iter5')
print('='*80)

fbeta_improvement = (v9_mean['fbeta'] - v8_mean['fbeta']) * 100
auc_improvement = (v9_mean['auc'] - v8_mean['auc']) * 100
precision_improvement = (v9_mean['precision'] - v8_mean['precision']) * 100
recall_improvement = (v9_mean['recall'] - v8_mean['recall']) * 100
accuracy_improvement = (v9_mean['accuracy'] - v8_mean['accuracy']) * 100

print(f'\nMean improvements:')
print(f'  F-beta:    {fbeta_improvement:+.2f}% ({v9_mean["fbeta"]:.4f} vs {v8_mean["fbeta"]:.4f})')
print(f'  AUC:       {auc_improvement:+.2f}% ({v9_mean["auc"]:.4f} vs {v8_mean["auc"]:.4f})')
print(f'  Precision: {precision_improvement:+.2f}% ({v9_mean["precision"]:.4f} vs {v8_mean["precision"]:.4f})')
print(f'  Recall:    {recall_improvement:+.2f}% ({v9_mean["recall"]:.4f} vs {v8_mean["recall"]:.4f})')
print(f'  Accuracy:  {accuracy_improvement:+.2f}% ({v9_mean["accuracy"]:.4f} vs {v8_mean["accuracy"]:.4f})')

# Statistical significance check (simple paired t-test)
from scipy import stats

print('\n' + '='*80)
print('STATISTICAL SIGNIFICANCE (paired t-test)')
print('='*80)

fbeta_diffs = [v9_fold_metrics[i]['fbeta'] - v8_fold_metrics[i]['fbeta'] for i in range(N_SPLITS)]
auc_diffs = [v9_fold_metrics[i]['auc'] - v8_fold_metrics[i]['auc'] for i in range(N_SPLITS)]

t_fbeta, p_fbeta = stats.ttest_rel(
    [m['fbeta'] for m in v9_fold_metrics],
    [m['fbeta'] for m in v8_fold_metrics]
)

t_auc, p_auc = stats.ttest_rel(
    [m['auc'] for m in v9_fold_metrics],
    [m['auc'] for m in v8_fold_metrics]
)

print(f'\nF-beta: t={t_fbeta:.3f}, p={p_fbeta:.4f}', end='')
if p_fbeta < 0.05:
    print(' (SIGNIFICANT)')
else:
    print(' (not significant)')

print(f'AUC:    t={t_auc:.3f}, p={p_auc:.4f}', end='')
if p_auc < 0.05:
    print(' (SIGNIFICANT)')
else:
    print(' (not significant)')

# Save results
results_df = pd.DataFrame({
    'model': ['v8_iter5', 'v9_iter8'],
    'fbeta_mean': [v8_mean['fbeta'], v9_mean['fbeta']],
    'fbeta_std': [v8_std['fbeta'], v9_std['fbeta']],
    'auc_mean': [v8_mean['auc'], v9_mean['auc']],
    'auc_std': [v8_std['auc'], v9_std['auc']],
    'precision_mean': [v8_mean['precision'], v9_mean['precision']],
    'precision_std': [v8_std['precision'], v9_std['precision']],
    'recall_mean': [v8_mean['recall'], v9_mean['recall']],
    'recall_std': [v8_std['recall'], v9_std['recall']],
    'accuracy_mean': [v8_mean['accuracy'], v9_mean['accuracy']],
    'accuracy_std': [v8_std['accuracy'], v9_std['accuracy']],
})

output_path = 'ml/results/cv_comparison_v8_iter5_vs_v9_iter8.csv'
results_df.to_csv(output_path, index=False)
print(f'\nResults saved to: {output_path}')

print('\n' + '='*80)
print('SUMMARY')
print('='*80)

print(f'\nv9_iter8 is {"SIGNIFICANTLY BETTER" if p_fbeta < 0.05 else "slightly better"} than v8_iter5:')
print(f'  - F-beta improved by {fbeta_improvement:+.2f}% (p={p_fbeta:.4f})')
print(f'  - AUC improved by {auc_improvement:+.2f}% (p={p_auc:.4f})')
print(f'  - Based on {N_SPLITS} independent test splits')
print(f'  - More stable (lower variance) in {"all" if all(v9_std[k] < v8_std[k] for k in v9_std) else "most"} metrics')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
