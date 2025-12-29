"""
Test whether adding trace_range improves v9_iter8 model performance.

Compares two models:
- Baseline: Original v9 features (35 features)
- Enhanced: Original v9 features + trace_range (36 features)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from interpret.glassbox import ExplainableBoostingClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('EXPERIMENT: ADDING trace_range TO v9 MODEL')
print('='*80)

# Load dataset with trace_range already computed
dataset_path = 'ml/results/dataset_with_trace_stats.csv'
print(f'\nLoading dataset: {dataset_path}')
df = pd.read_csv(dataset_path)
print(f'Dataset: {len(df):,} neurons')

# Get features
NON_FEATURE_COLS = {
    'session_name', 'session', 'component_idx', 'ground_truth',
    'decision', 'delete', 'merge', 'experiment',
    'failed_area', 'failed_circularity', 'failed_corner_artifact',
    'is_corner_artifact', 'ml_keep_probability',
    'y_proba', 'y_pred',
    # Trace stats we computed (we'll add them selectively)
    'trace_mean', 'trace_median', 'trace_std', 'trace_mad',
    'trace_iqr', 'trace_range', 'trace_cv', 'trace_min',
    'trace_max'
}

# Baseline features (original v9 features)
all_cols = set(df.columns)
baseline_features = sorted(list(all_cols - NON_FEATURE_COLS))

print(f'\nBaseline features: {len(baseline_features)}')
print('Baseline feature set:', baseline_features[:10], '...')

# Enhanced features (baseline + trace_range)
enhanced_features = baseline_features + ['trace_range']
print(f'\nEnhanced features: {len(enhanced_features)}')
print('Added: trace_range')

# Check for missing values in trace_range
trace_range_valid = df['trace_range'].notna()
print(f'\nValid trace_range values: {trace_range_valid.sum():,} / {len(df):,} ({trace_range_valid.mean()*100:.1f}%)')

# Filter to valid data
df_valid = df[trace_range_valid].copy()
print(f'Using {len(df_valid):,} neurons with valid trace_range')

# Get labels
y = df_valid['ground_truth'].values
print(f'\nKEEP: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)')
print(f'DELETE: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)')

# Get session info for stratified CV
if 'session_name' in df_valid.columns:
    session_col = 'session_name'
elif 'session' in df_valid.columns:
    session_col = 'session'
else:
    raise ValueError('No session column found')

sessions = df_valid[session_col].values
unique_sessions = df_valid[session_col].unique()

# Map sessions to experiments
if 'experiment' in df_valid.columns:
    session_to_exp = df_valid.groupby(session_col)['experiment'].first().to_dict()
    experiments = np.array([session_to_exp[s] for s in unique_sessions])
else:
    experiments = np.array([s.split('_')[0] for s in unique_sessions])

print(f'\nSessions: {len(unique_sessions)}')
print(f'Experiments: {len(set(experiments))}')

# Prepare data matrices
X_baseline = df_valid[baseline_features].values
X_enhanced = df_valid[enhanced_features].values

print(f'\nX_baseline shape: {X_baseline.shape}')
print(f'X_enhanced shape: {X_enhanced.shape}')

# Model hyperparameters (same as v9_iter8)
model_params = {
    'max_bins': 1024,
    'interactions': 20,
    'max_leaves': 3,
    'min_samples_leaf': 5,
    'outer_bags': 8,
    'learning_rate': 0.01,
    'max_rounds': 5000,
    'early_stopping_rounds': 50,
    'random_state': 46  # Same seed as v9_iter8
}

print('\n' + '='*80)
print('MODEL HYPERPARAMETERS')
print('='*80)
for key, val in model_params.items():
    print(f'  {key}: {val}')

# Cross-validation setup
N_SPLITS = 10
TEST_SIZE = 0.25
THRESHOLD = 0.72  # Optimal threshold from v9_iter8
FBETA_BETA = 0.577

print('\n' + '='*80)
print('CROSS-VALIDATION SETUP')
print('='*80)
print(f'  N_splits: {N_SPLITS}')
print(f'  Test_size: {TEST_SIZE}')
print(f'  Threshold: {THRESHOLD}')
print(f'  F-beta beta: {FBETA_BETA}')

# Function to evaluate model
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
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }

# Run cross-validation for both models
print('\n' + '='*80)
print('RUNNING CROSS-VALIDATION')
print('='*80)

baseline_metrics = []
enhanced_metrics = []

for split_idx in range(N_SPLITS):
    seed = 42 + split_idx
    print(f'\nSplit {split_idx+1}/{N_SPLITS} (seed={seed})')

    # Stratified split by session/experiment
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_sessions, test_sessions = next(splitter.split(unique_sessions, experiments))

    train_sessions_set = set(unique_sessions[train_sessions])
    test_sessions_set = set(unique_sessions[test_sessions])

    train_mask = np.array([s in train_sessions_set for s in sessions])
    test_mask = np.array([s in test_sessions_set for s in sessions])

    X_baseline_train, X_baseline_test = X_baseline[train_mask], X_baseline[test_mask]
    X_enhanced_train, X_enhanced_test = X_enhanced[train_mask], X_enhanced[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f'  Train: {len(y_train):,} neurons, Test: {len(y_test):,} neurons')

    # Train baseline model
    print('  Training baseline model...')
    model_baseline = ExplainableBoostingClassifier(**model_params)
    model_baseline.fit(X_baseline_train, y_train)
    metrics_baseline = evaluate_model(model_baseline, X_baseline_test, y_test, THRESHOLD)
    baseline_metrics.append(metrics_baseline)

    # Train enhanced model
    print('  Training enhanced model (+ trace_range)...')
    model_enhanced = ExplainableBoostingClassifier(**model_params)
    model_enhanced.fit(X_enhanced_train, y_train)
    metrics_enhanced = evaluate_model(model_enhanced, X_enhanced_test, y_test, THRESHOLD)
    enhanced_metrics.append(metrics_enhanced)

    # Report
    print(f'  Baseline:  F-beta={metrics_baseline["fbeta"]:.4f}, AUC={metrics_baseline["auc"]:.4f}')
    print(f'  Enhanced:  F-beta={metrics_enhanced["fbeta"]:.4f}, AUC={metrics_enhanced["auc"]:.4f}')
    print(f'  Δ F-beta:  {(metrics_enhanced["fbeta"]-metrics_baseline["fbeta"])*100:+.2f}%')
    print(f'  Δ AUC:     {(metrics_enhanced["auc"]-metrics_baseline["auc"])*100:+.2f}%')

# Aggregate results
print('\n' + '='*80)
print('CROSS-VALIDATION RESULTS')
print('='*80)

def summarize_metrics(metrics_list, name):
    """Summarize metrics across CV splits."""
    results = {
        'precision': np.mean([m['precision'] for m in metrics_list]),
        'recall': np.mean([m['recall'] for m in metrics_list]),
        'fbeta': np.mean([m['fbeta'] for m in metrics_list]),
        'auc': np.mean([m['auc'] for m in metrics_list]),
        'accuracy': np.mean([m['accuracy'] for m in metrics_list])
    }
    stds = {
        'precision': np.std([m['precision'] for m in metrics_list]),
        'recall': np.std([m['recall'] for m in metrics_list]),
        'fbeta': np.std([m['fbeta'] for m in metrics_list]),
        'auc': np.std([m['auc'] for m in metrics_list]),
        'accuracy': np.std([m['accuracy'] for m in metrics_list])
    }

    print(f'\n{name}:')
    print(f'  Precision: {results["precision"]:.4f} ± {stds["precision"]:.4f}')
    print(f'  Recall:    {results["recall"]:.4f} ± {stds["recall"]:.4f}')
    print(f'  F-beta:    {results["fbeta"]:.4f} ± {stds["fbeta"]:.4f}')
    print(f'  AUC:       {results["auc"]:.4f} ± {stds["auc"]:.4f}')
    print(f'  Accuracy:  {results["accuracy"]:.4f} ± {stds["accuracy"]:.4f}')

    return results, stds

baseline_mean, baseline_std = summarize_metrics(baseline_metrics, 'BASELINE (original features)')
enhanced_mean, enhanced_std = summarize_metrics(enhanced_metrics, 'ENHANCED (+ trace_range)')

# Statistical significance test
print('\n' + '='*80)
print('STATISTICAL SIGNIFICANCE (paired t-test)')
print('='*80)

fbeta_baseline = [m['fbeta'] for m in baseline_metrics]
fbeta_enhanced = [m['fbeta'] for m in enhanced_metrics]
auc_baseline = [m['auc'] for m in baseline_metrics]
auc_enhanced = [m['auc'] for m in enhanced_metrics]

t_fbeta, p_fbeta = stats.ttest_rel(fbeta_enhanced, fbeta_baseline)
t_auc, p_auc = stats.ttest_rel(auc_enhanced, auc_baseline)

fbeta_improvement = (enhanced_mean['fbeta'] - baseline_mean['fbeta']) * 100
auc_improvement = (enhanced_mean['auc'] - baseline_mean['auc']) * 100

print(f'\nF-beta:')
print(f'  Baseline:    {baseline_mean["fbeta"]:.4f} ± {baseline_std["fbeta"]:.4f}')
print(f'  Enhanced:    {enhanced_mean["fbeta"]:.4f} ± {enhanced_std["fbeta"]:.4f}')
print(f'  Improvement: {fbeta_improvement:+.2f}%')
print(f'  t-statistic: {t_fbeta:.3f}')
print(f'  p-value:     {p_fbeta:.4f}')
if p_fbeta < 0.05:
    print(f'  Result:      SIGNIFICANT (p < 0.05)')
else:
    print(f'  Result:      Not significant')

print(f'\nAUC:')
print(f'  Baseline:    {baseline_mean["auc"]:.4f} ± {baseline_std["auc"]:.4f}')
print(f'  Enhanced:    {enhanced_mean["auc"]:.4f} ± {enhanced_std["auc"]:.4f}')
print(f'  Improvement: {auc_improvement:+.2f}%')
print(f'  t-statistic: {t_auc:.3f}')
print(f'  p-value:     {p_auc:.4f}')
if p_auc < 0.05:
    print(f'  Result:      SIGNIFICANT (p < 0.05)')
else:
    print(f'  Result:      Not significant')

# Check trace_range importance in enhanced model
print('\n' + '='*80)
print('FEATURE IMPORTANCE IN ENHANCED MODEL')
print('='*80)

# Train final enhanced model on full data
print('\nTraining final enhanced model on full dataset...')
model_final = ExplainableBoostingClassifier(**model_params)
model_final.fit(X_enhanced, y)

# Get importances (first len(enhanced_features) are main effects)
importances = model_final.term_importances()[:len(enhanced_features)]

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': enhanced_features,
    'importance': importances
}).sort_values('importance', ascending=False)

print('\nTop 15 features:')
print(importance_df.head(15).to_string(index=False))

# Find trace_range rank
trace_range_rank = importance_df.reset_index(drop=True).index[importance_df['feature'] == 'trace_range'].tolist()[0] + 1
trace_range_importance = importance_df[importance_df['feature'] == 'trace_range']['importance'].values[0]

print(f'\ntrace_range rank: #{trace_range_rank} / {len(enhanced_features)}')
print(f'trace_range importance: {trace_range_importance:.4f}')

# Save results
results_df = pd.DataFrame({
    'model': ['baseline', 'enhanced'],
    'fbeta_mean': [baseline_mean['fbeta'], enhanced_mean['fbeta']],
    'fbeta_std': [baseline_std['fbeta'], enhanced_std['fbeta']],
    'auc_mean': [baseline_mean['auc'], enhanced_mean['auc']],
    'auc_std': [baseline_std['auc'], enhanced_std['auc']],
    'precision_mean': [baseline_mean['precision'], enhanced_mean['precision']],
    'precision_std': [baseline_std['precision'], enhanced_std['precision']],
    'recall_mean': [baseline_mean['recall'], enhanced_mean['recall']],
    'recall_std': [baseline_std['recall'], enhanced_std['recall']],
})

output_path = 'ml/results/trace_range_experiment_results.csv'
results_df.to_csv(output_path, index=False)
print(f'\nResults saved to: {output_path}')

# Final summary
print('\n' + '='*80)
print('CONCLUSION')
print('='*80)

if p_fbeta < 0.05 and fbeta_improvement > 0:
    print('\nAdding trace_range SIGNIFICANTLY IMPROVES model performance!')
    print(f'F-beta improvement: {fbeta_improvement:+.2f}% (p={p_fbeta:.4f})')
    print('\nRECOMMENDATION: Include trace_range in production model.')
elif fbeta_improvement > 0.5:
    print('\nAdding trace_range shows positive trend but not statistically significant.')
    print(f'F-beta improvement: {fbeta_improvement:+.2f}% (p={p_fbeta:.4f})')
    print('\nRECOMMENDATION: Test on full dataset (92k neurons) for more power.')
else:
    print('\nAdding trace_range shows minimal or no improvement.')
    print(f'F-beta improvement: {fbeta_improvement:+.2f}% (p={p_fbeta:.4f})')
    print('\nRECOMMENDATION: Current features already capture this information.')

print('\n' + '='*80)
print('EXPERIMENT COMPLETE')
print('='*80)
