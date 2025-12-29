"""
Cross-validation experiment: v9 baseline vs v9 + trace_range

Tests whether adding trace_range improves model performance
on the full dataset with 10-fold stratified CV.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from interpret.glassbox import ExplainableBoostingClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('CV EXPERIMENT: BASELINE vs BASELINE + trace_range')
print('='*80)

# Load dataset with trace_range
dataset_path = 'ml/results/training_dataset_v9_with_trace_range.csv'
print(f'\nLoading: {dataset_path}')
df = pd.read_csv(dataset_path)
print(f'Total neurons: {len(df):,}')

# Filter to neurons with trace_range
df_valid = df[df['trace_range'].notna()].copy()
print(f'Neurons with trace_range: {len(df_valid):,}')

# Get feature columns
NON_FEATURE_COLS = {
    'session_name', 'session', 'component_idx', 'ground_truth',
    'decision', 'delete', 'merge', 'experiment',
    'failed_area', 'failed_circularity', 'failed_corner_artifact',
    'is_corner_artifact', 'ml_keep_probability',
    'y_proba', 'y_pred', 'trace_range'
}

baseline_features = sorted([col for col in df_valid.columns if col not in NON_FEATURE_COLS])
enhanced_features = baseline_features + ['trace_range']

print(f'\nBaseline features: {len(baseline_features)}')
print(f'Enhanced features: {len(enhanced_features)} (+ trace_range)')

# Prepare data
y = df_valid['ground_truth'].values
X_baseline = df_valid[baseline_features].values
X_enhanced = df_valid[enhanced_features].values

print(f'\nDataset:')
print(f'  KEEP: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)')
print(f'  DELETE: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)')

# Session info for stratified CV
session_col = 'session_name' if 'session_name' in df_valid.columns else 'session'
sessions = df_valid[session_col].values
unique_sessions = df_valid[session_col].unique()

if 'experiment' in df_valid.columns:
    session_to_exp = df_valid.groupby(session_col)['experiment'].first().to_dict()
    experiments = np.array([session_to_exp[s] for s in unique_sessions])
else:
    experiments = np.array([s.split('_')[0] for s in unique_sessions])

print(f'\nSessions: {len(unique_sessions)}')
print(f'Experiments: {len(set(experiments))}')

# Model hyperparameters (match v9_iter8)
model_params = {
    'max_bins': 1024,
    'interactions': 20,
    'max_leaves': 3,
    'min_samples_leaf': 5,
    'outer_bags': 8,
    'learning_rate': 0.01,
    'max_rounds': 5000,
    'early_stopping_rounds': 50,
    'random_state': 46
}

# CV parameters
N_SPLITS = 10
TEST_SIZE = 0.25
THRESHOLD = 0.72
FBETA_BETA = 0.577

print('\n' + '='*80)
print('CROSS-VALIDATION SETUP')
print('='*80)
print(f'  Splits: {N_SPLITS}')
print(f'  Test size: {TEST_SIZE}')
print(f'  Threshold: {THRESHOLD}')
print(f'  F-beta β: {FBETA_BETA}')

def evaluate_model(model, X, y, threshold):
    """Evaluate model at threshold."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tp = ((y_pred == 1) & (y == 1)).sum()
    fp = ((y_pred == 1) & (y == 0)).sum()
    fn = ((y_pred == 0) & (y == 1)).sum()
    tn = ((y_pred == 0) & (y == 0)).sum()

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
        'accuracy': accuracy
    }

# Run CV
print('\n' + '='*80)
print('RUNNING CROSS-VALIDATION')
print('='*80)

baseline_metrics = []
enhanced_metrics = []

for split_idx in range(N_SPLITS):
    seed = 42 + split_idx
    print(f'\n[Split {split_idx+1}/{N_SPLITS}] seed={seed}')

    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_sessions, test_sessions = next(splitter.split(unique_sessions, experiments))

    train_sessions_set = set(unique_sessions[train_sessions])
    test_sessions_set = set(unique_sessions[test_sessions])

    train_mask = np.array([s in train_sessions_set for s in sessions])
    test_mask = np.array([s in test_sessions_set for s in sessions])

    X_baseline_train, X_baseline_test = X_baseline[train_mask], X_baseline[test_mask]
    X_enhanced_train, X_enhanced_test = X_enhanced[train_mask], X_enhanced[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f'  Train: {len(y_train):,}, Test: {len(y_test):,}')

    # Baseline model
    print('  Training baseline...')
    model_baseline = ExplainableBoostingClassifier(**model_params)
    model_baseline.fit(X_baseline_train, y_train)
    metrics_baseline = evaluate_model(model_baseline, X_baseline_test, y_test, THRESHOLD)
    baseline_metrics.append(metrics_baseline)

    # Enhanced model
    print('  Training enhanced (+trace_range)...')
    model_enhanced = ExplainableBoostingClassifier(**model_params)
    model_enhanced.fit(X_enhanced_train, y_train)
    metrics_enhanced = evaluate_model(model_enhanced, X_enhanced_test, y_test, THRESHOLD)
    enhanced_metrics.append(metrics_enhanced)

    # Report
    print(f'  Baseline:  F={metrics_baseline["fbeta"]:.4f}, AUC={metrics_baseline["auc"]:.4f}')
    print(f'  Enhanced:  F={metrics_enhanced["fbeta"]:.4f}, AUC={metrics_enhanced["auc"]:.4f}')
    print(f'  Δ F-beta:  {(metrics_enhanced["fbeta"]-metrics_baseline["fbeta"])*100:+.2f}%')

# Aggregate results
print('\n' + '='*80)
print('RESULTS')
print('='*80)

def summarize(metrics, name):
    results = {k: np.mean([m[k] for m in metrics]) for k in ['precision', 'recall', 'fbeta', 'auc', 'accuracy']}
    stds = {k: np.std([m[k] for m in metrics]) for k in ['precision', 'recall', 'fbeta', 'auc', 'accuracy']}

    print(f'\n{name}:')
    for k in ['fbeta', 'auc', 'precision', 'recall', 'accuracy']:
        print(f'  {k:10s}: {results[k]:.4f} ± {stds[k]:.4f}')

    return results, stds

baseline_mean, baseline_std = summarize(baseline_metrics, 'BASELINE')
enhanced_mean, enhanced_std = summarize(enhanced_metrics, 'ENHANCED (+trace_range)')

# Statistical test
print('\n' + '='*80)
print('STATISTICAL SIGNIFICANCE')
print('='*80)

fbeta_base = [m['fbeta'] for m in baseline_metrics]
fbeta_enh = [m['fbeta'] for m in enhanced_metrics]
auc_base = [m['auc'] for m in baseline_metrics]
auc_enh = [m['auc'] for m in enhanced_metrics]

t_fbeta, p_fbeta = stats.ttest_rel(fbeta_enh, fbeta_base)
t_auc, p_auc = stats.ttest_rel(auc_enh, auc_base)

fbeta_imp = (enhanced_mean['fbeta'] - baseline_mean['fbeta']) * 100
auc_imp = (enhanced_mean['auc'] - baseline_mean['auc']) * 100

print(f'\nF-beta:')
print(f'  Baseline:    {baseline_mean["fbeta"]:.4f} ± {baseline_std["fbeta"]:.4f}')
print(f'  Enhanced:    {enhanced_mean["fbeta"]:.4f} ± {enhanced_std["fbeta"]:.4f}')
print(f'  Improvement: {fbeta_imp:+.2f}%')
print(f'  p-value:     {p_fbeta:.4f}')
print(f'  Result:      {"SIGNIFICANT" if p_fbeta < 0.05 else "Not significant"}')

print(f'\nAUC:')
print(f'  Baseline:    {baseline_mean["auc"]:.4f} ± {baseline_std["auc"]:.4f}')
print(f'  Enhanced:    {enhanced_mean["auc"]:.4f} ± {enhanced_std["auc"]:.4f}')
print(f'  Improvement: {auc_imp:+.2f}%')
print(f'  p-value:     {p_auc:.4f}')
print(f'  Result:      {"SIGNIFICANT" if p_auc < 0.05 else "Not significant"}')

# Feature importance check
print('\n' + '='*80)
print('FEATURE IMPORTANCE IN ENHANCED MODEL')
print('='*80)

print('\nTraining final model on full dataset...')
model_final = ExplainableBoostingClassifier(**model_params)
model_final.fit(X_enhanced, y)

importances = model_final.term_importances()[:len(enhanced_features)]
imp_df = pd.DataFrame({
    'feature': enhanced_features,
    'importance': importances
}).sort_values('importance', ascending=False)

print('\nTop 20 features:')
for i, row in imp_df.head(20).iterrows():
    marker = ' ← NEW' if row['feature'] == 'trace_range' else ''
    print(f'  {i+1:2d}. {row["feature"]:25s} {row["importance"]:.4f}{marker}')

trace_range_rank = imp_df.reset_index(drop=True).index[imp_df['feature'] == 'trace_range'].tolist()[0] + 1
trace_range_imp = imp_df[imp_df['feature'] == 'trace_range']['importance'].values[0]

print(f'\ntrace_range:')
print(f'  Rank: #{trace_range_rank} / {len(enhanced_features)}')
print(f'  Importance: {trace_range_imp:.4f}')

# Save results
results_df = pd.DataFrame({
    'model': ['baseline', 'enhanced'],
    'fbeta_mean': [baseline_mean['fbeta'], enhanced_mean['fbeta']],
    'fbeta_std': [baseline_std['fbeta'], enhanced_std['fbeta']],
    'auc_mean': [baseline_mean['auc'], enhanced_mean['auc']],
    'auc_std': [baseline_std['auc'], enhanced_std['auc']],
})

output_path = 'ml/results/cv_trace_range_experiment.csv'
results_df.to_csv(output_path, index=False)
print(f'\nResults saved: {output_path}')

# Conclusion
print('\n' + '='*80)
print('CONCLUSION')
print('='*80)

if p_fbeta < 0.05:
    if fbeta_imp > 0:
        print(f'\ntrace_range SIGNIFICANTLY IMPROVES performance!')
        print(f'  F-beta: {fbeta_imp:+.2f}% (p={p_fbeta:.4f})')
        print(f'\nRECOMMENDATION: Add trace_range to production model.')
    else:
        print(f'\ntrace_range significantly DEGRADES performance.')
        print(f'  F-beta: {fbeta_imp:+.2f}% (p={p_fbeta:.4f})')
        print(f'\nRECOMMENDATION: Do NOT add trace_range.')
else:
    print(f'\nNo significant effect from adding trace_range.')
    print(f'  F-beta: {fbeta_imp:+.2f}% (p={p_fbeta:.4f})')
    if abs(fbeta_imp) > 0.5:
        print(f'\nRECOMMENDATION: Suggestive trend, test on full 92k dataset.')
    else:
        print(f'\nRECOMMENDATION: Current features already capture this.')

print('\n' + '='*80)
print('EXPERIMENT COMPLETE')
print('='*80)
