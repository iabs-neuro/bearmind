"""
Report top false negative and false positive neurons of v8 model.
Shows detailed analysis of high-confidence errors.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

THRESHOLD = 0.75

print('='*80)
print('v8 MODEL ERROR REPORT')
print('='*80)

# Load model and data
print('\nLoading v8 model and dataset...')
with open('production_models/ebm_v8.pkl', 'rb') as f:
    v8_model = pickle.load(f)

df = pd.read_csv('ml/results/training_dataset_v8.csv')
print(f'Dataset: {len(df):,} neurons')

# Prepare features
exclude_cols = {'ground_truth', 'session', 'experiment', 'component_idx', 'center',
                'distance_to_gt', 'is_corner_artifact', 'corr_groups'}
feature_cols = [c for c in df.columns if c not in exclude_cols
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

# Stratified train/test split (same as training)
sessions = df['session'].unique()
session_experiments = {s: s.split('_')[0] for s in sessions}
experiments = [session_experiments[s] for s in sessions]

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(sessions, experiments))
train_sessions = set(sessions[train_idx])
test_sessions = set(sessions[test_idx])

train_mask = df['session'].isin(train_sessions)
test_mask = df['session'].isin(test_sessions)

X_test = df.loc[test_mask, feature_cols].copy()
y_test = df.loc[test_mask, 'ground_truth'].values

print(f'Test set: {len(X_test):,} neurons')

# Get predictions
y_proba = v8_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= THRESHOLD).astype(int)

# Identify error types
tp_mask = (y_pred == 1) & (y_test == 1)  # True positive: correct KEEP
fp_mask = (y_pred == 1) & (y_test == 0)  # False positive: wrong KEEP (should DELETE)
fn_mask = (y_pred == 0) & (y_test == 1)  # False negative: wrong DELETE (should KEEP)
tn_mask = (y_pred == 0) & (y_test == 0)  # True negative: correct DELETE

n_tp = tp_mask.sum()
n_fp = fp_mask.sum()
n_fn = fn_mask.sum()
n_tn = tn_mask.sum()

print(f'\n{"="*80}')
print('CONFUSION MATRIX (threshold={THRESHOLD})')
print('='*80)
print(f'\nTrue Positives (correct KEEP):  {n_tp:,} ({n_tp/len(X_test)*100:.2f}%)')
print(f'False Positives (wrong KEEP):   {n_fp:,} ({n_fp/len(X_test)*100:.2f}%)  [PRECISION ERRORS]')
print(f'False Negatives (wrong DELETE): {n_fn:,} ({n_fn/len(X_test)*100:.2f}%)  [RECALL ERRORS]')
print(f'True Negatives (correct DELETE): {n_tn:,} ({n_tn/len(X_test)*100:.2f}%)')

precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f'\nPrecision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'F1 Score:  {f1:.4f}')

# Get test set dataframe
df_test = df.loc[test_mask].copy()
df_test['y_proba'] = y_proba
df_test['y_pred'] = y_pred
df_test['error_type'] = 'CORRECT'
df_test.loc[fp_mask, 'error_type'] = 'FP'
df_test.loc[fn_mask, 'error_type'] = 'FN'

# FALSE POSITIVES
print(f'\n{"="*80}')
print('TOP FALSE POSITIVES (model says KEEP, should DELETE)')
print('='*80)

df_fp = df_test[fp_mask].copy()
df_fp = df_fp.sort_values('y_proba', ascending=False)

print(f'\nTotal FP: {len(df_fp):,}')
print(f'High-confidence FP (prob > 0.85): {(df_fp["y_proba"] > 0.85).sum():,}')
print(f'Medium-confidence FP (0.75-0.85): {((df_fp["y_proba"] >= 0.75) & (df_fp["y_proba"] <= 0.85)).sum():,}')

print(f'\nTop 20 False Positives (highest confidence):')
print(f'\n{"#":<4} {"Session":<20} {"Comp":<6} {"Prob":<8} {"Key Features"}')
print('-'*80)

for i, (idx, row) in enumerate(df_fp.head(20).iterrows(), 1):
    session = row['session'][:20]
    comp_idx = int(row['component_idx'])
    prob = row['y_proba']

    # Key features
    snr = row.get('caiman_snr', np.nan)
    r2 = row.get('r2_score', np.nan)
    area = row.get('area', np.nan)

    print(f'{i:<4} {session:<20} {comp_idx:<6} {prob:.3f}    SNR={snr:.2f}, R2={r2:.3f}, Area={area:.0f}')

# Feature analysis for FP
print(f'\n{"="*80}')
print('FALSE POSITIVE FEATURE CHARACTERISTICS')
print('='*80)

print(f'\nComparing FP vs TP (correct KEEP) features:')
print(f'{"Feature":<25} {"FP Mean":<12} {"TP Mean":<12} {"Difference":<12} {"FP/TP Ratio"}')
print('-'*80)

df_tp = df_test[tp_mask].copy()

key_features = [
    'caiman_snr', 'caiman_r_score', 'r2_score', 'event_r2_score', 'snr_recon',
    'area', 'edge_distance', 'nn_distance_center', 'events_per_min',
    'trace_skewness', 'trace_kurtosis', 'noise_level', 'tau_decay'
]

for feat in key_features:
    if feat in df_fp.columns and feat in df_tp.columns:
        fp_mean = df_fp[feat].mean()
        tp_mean = df_tp[feat].mean()
        diff = fp_mean - tp_mean
        ratio = fp_mean / tp_mean if tp_mean != 0 else np.nan

        marker = ' [FP HIGHER]' if diff > 0 else ' [FP LOWER]'
        print(f'{feat:<25} {fp_mean:<12.3f} {tp_mean:<12.3f} {diff:<+12.3f} {ratio:.2f}x{marker}')

# FALSE NEGATIVES
print(f'\n{"="*80}')
print('TOP FALSE NEGATIVES (model says DELETE, should KEEP)')
print('='*80)

df_fn = df_test[fn_mask].copy()
df_fn = df_fn.sort_values('y_proba', ascending=True)

print(f'\nTotal FN: {len(df_fn):,}')
print(f'High-confidence FN (prob < 0.25): {(df_fn["y_proba"] < 0.25).sum():,}')
print(f'Medium-confidence FN (0.25-0.50): {((df_fn["y_proba"] >= 0.25) & (df_fn["y_proba"] < 0.50)).sum():,}')
print(f'Near-threshold FN (0.50-0.75): {((df_fn["y_proba"] >= 0.50) & (df_fn["y_proba"] < 0.75)).sum():,}')

print(f'\nTop 20 False Negatives (lowest confidence):')
print(f'\n{"#":<4} {"Session":<20} {"Comp":<6} {"Prob":<8} {"Key Features"}')
print('-'*80)

for i, (idx, row) in enumerate(df_fn.head(20).iterrows(), 1):
    session = row['session'][:20]
    comp_idx = int(row['component_idx'])
    prob = row['y_proba']

    # Key features
    snr = row.get('caiman_snr', np.nan)
    r2 = row.get('r2_score', np.nan)
    area = row.get('area', np.nan)

    print(f'{i:<4} {session:<20} {comp_idx:<6} {prob:.3f}    SNR={snr:.2f}, R2={r2:.3f}, Area={area:.0f}')

# Feature analysis for FN
print(f'\n{"="*80}')
print('FALSE NEGATIVE FEATURE CHARACTERISTICS')
print('='*80)

print(f'\nComparing FN vs TP (correct KEEP) features:')
print(f'{"Feature":<25} {"FN Mean":<12} {"TP Mean":<12} {"Difference":<12} {"FN/TP Ratio"}')
print('-'*80)

for feat in key_features:
    if feat in df_fn.columns and feat in df_tp.columns:
        fn_mean = df_fn[feat].mean()
        tp_mean = df_tp[feat].mean()
        diff = fn_mean - tp_mean
        ratio = fn_mean / tp_mean if tp_mean != 0 else np.nan

        marker = ' [FN HIGHER]' if diff > 0 else ' [FN LOWER]'
        print(f'{feat:<25} {fn_mean:<12.3f} {tp_mean:<12.3f} {diff:<+12.3f} {ratio:.2f}x{marker}')

# Session-level analysis
print(f'\n{"="*80}')
print('SESSIONS WITH MOST ERRORS')
print('='*80)

# FP by session
fp_by_session = df_fp.groupby('session').size().sort_values(ascending=False).head(10)
print(f'\nTop 10 sessions by FALSE POSITIVES:')
print(f'{"Session":<20} {"FP Count":<10} {"% of Session"}')
print('-'*50)
for session, count in fp_by_session.items():
    total_in_session = len(df_test[df_test['session'] == session])
    pct = count / total_in_session * 100 if total_in_session > 0 else 0
    print(f'{session:<20} {count:<10} {pct:.1f}%')

# FN by session
fn_by_session = df_fn.groupby('session').size().sort_values(ascending=False).head(10)
print(f'\nTop 10 sessions by FALSE NEGATIVES:')
print(f'{"Session":<20} {"FN Count":<10} {"% of Session"}')
print('-'*50)
for session, count in fn_by_session.items():
    total_in_session = len(df_test[df_test['session'] == session])
    pct = count / total_in_session * 100 if total_in_session > 0 else 0
    print(f'{session:<20} {count:<10} {pct:.1f}%')

# Save detailed error report
print(f'\n{"="*80}')
print('SAVING DETAILED ERROR REPORTS')
print('='*80)

# Top 100 FP
fp_report = df_fp.head(100)[['session', 'component_idx', 'y_proba', 'ground_truth'] + key_features].copy()
fp_report['error_type'] = 'FP'
fp_report.to_csv('ml/results/v8_top100_false_positives.csv', index=False)
print(f'\nSaved: ml/results/v8_top100_false_positives.csv')

# Top 100 FN
fn_report = df_fn.head(100)[['session', 'component_idx', 'y_proba', 'ground_truth'] + key_features].copy()
fn_report['error_type'] = 'FN'
fn_report.to_csv('ml/results/v8_top100_false_negatives.csv', index=False)
print(f'Saved: ml/results/v8_top100_false_negatives.csv')

# Combined report
combined_report = pd.concat([fp_report, fn_report], ignore_index=True)
combined_report.to_csv('ml/results/v8_error_report.csv', index=False)
print(f'Saved: ml/results/v8_error_report.csv ({len(combined_report)} neurons)')

print(f'\n{"="*80}')
print('KEY INSIGHTS')
print('='*80)

print(f'\n1. ERROR DISTRIBUTION:')
print(f'   - FP (precision errors): {n_fp:,} neurons ({n_fp/(n_fp+n_fn)*100:.1f}% of errors)')
print(f'   - FN (recall errors): {n_fn:,} neurons ({n_fn/(n_fp+n_fn)*100:.1f}% of errors)')
if n_fn > n_fp:
    print(f'   - Model is MORE likely to DELETE good neurons than KEEP bad ones')
else:
    print(f'   - Model is MORE likely to KEEP bad neurons than DELETE good ones')

print(f'\n2. HIGH-CONFIDENCE ERRORS:')
print(f'   - Very confident FP (prob > 0.85): {(df_fp["y_proba"] > 0.85).sum():,}')
print(f'   - Very confident FN (prob < 0.25): {(df_fn["y_proba"] < 0.25).sum():,}')
print(f'   - These are the hardest errors to fix')

print(f'\n3. FEATURE PATTERNS:')
fp_lower_snr = df_fp['caiman_snr'].mean() < df_tp['caiman_snr'].mean()
fn_lower_snr = df_fn['caiman_snr'].mean() < df_tp['caiman_snr'].mean()
if fp_lower_snr:
    print(f'   - FP neurons have LOWER SNR than TP (confused with good neurons)')
if fn_lower_snr:
    print(f'   - FN neurons have LOWER SNR than TP (borderline quality)')

print(f'\n{"="*80}')
