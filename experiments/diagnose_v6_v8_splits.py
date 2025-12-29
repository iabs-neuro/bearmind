"""
Diagnose train/test splits and evaluate models on full datasets.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

def compute_fbeta(precision, recall, beta=0.5773502691896257):
    """Compute F-beta score (beta=0.577 favors precision)."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

print('='*80)
print('DIAGNOSING v6_no3dm vs v8 TRAIN/TEST SPLITS')
print('='*80)

# Load datasets
v6 = pd.read_csv('ml/results/training_dataset_v6_no3dm.csv')
v8 = pd.read_csv('ml/results/training_dataset_v8.csv')

print(f'\nDataset sizes:')
print(f'  v6_no3dm: {len(v6):,} neurons')
print(f'  v8:       {len(v8):,} neurons')

# Check if same sessions
v6_sessions = set(v6['session'].unique())
v8_sessions = set(v8['session'].unique())
print(f'\nSession counts:')
print(f'  v6_no3dm: {len(v6_sessions)} sessions')
print(f'  v8:       {len(v8_sessions)} sessions')
print(f'  Sessions match: {v6_sessions == v8_sessions}')

# Overall class balance
print(f'\nOverall class balance:')
v6_gt = v6['ground_truth'].value_counts()
v8_gt = v8['ground_truth'].value_counts()
print(f'  v6_no3dm: KEEP={v6_gt.get(1, 0):,} ({v6_gt.get(1, 0)/len(v6)*100:.1f}%), DELETE={v6_gt.get(0, 0):,} ({v6_gt.get(0, 0)/len(v6)*100:.1f}%)')
print(f'  v8:       KEEP={v8_gt.get(1, 0):,} ({v8_gt.get(1, 0)/len(v8)*100:.1f}%), DELETE={v8_gt.get(0, 0):,} ({v8_gt.get(0, 0)/len(v8)*100:.1f}%)')

# Simulate same train/test split as grid search
def create_split(df, test_fraction=0.25, random_state=42):
    """Create stratified session split."""
    sessions = df['session'].unique()
    session_experiments = {s: s.split('_')[0] for s in sessions}
    experiments = [session_experiments[s] for s in sessions]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
    train_idx, test_idx = next(splitter.split(sessions, experiments))
    train_sessions = set(sessions[train_idx])
    test_sessions = set(sessions[test_idx])

    train_mask = df['session'].isin(train_sessions)
    test_mask = df['session'].isin(test_sessions)

    return train_mask, test_mask, train_sessions, test_sessions

print(f'\n{"="*80}')
print('TRAIN/TEST SPLIT ANALYSIS (test_fraction=0.25, seed=42)')
print('='*80)

# v6 split
v6_train_mask, v6_test_mask, v6_train_sessions, v6_test_sessions = create_split(v6)
v6_train = v6[v6_train_mask]
v6_test = v6[v6_test_mask]

print(f'\nv6_no3dm split:')
print(f'  Train: {len(v6_train_sessions)} sessions, {len(v6_train):,} neurons')
print(f'    KEEP: {v6_train["ground_truth"].sum():,} ({v6_train["ground_truth"].mean()*100:.1f}%)')
print(f'    DELETE: {(v6_train["ground_truth"] == 0).sum():,} ({(v6_train["ground_truth"] == 0).mean()*100:.1f}%)')
print(f'  Test: {len(v6_test_sessions)} sessions, {len(v6_test):,} neurons')
print(f'    KEEP: {v6_test["ground_truth"].sum():,} ({v6_test["ground_truth"].mean()*100:.1f}%)')
print(f'    DELETE: {(v6_test["ground_truth"] == 0).sum():,} ({(v6_test["ground_truth"] == 0).mean()*100:.1f}%)')

# v8 split
v8_train_mask, v8_test_mask, v8_train_sessions, v8_test_sessions = create_split(v8)
v8_train = v8[v8_train_mask]
v8_test = v8[v8_test_mask]

print(f'\nv8 split:')
print(f'  Train: {len(v8_train_sessions)} sessions, {len(v8_train):,} neurons')
print(f'    KEEP: {v8_train["ground_truth"].sum():,} ({v8_train["ground_truth"].mean()*100:.1f}%)')
print(f'    DELETE: {(v8_train["ground_truth"] == 0).sum():,} ({(v8_train["ground_truth"] == 0).mean()*100:.1f}%)')
print(f'  Test: {len(v8_test_sessions)} sessions, {len(v8_test):,} neurons')
print(f'    KEEP: {v8_test["ground_truth"].sum():,} ({v8_test["ground_truth"].mean()*100:.1f}%)')
print(f'    DELETE: {(v8_test["ground_truth"] == 0).sum():,} ({(v8_test["ground_truth"] == 0).mean()*100:.1f}%)')

# Check if same test sessions
print(f'\nTest sessions match: {v6_test_sessions == v8_test_sessions}')

# Class balance comparison
print(f'\n{"="*80}')
print('CLASS BALANCE COMPARISON')
print('='*80)
print(f'\n{"Dataset":<20} {"Train KEEP%":<15} {"Test KEEP%":<15} {"Difference"}')
print('-'*65)
print(f'{"v6_no3dm":<20} {v6_train["ground_truth"].mean()*100:<15.1f} {v6_test["ground_truth"].mean()*100:<15.1f} {(v6_test["ground_truth"].mean() - v6_train["ground_truth"].mean())*100:+.1f}')
print(f'{"v8":<20} {v8_train["ground_truth"].mean()*100:<15.1f} {v8_test["ground_truth"].mean()*100:<15.1f} {(v8_test["ground_truth"].mean() - v8_train["ground_truth"].mean())*100:+.1f}')

# Now evaluate best models on FULL datasets
print(f'\n{"="*80}')
print('EVALUATING BEST MODELS ON FULL DATASETS (like pr_comparison plots)')
print('='*80)

# Load best models
v6_model_path = 'ml/ebm_grid_search_v6_no3dm/ebm_best.pkl'
v8_model_path = 'ml/ebm_grid_search_v8/ebm_best.pkl'

with open(v6_model_path, 'rb') as f:
    v6_model = pickle.load(f)
with open(v8_model_path, 'rb') as f:
    v8_model = pickle.load(f)

print(f'\nLoaded models:')
print(f'  v6_no3dm: {v6_model_path}')
print(f'  v8:       {v8_model_path}')

# Prepare features
exclude_cols = {'ground_truth', 'session', 'experiment', 'component_idx', 'center',
                'distance_to_gt', 'is_corner_artifact', 'corr_groups'}

# v6 evaluation on full v6 dataset
v6_feature_cols = [c for c in v6.columns if c not in exclude_cols and c in v6_model.feature_names_in_]
X_v6_full = v6[v6_feature_cols].copy()
y_v6_full = v6['ground_truth'].values
y_v6_proba = v6_model.predict_proba(X_v6_full)[:, 1]

# v8 evaluation on full v8 dataset
v8_feature_cols = [c for c in v8.columns if c not in exclude_cols and c in v8_model.feature_names_in_]
X_v8_full = v8[v8_feature_cols].copy()
y_v8_full = v8['ground_truth'].values
y_v8_proba = v8_model.predict_proba(X_v8_full)[:, 1]

# Evaluate at key thresholds
print(f'\n{"Threshold":<12} {"Dataset":<15} {"Precision":<12} {"Recall":<12} {"F-beta":<12}')
print('-'*65)

for thresh in [0.5, 0.6, 0.7, 0.75, 0.8]:
    # v6
    y_v6_pred = (y_v6_proba >= thresh).astype(int)
    tp_v6 = ((y_v6_pred == 1) & (y_v6_full == 1)).sum()
    fp_v6 = ((y_v6_pred == 1) & (y_v6_full == 0)).sum()
    fn_v6 = ((y_v6_pred == 0) & (y_v6_full == 1)).sum()
    prec_v6 = tp_v6 / (tp_v6 + fp_v6) if (tp_v6 + fp_v6) > 0 else 0
    rec_v6 = tp_v6 / (tp_v6 + fn_v6) if (tp_v6 + fn_v6) > 0 else 0
    fb_v6 = compute_fbeta(prec_v6, rec_v6)

    # v8
    y_v8_pred = (y_v8_proba >= thresh).astype(int)
    tp_v8 = ((y_v8_pred == 1) & (y_v8_full == 1)).sum()
    fp_v8 = ((y_v8_pred == 1) & (y_v8_full == 0)).sum()
    fn_v8 = ((y_v8_pred == 0) & (y_v8_full == 1)).sum()
    prec_v8 = tp_v8 / (tp_v8 + fp_v8) if (tp_v8 + fp_v8) > 0 else 0
    rec_v8 = tp_v8 / (tp_v8 + fn_v8) if (tp_v8 + fn_v8) > 0 else 0
    fb_v8 = compute_fbeta(prec_v8, rec_v8)

    print(f'{thresh:<12.2f} {"v6_no3dm":<15} {prec_v6:<12.4f} {rec_v6:<12.4f} {fb_v6:<12.4f}')
    print(f'{"":<12} {"v8":<15} {prec_v8:<12.4f} {rec_v8:<12.4f} {fb_v8:<12.4f}')
    print(f'{"":<12} {"DIFFERENCE":<15} {prec_v8-prec_v6:<12.4f} {rec_v8-rec_v6:<12.4f} {fb_v8-fb_v6:<12.4f}')
    print()

# Find best F-beta threshold for each
print(f'\n{"="*80}')
print('BEST F-BETA THRESHOLD (on full datasets)')
print('='*80)

thresholds = np.linspace(0.3, 0.9, 100)

v6_best_fbeta = 0
v6_best_thresh = 0
for thresh in thresholds:
    y_pred = (y_v6_proba >= thresh).astype(int)
    tp = ((y_pred == 1) & (y_v6_full == 1)).sum()
    fp = ((y_pred == 1) & (y_v6_full == 0)).sum()
    fn = ((y_pred == 0) & (y_v6_full == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    fb = compute_fbeta(prec, rec)
    if fb > v6_best_fbeta:
        v6_best_fbeta = fb
        v6_best_thresh = thresh
        v6_best_prec = prec
        v6_best_rec = rec

v8_best_fbeta = 0
v8_best_thresh = 0
for thresh in thresholds:
    y_pred = (y_v8_proba >= thresh).astype(int)
    tp = ((y_pred == 1) & (y_v8_full == 1)).sum()
    fp = ((y_pred == 1) & (y_v8_full == 0)).sum()
    fn = ((y_pred == 0) & (y_v8_full == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    fb = compute_fbeta(prec, rec)
    if fb > v8_best_fbeta:
        v8_best_fbeta = fb
        v8_best_thresh = thresh
        v8_best_prec = prec
        v8_best_rec = rec

print(f'\nv6_no3dm best (FULL dataset):')
print(f'  F-beta: {v6_best_fbeta:.4f} at threshold {v6_best_thresh:.2f}')
print(f'  Precision: {v6_best_prec:.4f}')
print(f'  Recall: {v6_best_rec:.4f}')

print(f'\nv8 best (FULL dataset):')
print(f'  F-beta: {v8_best_fbeta:.4f} at threshold {v8_best_thresh:.2f}')
print(f'  Precision: {v8_best_prec:.4f}')
print(f'  Recall: {v8_best_rec:.4f}')

print(f'\nDifference:')
print(f'  F-beta: {v8_best_fbeta - v6_best_fbeta:+.4f} ({(v8_best_fbeta - v6_best_fbeta)/v6_best_fbeta*100:+.2f}%)')
print(f'  Precision: {v8_best_prec - v6_best_prec:+.4f}')
print(f'  Recall: {v8_best_rec - v6_best_rec:+.4f}')

print(f'\n{"="*80}')
print('SUMMARY')
print('='*80)
print(f'\nKey findings:')
print(f'1. Train/test class balance:')
print(f'   - v6_no3dm: train {v6_train["ground_truth"].mean()*100:.1f}% KEEP, test {v6_test["ground_truth"].mean()*100:.1f}% KEEP')
print(f'   - v8: train {v8_train["ground_truth"].mean()*100:.1f}% KEEP, test {v8_test["ground_truth"].mean()*100:.1f}% KEEP')
print(f'\n2. Grid search results (TEST SET only):')
print(f'   - May not reflect full dataset performance')
print(f'   - Class imbalance between train/test affects metrics')
print(f'\n3. Full dataset evaluation (like pr_comparison plots):')
print(f'   - v6_no3dm: F-beta={v6_best_fbeta:.4f}')
print(f'   - v8: F-beta={v8_best_fbeta:.4f}')
print(f'   - Difference: {v8_best_fbeta - v6_best_fbeta:+.4f}')

if v8_best_fbeta > v6_best_fbeta:
    print(f'\nv8 IS BETTER on full dataset evaluation!')
elif abs(v8_best_fbeta - v6_best_fbeta) < 0.001:
    print(f'\nv8 and v6_no3dm are EQUIVALENT on full dataset evaluation.')
else:
    print(f'\nv6_no3dm is better on full dataset evaluation.')
