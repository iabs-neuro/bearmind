"""
Compare v6, v7, v8 EBM models on full datasets.
"""
import pandas as pd
import numpy as np
import pickle

def compute_fbeta(precision, recall, beta=0.5773502691896257):
    """Compute F-beta score (beta=0.577 favors precision)."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

print('='*80)
print('FULL DATASET COMPARISON: v6_no3dm vs v7 vs v8')
print('='*80)

# Load datasets
v6 = pd.read_csv('ml/results/training_dataset_v6_no3dm.csv')
v7 = pd.read_csv('ml/results/training_dataset_v7.csv')
v8 = pd.read_csv('ml/results/training_dataset_v8.csv')

print(f'\nDatasets:')
print(f'  v6_no3dm: {len(v6):,} neurons, {len(v6.columns)} features')
print(f'  v7:       {len(v7):,} neurons, {len(v7.columns)} features')
print(f'  v8:       {len(v8):,} neurons, {len(v8.columns)} features')

# Load models
with open('ml/ebm_grid_search_v6_no3dm/ebm_best.pkl', 'rb') as f:
    v6_model = pickle.load(f)
with open('ml/ebm_grid_search_v7/ebm_best.pkl', 'rb') as f:
    v7_model = pickle.load(f)
with open('ml/ebm_grid_search_v8/ebm_best.pkl', 'rb') as f:
    v8_model = pickle.load(f)

print(f'\nModel features:')
print(f'  v6_no3dm: {len(v6_model.feature_names_in_)} features')
print(f'  v7:       {len(v7_model.feature_names_in_)} features')
print(f'  v8:       {len(v8_model.feature_names_in_)} features')

# Check for new features in v8
v6_features = set(v6_model.feature_names_in_)
v7_features = set(v7_model.feature_names_in_)
v8_features = set(v8_model.feature_names_in_)

new_in_v8 = v8_features - v6_features
if new_in_v8:
    print(f'\nNew features in v8: {sorted(new_in_v8)}')

# Prepare features
exclude_cols = {'ground_truth', 'session', 'experiment', 'component_idx', 'center',
                'distance_to_gt', 'is_corner_artifact', 'corr_groups'}

# v6 evaluation
v6_feature_cols = [c for c in v6.columns if c not in exclude_cols and c in v6_model.feature_names_in_]
X_v6 = v6[v6_feature_cols].copy()
y_v6 = v6['ground_truth'].values
y_v6_proba = v6_model.predict_proba(X_v6)[:, 1]

# v7 evaluation
v7_feature_cols = [c for c in v7.columns if c not in exclude_cols and c in v7_model.feature_names_in_]
X_v7 = v7[v7_feature_cols].copy()
y_v7 = v7['ground_truth'].values
y_v7_proba = v7_model.predict_proba(X_v7)[:, 1]

# v8 evaluation
v8_feature_cols = [c for c in v8.columns if c not in exclude_cols and c in v8_model.feature_names_in_]
X_v8 = v8[v8_feature_cols].copy()
y_v8 = v8['ground_truth'].values
y_v8_proba = v8_model.predict_proba(X_v8)[:, 1]

# Evaluate at key thresholds
print(f'\n{"="*80}')
print('PERFORMANCE AT KEY THRESHOLDS (on full datasets)')
print('='*80)

print(f'\n{"Threshold":<12} {"Version":<10} {"Precision":<12} {"Recall":<12} {"F-beta":<12}')
print('-'*60)

for thresh in [0.5, 0.6, 0.7, 0.75, 0.8]:
    # v6
    y_v6_pred = (y_v6_proba >= thresh).astype(int)
    tp = ((y_v6_pred == 1) & (y_v6 == 1)).sum()
    fp = ((y_v6_pred == 1) & (y_v6 == 0)).sum()
    fn = ((y_v6_pred == 0) & (y_v6 == 1)).sum()
    prec_v6 = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_v6 = tp / (tp + fn) if (tp + fn) > 0 else 0
    fb_v6 = compute_fbeta(prec_v6, rec_v6)

    # v7
    y_v7_pred = (y_v7_proba >= thresh).astype(int)
    tp = ((y_v7_pred == 1) & (y_v7 == 1)).sum()
    fp = ((y_v7_pred == 1) & (y_v7 == 0)).sum()
    fn = ((y_v7_pred == 0) & (y_v7 == 1)).sum()
    prec_v7 = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_v7 = tp / (tp + fn) if (tp + fn) > 0 else 0
    fb_v7 = compute_fbeta(prec_v7, rec_v7)

    # v8
    y_v8_pred = (y_v8_proba >= thresh).astype(int)
    tp = ((y_v8_pred == 1) & (y_v8 == 1)).sum()
    fp = ((y_v8_pred == 1) & (y_v8 == 0)).sum()
    fn = ((y_v8_pred == 0) & (y_v8 == 1)).sum()
    prec_v8 = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_v8 = tp / (tp + fn) if (tp + fn) > 0 else 0
    fb_v8 = compute_fbeta(prec_v8, rec_v8)

    print(f'{thresh:<12.2f} {"v6_no3dm":<10} {prec_v6:<12.4f} {rec_v6:<12.4f} {fb_v6:<12.4f}')
    print(f'{"":<12} {"v7":<10} {prec_v7:<12.4f} {rec_v7:<12.4f} {fb_v7:<12.4f}')
    print(f'{"":<12} {"v8":<10} {prec_v8:<12.4f} {rec_v8:<12.4f} {fb_v8:<12.4f}')

    # Find best
    best_fb = max(fb_v6, fb_v7, fb_v8)
    if fb_v6 == best_fb:
        winner = "v6_no3dm"
    elif fb_v7 == best_fb:
        winner = "v7"
    else:
        winner = "v8"
    print(f'{"":<12} {"BEST":<10} {winner}')
    print()

# Find best threshold for each
print(f'\n{"="*80}')
print('BEST F-BETA THRESHOLD FOR EACH VERSION')
print('='*80)

thresholds = np.linspace(0.3, 0.9, 100)

results = {}
for name, y_proba, y_true in [('v6_no3dm', y_v6_proba, y_v6),
                                ('v7', y_v7_proba, y_v7),
                                ('v8', y_v8_proba, y_v8)]:
    best_fbeta = 0
    best_thresh = 0
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        fb = compute_fbeta(prec, rec)
        if fb > best_fbeta:
            best_fbeta = fb
            best_thresh = thresh
            best_prec = prec
            best_rec = rec

    results[name] = {
        'fbeta': best_fbeta,
        'threshold': best_thresh,
        'precision': best_prec,
        'recall': best_rec
    }

print(f'\n{"Version":<12} {"F-beta":<12} {"Threshold":<12} {"Precision":<12} {"Recall"}')
print('-'*60)
for name in ['v6_no3dm', 'v7', 'v8']:
    r = results[name]
    print(f'{name:<12} {r["fbeta"]:<12.4f} {r["threshold"]:<12.2f} {r["precision"]:<12.4f} {r["recall"]:.4f}')

# Comparison
print(f'\n{"="*80}')
print('COMPARISON SUMMARY')
print('='*80)

v6_fb = results['v6_no3dm']['fbeta']
v7_fb = results['v7']['fbeta']
v8_fb = results['v8']['fbeta']

print(f'\nBest F-beta scores:')
print(f'  v6_no3dm: {v6_fb:.4f}')
print(f'  v7:       {v7_fb:.4f}')
print(f'  v8:       {v8_fb:.4f}')

print(f'\nv7 vs v6_no3dm: {v7_fb - v6_fb:+.4f} ({(v7_fb - v6_fb)/v6_fb*100:+.2f}%)')
print(f'v8 vs v6_no3dm: {v8_fb - v6_fb:+.4f} ({(v8_fb - v6_fb)/v6_fb*100:+.2f}%)')
print(f'v8 vs v7:       {v8_fb - v7_fb:+.4f} ({(v8_fb - v7_fb)/v7_fb*100:+.2f}%)')

# Ranking
ranking = sorted([('v6_no3dm', v6_fb), ('v7', v7_fb), ('v8', v8_fb)],
                 key=lambda x: x[1], reverse=True)

print(f'\nRanking by F-beta:')
for i, (name, fb) in enumerate(ranking, 1):
    print(f'  {i}. {name}: {fb:.4f}')

print(f'\nEvent detection methods:')
print(f'  v6_no3dm: Wavelet n=2')
print(f'  v7:       Threshold n=2')
print(f'  v8:       Hybrid kinetics + Wavelet n=3')

print(f'\nKey improvements in v8:')
print(f'  - Hybrid kinetics optimization (cascading tiers)')
print(f'  - Wavelet n=3 (vs n=2)')
print(f'  - New features: hurst_exponent, baseline_drift, kinetics_source')
print(f'  - Event detection rate: 98.2% vs 71.1% (v6/v7)')
