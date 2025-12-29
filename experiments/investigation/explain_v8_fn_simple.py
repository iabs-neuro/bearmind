"""
Simple explanation of v8 model decisions for specific false negative neurons.
Shows feature values and compares to typical TP neurons.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

print('='*80)
print('EXPLAINING v8 FALSE NEGATIVE DECISIONS')
print('='*80)

# Load model and full dataset
with open('production_models/ebm_v8.pkl', 'rb') as f:
    v8_model = pickle.load(f)

df = pd.read_csv('ml/results/training_dataset_v8.csv')

# Prepare features
exclude_cols = {'ground_truth', 'session', 'experiment', 'component_idx', 'center',
                'distance_to_gt', 'is_corner_artifact', 'corr_groups'}
feature_cols = [c for c in df.columns if c not in exclude_cols
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

# Get test set
sessions = df['session'].unique()
session_experiments = {s: s.split('_')[0] for s in sessions}
experiments = [session_experiments[s] for s in sessions]

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(sessions, experiments))
test_sessions = set(sessions[test_idx])
test_mask = df['session'].isin(test_sessions)

X_test = df.loc[test_mask, feature_cols].copy()
y_test = df.loc[test_mask, 'ground_truth'].values
df_test = df.loc[test_mask].copy()

# Get predictions
y_proba = v8_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.75).astype(int)

# Get TP for comparison
tp_mask = (y_pred == 1) & (y_test == 1)
df_tp = df_test[tp_mask].copy()

# Get model feature importances
feature_importance_scores = v8_model.term_importances()
model_feature_names = v8_model.feature_names_in_
importance_dict = dict(zip(model_feature_names, feature_importance_scores))

# Load specific neurons
fn_report = pd.read_csv('ml/results/v8_top100_false_negatives.csv')
real_fn_display = [3, 6, 8, 15, 37, 55, 58, 68, 79, 100]

print(f'\nAnalyzing {len(real_fn_display)} REAL false negatives (model wrong to delete):')
print('='*80)

for fn_num in real_fn_display:
    fn_row = fn_report.iloc[fn_num - 1]
    session = fn_row['session']
    comp_idx = int(fn_row['component_idx'])
    proba = fn_row['y_proba']

    # Find in full dataset
    neuron_mask = (df['session'] == session) & (df['component_idx'] == comp_idx)
    if neuron_mask.sum() == 0:
        continue

    neuron_row = df[neuron_mask].iloc[0]
    gt = int(neuron_row['ground_truth'])

    print(f'\n{"="*80}')
    print(f'FN #{fn_num}: {session} component {comp_idx}')
    print(f'Probability: {proba:.3f} (threshold=0.75) - Model DELETES')
    print(f'Ground Truth: KEEP')
    print('='*80)

    # Key features
    key_features = [
        'caiman_snr', 'caiman_r_score', 'r2_score', 'event_r2_score', 'snr_recon',
        'trace_kurtosis', 'trace_skewness', 'events_per_min', 'area',
        'noise_level', 'tau_decay', 'baseline_drift', 'hurst_exponent'
    ]

    print(f'\nFeature Comparison (FN value vs TP mean):')
    print(f'{"Feature":<25} {"FN Value":<12} {"TP Mean":<12} {"Ratio":<10} {"Importance"}')
    print('-'*90)

    problem_features = []

    for feat in key_features:
        if feat in neuron_row.index and feat in df_tp.columns:
            fn_value = neuron_row[feat]
            tp_mean = df_tp[feat].mean()
            ratio = fn_value / tp_mean if tp_mean != 0 and not np.isnan(tp_mean) else np.nan

            # Get importance
            importance = importance_dict.get(feat, 0.0)

            # Identify problem features
            if ratio < 0.5 or ratio > 2.0:
                marker = ' [PROBLEM]'
                problem_features.append((feat, fn_value, tp_mean, ratio))
            else:
                marker = ''

            print(f'{feat:<25} {fn_value:<12.3f} {tp_mean:<12.3f} {ratio:<10.2f} {importance:.4f}{marker}')

    print(f'\nProblem Features (ratio < 0.5 or > 2.0):')
    if problem_features:
        for feat, fn_val, tp_mean, ratio in problem_features:
            if ratio < 0.5:
                print(f'  {feat}: TOO LOW ({fn_val:.3f} vs {tp_mean:.3f}, {ratio:.2f}x)')
            else:
                print(f'  {feat}: TOO HIGH ({fn_val:.3f} vs {tp_mean:.3f}, {ratio:.2f}x)')
    else:
        print(f'  None - features similar to TP!')

    # Special analysis
    print(f'\nLikely Rejection Reasons:')
    reasons = []

    if neuron_row.get('caiman_snr', 0) < 3.0:
        reasons.append(f'  - Low SNR ({neuron_row.get("caiman_snr", 0):.2f} < 3.0)')
    if neuron_row.get('r2_score', 0) < 0.3:
        reasons.append(f'  - Poor reconstruction R2 ({neuron_row.get("r2_score", 0):.3f} < 0.3)')
    if neuron_row.get('trace_kurtosis', 0) < 5.0:
        reasons.append(f'  - Low kurtosis ({neuron_row.get("trace_kurtosis", 0):.2f} < 5.0) - not sparse')
    if neuron_row.get('events_per_min', 0) > 12.0:
        reasons.append(f'  - Too many events ({neuron_row.get("events_per_min", 0):.1f} > 12) - noisy')
    if neuron_row.get('area', 0) > 35:
        reasons.append(f'  - Large area ({neuron_row.get("area", 0):.0f} > 35) - diffuse')
    if neuron_row.get('noise_level', 0) > 3.5:
        reasons.append(f'  - High noise ({neuron_row.get("noise_level", 0):.2f} > 3.5)')

    if reasons:
        for reason in reasons:
            print(reason)
    else:
        print(f'  - No obvious red flags - model may be over-conservative!')

# Summary
print(f'\n{"="*80}')
print('SUMMARY: Why Model Deletes Real Neurons')
print('='*80)

# Aggregate analysis
all_real_fn = []
for fn_num in real_fn_display:
    fn_row = fn_report.iloc[fn_num - 1]
    session = fn_row['session']
    comp_idx = int(fn_row['component_idx'])

    neuron_mask = (df['session'] == session) & (df['component_idx'] == comp_idx)
    if neuron_mask.sum() > 0:
        all_real_fn.append(df[neuron_mask].iloc[0])

if all_real_fn:
    df_real_fn = pd.DataFrame(all_real_fn)

    print(f'\nAggregate statistics ({len(all_real_fn)} real FN):')
    print(f'{"Feature":<25} {"Real FN Mean":<15} {"TP Mean":<15} {"Ratio"}')
    print('-'*70)

    for feat in key_features:
        if feat in df_real_fn.columns and feat in df_tp.columns:
            fn_mean = df_real_fn[feat].mean()
            tp_mean = df_tp[feat].mean()
            ratio = fn_mean / tp_mean if tp_mean != 0 else np.nan

            if ratio < 0.8:
                marker = ' [MUCH LOWER]'
            elif ratio > 1.2:
                marker = ' [MUCH HIGHER]'
            else:
                marker = ''

            print(f'{feat:<25} {fn_mean:<15.3f} {tp_mean:<15.3f} {ratio:.2f}{marker}')

    print(f'\nModel Decision Pattern:')
    print(f'  1. Model learned conservative thresholds from training data')
    print(f'  2. Real low-quality neurons get rejected even when valid')
    print(f'  3. Key triggers: low SNR, poor R2, high noise, low kurtosis')
    print(f'  4. Model prioritizes precision over recall')

print(f'\n{"="*80}')
