"""
Explain v8 model decisions for specific false negative neurons.
Uses EBM's local explanations to understand why the model rejected real neurons.
"""
import pickle
import numpy as np
import pandas as pd
from interpret import show

print('='*80)
print('EXPLAINING v8 FALSE NEGATIVE DECISIONS')
print('='*80)

# Load model and data
with open('production_models/ebm_v8.pkl', 'rb') as f:
    v8_model = pickle.load(f)

# Load FN report
df_fn = pd.read_csv('ml/results/v8_top100_false_negatives.csv')

# Specific real FN cases to analyze
real_fn_indices = [2, 5, 7, 14, 36, 54, 57, 67, 78, 99]  # 0-indexed (subtract 1 from display numbers)

print(f'\nAnalyzing {len(real_fn_indices)} REAL false negatives (model wrong to delete):')
print('='*80)

# Get features
exclude_cols = {'ground_truth', 'session', 'experiment', 'component_idx', 'center',
                'distance_to_gt', 'is_corner_artifact', 'corr_groups', 'error_type',
                'y_proba', 'y_pred'}
feature_cols = [c for c in df_fn.columns if c not in exclude_cols
                and df_fn[c].dtype in ['float64', 'float32', 'int64', 'int32']]

for idx in real_fn_indices:
    row = df_fn.iloc[idx]
    session = row['session']
    comp_idx = int(row['component_idx'])
    proba = row['y_proba']
    gt = int(row['ground_truth'])

    print(f'\n{"="*80}')
    print(f'FN #{idx+1}: {session} component {comp_idx}')
    print(f'Probability: {proba:.3f} (threshold=0.75) - Model says DELETE')
    print(f'Ground Truth: {"KEEP" if gt else "DELETE"}')
    print('='*80)

    # Get feature values for this neuron
    X_neuron = row[feature_cols].values.reshape(1, -1)

    # Get local explanation
    ebm_local = v8_model.explain_local(X_neuron, None)

    # Get feature contributions (scores)
    feature_names = ebm_local.data()['names']
    feature_scores = ebm_local.data()['scores'][0]  # For first (only) sample

    # Sort by contribution (most negative = pushing DELETE)
    contributions = list(zip(feature_names, feature_scores))
    contributions.sort(key=lambda x: x[1])

    print(f'\nTop 10 features PUSHING DELETE (negative contribution):')
    print(f'{"Feature":<30} {"Score":<12} {"Value"}')
    print('-'*80)

    for feat_name, score in contributions[:10]:
        if feat_name in feature_cols:
            feat_idx = feature_cols.index(feat_name)
            feat_value = X_neuron[0, feat_idx]
            print(f'{feat_name:<30} {score:<12.4f} {feat_value:.3f}')
        else:
            print(f'{feat_name:<30} {score:<12.4f} (interaction)')

    print(f'\nTop 5 features PUSHING KEEP (positive contribution):')
    print(f'{"Feature":<30} {"Score":<12} {"Value"}')
    print('-'*80)

    for feat_name, score in contributions[-5:]:
        if feat_name in feature_cols:
            feat_idx = feature_cols.index(feat_name)
            feat_value = X_neuron[0, feat_idx]
            print(f'{feat_name:<30} {score:<12.4f} {feat_value:.3f}')
        else:
            print(f'{feat_name:<30} {score:<12.4f} (interaction)')

    # Show key feature values
    print(f'\nKey Feature Values:')
    key_features = ['caiman_snr', 'r2_score', 'trace_kurtosis', 'trace_skewness',
                   'events_per_min', 'area', 'event_r2_score', 'noise_level']
    for feat in key_features:
        if feat in feature_cols:
            feat_idx = feature_cols.index(feat)
            feat_value = X_neuron[0, feat_idx]
            print(f'  {feat:<25} {feat_value:.3f}')

    # Calculate intercept and total
    intercept = ebm_local.data()['extra']['names'][0]
    intercept_score = ebm_local.data()['extra']['scores'][0]
    total_score = intercept_score + sum(score for _, score in contributions)

    print(f'\nScore Breakdown:')
    print(f'  Intercept: {intercept_score:.4f}')
    print(f'  Feature contributions sum: {sum(score for _, score in contributions):.4f}')
    print(f'  Total logit score: {total_score:.4f}')
    print(f'  Probability: {proba:.4f} (sigmoid of logit)')

# Summary analysis
print(f'\n{"="*80}')
print('SUMMARY: Why Model Rejects Real Neurons')
print('='*80)

# Analyze common patterns
real_fn_df = df_fn.iloc[real_fn_indices]

print(f'\nCommon characteristics of REAL FN (n={len(real_fn_indices)}):')
print(f'{"Feature":<25} {"Mean":<12} {"vs TP Mean":<15} {"Typical Pattern"}')
print('-'*80)

# Load full dataset to get TP stats
df_full = pd.read_csv('ml/results/training_dataset_v8.csv')
from sklearn.model_selection import StratifiedShuffleSplit

sessions = df_full['session'].unique()
session_experiments = {s: s.split('_')[0] for s in sessions}
experiments = [session_experiments[s] for s in sessions]

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(sessions, experiments))
train_sessions = set(sessions[train_idx])
test_sessions = set(sessions[test_idx])
test_mask = df_full['session'].isin(test_sessions)

X_test = df_full.loc[test_mask, feature_cols].copy()
y_test = df_full.loc[test_mask, 'ground_truth'].values
y_proba = v8_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.75).astype(int)
tp_mask = (y_pred == 1) & (y_test == 1)

df_tp = df_full.loc[test_mask][tp_mask]

key_features = ['caiman_snr', 'r2_score', 'trace_kurtosis', 'trace_skewness',
               'events_per_min', 'area', 'event_r2_score', 'noise_level',
               'tau_decay', 'baseline_drift']

for feat in key_features:
    if feat in real_fn_df.columns and feat in df_tp.columns:
        fn_mean = real_fn_df[feat].mean()
        tp_mean = df_tp[feat].mean()
        ratio = fn_mean / tp_mean if tp_mean != 0 else np.nan

        if ratio < 0.8:
            pattern = 'MUCH LOWER'
        elif ratio < 0.95:
            pattern = 'Lower'
        elif ratio > 1.2:
            pattern = 'MUCH HIGHER'
        elif ratio > 1.05:
            pattern = 'Higher'
        else:
            pattern = 'Similar'

        print(f'{feat:<25} {fn_mean:<12.3f} {tp_mean:<15.3f} {pattern} ({ratio:.2f}x)')

print(f'\nModel Decision Pattern:')
print(f'  - Model is OVER-PENALIZING low-quality features')
print(f'  - Even when neurons are real, weak signals trigger rejection')
print(f'  - Conservative bias: "when in doubt, delete"')

print(f'\n{"="*80}')
