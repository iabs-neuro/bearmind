"""
Explain v8 model decisions for specific false negative neurons.
Uses EBM's local explanations to understand why the model rejected real neurons.
FIXED: Load full dataset with all features.
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

# Prepare features (same as training)
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
train_sessions = set(sessions[train_idx])
test_sessions = set(sessions[test_idx])
test_mask = df['session'].isin(test_sessions)

X_test = df.loc[test_mask, feature_cols].copy()
y_test = df.loc[test_mask, 'ground_truth'].values
df_test = df.loc[test_mask].copy()

# Get predictions
y_proba = v8_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.75).astype(int)

# Identify FN
fn_mask = (y_pred == 0) & (y_test == 1)
df_fn = df_test[fn_mask].copy()
df_fn['y_proba'] = y_proba[fn_mask]
df_fn = df_fn.sort_values('y_proba')

# Load the specific neurons from the FN report
fn_report = pd.read_csv('ml/results/v8_top100_false_negatives.csv')

# Specific real FN cases to analyze (1-indexed display numbers)
real_fn_display = [3, 6, 8, 15, 37, 55, 58, 68, 79, 100]

print(f'\nAnalyzing {len(real_fn_display)} REAL false negatives (model wrong to delete):')
print('='*80)

for fn_num in real_fn_display:
    # Get neuron info from report
    fn_row = fn_report.iloc[fn_num - 1]
    session = fn_row['session']
    comp_idx = int(fn_row['component_idx'])
    proba = fn_row['y_proba']

    # Find this neuron in full dataset
    neuron_mask = (df['session'] == session) & (df['component_idx'] == comp_idx)
    if neuron_mask.sum() == 0:
        print(f'\nFN #{fn_num}: {session} component {comp_idx} - NOT FOUND in dataset')
        continue

    neuron_row = df[neuron_mask].iloc[0]
    X_neuron = neuron_row[feature_cols].values.reshape(1, -1)
    gt = int(neuron_row['ground_truth'])

    print(f'\n{"="*80}')
    print(f'FN #{fn_num}: {session} component {comp_idx}')
    print(f'Probability: {proba:.3f} (threshold=0.75) - Model says DELETE')
    print(f'Ground Truth: {"KEEP" if gt else "DELETE"}')
    print('='*80)

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
        # Check if it's an interaction
        if ' x ' in feat_name:
            print(f'{feat_name:<30} {score:<12.4f} (interaction)')
        elif feat_name in feature_cols:
            feat_idx = feature_cols.index(feat_name)
            feat_value = X_neuron[0, feat_idx]
            print(f'{feat_name:<30} {score:<12.4f} {feat_value:.3f}')
        else:
            print(f'{feat_name:<30} {score:<12.4f} ???')

    print(f'\nTop 5 features PUSHING KEEP (positive contribution):')
    print(f'{"Feature":<30} {"Score":<12} {"Value"}')
    print('-'*80)

    for feat_name, score in list(reversed(contributions))[:5]:
        if ' x ' in feat_name:
            print(f'{feat_name:<30} {score:<12.4f} (interaction)')
        elif feat_name in feature_cols:
            feat_idx = feature_cols.index(feat_name)
            feat_value = X_neuron[0, feat_idx]
            print(f'{feat_name:<30} {score:<12.4f} {feat_value:.3f}')
        else:
            print(f'{feat_name:<30} {score:<12.4f} ???')

    # Show key feature values
    print(f'\nKey Feature Values:')
    key_features = ['caiman_snr', 'r2_score', 'trace_kurtosis', 'trace_skewness',
                   'events_per_min', 'area', 'event_r2_score', 'noise_level']
    for feat in key_features:
        if feat in feature_cols:
            feat_idx = feature_cols.index(feat)
            feat_value = X_neuron[0, feat_idx]
            print(f'  {feat:<25} {feat_value:.3f}')

    # Calculate total
    intercept_score = ebm_local.data()['extra']['scores'][0]
    total_score = intercept_score + sum(score for _, score in contributions)

    print(f'\nScore Breakdown:')
    print(f'  Intercept: {intercept_score:.4f}')
    print(f'  Feature contributions sum: {sum(score for _, score in contributions):.4f}')
    print(f'  Total logit score: {total_score:.4f}')
    print(f'  Probability: {proba:.4f}')

print(f'\n{"="*80}')
print('DONE')
print('='*80)
