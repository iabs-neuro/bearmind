"""
Retrain v8 model on corrected dataset.
Uses same hyperparameters as original v8 but different seed.
Generates error reports and visualizations for next iteration.
"""
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, fbeta_score
from interpret.glassbox import ExplainableBoostingClassifier

FBETA_BETA = 0.5773502691896257
THRESHOLD = 0.75
RANDOM_SEED = 43  # Changed from 42

print('='*80)
print('RETRAINING v8 ON CORRECTED DATASET (SEED=43)')
print('='*80)

# Load corrected dataset
df = pd.read_csv('ml/results/training_dataset_v8_corrected.csv')
print(f'\nDataset: {len(df):,} neurons')
print(f'Class balance: {df["ground_truth"].mean()*100:.2f}% KEEP')

# Load original v8 for comparison
with open('production_models/ebm_v8.pkl', 'rb') as f:
    v8_original = pickle.load(f)

print(f'\nOriginal v8 hyperparameters:')
print(f'  max_bins: {v8_original.max_bins}')
print(f'  interactions: {v8_original.interactions}')
print(f'  max_leaves: {v8_original.max_leaves}')
print(f'  min_samples_leaf: {v8_original.min_samples_leaf}')
print(f'  random_state: {v8_original.random_state}')

# Prepare data
exclude_cols = {'ground_truth', 'session', 'experiment', 'component_idx', 'center',
                'distance_to_gt', 'is_corner_artifact', 'corr_groups'}

feature_cols = [c for c in df.columns if c not in exclude_cols
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

print(f'\nFeatures: {len(feature_cols)}')

# Stratified train/test split
sessions = df['session'].unique()
session_experiments = {s: s.split('_')[0] for s in sessions}
experiments = [session_experiments[s] for s in sessions]

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
train_idx, test_idx = next(splitter.split(sessions, experiments))
train_sessions = set(sessions[train_idx])
test_sessions = set(sessions[test_idx])

train_mask = df['session'].isin(train_sessions)
test_mask = df['session'].isin(test_sessions)

X_train = df.loc[train_mask, feature_cols].copy()
y_train = df.loc[train_mask, 'ground_truth'].values
X_test = df.loc[test_mask, feature_cols].copy()
y_test = df.loc[test_mask, 'ground_truth'].values

print(f'\nTrain: {len(X_train):,} neurons ({len(train_sessions)} sessions, KEEP: {y_train.mean()*100:.2f}%)')
print(f'Test:  {len(X_test):,} neurons ({len(test_sessions)} sessions, KEEP: {y_test.mean()*100:.2f}%)')

# Train new model (same hyperparameters, different seed)
print(f'\n{"="*80}')
print(f'TRAINING v8_corrected_iter1 (seed={RANDOM_SEED})')
print('='*80)

v8_corrected = ExplainableBoostingClassifier(
    feature_names=feature_cols,
    max_bins=1024,
    max_interaction_bins=64,
    interactions=20,
    outer_bags=8,
    inner_bags=0,
    learning_rate=0.01,
    validation_size=0.15,
    early_stopping_rounds=50,
    early_stopping_tolerance=1e-4,
    max_rounds=5000,
    min_samples_leaf=2,
    max_leaves=3,
    random_state=RANDOM_SEED
)

print(f'\nTraining model...')
start_time = time.time()
v8_corrected.fit(X_train, y_train)
train_time = time.time() - start_time
print(f'Training completed in {train_time:.1f}s')

# Evaluate
print(f'\n{"="*80}')
print('EVALUATION')
print('='*80)

y_train_proba = v8_corrected.predict_proba(X_train)[:, 1]
y_test_proba = v8_corrected.predict_proba(X_test)[:, 1]

y_train_pred = (y_train_proba >= THRESHOLD).astype(int)
y_test_pred = (y_test_proba >= THRESHOLD).astype(int)

def compute_metrics(y_true, y_pred, y_proba):
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    fbeta = fbeta_score(y_true, y_pred, beta=FBETA_BETA, average='binary', zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    acc = (y_pred == y_true).mean()
    return {
        'precision': prec,
        'recall': rec,
        'fbeta': fbeta,
        'auc': auc,
        'accuracy': acc
    }

train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)
test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

print(f'\nTRAIN SET (threshold={THRESHOLD}):')
for metric, value in train_metrics.items():
    print(f'  {metric.capitalize():<12} {value:.4f}')

print(f'\nTEST SET (threshold={THRESHOLD}):')
for metric, value in test_metrics.items():
    print(f'  {metric.capitalize():<12} {value:.4f}')

# Save model
output_path = 'production_models/ebm_v8_corrected_iter1.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(v8_corrected, f)
print(f'\nModel saved to: {output_path}')

# Generate error reports
print(f'\n{"="*80}')
print('GENERATING ERROR REPORTS')
print('='*80)

# Add predictions to test set
df_test = df.loc[test_mask].copy()
df_test['y_proba'] = y_test_proba
df_test['y_pred'] = y_test_pred

# Identify errors
tp_mask = (y_test_pred == 1) & (y_test == 1)
fp_mask = (y_test_pred == 1) & (y_test == 0)
fn_mask = (y_test_pred == 0) & (y_test == 1)
tn_mask = (y_test_pred == 0) & (y_test == 0)

n_tp = tp_mask.sum()
n_fp = fp_mask.sum()
n_fn = fn_mask.sum()
n_tn = tn_mask.sum()

print(f'\nConfusion Matrix:')
print(f'  True Positives:  {n_tp:,}')
print(f'  False Positives: {n_fp:,}')
print(f'  False Negatives: {n_fn:,}')
print(f'  True Negatives:  {n_tn:,}')

# Top 100 FP and FN
df_fp = df_test[fp_mask].sort_values('y_proba', ascending=False).head(100)
df_fn = df_test[fn_mask].sort_values('y_proba', ascending=True).head(100)

df_fp.to_csv('ml/results/v8_corrected_iter1_top100_fp.csv', index=False)
df_fn.to_csv('ml/results/v8_corrected_iter1_top100_fn.csv', index=False)

print(f'\nError reports saved:')
print(f'  ml/results/v8_corrected_iter1_top100_fp.csv ({len(df_fp)} neurons)')
print(f'  ml/results/v8_corrected_iter1_top100_fn.csv ({len(df_fn)} neurons)')

print(f'\n{"="*80}')
print('NEXT STEPS')
print('='*80)
print(f'\n1. Visualize errors:')
print(f'   python visualize_v8_corrected_iter1_errors.py')
print(f'\n2. Review visualizations and identify:')
print(f'   - Real FP errors (model wrong to KEEP)')
print(f'   - Real FN errors (model wrong to DELETE)')
print(f'\n3. Apply corrections and retrain (iteration 2)')
print(f'\n{"="*80}')
