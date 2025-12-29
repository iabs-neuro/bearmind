"""
Retrain v8 model with max_leaves=5 (increased from 3) to improve performance.
Compare to original v8 model to see if flexibility helps.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, fbeta_score
from interpret.glassbox import ExplainableBoostingClassifier
import time

FBETA_BETA = 0.5773502691896257
THRESHOLD = 0.75

print('='*80)
print('RETRAINING v8 WITH max_leaves=5')
print('='*80)

# Load v8 dataset
df = pd.read_csv('ml/results/training_dataset_v8.csv')
print(f'\nDataset: {len(df):,} neurons')
print(f'Class balance: {df["ground_truth"].mean()*100:.1f}% KEEP')

# Load original v8 model for reference
with open('production_models/ebm_v8.pkl', 'rb') as f:
    v8_original = pickle.load(f)

print(f'\nOriginal v8 hyperparameters:')
print(f'  max_bins: {v8_original.max_bins}')
print(f'  interactions: {v8_original.interactions}')
print(f'  max_leaves: {v8_original.max_leaves}')
print(f'  min_samples_leaf: {v8_original.min_samples_leaf}')

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

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(sessions, experiments))
train_sessions = set(sessions[train_idx])
test_sessions = set(sessions[test_idx])

train_mask = df['session'].isin(train_sessions)
test_mask = df['session'].isin(test_sessions)

X_train = df.loc[train_mask, feature_cols].copy()
y_train = df.loc[train_mask, 'ground_truth'].values
X_test = df.loc[test_mask, feature_cols].copy()
y_test = df.loc[test_mask, 'ground_truth'].values

print(f'\nTrain: {len(X_train):,} neurons ({len(train_sessions)} sessions, KEEP: {y_train.mean()*100:.1f}%)')
print(f'Test:  {len(X_test):,} neurons ({len(test_sessions)} sessions, KEEP: {y_test.mean()*100:.1f}%)')

# Train new v8 model with max_leaves=5
print(f'\n{"="*80}')
print('TRAINING NEW v8 MODEL (max_leaves=5)')
print('='*80)

v8_improved = ExplainableBoostingClassifier(
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
    max_leaves=5,  # CHANGED FROM 3 TO 5
    random_state=42
)

print(f'\nTraining model...')
start_time = time.time()
v8_improved.fit(X_train, y_train)
train_time = time.time() - start_time
print(f'Training completed in {train_time:.1f}s')

# Evaluate both models
print(f'\n{"="*80}')
print('EVALUATION: Original v8 vs Improved v8')
print('='*80)

# Original v8
X_train_orig = X_train[[c for c in X_train.columns if c in v8_original.feature_names_in_]].copy()
X_test_orig = X_test[[c for c in X_test.columns if c in v8_original.feature_names_in_]].copy()

y_train_proba_orig = v8_original.predict_proba(X_train_orig)[:, 1]
y_test_proba_orig = v8_original.predict_proba(X_test_orig)[:, 1]

y_train_pred_orig = (y_train_proba_orig >= THRESHOLD).astype(int)
y_test_pred_orig = (y_test_proba_orig >= THRESHOLD).astype(int)

# Improved v8
y_train_proba_new = v8_improved.predict_proba(X_train)[:, 1]
y_test_proba_new = v8_improved.predict_proba(X_test)[:, 1]

y_train_pred_new = (y_train_proba_new >= THRESHOLD).astype(int)
y_test_pred_new = (y_test_proba_new >= THRESHOLD).astype(int)

# Compute metrics
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

train_metrics_orig = compute_metrics(y_train, y_train_pred_orig, y_train_proba_orig)
test_metrics_orig = compute_metrics(y_test, y_test_pred_orig, y_test_proba_orig)

train_metrics_new = compute_metrics(y_train, y_train_pred_new, y_train_proba_new)
test_metrics_new = compute_metrics(y_test, y_test_pred_new, y_test_proba_new)

print(f'\nTRAIN SET (threshold={THRESHOLD}):')
print(f'\n{"Metric":<15} {"Original v8":<15} {"Improved v8":<15} {"Difference"}')
print('-'*65)
for metric in ['precision', 'recall', 'fbeta', 'auc', 'accuracy']:
    orig = train_metrics_orig[metric]
    new = train_metrics_new[metric]
    diff = new - orig
    print(f'{metric.capitalize():<15} {orig:<15.4f} {new:<15.4f} {diff:+.4f}')

print(f'\nTEST SET (threshold={THRESHOLD}):')
print(f'\n{"Metric":<15} {"Original v8":<15} {"Improved v8":<15} {"Difference"}')
print('-'*65)
for metric in ['precision', 'recall', 'fbeta', 'auc', 'accuracy']:
    orig = test_metrics_orig[metric]
    new = test_metrics_new[metric]
    diff = new - orig
    marker = ' [BETTER]' if diff > 0.001 else (' [WORSE]' if diff < -0.001 else '')
    print(f'{metric.capitalize():<15} {orig:<15.4f} {new:<15.4f} {diff:+.4f}{marker}')

# Overall improvement
test_fbeta_improvement = test_metrics_new['fbeta'] - test_metrics_orig['fbeta']
test_auc_improvement = test_metrics_new['auc'] - test_metrics_orig['auc']

print(f'\n{"="*80}')
print('IMPROVEMENT SUMMARY')
print('='*80)

print(f'\nTest set improvement:')
print(f'  F-beta: {test_fbeta_improvement:+.4f} ({test_fbeta_improvement/test_metrics_orig["fbeta"]*100:+.2f}%)')
print(f'  ROC AUC: {test_auc_improvement:+.4f} ({test_auc_improvement/test_metrics_orig["auc"]*100:+.2f}%)')
print(f'  Accuracy: {test_metrics_new["accuracy"] - test_metrics_orig["accuracy"]:+.4f}')

if test_fbeta_improvement > 0.001 and test_auc_improvement > 0:
    print(f'\n[SUCCESS] Improved v8 is BETTER than original v8!')
    print(f'Increasing max_leaves from 3 to 5 improved performance.')

    # Save improved model
    output_path = 'production_models/ebm_v8_improved.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(v8_improved, f)
    print(f'\nImproved model saved to: {output_path}')

elif abs(test_fbeta_improvement) < 0.001 and abs(test_auc_improvement) < 0.001:
    print(f'\n[NEUTRAL] Improved v8 performs EQUIVALENTLY to original v8.')
    print(f'max_leaves=5 does not significantly change performance.')
else:
    print(f'\n[WARNING] Improved v8 may be WORSE than original v8.')
    print(f'max_leaves=5 may have introduced overfitting.')

# Evaluate on v7 advantage neurons
print(f'\n{"="*80}')
print('IMPACT ON v7 ADVANTAGE NEURONS')
print('='*80)

# Load v7 wins analysis
v7_wins = pd.read_csv('ml/results/v7_wins_detailed.csv')
print(f'\nv7 advantage neurons: {len(v7_wins):,}')

# Get neuron IDs
v7_wins['neuron_id'] = v7_wins['session'] + '_' + v7_wins['component_idx'].astype(str)
df['neuron_id'] = df['session'] + '_' + df['component_idx'].astype(str)

# Merge to get full features
v7_wins_full = v7_wins[['neuron_id', 'ground_truth']].merge(
    df[['neuron_id'] + feature_cols],
    on='neuron_id',
    how='inner'
)

print(f'Matched neurons: {len(v7_wins_full):,}')

if len(v7_wins_full) > 0:
    X_v7_wins = v7_wins_full[feature_cols].copy()
    y_v7_wins = v7_wins_full['ground_truth'].values

    # Original v8 predictions
    X_v7_wins_orig = X_v7_wins[[c for c in X_v7_wins.columns if c in v8_original.feature_names_in_]].copy()
    y_v7_wins_proba_orig = v8_original.predict_proba(X_v7_wins_orig)[:, 1]
    y_v7_wins_pred_orig = (y_v7_wins_proba_orig >= THRESHOLD).astype(int)
    orig_correct = (y_v7_wins_pred_orig == y_v7_wins).sum()

    # Improved v8 predictions
    y_v7_wins_proba_new = v8_improved.predict_proba(X_v7_wins)[:, 1]
    y_v7_wins_pred_new = (y_v7_wins_proba_new >= THRESHOLD).astype(int)
    new_correct = (y_v7_wins_pred_new == y_v7_wins).sum()

    print(f'\nAccuracy on v7 advantage neurons:')
    print(f'  Original v8: {orig_correct:,}/{len(v7_wins_full):,} ({orig_correct/len(v7_wins_full)*100:.2f}%)')
    print(f'  Improved v8: {new_correct:,}/{len(v7_wins_full):,} ({new_correct/len(v7_wins_full)*100:.2f}%)')
    print(f'  Improvement: {new_correct - orig_correct:+,} neurons ({(new_correct - orig_correct)/len(v7_wins_full)*100:+.2f}%)')

    if new_correct > orig_correct:
        print(f'\n[SUCCESS] Improved v8 fixes {new_correct - orig_correct:,} cases where v7 beat original v8!')
    elif new_correct == orig_correct:
        print(f'\n[NEUTRAL] No change on v7 advantage neurons.')
    else:
        print(f'\n[WARNING] Improved v8 is worse on {orig_correct - new_correct:,} v7 advantage neurons.')

print(f'\n{"="*80}')
print('CONCLUSION')
print('='*80)

if test_fbeta_improvement > 0.001:
    print(f'\nIncreasing max_leaves from 3 to 5 IMPROVES v8 performance.')
    print(f'Key improvements:')
    print(f'  - Test F-beta: {test_fbeta_improvement:+.4f}')
    print(f'  - Test ROC AUC: {test_auc_improvement:+.4f}')
    print(f'\nRECOMMENDATION: Deploy improved v8 model (max_leaves=5)')
else:
    print(f'\nIncreasing max_leaves from 3 to 5 does NOT significantly improve v8.')
    print(f'Other approaches may be needed:')
    print(f'  1. Adjust threshold (try 0.72-0.73)')
    print(f'  2. Feature engineering (blend v7/v8 event features)')
    print(f'  3. Ensemble v7 + v8 probabilities')
