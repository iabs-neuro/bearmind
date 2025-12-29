"""
Train v8 with v7 probability as an additional feature (stacking approach).
This allows v8 to learn when to trust v7's predictions and when to override them.
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
print('TRAINING v8 WITH v7 PROBABILITY AS FEATURE (Stacking)')
print('='*80)

# Load datasets
v7_df = pd.read_csv('ml/results/training_dataset_v7.csv')
v8_df = pd.read_csv('ml/results/training_dataset_v8.csv')

print(f'\nv7 dataset: {len(v7_df):,} neurons')
print(f'v8 dataset: {len(v8_df):,} neurons')

# Load models
with open('production_models/ebm_v7.pkl', 'rb') as f:
    v7_model = pickle.load(f)
with open('production_models/ebm_v8.pkl', 'rb') as f:
    v8_original = pickle.load(f)

print(f'\nOriginal v8 features: {len(v8_original.feature_names_in_)}')

# Generate v7 probabilities for all v8 neurons
print(f'\n{"="*80}')
print('GENERATING v7 PROBABILITIES FOR v8 DATASET')
print('='*80)

# Match neurons
v7_df['neuron_id'] = v7_df['session'] + '_' + v7_df['component_idx'].astype(str)
v8_df['neuron_id'] = v8_df['session'] + '_' + v8_df['component_idx'].astype(str)

# Get v7 predictions
exclude_cols = {'ground_truth', 'session', 'experiment', 'component_idx', 'center',
                'distance_to_gt', 'is_corner_artifact', 'corr_groups', 'neuron_id'}

v7_features = [c for c in v7_df.columns if c not in exclude_cols and c in v7_model.feature_names_in_]
X_v7 = v7_df[v7_features].copy()
v7_proba = v7_model.predict_proba(X_v7)[:, 1]

v7_df['v7_probability'] = v7_proba

# Merge v7 probability into v8 dataset
v8_with_v7 = v8_df.merge(
    v7_df[['neuron_id', 'v7_probability']],
    on='neuron_id',
    how='left'
)

# Check coverage
v7_prob_available = v8_with_v7['v7_probability'].notna().sum()
print(f'\nv7 probability available for: {v7_prob_available:,}/{len(v8_with_v7):,} neurons ({v7_prob_available/len(v8_with_v7)*100:.1f}%)')

if v7_prob_available < len(v8_with_v7):
    # Fill missing values with 0.5 (neutral)
    print(f'Filling {len(v8_with_v7) - v7_prob_available:,} missing values with 0.5')
    v8_with_v7['v7_probability'].fillna(0.5, inplace=True)

# Prepare features
v8_features = [c for c in v8_with_v7.columns if c not in exclude_cols
               and v8_with_v7[c].dtype in ['float64', 'float32', 'int64', 'int32']]

# Add v7_probability to feature list
if 'v7_probability' not in v8_features:
    v8_features.append('v7_probability')

print(f'\nNew v8 features (with v7_probability): {len(v8_features)}')
print(f'Added feature: v7_probability')

# Stratified train/test split
sessions = v8_with_v7['session'].unique()
session_experiments = {s: s.split('_')[0] for s in sessions}
experiments = [session_experiments[s] for s in sessions]

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(sessions, experiments))
train_sessions = set(sessions[train_idx])
test_sessions = set(sessions[test_idx])

train_mask = v8_with_v7['session'].isin(train_sessions)
test_mask = v8_with_v7['session'].isin(test_sessions)

X_train = v8_with_v7.loc[train_mask, v8_features].copy()
y_train = v8_with_v7.loc[train_mask, 'ground_truth'].values
X_test = v8_with_v7.loc[test_mask, v8_features].copy()
y_test = v8_with_v7.loc[test_mask, 'ground_truth'].values

print(f'\nTrain: {len(X_train):,} neurons ({len(train_sessions)} sessions, KEEP: {y_train.mean()*100:.1f}%)')
print(f'Test:  {len(X_test):,} neurons ({len(test_sessions)} sessions, KEEP: {y_test.mean()*100:.1f}%)')

# Train new stacked model
print(f'\n{"="*80}')
print('TRAINING STACKED v8 MODEL')
print('='*80)

v8_stacked = ExplainableBoostingClassifier(
    feature_names=v8_features,
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
    random_state=42
)

print(f'\nTraining stacked model with v7_probability feature...')
start_time = time.time()
v8_stacked.fit(X_train, y_train)
train_time = time.time() - start_time
print(f'Training completed in {train_time:.1f}s')

# Check v7_probability importance
feature_importances = v8_stacked.term_importances()
feature_names = v8_stacked.feature_names_in_
v7_prob_idx = list(feature_names).index('v7_probability')
v7_prob_importance = feature_importances[v7_prob_idx]

print(f'\nv7_probability feature importance: {v7_prob_importance:.4f}')
print(f'Rank: {sorted(enumerate(feature_importances[:len(feature_names)]), key=lambda x: x[1], reverse=True).index((v7_prob_idx, v7_prob_importance)) + 1}/{len(feature_names)}')

# Evaluate
print(f'\n{"="*80}')
print('EVALUATION: Original v8 vs Stacked v8')
print('='*80)

# Original v8
X_train_orig = X_train[[c for c in X_train.columns if c in v8_original.feature_names_in_]].copy()
X_test_orig = X_test[[c for c in X_test.columns if c in v8_original.feature_names_in_]].copy()

y_train_proba_orig = v8_original.predict_proba(X_train_orig)[:, 1]
y_test_proba_orig = v8_original.predict_proba(X_test_orig)[:, 1]

y_train_pred_orig = (y_train_proba_orig >= THRESHOLD).astype(int)
y_test_pred_orig = (y_test_proba_orig >= THRESHOLD).astype(int)

# Stacked v8
y_train_proba_stacked = v8_stacked.predict_proba(X_train)[:, 1]
y_test_proba_stacked = v8_stacked.predict_proba(X_test)[:, 1]

y_train_pred_stacked = (y_train_proba_stacked >= THRESHOLD).astype(int)
y_test_pred_stacked = (y_test_proba_stacked >= THRESHOLD).astype(int)

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

train_metrics_stacked = compute_metrics(y_train, y_train_pred_stacked, y_train_proba_stacked)
test_metrics_stacked = compute_metrics(y_test, y_test_pred_stacked, y_test_proba_stacked)

print(f'\nTRAIN SET (threshold={THRESHOLD}):')
print(f'\n{"Metric":<15} {"Original v8":<15} {"Stacked v8":<15} {"Difference"}')
print('-'*65)
for metric in ['precision', 'recall', 'fbeta', 'auc', 'accuracy']:
    orig = train_metrics_orig[metric]
    stacked = train_metrics_stacked[metric]
    diff = stacked - orig
    print(f'{metric.capitalize():<15} {orig:<15.4f} {stacked:<15.4f} {diff:+.4f}')

print(f'\nTEST SET (threshold={THRESHOLD}):')
print(f'\n{"Metric":<15} {"Original v8":<15} {"Stacked v8":<15} {"Difference"}')
print('-'*65)
for metric in ['precision', 'recall', 'fbeta', 'auc', 'accuracy']:
    orig = test_metrics_orig[metric]
    stacked = test_metrics_stacked[metric]
    diff = stacked - orig
    marker = ' [BETTER]' if diff > 0.001 else (' [WORSE]' if diff < -0.001 else '')
    print(f'{metric.capitalize():<15} {orig:<15.4f} {stacked:<15.4f} {diff:+.4f}{marker}')

# Overall improvement
test_fbeta_improvement = test_metrics_stacked['fbeta'] - test_metrics_orig['fbeta']
test_auc_improvement = test_metrics_stacked['auc'] - test_metrics_orig['auc']

print(f'\n{"="*80}')
print('IMPROVEMENT SUMMARY')
print('='*80)

print(f'\nTest set improvement:')
print(f'  F-beta: {test_fbeta_improvement:+.4f} ({test_fbeta_improvement/test_metrics_orig["fbeta"]*100:+.2f}%)')
print(f'  ROC AUC: {test_auc_improvement:+.4f} ({test_auc_improvement/test_metrics_orig["auc"]*100:+.2f}%)')
print(f'  Accuracy: {test_metrics_stacked["accuracy"] - test_metrics_orig["accuracy"]:+.4f}')

if test_fbeta_improvement > 0.001 and test_auc_improvement > 0:
    print(f'\n[SUCCESS] Stacked v8 is BETTER than original v8!')
    print(f'Adding v7_probability as a feature improved performance.')

    # Save stacked model
    output_path = 'production_models/ebm_v8_stacked.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(v8_stacked, f)
    print(f'\nStacked model saved to: {output_path}')
    print(f'NOTE: This model requires v7_probability as input at inference time.')

elif abs(test_fbeta_improvement) < 0.001 and abs(test_auc_improvement) < 0.001:
    print(f'\n[NEUTRAL] Stacked v8 performs EQUIVALENTLY to original v8.')
    print(f'v7_probability does not add significant value.')
else:
    print(f'\n[WARNING] Stacked v8 may be WORSE than original v8.')

# Evaluate on v7 advantage neurons
print(f'\n{"="*80}')
print('IMPACT ON v7 ADVANTAGE NEURONS')
print('='*80)

# Load v7 wins
v7_wins_df = pd.read_csv('ml/results/v7_wins_detailed.csv')
print(f'\nv7 advantage neurons: {len(v7_wins_df):,}')

v7_wins_df['neuron_id'] = v7_wins_df['session'] + '_' + v7_wins_df['component_idx'].astype(str)

# Merge to get full features
v7_wins_full = v7_wins_df[['neuron_id', 'ground_truth']].merge(
    v8_with_v7[['neuron_id'] + v8_features],
    on='neuron_id',
    how='inner'
)

print(f'Matched neurons: {len(v7_wins_full):,}')

if len(v7_wins_full) > 0:
    X_v7_wins = v7_wins_full[v8_features].copy()
    y_v7_wins = v7_wins_full['ground_truth'].values

    # Original v8 predictions
    X_v7_wins_orig = X_v7_wins[[c for c in X_v7_wins.columns if c in v8_original.feature_names_in_]].copy()
    y_v7_wins_proba_orig = v8_original.predict_proba(X_v7_wins_orig)[:, 1]
    y_v7_wins_pred_orig = (y_v7_wins_proba_orig >= THRESHOLD).astype(int)
    orig_correct = (y_v7_wins_pred_orig == y_v7_wins).sum()

    # Stacked v8 predictions
    y_v7_wins_proba_stacked = v8_stacked.predict_proba(X_v7_wins)[:, 1]
    y_v7_wins_pred_stacked = (y_v7_wins_proba_stacked >= THRESHOLD).astype(int)
    stacked_correct = (y_v7_wins_pred_stacked == y_v7_wins).sum()

    print(f'\nAccuracy on v7 advantage neurons:')
    print(f'  Original v8: {orig_correct:,}/{len(v7_wins_full):,} ({orig_correct/len(v7_wins_full)*100:.2f}%)')
    print(f'  Stacked v8:  {stacked_correct:,}/{len(v7_wins_full):,} ({stacked_correct/len(v7_wins_full)*100:.2f}%)')
    print(f'  Improvement: {stacked_correct - orig_correct:+,} neurons ({(stacked_correct - orig_correct)/len(v7_wins_full)*100:+.2f}%)')

    if stacked_correct > orig_correct:
        print(f'\n[SUCCESS] Stacked v8 fixes {stacked_correct - orig_correct:,} cases where v7 beat original v8!')

        # Analyze which cases were fixed
        v7_wins_full['orig_correct'] = (y_v7_wins_pred_orig == y_v7_wins)
        v7_wins_full['stacked_correct'] = (y_v7_wins_pred_stacked == y_v7_wins)
        v7_wins_full['fixed_by_stacking'] = (~v7_wins_full['orig_correct']) & (v7_wins_full['stacked_correct'])

        fixed = v7_wins_full[v7_wins_full['fixed_by_stacking']]
        print(f'\nCharacteristics of fixed neurons:')
        print(f'  Average v7_probability: {fixed["v7_probability"].mean():.3f}')
        print(f'  Ground truth distribution: KEEP={((fixed["ground_truth"] == 1).sum())/len(fixed)*100:.1f}%, DELETE={(fixed["ground_truth"] == 0).sum()/len(fixed)*100:.1f}%')

    elif stacked_correct == orig_correct:
        print(f'\n[NEUTRAL] No improvement on v7 advantage neurons.')
    else:
        print(f'\n[WARNING] Stacked v8 is worse on {orig_correct - stacked_correct:,} v7 advantage neurons.')

print(f'\n{"="*80}')
print('FEATURE IMPORTANCE ANALYSIS')
print('='*80)

# Top features
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances[:len(feature_names)]
}).sort_values('importance', ascending=False)

print(f'\nTop 10 most important features:')
print(f'\n{"Rank":<6} {"Feature":<30} {"Importance"}')
print('-'*55)
for i, row in feature_importance_df.head(10).iterrows():
    marker = ' [v7]' if row['feature'] == 'v7_probability' else ''
    print(f'{i+1:<6} {row["feature"]:<30} {row["importance"]:.4f}{marker}')

print(f'\n{"="*80}')
print('CONCLUSION')
print('='*80)

if test_fbeta_improvement > 0.002:
    print(f'\nAdding v7_probability as a feature SIGNIFICANTLY IMPROVES v8!')
    print(f'Key benefits:')
    print(f'  1. Test F-beta: {test_fbeta_improvement:+.4f} ({test_fbeta_improvement/test_metrics_orig["fbeta"]*100:+.2f}%)')
    print(f'  2. Test ROC AUC: {test_auc_improvement:+.4f}')
    print(f'  3. v7_probability importance rank: {feature_importance_df[feature_importance_df["feature"] == "v7_probability"].index[0] + 1}/{len(feature_names)}')

    if stacked_correct > orig_correct:
        print(f'  4. Fixes {stacked_correct - orig_correct:,} v7 advantage cases')

    print(f'\nRECOMMENDATION: Deploy stacked v8 model')
    print(f'NOTE: Requires v7 model to generate v7_probability at inference time')

elif test_fbeta_improvement > 0:
    print(f'\nAdding v7_probability provides MARGINAL improvement.')
    print(f'Consider whether added complexity is worth {test_fbeta_improvement:+.4f} F-beta gain.')

else:
    print(f'\nAdding v7_probability does NOT improve v8.')
    print(f'Possible reasons:')
    print(f'  1. v8 already captures information v7 provides')
    print(f'  2. v7 and v8 make same mistakes (94% agreement)')
    print(f'  3. Stacking may need more sophisticated approach (e.g., train on v7 errors only)')
