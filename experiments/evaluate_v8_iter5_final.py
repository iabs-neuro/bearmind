"""
Comprehensive evaluation of v8_corrected_iter5 model.
- Cross-validation with stratified k-fold
- Threshold optimization for f-beta
- Full dataset and test set evaluation
- Detailed performance reporting
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    fbeta_score,
    confusion_matrix,
    precision_recall_curve
)
import matplotlib.pyplot as plt
from pathlib import Path

FBETA_BETA = 0.5773502691896257
CURRENT_THRESHOLD = 0.75
RANDOM_SEED = 45
N_FOLDS = 5

print('='*80)
print('COMPREHENSIVE EVALUATION: v8_corrected_iter5')
print('='*80)

# Load model
print('\nLoading model...')
with open('production_models/ebm_v8_corrected_iter5.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataset
print('Loading dataset...')
df = pd.read_csv('ml/results/training_dataset_v8_corrected_iter5.csv')
print(f'Dataset: {len(df):,} neurons from {df["session"].nunique()} sessions')
print(f'Class balance: {df["ground_truth"].mean()*100:.2f}% KEEP')

# Prepare data
exclude_cols = {'ground_truth', 'session', 'experiment', 'component_idx', 'center',
                'distance_to_gt', 'is_corner_artifact', 'corr_groups'}

feature_cols = [c for c in df.columns if c not in exclude_cols
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

print(f'Features: {len(feature_cols)}')

X = df[feature_cols].values
y = df['ground_truth'].values

# ============================================================================
# PART 1: CROSS-VALIDATION
# ============================================================================
print(f'\n{"="*80}')
print(f'PART 1: {N_FOLDS}-FOLD CROSS-VALIDATION')
print('='*80)

# Get session-level stratification
sessions = df['session'].unique()
session_experiments = {s: s.split('_')[0] for s in sessions}
session_labels = df.groupby('session')['ground_truth'].mean().to_dict()

# Create fold assignments at session level
session_to_fold = {}
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# Use experiment as stratification variable
session_exp_list = [session_experiments[s] for s in sessions]
session_label_list = [int(session_labels[s] > 0.5) for s in sessions]

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(sessions, session_exp_list)):
    for idx in test_idx:
        session_to_fold[sessions[idx]] = fold_idx

# Map neurons to folds
df['fold'] = df['session'].map(session_to_fold)

print(f'\nFold distribution:')
for fold in range(N_FOLDS):
    fold_mask = df['fold'] == fold
    n_neurons = fold_mask.sum()
    n_sessions = df.loc[fold_mask, 'session'].nunique()
    keep_pct = df.loc[fold_mask, 'ground_truth'].mean() * 100
    print(f'  Fold {fold}: {n_neurons:,} neurons, {n_sessions} sessions, {keep_pct:.1f}% KEEP')

# Cross-validation loop
cv_results = []

for fold in range(N_FOLDS):
    print(f'\n--- Fold {fold+1}/{N_FOLDS} ---')

    train_mask = df['fold'] != fold
    test_mask = df['fold'] == fold

    X_train_fold = df.loc[train_mask, feature_cols].values
    y_train_fold = df.loc[train_mask, 'ground_truth'].values
    X_test_fold = df.loc[test_mask, feature_cols].values
    y_test_fold = df.loc[test_mask, 'ground_truth'].values

    # Train model
    from interpret.glassbox import ExplainableBoostingClassifier
    model_fold = ExplainableBoostingClassifier(
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
        random_state=RANDOM_SEED + fold
    )

    model_fold.fit(X_train_fold, y_train_fold)

    # Predict
    y_pred_proba = model_fold.predict_proba(X_test_fold)[:, 1]
    y_pred = (y_pred_proba >= CURRENT_THRESHOLD).astype(int)

    # Metrics
    prec, rec, _, _ = precision_recall_fscore_support(
        y_test_fold, y_pred, average='binary', zero_division=0
    )
    fbeta = fbeta_score(y_test_fold, y_pred, beta=FBETA_BETA, average='binary', zero_division=0)
    auc = roc_auc_score(y_test_fold, y_pred_proba)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred).ravel()

    cv_results.append({
        'fold': fold,
        'precision': prec,
        'recall': rec,
        'fbeta': fbeta,
        'auc': auc,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    })

    print(f'  F-beta: {fbeta:.4f}, AUC: {auc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}')
    print(f'  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')

# Aggregate CV results
cv_df = pd.DataFrame(cv_results)
print(f'\n{"="*80}')
print('CROSS-VALIDATION SUMMARY (threshold=0.75)')
print('='*80)
print(f'\nF-beta:    {cv_df["fbeta"].mean():.4f} ± {cv_df["fbeta"].std():.4f}')
print(f'AUC:       {cv_df["auc"].mean():.4f} ± {cv_df["auc"].std():.4f}')
print(f'Precision: {cv_df["precision"].mean():.4f} ± {cv_df["precision"].std():.4f}')
print(f'Recall:    {cv_df["recall"].mean():.4f} ± {cv_df["recall"].std():.4f}')
print(f'\nTotal FP:  {cv_df["fp"].sum():,}')
print(f'Total FN:  {cv_df["fn"].sum():,}')
print(f'Total Errors: {cv_df["fp"].sum() + cv_df["fn"].sum():,}')

# Save CV results
cv_df.to_csv('ml/results/v8_iter5_cv_results.csv', index=False)
print(f'\nCV results saved: ml/results/v8_iter5_cv_results.csv')

# ============================================================================
# PART 2: THRESHOLD OPTIMIZATION
# ============================================================================
print(f'\n{"="*80}')
print('PART 2: THRESHOLD OPTIMIZATION')
print('='*80)

# Use the same train/test split as iter5 training (seed=45)
sessions = df['session'].unique()
session_experiments = {s: s.split('_')[0] for s in sessions}
experiments = [session_experiments[s] for s in sessions]

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
train_idx, test_idx = next(splitter.split(sessions, experiments))
train_sessions = set(sessions[train_idx])
test_sessions = set(sessions[test_idx])

train_mask = df['session'].isin(train_sessions)
test_mask = df['session'].isin(test_sessions)

X_train = df.loc[train_mask, feature_cols].values
y_train = df.loc[train_mask, 'ground_truth'].values
X_test = df.loc[test_mask, feature_cols].values
y_test = df.loc[test_mask, 'ground_truth'].values

print(f'\nTrain: {len(X_train):,} neurons ({len(train_sessions)} sessions)')
print(f'Test:  {len(X_test):,} neurons ({len(test_sessions)} sessions)')

# Get predictions
y_test_proba = model.predict_proba(X_test)[:, 1]

# Test thresholds from 0.5 to 0.95
thresholds = np.arange(0.50, 0.96, 0.01)
threshold_results = []

print(f'\nTesting {len(thresholds)} thresholds...')
for thresh in thresholds:
    y_pred = (y_test_proba >= thresh).astype(int)

    prec, rec, _, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    fbeta = fbeta_score(y_test, y_pred, beta=FBETA_BETA, average='binary', zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    threshold_results.append({
        'threshold': thresh,
        'fbeta': fbeta,
        'precision': prec,
        'recall': rec,
        'fp': fp,
        'fn': fn,
        'total_errors': fp + fn
    })

threshold_df = pd.DataFrame(threshold_results)

# Find best threshold
best_idx = threshold_df['fbeta'].idxmax()
best_threshold = threshold_df.loc[best_idx, 'threshold']
best_fbeta = threshold_df.loc[best_idx, 'fbeta']

print(f'\n{"="*80}')
print('THRESHOLD OPTIMIZATION RESULTS')
print('='*80)
print(f'\nBest threshold: {best_threshold:.2f}')
print(f'Best F-beta: {best_fbeta:.4f}')

# Compare current vs best threshold
current_idx = (threshold_df['threshold'] - CURRENT_THRESHOLD).abs().idxmin()
current_fbeta = threshold_df.loc[current_idx, 'fbeta']

print(f'\nCurrent threshold (0.75):')
print(f'  F-beta: {current_fbeta:.4f}')
print(f'  FP: {threshold_df.loc[current_idx, "fp"]:,}')
print(f'  FN: {threshold_df.loc[current_idx, "fn"]:,}')
print(f'  Total errors: {threshold_df.loc[current_idx, "total_errors"]:,}')

print(f'\nBest threshold ({best_threshold:.2f}):')
print(f'  F-beta: {best_fbeta:.4f}')
print(f'  FP: {threshold_df.loc[best_idx, "fp"]:,}')
print(f'  FN: {threshold_df.loc[best_idx, "fn"]:,}')
print(f'  Total errors: {threshold_df.loc[best_idx, "total_errors"]:,}')

print(f'\nImprovement: {(best_fbeta - current_fbeta):.4f} ({(best_fbeta/current_fbeta - 1)*100:+.2f}%)')

# Save threshold results
threshold_df.to_csv('ml/results/v8_iter5_threshold_optimization.csv', index=False)
print(f'\nThreshold results saved: ml/results/v8_iter5_threshold_optimization.csv')

# Plot threshold curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(threshold_df['threshold'], threshold_df['fbeta'], 'b-', linewidth=2)
plt.axvline(CURRENT_THRESHOLD, color='orange', linestyle='--', label=f'Current (0.75): {current_fbeta:.4f}')
plt.axvline(best_threshold, color='green', linestyle='--', label=f'Best ({best_threshold:.2f}): {best_fbeta:.4f}')
plt.xlabel('Threshold')
plt.ylabel('F-beta Score')
plt.title('F-beta vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(threshold_df['threshold'], threshold_df['fp'], 'r-', linewidth=2, label='False Positives')
plt.plot(threshold_df['threshold'], threshold_df['fn'], 'orange', linewidth=2, label='False Negatives')
plt.plot(threshold_df['threshold'], threshold_df['total_errors'], 'k--', linewidth=2, label='Total Errors')
plt.axvline(CURRENT_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
plt.axvline(best_threshold, color='green', linestyle='--', alpha=0.5)
plt.xlabel('Threshold')
plt.ylabel('Error Count')
plt.title('Errors vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml/results/v8_iter5_threshold_analysis.png', dpi=150)
print(f'Threshold plot saved: ml/results/v8_iter5_threshold_analysis.png')
plt.close()

# ============================================================================
# PART 3: FULL DATASET EVALUATION
# ============================================================================
print(f'\n{"="*80}')
print('PART 3: FULL DATASET EVALUATION')
print('='*80)

# Apply model to full dataset
print(f'\nApplying model to full dataset ({len(df):,} neurons)...')
X_full = df[feature_cols].values
y_full = df['ground_truth'].values

y_full_proba = model.predict_proba(X_full)[:, 1]

# Predictions at both thresholds
y_pred_current = (y_full_proba >= CURRENT_THRESHOLD).astype(int)
y_pred_best = (y_full_proba >= best_threshold).astype(int)

# Add predictions to dataframe
df['y_proba'] = y_full_proba
df['y_pred_0.75'] = y_pred_current
df[f'y_pred_{best_threshold:.2f}'] = y_pred_best

# Metrics at current threshold
print(f'\nFull dataset - Current threshold (0.75):')
prec, rec, _, _ = precision_recall_fscore_support(y_full, y_pred_current, average='binary', zero_division=0)
fbeta_full = fbeta_score(y_full, y_pred_current, beta=FBETA_BETA, average='binary', zero_division=0)
auc_full = roc_auc_score(y_full, y_full_proba)

tn, fp, fn, tp = confusion_matrix(y_full, y_pred_current).ravel()

print(f'  F-beta: {fbeta_full:.4f}')
print(f'  AUC: {auc_full:.4f}')
print(f'  Precision: {prec:.4f}')
print(f'  Recall: {rec:.4f}')
print(f'  TP: {tp:,}, FP: {fp:,}, FN: {fn:,}, TN: {tn:,}')
print(f'  Total errors: {fp + fn:,}')

# Metrics at best threshold
print(f'\nFull dataset - Best threshold ({best_threshold:.2f}):')
prec_best, rec_best, _, _ = precision_recall_fscore_support(y_full, y_pred_best, average='binary', zero_division=0)
fbeta_full_best = fbeta_score(y_full, y_pred_best, beta=FBETA_BETA, average='binary', zero_division=0)

tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_full, y_pred_best).ravel()

print(f'  F-beta: {fbeta_full_best:.4f}')
print(f'  Precision: {prec_best:.4f}')
print(f'  Recall: {rec_best:.4f}')
print(f'  TP: {tp_b:,}, FP: {fp_b:,}, FN: {fn_b:,}, TN: {tn_b:,}')
print(f'  Total errors: {fp_b + fn_b:,}')

# Save full dataset with predictions
output_path = 'ml/results/training_dataset_v8_iter5_with_predictions.csv'
df.to_csv(output_path, index=False)
print(f'\nFull dataset with predictions saved: {output_path}')

# ============================================================================
# PART 4: TEST SET DETAILED EVALUATION
# ============================================================================
print(f'\n{"="*80}')
print('PART 4: TEST SET DETAILED EVALUATION')
print('='*80)

# Test set evaluation at best threshold
y_test_pred_best = (y_test_proba >= best_threshold).astype(int)

print(f'\nTest set ({len(y_test):,} neurons):')
print(f'\n--- Current threshold (0.75) ---')
y_test_pred_current = (y_test_proba >= CURRENT_THRESHOLD).astype(int)

prec_t, rec_t, _, _ = precision_recall_fscore_support(y_test, y_test_pred_current, average='binary', zero_division=0)
fbeta_t = fbeta_score(y_test, y_test_pred_current, beta=FBETA_BETA, average='binary', zero_division=0)
auc_t = roc_auc_score(y_test, y_test_proba)

tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_test_pred_current).ravel()

print(f'F-beta: {fbeta_t:.4f}')
print(f'AUC: {auc_t:.4f}')
print(f'Precision: {prec_t:.4f}')
print(f'Recall: {rec_t:.4f}')
print(f'TP: {tp_t:,}, FP: {fp_t:,}, FN: {fn_t:,}, TN: {tn_t:,}')

print(f'\n--- Best threshold ({best_threshold:.2f}) ---')
prec_tb, rec_tb, _, _ = precision_recall_fscore_support(y_test, y_test_pred_best, average='binary', zero_division=0)
fbeta_tb = fbeta_score(y_test, y_test_pred_best, beta=FBETA_BETA, average='binary', zero_division=0)

tn_tb, fp_tb, fn_tb, tp_tb = confusion_matrix(y_test, y_test_pred_best).ravel()

print(f'F-beta: {fbeta_tb:.4f}')
print(f'Precision: {prec_tb:.4f}')
print(f'Recall: {rec_tb:.4f}')
print(f'TP: {tp_tb:,}, FP: {fp_tb:,}, FN: {fn_tb:,}, TN: {tn_tb:,}')

# ============================================================================
# PART 5: FEATURE IMPORTANCE
# ============================================================================
print(f'\n{"="*80}')
print('PART 5: FEATURE IMPORTANCE')
print('='*80)

# Get feature importances
feature_importances = model.term_importances()

# Extract main effects (non-interaction features)
main_effects = []
for name, importance in zip(feature_importances['names'], feature_importances['scores']):
    if ' x ' not in name:  # Main effect, not interaction
        main_effects.append({'feature': name, 'importance': importance})

main_effects_df = pd.DataFrame(main_effects).sort_values('importance', ascending=False)

print(f'\nTop 20 Most Important Features:')
for i, row in main_effects_df.head(20).iterrows():
    print(f'  {row["feature"]:<30} {row["importance"]:>8.4f}')

# Save feature importance
main_effects_df.to_csv('ml/results/v8_iter5_feature_importance.csv', index=False)
print(f'\nFeature importance saved: ml/results/v8_iter5_feature_importance.csv')

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f'\n{"="*80}')
print('FINAL SUMMARY: v8_corrected_iter5')
print('='*80)

print(f'\n[CROSS-VALIDATION ({N_FOLDS}-fold)]')
print(f'F-beta (threshold=0.75): {cv_df["fbeta"].mean():.4f} ± {cv_df["fbeta"].std():.4f}')
print(f'AUC:                     {cv_df["auc"].mean():.4f} ± {cv_df["auc"].std():.4f}')

print(f'\n[THRESHOLD OPTIMIZATION]')
print(f'Current threshold: 0.75 → F-beta: {current_fbeta:.4f}')
print(f'Best threshold:    {best_threshold:.2f} → F-beta: {best_fbeta:.4f}')
print(f'Improvement:       {(best_fbeta - current_fbeta):.4f} ({(best_fbeta/current_fbeta - 1)*100:+.2f}%)')

print(f'\n[TEST SET PERFORMANCE]')
print(f'At threshold 0.75:       F-beta={fbeta_t:.4f}, FP={fp_t:,}, FN={fn_t:,}, Errors={fp_t+fn_t:,}')
print(f'At threshold {best_threshold:.2f}:      F-beta={fbeta_tb:.4f}, FP={fp_tb:,}, FN={fn_tb:,}, Errors={fp_tb+fn_tb:,}')

print(f'\n[FULL DATASET PERFORMANCE]')
print(f'Total neurons: {len(df):,}')
print(f'At threshold 0.75:       F-beta={fbeta_full:.4f}, Errors={fp+fn:,}')
print(f'At threshold {best_threshold:.2f}:      F-beta={fbeta_full_best:.4f}, Errors={fp_b+fn_b:,}')

print(f'\n[TOP 5 FEATURES]')
for i, row in main_effects_df.head(5).iterrows():
    print(f'  {i+1}. {row["feature"]:<30} {row["importance"]:>8.4f}')

print(f'\n{"="*80}')
print('All results saved to ml/results/')
print('='*80)
