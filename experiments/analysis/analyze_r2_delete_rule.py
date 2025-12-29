"""
Ultrathink: What effect will the delete rule "r2_score < 0" have on v9 dataset?

Comprehensive analysis of applying r2_score < 0 as a deletion criterion.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

print('='*80)
print('ANALYSIS: DELETE RULE "r2_score < 0" ON V9 DATASET')
print('='*80)

# Load v9 dataset
v9 = pd.read_csv('ml/results/training_dataset_v9.csv')
print(f'\nv9 dataset: {len(v9):,} neurons')

# Handle NaN values in r2_score
r2_nan_count = v9['r2_score'].isna().sum()
print(f'NaN r2_score values: {r2_nan_count} ({100*r2_nan_count/len(v9):.2f}%)')

# For this analysis, we need to decide how to handle NaN
# Option 1: Treat NaN as "failed reconstruction" → should delete
# Option 2: Exclude NaN from rule (only apply to computed values)
# Let's analyze both

print('\n' + '='*80)
print('STRATEGY A: EXCLUDE NaN (only apply rule to valid r2_score)')
print('='*80)

v9_valid = v9[v9['r2_score'].notna()].copy()
print(f'\nNeurons with valid r2_score: {len(v9_valid):,}')

# Apply rule
v9_valid['rule_delete'] = (v9_valid['r2_score'] < 0).astype(int)
v9_valid['rule_keep'] = 1 - v9_valid['rule_delete']

# Compare with ground truth
rule_delete_count = v9_valid['rule_delete'].sum()
print(f'\nNeurons flagged by rule (r2 < 0): {rule_delete_count:,} ({100*rule_delete_count/len(v9_valid):.2f}%)')

# Confusion matrix
y_true = v9_valid['ground_truth']
y_pred = v9_valid['rule_keep']

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f'\nConfusion Matrix (valid r2_score only):')
print(f'  True Positives (correctly kept):  {tp:,}')
print(f'  True Negatives (correctly deleted): {tn:,}')
print(f'  False Positives (kept but should delete): {fp:,}')
print(f'  False Negatives (deleted but should keep): {fn:,}')

# Metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f'\nMetrics:')
print(f'  Precision (of KEEP): {precision:.4f}')
print(f'  Recall (of KEEP): {recall:.4f}')
print(f'  Specificity (DELETE accuracy): {specificity:.4f}')
print(f'  F1 Score: {f1:.4f}')

# Breakdown by experiment
print(f'\n' + '-'*80)
print('BREAKDOWN BY EXPERIMENT')
print('-'*80)

for exp in v9_valid['experiment'].unique():
    exp_data = v9_valid[v9_valid['experiment'] == exp]
    exp_flagged = exp_data['rule_delete'].sum()
    exp_gt_delete = (1 - exp_data['ground_truth']).sum()

    # How many of flagged are actually bad?
    flagged_and_bad = ((exp_data['rule_delete'] == 1) & (exp_data['ground_truth'] == 0)).sum()

    print(f'\n{exp}:')
    print(f'  Total: {len(exp_data):,}')
    print(f'  Flagged by rule: {exp_flagged:,} ({100*exp_flagged/len(exp_data):.2f}%)')
    print(f'  Ground truth DELETE: {exp_gt_delete:,}')
    print(f'  Overlap (flagged AND bad): {flagged_and_bad:,}')
    if exp_flagged > 0:
        print(f'  Precision of rule: {100*flagged_and_bad/exp_flagged:.1f}%')

# Analyze neurons flagged by rule
print(f'\n' + '='*80)
print('CHARACTERISTICS OF FLAGGED NEURONS (r2 < 0)')
print('='*80)

flagged = v9_valid[v9_valid['rule_delete'] == 1].copy()
not_flagged = v9_valid[v9_valid['rule_delete'] == 0].copy()

print(f'\nFlagged neurons: {len(flagged):,}')
print(f'  Ground truth KEEP: {flagged["ground_truth"].sum():,}')
print(f'  Ground truth DELETE: {(1-flagged["ground_truth"]).sum():,}')

# What do flagged KEEP neurons look like? (False negatives)
flagged_keep = flagged[flagged['ground_truth'] == 1]
print(f'\nFalse Negatives (flagged but should KEEP): {len(flagged_keep):,}')

if len(flagged_keep) > 0:
    print(f'  Mean caiman_snr: {flagged_keep["caiman_snr"].mean():.2f}')
    print(f'  Mean r2_score: {flagged_keep["r2_score"].mean():.2f}')
    print(f'  Mean event_snr: {flagged_keep["event_snr"].mean():.2f}')
    print(f'  Mean events_fraction: {flagged_keep["events_fraction"].mean():.4f}')

# Distribution of r2_score
print(f'\n' + '='*80)
print('R2_SCORE DISTRIBUTION')
print('='*80)

r2_percentiles = v9_valid['r2_score'].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
print(f'\nPercentiles:')
for p, val in r2_percentiles.items():
    print(f'  {int(p*100):2d}th: {val:7.3f}')

# How many have negative r2?
neg_r2 = (v9_valid['r2_score'] < 0).sum()
print(f'\nNegative r2_score: {neg_r2:,} ({100*neg_r2/len(v9_valid):.2f}%)')

# Break down by ground truth
neg_r2_keep = v9_valid[(v9_valid['r2_score'] < 0) & (v9_valid['ground_truth'] == 1)]
neg_r2_delete = v9_valid[(v9_valid['r2_score'] < 0) & (v9_valid['ground_truth'] == 0)]

print(f'  Among negative r2: KEEP={len(neg_r2_keep):,}, DELETE={len(neg_r2_delete):,}')
print(f'  Ratio: {100*len(neg_r2_delete)/neg_r2:.1f}% are actually bad')

print('\n' + '='*80)
print('STRATEGY B: TREAT NaN AS FAILED (NaN → should delete)')
print('='*80)

# Apply rule with NaN treated as "should delete"
v9['rule_delete_strict'] = ((v9['r2_score'] < 0) | v9['r2_score'].isna()).astype(int)
v9['rule_keep_strict'] = 1 - v9['rule_delete_strict']

flagged_strict = v9['rule_delete_strict'].sum()
print(f'\nNeurons flagged (r2 < 0 OR NaN): {flagged_strict:,} ({100*flagged_strict/len(v9):.2f}%)')

# Confusion matrix
y_true_all = v9['ground_truth']
y_pred_strict = v9['rule_keep_strict']

cm_strict = confusion_matrix(y_true_all, y_pred_strict)
tn_s, fp_s, fn_s, tp_s = cm_strict.ravel()

print(f'\nConfusion Matrix (including NaN as delete):')
print(f'  True Positives (correctly kept):  {tp_s:,}')
print(f'  True Negatives (correctly deleted): {tn_s:,}')
print(f'  False Positives (kept but should delete): {fp_s:,}')
print(f'  False Negatives (deleted but should keep): {fn_s:,}')

precision_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0
recall_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
specificity_s = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0

print(f'\nMetrics:')
print(f'  Precision (of KEEP): {precision_s:.4f}')
print(f'  Recall (of KEEP): {recall_s:.4f}')
print(f'  Specificity (DELETE accuracy): {specificity_s:.4f}')

# Compare strategies
print('\n' + '='*80)
print('STRATEGY COMPARISON')
print('='*80)

print(f'\nStrategy A (ignore NaN):')
print(f'  Flagged: {rule_delete_count:,}')
print(f'  False Negatives: {fn:,} (good neurons deleted)')
print(f'  Specificity: {specificity:.4f}')

print(f'\nStrategy B (NaN → delete):')
print(f'  Flagged: {flagged_strict:,}')
print(f'  False Negatives: {fn_s:,} (good neurons deleted)')
print(f'  Specificity: {specificity_s:.4f}')

# Visualization
print('\n' + '='*80)
print('GENERATING VISUALIZATIONS')
print('='*80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. R2 distribution by ground truth
ax = axes[0, 0]
keep_r2 = v9_valid[v9_valid['ground_truth'] == 1]['r2_score']
delete_r2 = v9_valid[v9_valid['ground_truth'] == 0]['r2_score']

ax.hist([keep_r2, delete_r2], bins=50, alpha=0.7, label=['KEEP', 'DELETE'])
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='r2=0 threshold')
ax.set_xlabel('r2_score')
ax.set_ylabel('Count')
ax.set_title('R² Distribution by Ground Truth')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. ROC-like curve for different thresholds
ax = axes[0, 1]
thresholds = np.linspace(-2, 1, 100)
specificities = []
recalls = []

for thresh in thresholds:
    pred = (v9_valid['r2_score'] >= thresh).astype(int)
    cm_temp = confusion_matrix(v9_valid['ground_truth'], pred)
    tn_t, fp_t, fn_t, tp_t = cm_temp.ravel()
    spec = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
    rec = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    specificities.append(spec)
    recalls.append(rec)

ax.plot(thresholds, specificities, label='Specificity (DELETE accuracy)', linewidth=2)
ax.plot(thresholds, recalls, label='Recall (KEEP coverage)', linewidth=2)
ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='r2=0')
ax.set_xlabel('r2_score threshold')
ax.set_ylabel('Score')
ax.set_title('Threshold Analysis')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Feature comparison: flagged vs not flagged
ax = axes[1, 0]
features_to_compare = ['caiman_snr', 'event_snr', 'events_fraction', 'r2_score']
x_pos = np.arange(len(features_to_compare))

flagged_means = [flagged[f].mean() for f in features_to_compare]
not_flagged_means = [not_flagged[f].mean() for f in features_to_compare]

# Normalize for visualization
flagged_norm = [flagged_means[i] / not_flagged_means[i] if not_flagged_means[i] != 0 else 0
                for i in range(len(features_to_compare))]

ax.bar(x_pos, flagged_norm, alpha=0.7)
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Equal to non-flagged')
ax.set_xticks(x_pos)
ax.set_xticklabels(features_to_compare, rotation=45, ha='right')
ax.set_ylabel('Ratio (flagged / not flagged)')
ax.set_title('Feature Comparison: Flagged vs Not Flagged')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Experiment breakdown
ax = axes[1, 1]
exp_names = []
exp_fp_rates = []

for exp in ['NOF', 'RFC', 'FOF', 'LNOF']:
    if exp in v9_valid['experiment'].values:
        exp_data = v9_valid[v9_valid['experiment'] == exp]
        flagged_count = (exp_data['r2_score'] < 0).sum()
        total_count = len(exp_data)
        exp_names.append(exp)
        exp_fp_rates.append(100 * flagged_count / total_count)

ax.bar(exp_names, exp_fp_rates, alpha=0.7, color=['blue', 'orange', 'green', 'red'])
ax.set_ylabel('% Flagged by r2 < 0')
ax.set_title('Flagging Rate by Experiment')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = 'output/r2_delete_rule_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'\nSaved: {output_path}')
plt.close()

# Final recommendation
print('\n' + '='*80)
print('RECOMMENDATION')
print('='*80)

fn_rate = 100 * fn / (tp + fn)
precision_pct = 100 * precision

print(f'''
Rule: "DELETE if r2_score < 0"

Impact on v9 dataset ({len(v9_valid):,} neurons with valid r2):
- Flags {rule_delete_count:,} neurons ({100*rule_delete_count/len(v9_valid):.2f}%)
- Correctly identifies {tn:,} bad neurons
- Incorrectly flags {fn:,} good neurons (FN rate: {fn_rate:.2f}%)
- Precision: {precision_pct:.1f}%

Trade-off:
✓ High specificity ({100*specificity:.1f}%) - good at catching bad neurons
✓ Simple, interpretable rule
✗ Deletes {fn:,} legitimate neurons ({fn_rate:.2f}% of KEEP set)

Recommendation:
- As a HARD RULE: Too aggressive, loses {fn:,} good neurons
- As a FEATURE: Excellent! Strong predictor (d=-1.24 from LNOF FP analysis)
- Best use: Include r2_score in ML model, not as standalone filter

Alternative: Use softer threshold (r2 < -0.5) or combine with other metrics.
''')

print('='*80)
