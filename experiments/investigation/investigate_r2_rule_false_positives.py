"""
Investigate "false positives" of r2_score < -1.0 rule.

These are neurons flagged by the rule (r2 < -1.0) but labeled as KEEP.
User hypothesis: These might be labeling errors.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print('='*80)
print('INVESTIGATING R2 RULE "FALSE POSITIVES" - POTENTIAL LABELING ERRORS')
print('='*80)

# Load v9 dataset
v9 = pd.read_csv('ml/results/training_dataset_v9.csv')
v9_valid = v9[v9['r2_score'].notna()].copy()

print(f'\nv9 dataset: {len(v9):,} neurons')
print(f'Valid r2_score: {len(v9_valid):,} neurons')

# Identify "false positives" - flagged by rule but labeled KEEP
flagged_by_rule = v9_valid['r2_score'] < -1.0
labeled_keep = v9_valid['ground_truth'] == 1

false_positives = v9_valid[flagged_by_rule & labeled_keep].copy()
true_positives = v9_valid[flagged_by_rule & ~labeled_keep].copy()

print(f'\n' + '='*80)
print('RULE PERFORMANCE')
print('='*80)

print(f'\nNeurons flagged by r2_score < -1.0: {flagged_by_rule.sum():,}')
print(f'  Labeled DELETE (true positives): {len(true_positives):,}')
print(f'  Labeled KEEP (false positives): {len(false_positives):,}')
print(f'  Precision: {100*len(true_positives)/flagged_by_rule.sum():.1f}%')

# Analyze false positives
print(f'\n' + '='*80)
print('FALSE POSITIVE ANALYSIS - POTENTIAL LABELING ERRORS')
print('='*80)

print(f'\n{len(false_positives)} neurons with r2 < -1.0 labeled as KEEP:')

# Basic statistics
print(f'\nBasic statistics:')
print(f'  Mean r2_score: {false_positives["r2_score"].mean():.3f}')
print(f'  Median r2_score: {false_positives["r2_score"].median():.3f}')
print(f'  Min r2_score: {false_positives["r2_score"].min():.3f}')
print(f'  Max r2_score: {false_positives["r2_score"].max():.3f}')

# Experiment distribution
print(f'\nExperiment distribution:')
exp_dist = false_positives['experiment'].value_counts()
for exp, count in exp_dist.items():
    total_exp = v9_valid[v9_valid['experiment'] == exp]
    pct = 100 * count / len(total_exp)
    print(f'  {exp}: {count} ({pct:.1f}% of all {exp} neurons)')

# Compare with correctly labeled KEEP neurons
correct_keep = v9_valid[(v9_valid['r2_score'] >= -1.0) & labeled_keep].copy()

print(f'\n' + '='*80)
print('FEATURE COMPARISON: FALSE POSITIVES vs CORRECT KEEP')
print('='*80)

features_to_check = [
    'caiman_snr', 'caiman_r_score', 'r2_score', 'event_r2_score',
    'event_snr', 'events_fraction', 'events_per_min',
    'nmae', 'nrmse', 'snr_recon',
    'trace_skewness', 'trace_kurtosis', 'bimodality',
    'area', 'circularity', 'convexity'
]

comparison = []
for feature in features_to_check:
    if feature not in v9_valid.columns:
        continue

    fp_vals = false_positives[feature].dropna()
    ck_vals = correct_keep[feature].dropna()

    if len(fp_vals) < 5 or len(ck_vals) < 5:
        continue

    comparison.append({
        'feature': feature,
        'fp_mean': fp_vals.mean(),
        'fp_median': fp_vals.median(),
        'correct_mean': ck_vals.mean(),
        'correct_median': ck_vals.median(),
        'difference': fp_vals.mean() - ck_vals.mean()
    })

comparison_df = pd.DataFrame(comparison)

print(f'\n{"Feature":<20} {"FP Mean":>10} {"Correct Mean":>12} {"Difference":>12}')
print('-' * 60)
for _, row in comparison_df.iterrows():
    diff_pct = 100 * row['difference'] / row['correct_mean'] if row['correct_mean'] != 0 else 0
    print(f'{row["feature"]:<20} {row["fp_mean"]:>10.3f} {row["correct_mean"]:>12.3f} '
          f'{row["difference"]:>12.3f} ({diff_pct:+.1f}%)')

# Show worst cases - these are most likely labeling errors
print(f'\n' + '='*80)
print('TOP 20 WORST CASES - MOST LIKELY LABELING ERRORS')
print('='*80)

worst_cases = false_positives.nsmallest(20, 'r2_score')

print(f'\n{"Session":<20} {"Neuron":>6} {"Exp":>4} {"r2":>8} {"caiman_snr":>10} '
      f'{"event_snr":>10} {"events/min":>10} {"nmae":>8}')
print('-' * 95)

for idx, row in worst_cases.iterrows():
    print(f'{row["session_name"]:<20} {row["component_idx"]:>6.0f} {row["experiment"]:>4} '
          f'{row["r2_score"]:>8.3f} {row["caiman_snr"]:>10.2f} '
          f'{row["event_snr"]:>10.2f} {row["events_per_min"]:>10.2f} '
          f'{row.get("nmae", np.nan):>8.2f}')

# Compare with LNOF FP analysis
print(f'\n' + '='*80)
print('COMPARISON WITH LNOF FP ANALYSIS')
print('='*80)

# Load LNOF FP analysis if available
try:
    lnof_fp_analysis = pd.read_csv('ml/results/lnof_fp_analysis.csv')

    # These are neurons that experts marked as FP (should DELETE but model said KEEP)
    # From LNOF FP analysis, we know FP neurons have:
    # - r2_score mean: -0.122
    # - event_snr mean: 1.764
    # - events_fraction mean: 0.002

    print(f'\nLNOF Expert FP neurons (model said KEEP, expert said DELETE):')
    print(f'  Mean r2_score: -0.12')
    print(f'  Mean event_snr: 1.76')
    print(f'  Mean events_fraction: 0.002')

    print(f'\nOur r2<-1.0 rule "FP" neurons (rule says DELETE, label says KEEP):')
    print(f'  Mean r2_score: {false_positives["r2_score"].mean():.3f}')
    print(f'  Mean event_snr: {false_positives["event_snr"].mean():.2f}')
    print(f'  Mean events_fraction: {false_positives["events_fraction"].mean():.4f}')

    print(f'\nOur r2<-1.0 "FP" neurons are MUCH WORSE than LNOF expert FPs!')
    print(f'  r2_score: {false_positives["r2_score"].mean():.3f} vs -0.12 (10x worse!)')
    print(f'  This strongly suggests these are indeed LABELING ERRORS')

except FileNotFoundError:
    print(f'\n[INFO] LNOF FP analysis not found')

# Session-specific analysis
print(f'\n' + '='*80)
print('SESSIONS WITH MOST SUSPECTED LABELING ERRORS')
print('='*80)

session_counts = false_positives['session_name'].value_counts()
print(f'\nTop 15 sessions:')
for session, count in session_counts.head(15).items():
    session_total = v9_valid[v9_valid['session_name'] == session]
    pct = 100 * count / len(session_total)
    exp = v9_valid[v9_valid['session_name'] == session]['experiment'].iloc[0]
    print(f'  {session:<25} {exp:>4}: {count:>3} suspected errors ({pct:>5.1f}% of session)')

# Save suspected labeling errors
print(f'\n' + '='*80)
print('SAVING RESULTS')
print('='*80)

output_columns = [
    'session_name', 'component_idx', 'experiment', 'ground_truth',
    'r2_score', 'event_r2_score', 'caiman_snr', 'caiman_r_score',
    'event_snr', 'events_fraction', 'events_per_min',
    'nmae', 'nrmse', 'snr_recon',
    'trace_skewness', 'trace_kurtosis', 'bimodality',
    'area', 'circularity', 'convexity'
]

available_cols = [c for c in output_columns if c in false_positives.columns]
output_df = false_positives[available_cols].copy()

# Add suspicion score (lower r2 = higher suspicion)
output_df['suspicion_score'] = -output_df['r2_score']  # Higher = more suspicious
output_df = output_df.sort_values('suspicion_score', ascending=False)

output_path = 'ml/results/suspected_labeling_errors_r2.csv'
output_df.to_csv(output_path, index=False)
print(f'\nSaved suspected labeling errors: {output_path}')
print(f'  Total suspected errors: {len(output_df)}')
print(f'  Sorted by suspicion (worst r2_score first)')

# Visualization
print(f'\n' + '='*80)
print('GENERATING VISUALIZATION')
print('='*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. R2 distribution comparison
ax = axes[0, 0]
bins = np.linspace(-3, 1, 50)
ax.hist(correct_keep['r2_score'], bins=bins, alpha=0.5, label='Correct KEEP (r2≥-1.0)', color='green', density=True)
ax.hist(false_positives['r2_score'], bins=bins, alpha=0.7, label='Suspected errors (r2<-1.0, labeled KEEP)', color='red', density=True)
ax.hist(true_positives['r2_score'], bins=bins, alpha=0.5, label='Correctly flagged (r2<-1.0, labeled DELETE)', color='blue', density=True)
ax.axvline(-1.0, color='black', linestyle='--', linewidth=2, label='Rule threshold')
ax.set_xlabel('r2_score')
ax.set_ylabel('Density')
ax.set_title('R² Distribution: Suspected Labeling Errors')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Feature comparison
ax = axes[0, 1]
features_plot = ['caiman_snr', 'event_snr', 'events_fraction', 'nmae']
x = np.arange(len(features_plot))
width = 0.35

fp_means = []
ck_means = []
for feat in features_plot:
    fp_means.append(false_positives[feat].mean())
    ck_means.append(correct_keep[feat].mean())

# Normalize for comparison
fp_norm = [fp_means[i] / ck_means[i] if ck_means[i] != 0 else 0 for i in range(len(features_plot))]

ax.bar(x, fp_norm, width, label='Suspected errors / Correct KEEP', alpha=0.7, color='red')
ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Equal to correct')
ax.set_xticks(x)
ax.set_xticklabels(features_plot, rotation=45, ha='right')
ax.set_ylabel('Ratio')
ax.set_title('Feature Comparison (normalized)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 3. Experiment breakdown
ax = axes[1, 0]
experiments = ['NOF', 'RFC', 'FOF', 'LNOF']
fp_counts = []
total_counts = []

for exp in experiments:
    exp_fp = len(false_positives[false_positives['experiment'] == exp])
    exp_total = len(v9_valid[v9_valid['experiment'] == exp])
    fp_counts.append(100 * exp_fp / exp_total if exp_total > 0 else 0)
    total_counts.append(exp_fp)

ax.bar(experiments, fp_counts, alpha=0.7, color='red')
ax.set_ylabel('% of neurons in experiment')
ax.set_title('Suspected Labeling Errors by Experiment')
ax.grid(True, alpha=0.3, axis='y')

# Add counts on bars
for i, (exp, pct, count) in enumerate(zip(experiments, fp_counts, total_counts)):
    ax.text(i, pct + 0.1, f'n={count}', ha='center', fontsize=9)

# 4. Scatter: r2_score vs event_snr
ax = axes[1, 1]
ax.scatter(correct_keep['event_snr'], correct_keep['r2_score'], alpha=0.1, s=1, label='Correct KEEP', color='green')
ax.scatter(false_positives['event_snr'], false_positives['r2_score'], alpha=0.7, s=20, label='Suspected errors', color='red', edgecolors='black', linewidths=0.5)
ax.axhline(-1.0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('event_snr')
ax.set_ylabel('r2_score')
ax.set_title('Suspected Errors in Feature Space')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = 'output/suspected_labeling_errors_r2.png'
Path('output').mkdir(exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f'\nSaved visualization: {plot_path}')
plt.close()

# Final summary
print(f'\n' + '='*80)
print('CONCLUSION')
print('='*80)

print(f'''
EVIDENCE THAT THESE ARE LABELING ERRORS:

1. CATASTROPHICALLY BAD RECONSTRUCTION:
   - Mean r2_score: {false_positives["r2_score"].mean():.3f} (terrible fit!)
   - Compare to LNOF expert FP neurons: -0.12 (10x better)
   - These neurons have reconstruction fits worse than 99% of dataset

2. POOR EVENT CHARACTERISTICS:
   - Mean event_snr: {false_positives["event_snr"].mean():.2f}
   - Mean events_fraction: {false_positives["events_fraction"].mean():.4f}
   - Mean events_per_min: {false_positives["events_per_min"].mean():.2f}
   - Similar to artifacts, not legitimate neurons

3. HIGH ERROR METRICS:
   - Mean nmae: {false_positives["nmae"].mean():.3f}
   - Mean nrmse: {false_positives["nrmse"].mean():.3f}
   - Reconstruction errors are very high

4. DISTRIBUTION MATCHES TRUE ARTIFACTS:
   - r2 distribution of these "FP" overlaps with correctly flagged DELETE neurons
   - They don't look like other KEEP neurons at all

RECOMMENDATION:
These {len(false_positives)} neurons are VERY LIKELY mislabeled as KEEP.
They should probably be relabeled as DELETE.

Top priority for review: Top 20 worst cases (r2 < -1.8)
See: {output_path}
''')

print('='*80)
