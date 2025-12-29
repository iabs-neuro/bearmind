"""
Find optimal hard thresholds for deletion rules based on v9 dataset.

Strategy:
1. For each metric, find threshold that maximizes artifacts caught while minimizing FN
2. Test single-metric and multi-metric rules
3. Provide recommendations for DEFAULT_DELETION_RULES
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_curve, auc

print('='*80)
print('FINDING OPTIMAL HARD THRESHOLDS FOR DELETION RULES')
print('='*80)

# Load v9 dataset
v9 = pd.read_csv('ml/results/training_dataset_v9.csv')
print(f'\nv9 dataset: {len(v9):,} neurons')

# Separate by ground truth
keep_neurons = v9[v9['ground_truth'] == 1].copy()
delete_neurons = v9[v9['ground_truth'] == 0].copy()

print(f'  KEEP: {len(keep_neurons):,}')
print(f'  DELETE: {len(delete_neurons):,}')

# Metrics to analyze (from LNOF FP analysis - top discriminators)
metrics_to_analyze = [
    ('r2_score', 'less'),           # Lower is worse
    ('snr_recon', 'less'),          # Lower is worse
    ('event_r2_score', 'less'),     # Lower is worse
    ('event_snr', 'less'),          # Lower is worse
    ('events_per_min', 'less'),     # Lower is worse (fewer events = worse)
    ('events_fraction', 'less'),    # Lower is worse
    ('nmae', 'greater'),            # Higher is worse (error metric)
    ('nrmse', 'greater'),           # Higher is worse (error metric)
    ('caiman_snr', 'less'),         # Lower is worse
]

print('\n' + '='*80)
print('ANALYZING EACH METRIC FOR OPTIMAL THRESHOLD')
print('='*80)

results = []

for metric, direction in metrics_to_analyze:
    if metric not in v9.columns:
        continue

    # Get valid values
    keep_valid = keep_neurons[keep_neurons[metric].notna()][metric].values
    delete_valid = delete_neurons[delete_neurons[metric].notna()][metric].values

    if len(keep_valid) < 10 or len(delete_valid) < 10:
        continue

    print(f'\n{metric} ({direction} is worse):')
    print(f'  KEEP: mean={keep_valid.mean():.4f}, median={np.median(keep_valid):.4f}, '
          f'min={keep_valid.min():.4f}, max={keep_valid.max():.4f}')
    print(f'  DELETE: mean={delete_valid.mean():.4f}, median={np.median(delete_valid):.4f}, '
          f'min={delete_valid.min():.4f}, max={delete_valid.max():.4f}')

    # Find thresholds with different FN rate targets
    fn_targets = [0.001, 0.005, 0.01, 0.02, 0.05]  # 0.1%, 0.5%, 1%, 2%, 5% FN rate

    for fn_target in fn_targets:
        if direction == 'less':
            # Find threshold where X% of KEEP neurons fall below it
            threshold = np.percentile(keep_valid, fn_target * 100)
            # Count how many DELETE neurons are below threshold
            caught = (delete_valid < threshold).sum()
        else:  # greater
            # Find threshold where X% of KEEP neurons fall above it
            threshold = np.percentile(keep_valid, (1 - fn_target) * 100)
            # Count how many DELETE neurons are above threshold
            caught = (delete_valid > threshold).sum()

        coverage = 100 * caught / len(delete_valid)
        fn_actual = fn_target * 100

        results.append({
            'metric': metric,
            'direction': direction,
            'fn_target_pct': fn_actual,
            'threshold': threshold,
            'delete_caught': caught,
            'coverage_pct': coverage,
            'total_delete': len(delete_valid),
        })

        if fn_target <= 0.01:  # Only print for very conservative thresholds
            op = '<' if direction == 'less' else '>'
            print(f'  FN={fn_actual:.1f}%: {metric}{op}{threshold:.4f} catches '
                  f'{caught:,} ({coverage:.1f}% of bad neurons)')

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Find best single-metric rules (high coverage, low FN)
print('\n' + '='*80)
print('BEST SINGLE-METRIC RULES (FN <= 1%, sorted by coverage)')
print('='*80)

best_rules = results_df[results_df['fn_target_pct'] <= 1.0].copy()
best_rules = best_rules.sort_values('coverage_pct', ascending=False)

print(f'\n{"Metric":<20} {"Operator":<8} {"Threshold":<12} {"Coverage":<10} {"Caught":<10} {"FN Rate":<10}')
print('-' * 85)

for _, row in best_rules.head(15).iterrows():
    op = '<' if row['direction'] == 'less' else '>'
    print(f'{row["metric"]:<20} {op:<8} {row["threshold"]:<12.4f} '
          f'{row["coverage_pct"]:<10.2f}% {row["delete_caught"]:<10,} '
          f'{row["fn_target_pct"]:<10.2f}%')

# Test multi-metric combinations
print('\n' + '='*80)
print('TESTING MULTI-METRIC COMBINATIONS')
print('='*80)

# Prepare data for combinations
v9_valid = v9.copy()
for metric, _ in metrics_to_analyze:
    if metric in v9_valid.columns:
        v9_valid = v9_valid[v9_valid[metric].notna()]

print(f'\nNeurons with all metrics valid: {len(v9_valid):,}')

# Test promising combinations based on LNOF FP analysis
combinations = [
    # Reconstruction quality + event quality
    [('r2_score', '<', -0.5), ('events_per_min', '<', 2.0)],
    [('snr_recon', '<', 0.8), ('events_fraction', '<', 0.001)],
    [('r2_score', '<', 0.0), ('event_snr', '<', 1.2)],

    # Error metrics + event quality
    [('nmae', '>', 1.3), ('events_per_min', '<', 2.0)],
    [('nrmse', '>', 1.4), ('events_fraction', '<', 0.001)],

    # Triple combinations
    [('r2_score', '<', -0.3), ('snr_recon', '<', 1.0), ('events_per_min', '<', 2.5)],
    [('nmae', '>', 1.2), ('event_snr', '<', 1.3), ('events_fraction', '<', 0.002)],
]

combo_results = []

for combo in combinations:
    # Build mask for deletion
    delete_mask = pd.Series(True, index=v9_valid.index)

    for metric, op, threshold in combo:
        if metric not in v9_valid.columns:
            delete_mask = pd.Series(False, index=v9_valid.index)
            break

        if op == '<':
            delete_mask &= (v9_valid[metric] < threshold)
        elif op == '>':
            delete_mask &= (v9_valid[metric] > threshold)
        elif op == '<=':
            delete_mask &= (v9_valid[metric] <= threshold)
        elif op == '>=':
            delete_mask &= (v9_valid[metric] >= threshold)

    # Calculate metrics
    flagged = delete_mask.sum()
    tp = ((delete_mask) & (v9_valid['ground_truth'] == 0)).sum()  # Correctly flagged bad
    fp = ((delete_mask) & (v9_valid['ground_truth'] == 1)).sum()  # Incorrectly flagged good

    total_bad = (v9_valid['ground_truth'] == 0).sum()
    total_good = (v9_valid['ground_truth'] == 1).sum()

    precision = 100 * tp / flagged if flagged > 0 else 0
    coverage = 100 * tp / total_bad if total_bad > 0 else 0
    fn_rate = 100 * fp / total_good if total_good > 0 else 0

    combo_results.append({
        'rule': ' AND '.join([f'{m}{o}{t}' for m, o, t in combo]),
        'flagged': flagged,
        'tp': tp,
        'fp': fp,
        'precision': precision,
        'coverage': coverage,
        'fn_rate': fn_rate,
    })

combo_df = pd.DataFrame(combo_results)
combo_df = combo_df.sort_values('coverage', ascending=False)

print(f'\n{"Rule":<80} {"Flagged":<10} {"Prec":<8} {"Cov":<8} {"FN%":<8}')
print('-' * 115)

for _, row in combo_df.iterrows():
    print(f'{row["rule"]:<80} {row["flagged"]:<10,} '
          f'{row["precision"]:<8.1f}% {row["coverage"]:<8.1f}% {row["fn_rate"]:<8.2f}%')

# Visualization
print('\n' + '='*80)
print('GENERATING VISUALIZATIONS')
print('='*80)

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for idx, (metric, direction) in enumerate(metrics_to_analyze[:9]):
    if metric not in v9.columns:
        continue

    ax = axes[idx]

    keep_vals = keep_neurons[keep_neurons[metric].notna()][metric].values
    delete_vals = delete_neurons[delete_neurons[metric].notna()][metric].values

    # Determine range for bins
    all_vals = np.concatenate([keep_vals, delete_vals])
    vmin, vmax = np.percentile(all_vals, [1, 99])

    bins = np.linspace(vmin, vmax, 50)

    ax.hist(delete_vals, bins=bins, alpha=0.6, label='DELETE', color='red', density=True)
    ax.hist(keep_vals, bins=bins, alpha=0.6, label='KEEP', color='green', density=True)

    # Mark thresholds for different FN rates
    metric_results = results_df[results_df['metric'] == metric]

    for fn_target in [0.001, 0.01]:
        row = metric_results[metric_results['fn_target_pct'] == fn_target * 100]
        if len(row) > 0:
            threshold = row.iloc[0]['threshold']
            coverage = row.iloc[0]['coverage_pct']
            label = f'FN={fn_target*100:.1f}% (cov={coverage:.1f}%)'
            linestyle = '--' if fn_target == 0.01 else ':'
            ax.axvline(threshold, color='black', linestyle=linestyle,
                      linewidth=2, alpha=0.7, label=label)

    ax.set_xlabel(metric, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{metric} distribution', fontsize=11, pad=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'output/optimal_thresholds_analysis.png'
Path('output').mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'\nSaved: {output_path}')
plt.close()

# Save results
print('\n' + '='*80)
print('SAVING RESULTS')
print('='*80)

results_df.to_csv('ml/results/threshold_candidates.csv', index=False)
print('\nSaved: ml/results/threshold_candidates.csv')

combo_df.to_csv('ml/results/combination_rules.csv', index=False)
print('Saved: ml/results/combination_rules.csv')

# Final recommendations
print('\n' + '='*80)
print('RECOMMENDATIONS FOR HARD THRESHOLDS')
print('='*80)

print('''
Based on the analysis, here are recommended hard thresholds for DEFAULT_DELETION_RULES:

ULTRA-CONSERVATIVE (FN <= 0.1%, catches worst artifacts only):
''')

ultra_conservative = best_rules[best_rules['fn_target_pct'] <= 0.1].nlargest(5, 'coverage_pct')
for _, row in ultra_conservative.iterrows():
    op = '<' if row['direction'] == 'less' else '>'
    print(f'  {row["metric"]}{op}{row["threshold"]:.4f}  '
          f'# Catches {row["coverage_pct"]:.1f}% of bad neurons, FN={row["fn_target_pct"]:.2f}%')

print('''
CONSERVATIVE (FN <= 0.5%, good balance):
''')

conservative = best_rules[best_rules['fn_target_pct'] <= 0.5].nlargest(5, 'coverage_pct')
for _, row in conservative.iterrows():
    op = '<' if row['direction'] == 'less' else '>'
    print(f'  {row["metric"]}{op}{row["threshold"]:.4f}  '
          f'# Catches {row["coverage_pct"]:.1f}% of bad neurons, FN={row["fn_target_pct"]:.2f}%')

print('''
MODERATE (FN <= 1%, higher coverage):
''')

moderate = best_rules[best_rules['fn_target_pct'] <= 1.0].nlargest(5, 'coverage_pct')
for _, row in moderate.iterrows():
    op = '<' if row['direction'] == 'less' else '>'
    print(f'  {row["metric"]}{op}{row["threshold"]:.4f}  '
          f'# Catches {row["coverage_pct"]:.1f}% of bad neurons, FN={row["fn_target_pct"]:.2f}%')

print('''
BEST COMBINATIONS (consider for complex rules):
''')

best_combos = combo_df[combo_df['fn_rate'] <= 1.0].nlargest(3, 'coverage')
for _, row in best_combos.iterrows():
    print(f'  {row["rule"]}')
    print(f'    Precision={row["precision"]:.1f}%, Coverage={row["coverage"]:.1f}%, FN={row["fn_rate"]:.2f}%')
    print()

print('='*80)
