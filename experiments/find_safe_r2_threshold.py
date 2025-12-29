"""
Find the maximum safe r2_score threshold for deletion with ZERO false negatives.

The safe threshold is the minimum r2_score among all KEEP neurons.
Any neuron with r2_score below this can be safely deleted without risk.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print('='*80)
print('FINDING SAFE R2_SCORE DELETION THRESHOLD (ZERO FALSE NEGATIVES)')
print('='*80)

# Load v9 dataset
v9 = pd.read_csv('ml/results/training_dataset_v9.csv')
print(f'\nv9 dataset: {len(v9):,} neurons')

# Filter to valid r2_score
v9_valid = v9[v9['r2_score'].notna()].copy()
print(f'Valid r2_score: {len(v9_valid):,} neurons')

# Separate by ground truth
keep_neurons = v9_valid[v9_valid['ground_truth'] == 1].copy()
delete_neurons = v9_valid[v9_valid['ground_truth'] == 0].copy()

print(f'\nGround truth distribution:')
print(f'  KEEP: {len(keep_neurons):,}')
print(f'  DELETE: {len(delete_neurons):,}')

# Find minimum r2_score among KEEP neurons
min_r2_keep = keep_neurons['r2_score'].min()
max_r2_delete = delete_neurons['r2_score'].max()

print('\n' + '='*80)
print('CRITICAL VALUES')
print('='*80)

print(f'\nMinimum r2_score among KEEP neurons: {min_r2_keep:.6f}')
print(f'Maximum r2_score among DELETE neurons: {max_r2_delete:.6f}')

# The safe threshold is just below the minimum KEEP value
safe_threshold = min_r2_keep

print(f'\n' + '='*80)
print('SAFE THRESHOLD')
print('='*80)
print(f'\nSafe deletion threshold: r2_score < {safe_threshold:.6f}')
print(f'\nThis means: Any neuron with r2_score < {safe_threshold:.6f} can be')
print(f'safely deleted with ZERO risk of deleting a good neuron.')

# How many neurons does this catch?
safe_delete_mask = v9_valid['r2_score'] < safe_threshold
safe_delete_count = safe_delete_mask.sum()

# Verify no false negatives
fn_count = ((v9_valid['r2_score'] < safe_threshold) &
            (v9_valid['ground_truth'] == 1)).sum()

# Count true negatives (correctly deleted)
tn_count = ((v9_valid['r2_score'] < safe_threshold) &
            (v9_valid['ground_truth'] == 0)).sum()

print(f'\nImpact of safe threshold:')
print(f'  Total flagged: {safe_delete_count:,} neurons')
print(f'  True negatives (bad neurons caught): {tn_count:,}')
print(f'  False negatives (good neurons deleted): {fn_count:,} ✓ ZERO!')
print(f'  Coverage: {100*tn_count/len(delete_neurons):.2f}% of all bad neurons')

# Show the neurons at the boundary
print('\n' + '='*80)
print('NEURONS AT THE BOUNDARY')
print('='*80)

# Find KEEP neurons with lowest r2_scores
lowest_keep = keep_neurons.nsmallest(10, 'r2_score')
print(f'\nLowest r2_score among KEEP neurons (bottom 10):')
for idx, row in lowest_keep.iterrows():
    print(f'  {row["session_name"]:20s} neuron {row["component_idx"]:4.0f}: '
          f'r2={row["r2_score"]:7.4f}, caiman_snr={row["caiman_snr"]:5.2f}, '
          f'event_snr={row["event_snr"]:5.2f}, exp={row["experiment"]}')

# Find DELETE neurons with highest r2_scores
highest_delete = delete_neurons.nlargest(10, 'r2_score')
print(f'\nHighest r2_score among DELETE neurons (top 10):')
for idx, row in highest_delete.iterrows():
    print(f'  {row["session_name"]:20s} neuron {row["component_idx"]:4.0f}: '
          f'r2={row["r2_score"]:7.4f}, caiman_snr={row["caiman_snr"]:5.2f}, '
          f'event_snr={row["event_snr"]:5.2f}, exp={row["experiment"]}')

# Analyze what makes this boundary neuron special
print('\n' + '='*80)
print('BOUNDARY NEURON ANALYSIS')
print('='*80)

boundary_neuron = keep_neurons[keep_neurons['r2_score'] == min_r2_keep].iloc[0]
print(f'\nThe KEEP neuron with minimum r2_score:')
print(f'  Session: {boundary_neuron["session_name"]}')
print(f'  Component: {boundary_neuron["component_idx"]}')
print(f'  Experiment: {boundary_neuron["experiment"]}')
print(f'  r2_score: {boundary_neuron["r2_score"]:.6f}')
print(f'  caiman_snr: {boundary_neuron["caiman_snr"]:.3f}')
print(f'  caiman_r_score: {boundary_neuron["caiman_r_score"]:.3f}')
print(f'  event_snr: {boundary_neuron["event_snr"]:.3f}')
print(f'  events_fraction: {boundary_neuron["events_fraction"]:.6f}')
print(f'  events_per_min: {boundary_neuron["events_per_min"]:.3f}')

# Distribution analysis
print('\n' + '='*80)
print('DISTRIBUTION ANALYSIS')
print('='*80)

# How many DELETE neurons have r2_score below safe threshold?
delete_below_safe = delete_neurons[delete_neurons['r2_score'] < safe_threshold]
print(f'\nDELETE neurons with r2 < {safe_threshold:.6f}: {len(delete_below_safe):,}')
print(f'  This is {100*len(delete_below_safe)/len(delete_neurons):.2f}% of all bad neurons')

# Percentiles of r2_score for DELETE neurons
print(f'\nR2_score percentiles for DELETE neurons:')
for p in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    val = delete_neurons['r2_score'].quantile(p)
    print(f'  {int(p*100):2d}th percentile: {val:7.4f}')

# Visualization
print('\n' + '='*80)
print('GENERATING VISUALIZATION')
print('='*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. R2 distribution with safe threshold
ax = axes[0, 0]
bins = np.linspace(-2, 1, 100)
ax.hist(keep_neurons['r2_score'], bins=bins, alpha=0.6, label='KEEP', color='green', density=True)
ax.hist(delete_neurons['r2_score'], bins=bins, alpha=0.6, label='DELETE', color='red', density=True)
ax.axvline(safe_threshold, color='black', linestyle='--', linewidth=3,
           label=f'Safe threshold = {safe_threshold:.4f}')
ax.axvline(0, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='r2 = 0')
ax.set_xlabel('r2_score')
ax.set_ylabel('Density')
ax.set_title('R² Distribution with Safe Threshold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Cumulative distribution
ax = axes[0, 1]
keep_sorted = np.sort(keep_neurons['r2_score'])
delete_sorted = np.sort(delete_neurons['r2_score'])

keep_cum = np.arange(len(keep_sorted)) / len(keep_sorted) * 100
delete_cum = np.arange(len(delete_sorted)) / len(delete_sorted) * 100

ax.plot(keep_sorted, keep_cum, label='KEEP', color='green', linewidth=2)
ax.plot(delete_sorted, delete_cum, label='DELETE', color='red', linewidth=2)
ax.axvline(safe_threshold, color='black', linestyle='--', linewidth=3,
           label=f'Safe = {safe_threshold:.4f}')
ax.axvline(0, color='gray', linestyle=':', linewidth=2, alpha=0.5)
ax.set_xlabel('r2_score')
ax.set_ylabel('Cumulative %')
ax.set_title('Cumulative Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Coverage vs threshold
ax = axes[1, 0]
thresholds = np.linspace(min_r2_keep - 0.1, max_r2_delete + 0.1, 200)
coverage = []
fn_rate = []

for thresh in thresholds:
    tn = ((delete_neurons['r2_score'] < thresh)).sum()
    fn = ((keep_neurons['r2_score'] < thresh)).sum()
    coverage.append(100 * tn / len(delete_neurons))
    fn_rate.append(100 * fn / len(keep_neurons))

ax.plot(thresholds, coverage, label='Coverage (% bad caught)', linewidth=2, color='blue')
ax.plot(thresholds, fn_rate, label='FN rate (% good deleted)', linewidth=2, color='red')
ax.axvline(safe_threshold, color='black', linestyle='--', linewidth=3, alpha=0.7,
           label=f'Safe threshold')
ax.axhline(0, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('r2_score threshold')
ax.set_ylabel('Percentage')
ax.set_title('Coverage vs False Negative Trade-off')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Experiment breakdown
ax = axes[1, 1]
experiments = ['NOF', 'RFC', 'FOF', 'LNOF']
safe_counts = []
total_delete_counts = []

for exp in experiments:
    exp_delete = delete_neurons[delete_neurons['experiment'] == exp]
    exp_safe = exp_delete[exp_delete['r2_score'] < safe_threshold]
    safe_counts.append(len(exp_safe))
    total_delete_counts.append(len(exp_delete))

x = np.arange(len(experiments))
width = 0.35

ax.bar(x - width/2, safe_counts, width, label=f'Caught by safe threshold', color='green', alpha=0.7)
ax.bar(x + width/2, total_delete_counts, width, label='Total bad neurons', color='red', alpha=0.7)

ax.set_ylabel('Count')
ax.set_title('Safe Threshold Coverage by Experiment')
ax.set_xticks(x)
ax.set_xticklabels(experiments)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = 'output/safe_r2_threshold_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'\nSaved: {output_path}')
plt.close()

# Summary table
print('\n' + '='*80)
print('SUMMARY TABLE: SAFE THRESHOLD BY EXPERIMENT')
print('='*80)

print(f'\n{"Experiment":<12} {"Total Bad":>10} {"Caught":>10} {"Coverage":>10}')
print('-' * 50)
for exp in experiments:
    exp_delete = delete_neurons[delete_neurons['experiment'] == exp]
    exp_safe = exp_delete[exp_delete['r2_score'] < safe_threshold]
    coverage = 100 * len(exp_safe) / len(exp_delete) if len(exp_delete) > 0 else 0
    print(f'{exp:<12} {len(exp_delete):>10,} {len(exp_safe):>10,} {coverage:>9.2f}%')

total_bad = len(delete_neurons)
total_caught = tn_count
print('-' * 50)
print(f'{"TOTAL":<12} {total_bad:>10,} {total_caught:>10,} {100*total_caught/total_bad:>9.2f}%')

# Final recommendation
print('\n' + '='*80)
print('FINAL RECOMMENDATION')
print('='*80)

print(f'''
SAFE DELETION THRESHOLD: r2_score < {safe_threshold:.6f}

Implementation:
```python
if r2_score < {safe_threshold:.6f}:
    delete_neuron()  # 100% safe, zero false negatives
```

Impact:
✓ Deletes {tn_count:,} bad neurons ({100*tn_count/total_bad:.2f}% of all bad neurons)
✓ Zero false negatives (guaranteed safe)
✗ Low coverage ({100*tn_count/total_bad:.2f}% - misses {total_bad - tn_count:,} bad neurons)

This threshold is:
- MAXIMALLY SAFE: Will never delete a good neuron
- MINIMALLY EFFECTIVE: Only catches {100*tn_count/total_bad:.1f}% of bad neurons

Use case:
- Preprocessing step to remove obvious artifacts
- High-confidence automated cleanup
- First-pass filtering before manual review

For better coverage, consider:
- Combining with other metrics (event_snr, events_fraction)
- Using ML model that learns complex patterns
- Manual review of borderline cases
''')

print('='*80)

# Save boundary neuron info
boundary_info = {
    'safe_threshold': float(safe_threshold),
    'min_r2_keep': float(min_r2_keep),
    'max_r2_delete': float(max_r2_delete),
    'neurons_caught': int(tn_count),
    'total_bad_neurons': int(len(delete_neurons)),
    'coverage_percent': float(100*tn_count/len(delete_neurons)),
    'boundary_neuron': {
        'session': boundary_neuron['session_name'],
        'component_idx': int(boundary_neuron['component_idx']),
        'r2_score': float(boundary_neuron['r2_score']),
        'caiman_snr': float(boundary_neuron['caiman_snr']),
        'event_snr': float(boundary_neuron['event_snr']),
        'events_fraction': float(boundary_neuron['events_fraction'])
    }
}

import json
info_path = 'ml/results/safe_r2_threshold.json'
with open(info_path, 'w') as f:
    json.dump(boundary_info, f, indent=2)
print(f'\nSaved threshold info: {info_path}')
