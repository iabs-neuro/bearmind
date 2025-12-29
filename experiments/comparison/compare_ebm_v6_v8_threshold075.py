"""
Compare EBM grid search results between v6_no3dm and v8 at threshold=0.75.
"""
import pandas as pd
import numpy as np

print('='*80)
print('EBM GRID SEARCH COMPARISON: v6_no3dm vs v8 (threshold=0.75)')
print('='*80)

# Load results
v6 = pd.read_csv('ml/ebm_grid_search_v6_no3dm/ebm_grid_search_results.csv')
v8 = pd.read_csv('ml/ebm_grid_search_v8/ebm_grid_search_results.csv')

print(f'\nResults loaded:')
print(f'  v6_no3dm: {len(v6)} evaluations')
print(f'  v8:       {len(v8)} evaluations')

# Find closest threshold to 0.75
v6_thresholds = sorted(v6['threshold'].unique())
v8_thresholds = sorted(v8['threshold'].unique())

print(f'\nAvailable thresholds:')
print(f'  v6_no3dm: {v6_thresholds}')
print(f'  v8:       {v8_thresholds}')

# Use 0.7 if 0.75 not available
target_threshold = 0.7 if 0.75 not in v6_thresholds else 0.75

print(f'\nUsing threshold: {target_threshold}')

# Filter to target threshold
v6_t = v6[v6['threshold'] == target_threshold].copy()
v8_t = v8[v8['threshold'] == target_threshold].copy()

print(f'\nAt threshold={target_threshold}:')
print(f'  v6_no3dm: {len(v6_t)} configurations')
print(f'  v8:       {len(v8_t)} configurations')

# Best models at this threshold
v6_best = v6_t.loc[v6_t['test_fbeta'].idxmax()]
v8_best = v8_t.loc[v8_t['test_fbeta'].idxmax()]

print(f'\n{"="*80}')
print(f'BEST MODEL COMPARISON (threshold={target_threshold})')
print('='*80)

print(f'\n{"Metric":<25} {"v6_no3dm":<15} {"v8":<15} {"Difference":<15} {"Improvement"}')
print('-'*80)

metrics = [
    ('test_fbeta', 'F-beta Score'),
    ('test_precision', 'Precision'),
    ('test_recall', 'Recall'),
    ('test_auc', 'AUC'),
    ('train_fbeta', 'Train F-beta'),
    ('train_precision', 'Train Precision'),
    ('train_recall', 'Train Recall'),
]

for col, label in metrics:
    v6_val = v6_best[col]
    v8_val = v8_best[col]
    diff = v8_val - v6_val
    pct = (diff / v6_val * 100) if v6_val != 0 else 0
    print(f'{label:<25} {v6_val:<15.4f} {v8_val:<15.4f} {diff:<15.4f} {pct:+.2f}%')

print(f'\n{"="*80}')
print('BEST MODEL CONFIGURATIONS')
print('='*80)

print(f'\nv6_no3dm best:')
print(f'  max_bins: {v6_best["max_bins"]}')
print(f'  interactions: {v6_best["interactions"]}')
print(f'  greedy_ratio: {v6_best.get("greedy_ratio", "N/A")}')
print(f'  smoothing_rounds: {v6_best.get("smoothing_rounds", "N/A")}')
print(f'  min_samples_leaf: {v6_best["min_samples_leaf"]}')
print(f'  max_leaves: {v6_best["max_leaves"]}')

print(f'\nv8 best:')
print(f'  max_bins: {v8_best["max_bins"]}')
print(f'  interactions: {v8_best["interactions"]}')
print(f'  greedy_ratio: {v8_best.get("greedy_ratio", "N/A")}')
print(f'  smoothing_rounds: {v8_best.get("smoothing_rounds", "N/A")}')
print(f'  min_samples_leaf: {v8_best["min_samples_leaf"]}')
print(f'  max_leaves: {v8_best["max_leaves"]}')

# Overall statistics at this threshold
print(f'\n{"="*80}')
print(f'OVERALL STATISTICS (all configurations at threshold={target_threshold})')
print('='*80)

print(f'\n{"Metric":<25} {"v6_no3dm Mean":<15} {"v8 Mean":<15} {"Difference":<15}')
print('-'*75)

for col, label in metrics:
    v6_mean = v6_t[col].mean()
    v8_mean = v8_t[col].mean()
    diff = v8_mean - v6_mean
    print(f'{label:<25} {v6_mean:<15.4f} {v8_mean:<15.4f} {diff:<15.4f}')

# Top 5 models comparison at this threshold
print(f'\n{"="*80}')
print(f'TOP 5 MODELS BY F-BETA (threshold={target_threshold})')
print('='*80)

print('\nv6_no3dm:')
v6_top5 = v6_t.nlargest(5, 'test_fbeta')[
    ['max_bins', 'interactions', 'max_leaves', 'test_precision', 'test_recall', 'test_fbeta']
]
print(v6_top5.to_string(index=False))

print('\nv8:')
v8_top5 = v8_t.nlargest(5, 'test_fbeta')[
    ['max_bins', 'interactions', 'max_leaves', 'test_precision', 'test_recall', 'test_fbeta']
]
print(v8_top5.to_string(index=False))

# Compare across all thresholds for context
print(f'\n{"="*80}')
print('PRECISION-RECALL TRADE-OFF COMPARISON (best configs at each threshold)')
print('='*80)

# Get best overall config for each version (by F-beta at 0.5)
v6_t05 = v6[v6['threshold'] == 0.5]
v8_t05 = v8[v8['threshold'] == 0.5]
v6_best_overall = v6_t05.loc[v6_t05['test_fbeta'].idxmax()]
v8_best_overall = v8_t05.loc[v8_t05['test_fbeta'].idxmax()]

v6_best_config = v6[
    (v6['max_bins'] == v6_best_overall['max_bins']) &
    (v6['interactions'] == v6_best_overall['interactions']) &
    (v6['max_leaves'] == v6_best_overall['max_leaves'])
].sort_values('threshold')

v8_best_config = v8[
    (v8['max_bins'] == v8_best_overall['max_bins']) &
    (v8['interactions'] == v8_best_overall['interactions']) &
    (v8['max_leaves'] == v8_best_overall['max_leaves'])
].sort_values('threshold')

print(f'\n{"Threshold":<12} {"v6 Precision":<15} {"v6 Recall":<15} {"v6 F-beta":<15} {"v8 Precision":<15} {"v8 Recall":<15} {"v8 F-beta"}')
print('-'*105)

for thresh in sorted(v6['threshold'].unique()):
    v6_row = v6_best_config[v6_best_config['threshold'] == thresh]
    v8_row = v8_best_config[v8_best_config['threshold'] == thresh]

    if len(v6_row) > 0 and len(v8_row) > 0:
        v6_prec = v6_row['test_precision'].values[0]
        v6_rec = v6_row['test_recall'].values[0]
        v6_fb = v6_row['test_fbeta'].values[0]
        v8_prec = v8_row['test_precision'].values[0]
        v8_rec = v8_row['test_recall'].values[0]
        v8_fb = v8_row['test_fbeta'].values[0]

        highlight = ' <-- TARGET' if thresh == target_threshold else ''
        print(f'{thresh:<12.1f} {v6_prec:<15.4f} {v6_rec:<15.4f} {v6_fb:<15.4f} {v8_prec:<15.4f} {v8_rec:<15.4f} {v8_fb:<15.4f}{highlight}')

print(f'\n{"="*80}')
print('SUMMARY')
print('='*80)

fbeta_improvement = (v8_best['test_fbeta'] - v6_best['test_fbeta']) / v6_best['test_fbeta'] * 100
prec_improvement = (v8_best['test_precision'] - v6_best['test_precision']) / v6_best['test_precision'] * 100
rec_improvement = (v8_best['test_recall'] - v6_best['test_recall']) / v6_best['test_recall'] * 100

print(f'\nv8 improvements over v6_no3dm at threshold={target_threshold}:')
print(f'  F-beta score: {fbeta_improvement:+.2f}%')
print(f'  Precision:    {prec_improvement:+.2f}%')
print(f'  Recall:       {rec_improvement:+.2f}%')

print(f'\nAbsolute differences:')
print(f'  F-beta: {v8_best["test_fbeta"]:.4f} vs {v6_best["test_fbeta"]:.4f} (diff: {v8_best["test_fbeta"] - v6_best["test_fbeta"]:+.4f})')
print(f'  Precision: {v8_best["test_precision"]:.4f} vs {v6_best["test_precision"]:.4f} (diff: {v8_best["test_precision"] - v6_best["test_precision"]:+.4f})')
print(f'  Recall: {v8_best["test_recall"]:.4f} vs {v6_best["test_recall"]:.4f} (diff: {v8_best["test_recall"] - v6_best["test_recall"]:+.4f})')

if fbeta_improvement > 0.5:
    print(f'\nv8 dataset produces BETTER models at threshold={target_threshold}')
    print(f'The improved event detection quality translates to better ML performance.')
elif abs(fbeta_improvement) < 0.5:
    print(f'\nv8 and v6_no3dm produce COMPARABLE models at threshold={target_threshold}')
    print(f'Both datasets are suitable for training.')
else:
    print(f'\nv6_no3dm produces slightly better models at threshold={target_threshold}')

# Key insight about high threshold
print(f'\nHigh threshold ({target_threshold}) interpretation:')
print(f'  - Higher precision requirement (fewer false positives)')
print(f'  - Lower recall (some good neurons classified as DELETE)')
print(f'  - Better for conservative neuron selection')
print(f'  - Appropriate when manual review capacity is limited')
