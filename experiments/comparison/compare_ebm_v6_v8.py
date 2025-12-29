"""
Compare EBM grid search results between v6_no3dm and v8.
"""
import pandas as pd
import numpy as np

print('='*80)
print('EBM GRID SEARCH COMPARISON: v6_no3dm vs v8')
print('='*80)

# Load results
v6 = pd.read_csv('ml/ebm_grid_search_v6_no3dm/ebm_grid_search_results.csv')
v8 = pd.read_csv('ml/ebm_grid_search_v8/ebm_grid_search_results.csv')

print(f'\nResults loaded:')
print(f'  v6_no3dm: {len(v6)} evaluations')
print(f'  v8:       {len(v8)} evaluations')

# Filter to threshold=0.5 for comparison
v6_t05 = v6[v6['threshold'] == 0.5].copy()
v8_t05 = v8[v8['threshold'] == 0.5].copy()

print(f'\nAt threshold=0.5:')
print(f'  v6_no3dm: {len(v6_t05)} configurations')
print(f'  v8:       {len(v8_t05)} configurations')

# Best models
v6_best = v6_t05.loc[v6_t05['test_fbeta'].idxmax()]
v8_best = v8_t05.loc[v8_t05['test_fbeta'].idxmax()]

print(f'\n{"="*80}')
print('BEST MODEL COMPARISON (threshold=0.5)')
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

# Overall statistics
print(f'\n{"="*80}')
print('OVERALL STATISTICS (all configurations at threshold=0.5)')
print('='*80)

print(f'\n{"Metric":<25} {"v6_no3dm Mean":<15} {"v8 Mean":<15} {"Difference":<15}')
print('-'*75)

for col, label in metrics:
    v6_mean = v6_t05[col].mean()
    v8_mean = v8_t05[col].mean()
    diff = v8_mean - v6_mean
    print(f'{label:<25} {v6_mean:<15.4f} {v8_mean:<15.4f} {diff:<15.4f}')

# Top 5 models comparison
print(f'\n{"="*80}')
print('TOP 5 MODELS BY F-BETA')
print('='*80)

print('\nv6_no3dm:')
v6_top5 = v6_t05.nlargest(5, 'test_fbeta')[
    ['max_bins', 'interactions', 'max_leaves', 'test_precision', 'test_recall', 'test_fbeta']
]
print(v6_top5.to_string(index=False))

print('\nv8:')
v8_top5 = v8_t05.nlargest(5, 'test_fbeta')[
    ['max_bins', 'interactions', 'max_leaves', 'test_precision', 'test_recall', 'test_fbeta']
]
print(v8_top5.to_string(index=False))

# Precision-Recall trade-off comparison
print(f'\n{"="*80}')
print('PRECISION-RECALL TRADE-OFF (best configs)')
print('='*80)

# Get best config for each version
v6_best_config = v6[
    (v6['max_bins'] == v6_best['max_bins']) &
    (v6['interactions'] == v6_best['interactions']) &
    (v6['max_leaves'] == v6_best['max_leaves'])
].sort_values('threshold')

v8_best_config = v8[
    (v8['max_bins'] == v8_best['max_bins']) &
    (v8['interactions'] == v8_best['interactions']) &
    (v8['max_leaves'] == v8_best['max_leaves'])
].sort_values('threshold')

print(f'\n{"Threshold":<12} {"v6 Precision":<15} {"v6 Recall":<15} {"v8 Precision":<15} {"v8 Recall":<15}')
print('-'*75)

for thresh in sorted(v6['threshold'].unique()):
    v6_row = v6_best_config[v6_best_config['threshold'] == thresh]
    v8_row = v8_best_config[v8_best_config['threshold'] == thresh]

    if len(v6_row) > 0 and len(v8_row) > 0:
        v6_prec = v6_row['test_precision'].values[0]
        v6_rec = v6_row['test_recall'].values[0]
        v8_prec = v8_row['test_precision'].values[0]
        v8_rec = v8_row['test_recall'].values[0]
        print(f'{thresh:<12.1f} {v6_prec:<15.4f} {v6_rec:<15.4f} {v8_prec:<15.4f} {v8_rec:<15.4f}')

print(f'\n{"="*80}')
print('SUMMARY')
print('='*80)

fbeta_improvement = (v8_best['test_fbeta'] - v6_best['test_fbeta']) / v6_best['test_fbeta'] * 100
prec_improvement = (v8_best['test_precision'] - v6_best['test_precision']) / v6_best['test_precision'] * 100
rec_improvement = (v8_best['test_recall'] - v6_best['test_recall']) / v6_best['test_recall'] * 100

print(f'\nv8 improvements over v6_no3dm (best models):')
print(f'  F-beta score: {fbeta_improvement:+.2f}%')
print(f'  Precision:    {prec_improvement:+.2f}%')
print(f'  Recall:       {rec_improvement:+.2f}%')

if fbeta_improvement > 0:
    print(f'\nv8 dataset with hybrid kinetics + wavelet n=3 produces BETTER models')
    print(f'The improved event detection quality translates to better ML performance.')
elif abs(fbeta_improvement) < 0.5:
    print(f'\nv8 and v6_no3dm produce COMPARABLE models')
    print(f'Both datasets are suitable for training.')
else:
    print(f'\nv6_no3dm produces slightly better models')
    print(f'Further investigation recommended.')
