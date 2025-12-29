import pandas as pd
import numpy as np

v9 = pd.read_csv('ml/results/training_dataset_v9.csv')

# Filter valid event_r2_score
v9_valid = v9[v9['event_r2_score'].notna()].copy()

keep_neurons = v9_valid[v9_valid['ground_truth'] == 1]
delete_neurons = v9_valid[v9_valid['ground_truth'] == 0]

thresholds = [0.15, 0.10, 0.05, 0.0]

print('Comparing event_r2_score thresholds:\n')
print(f'{"Threshold":<12} {"Flagged":<10} {"TP":<8} {"FP":<8} {"Precision":<10} {"Coverage":<10} {"FN%":<10} {"Ratio":<8}')
print('-' * 90)

for threshold in thresholds:
    # Calculate metrics
    flagged = (v9_valid['event_r2_score'] < threshold).sum()
    tp = ((v9_valid['event_r2_score'] < threshold) & (v9_valid['ground_truth'] == 0)).sum()
    fp = ((v9_valid['event_r2_score'] < threshold) & (v9_valid['ground_truth'] == 1)).sum()

    total_bad = len(delete_neurons)
    total_good = len(keep_neurons)

    precision = 100 * tp / flagged if flagged > 0 else 0
    coverage = 100 * tp / total_bad
    fn_rate = 100 * fp / total_good
    ratio = tp / fp if fp > 0 else float('inf')

    print(f'< {threshold:<10.2f} {flagged:<10,} {tp:<8,} {fp:<8,} {precision:<10.2f}% {coverage:<10.2f}% {fn_rate:<10.3f}% {ratio:<8.1f}')

print('\n' + '='*90)
print('DETAILED BREAKDOWN:\n')

for threshold in thresholds:
    flagged = (v9_valid['event_r2_score'] < threshold).sum()
    tp = ((v9_valid['event_r2_score'] < threshold) & (v9_valid['ground_truth'] == 0)).sum()
    fp = ((v9_valid['event_r2_score'] < threshold) & (v9_valid['ground_truth'] == 1)).sum()

    total_bad = len(delete_neurons)
    total_good = len(keep_neurons)

    precision = 100 * tp / flagged if flagged > 0 else 0
    coverage = 100 * tp / total_bad
    fn_rate = 100 * fp / total_good
    ratio = tp / fp if fp > 0 else float('inf')

    print(f'event_r2_score < {threshold}:')
    print(f'  Total flagged: {flagged:,}')
    print(f'  True positives (bad caught): {tp:,}')
    print(f'  False positives (good lost): {fp:,}')
    print(f'  Precision: {precision:.2f}%')
    print(f'  Coverage: {coverage:.2f}% ({tp:,} of {total_bad:,} bad neurons)')
    print(f'  FN rate: {fn_rate:.4f}% ({fp:,} of {total_good:,} good neurons)')
    print(f'  Ratio: Catch {ratio:.1f} bad neurons for every 1 good neuron lost')
    print()
