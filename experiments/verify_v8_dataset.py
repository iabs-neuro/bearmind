"""Verify training dataset v8 quality and metrics."""
import pandas as pd
import numpy as np

df = pd.read_csv('ml/results/training_dataset_v8.csv')

print('DATASET V8 QUALITY VERIFICATION')
print('='*80)
print(f'\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns')
print(f'\nColumn count verification:')
print(f'  Expected: 43 columns')
print(f'  Actual: {len(df.columns)} columns')
print(f'  Match: {len(df.columns) == 43}')

print(f'\nNEW METRICS VERIFICATION:')
new_metrics = ['hurst_exponent', 'baseline_drift', 'kinetics_source']
for metric in new_metrics:
    present = metric in df.columns
    if present:
        non_null = df[metric].notna().sum()
        pct = non_null / len(df) * 100
        if metric == 'kinetics_source':
            unique_vals = df[metric].unique()
            print(f'  {metric:<20} Present: YES, Non-null: {non_null}/{len(df)} ({pct:.1f}%)')
            print(f'    Unique values: {list(unique_vals)}')
        else:
            mean_val = df[metric].mean()
            print(f'  {metric:<20} Present: YES, Non-null: {non_null}/{len(df)} ({pct:.1f}%), Mean: {mean_val:.4f}')
    else:
        print(f'  {metric:<20} Present: NO')

print(f'\nGROUND TRUTH DISTRIBUTION:')
gt_counts = df['ground_truth'].value_counts()
print(f'  KEEP (1): {gt_counts.get(1, 0)} ({gt_counts.get(1, 0)/len(df)*100:.1f}%)')
print(f'  DELETE (0): {gt_counts.get(0, 0)} ({gt_counts.get(0, 0)/len(df)*100:.1f}%)')

print(f'\nEXPERIMENT DISTRIBUTION:')
exp_counts = df['experiment'].value_counts()
for exp, count in exp_counts.items():
    print(f'  {exp}: {count} neurons ({count/len(df)*100:.1f}%)')

print(f'\nKEY METRICS STATISTICS (all neurons):')
key_metrics = ['r2_score', 'event_r2_score', 'events_per_min', 't_rise', 't_off', 'hurst_exponent', 'baseline_drift']
for metric in key_metrics:
    if metric in df.columns:
        mean_val = df[metric].mean()
        median_val = df[metric].median()
        min_val = df[metric].min()
        max_val = df[metric].max()
        print(f'  {metric:<20} mean={mean_val:8.4f}, median={median_val:8.4f}, range=[{min_val:.4f}, {max_val:.4f}]')

print(f'\nCOMPARISON WITH v7:')
try:
    v7 = pd.read_csv('ml/results/training_dataset_v7.csv')
    print(f'  v7 shape: {v7.shape[0]} rows x {v7.shape[1]} columns')
    print(f'  v8 shape: {df.shape[0]} rows x {df.shape[1]} columns')
    print(f'  Row difference: {df.shape[0] - v7.shape[0]:+d}')
    print(f'  Column difference: {df.shape[1] - v7.shape[1]:+d}')
except:
    print('  v7 dataset not found for comparison')
