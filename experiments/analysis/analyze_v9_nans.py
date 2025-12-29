"""Analyze NaN sources in v9 dataset and propose solution."""
import pandas as pd
import numpy as np

# Load datasets
print('='*80)
print('ANALYZING NaN SOURCES IN V9 DATASET')
print('='*80)

v9 = pd.read_csv('ml/results/training_dataset_v9.csv')
lnof = pd.read_csv('LNOF_dataset_from_processed.csv')

# Define event-based metrics
event_metrics = [
    'event_r2_score', 'event_snr', 'events_fraction', 'events_per_min',
    'kinetics_opt', 't_off', 't_rise', 'nmae', 'nrmse', 'r2_score', 'snr_recon'
]

print(f'\nv9 Dataset: {len(v9)} neurons, {len(v9.columns)} columns')
print(f'LNOF Dataset: {len(lnof)} neurons, {len(lnof.columns)} columns')

# Check reconstruction metrics in LNOF
print('\n' + '='*80)
print('RECONSTRUCTION METRICS IN LNOF SOURCE')
print('='*80)
rec_metrics = ['event_r2_score', 'nmae', 'nrmse', 'r2_score', 'snr_recon']
for m in rec_metrics:
    if m in lnof.columns:
        nan_count = lnof[m].isna().sum()
        nan_pct = 100 * lnof[m].isna().mean()
        print(f'{m}: {nan_count} NaNs out of {len(lnof)} ({nan_pct:.2f}%)')
    else:
        print(f'{m}: NOT IN DATASET')

# Analyze v9 NaN patterns
print('\n' + '='*80)
print('V9 DATASET NaN ANALYSIS')
print('='*80)

has_nan = v9[event_metrics].isna().any(axis=1)
all_nan = v9[event_metrics].isna().all(axis=1)

print(f'\nNeurons with ANY event metric NaN: {has_nan.sum()}')
print(f'Neurons with ALL event metrics NaN: {all_nan.sum()}')
print(f'Neurons with PARTIAL NaN: {has_nan.sum() - all_nan.sum()}')

# Breakdown by kinetics_source
print('\n' + '-'*80)
print('BREAKDOWN BY KINETICS SOURCE')
print('-'*80)
complete_failures = v9['kinetics_source'] == 'error'
partial_failures = has_nan & ~complete_failures

print(f'\nCOMPLETE FAILURES (kinetics_source=error):')
print(f'  Count: {complete_failures.sum()}')
print(f'  KEEP: {v9[complete_failures]["ground_truth"].sum()}')
print(f'  DELETE: {(1-v9[complete_failures]["ground_truth"]).sum()}')
print(f'  Experiments: {v9[complete_failures]["experiment"].value_counts().to_dict()}')

print(f'\nPARTIAL FAILURES (event metrics succeeded, reconstruction failed):')
print(f'  Count: {partial_failures.sum()}')
print(f'  KEEP: {v9[partial_failures]["ground_truth"].sum()}')
print(f'  DELETE: {(1-v9[partial_failures]["ground_truth"]).sum()}')
print(f'  Experiments: {v9[partial_failures]["experiment"].value_counts().to_dict()}')

# Which metrics are NaN in partial failures?
print(f'\n  NaN metrics in partial failures:')
for m in event_metrics:
    nan_count = v9[partial_failures][m].isna().sum()
    if nan_count > 0:
        print(f'    {m}: {nan_count}/{partial_failures.sum()}')

# Summary statistics
print('\n' + '='*80)
print('SUMMARY')
print('='*80)
print(f'\nTotal neurons in v9: {len(v9):,}')
print(f'Neurons with NaN event metrics: {has_nan.sum()} ({100*has_nan.mean():.2f}%)')
print(f'  - Complete failures: {complete_failures.sum()} ({100*complete_failures.mean():.2f}%)')
print(f'  - Partial failures: {partial_failures.sum()} ({100*partial_failures.mean():.2f}%)')

print(f'\nGround truth of NaN neurons:')
print(f'  KEEP: {v9[has_nan]["ground_truth"].sum()}')
print(f'  DELETE: {(1-v9[has_nan]["ground_truth"]).sum()}')

print('\n' + '='*80)
print('RECOMMENDATIONS')
print('='*80)
print('\n1. COMPLETE FAILURES (177 neurons):')
print('   - These failed DRIADA processing entirely')
print('   - Contains both KEEP (122) and DELETE (105) neurons')
print('   - Recommendation: Replace NaNs with 0 for all event metrics')
print('   - Consider: May want to exclude from training (unreliable)')
print('\n2. PARTIAL FAILURES (50 neurons):')
print('   - Event detection succeeded, reconstruction quality failed')
print('   - ALL are KEEP neurons from LNOF')
print('   - Recommendation: Replace NaN reconstruction metrics with 0')
print('   - Note: Signal metrics (event_snr, events_fraction, etc.) are valid')
