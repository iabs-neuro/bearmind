"""
Compare event metrics between v7 (threshold) and v8 (hybrid wavelet) for a session.
"""
import pandas as pd
import numpy as np

session = 'FOF_F05_1D'

# Load v8 metrics (hybrid kinetics, wavelet n=3)
v8_path = f'data/capcan_validation_99_v8/capcan_artifacts_{session}/metrics_init.csv'
v8 = pd.read_csv(v8_path)

# Load v7 metrics (threshold n=2)
v7_path = f'data/capcan_validation_99_v7/capcan_artifacts_{session}/metrics_init.csv'
v7 = pd.read_csv(v7_path)

print(f'Session: {session}')
print('='*80)
print(f'\nNeuron counts:')
print(f'  v7 (threshold n=2):       {len(v7)} neurons')
print(f'  v8 (hybrid wavelet n=3):  {len(v8)} neurons')

# Filter to good neurons (t_off > -1)
v7_good = v7[v7['t_off'] > -1]
v8_good = v8[v8['t_off'] > -1]

print(f'\nGood neurons (t_off > -1):')
print(f'  v7: {len(v7_good)} / {len(v7)} ({len(v7_good)/len(v7)*100:.1f}%)')
print(f'  v8: {len(v8_good)} / {len(v8)} ({len(v8_good)/len(v8)*100:.1f}%)')

print(f'\n{"="*80}')
print('EVENT-BASED METRICS COMPARISON (Good neurons only)')
print('='*80)

metrics = [
    ('events_per_min', 'Events per Minute'),
    ('events_fraction', 'Events Fraction'),
    ('t_rise', 'Rise Time (sec)'),
    ('t_off', 'Decay Time (sec)'),
    ('event_snr', 'Event SNR'),
    ('r2_score', 'R2 Score'),
    ('event_r2_score', 'Event R2 Score'),
]

print(f"\n{'Metric':<25} {'v7 Mean':<12} {'v8 Mean':<12} {'Difference':<12} {'Change'}")
print('-'*80)

for metric, label in metrics:
    if metric in v7_good.columns and metric in v8_good.columns:
        v7_mean = v7_good[metric].mean()
        v8_mean = v8_good[metric].mean()
        diff = v8_mean - v7_mean
        pct = (diff / v7_mean * 100) if v7_mean != 0 else 0
        print(f'{label:<25} {v7_mean:<12.4f} {v8_mean:<12.4f} {diff:+12.4f} {pct:+6.1f}%')

# Check for kinetics_source in v8
if 'kinetics_source' in v8_good.columns:
    print(f'\n{"="*80}')
    print('v8 KINETICS SOURCE BREAKDOWN (Good neurons only)')
    print('='*80)
    tier_counts = v8_good['kinetics_source'].value_counts()
    for source, count in tier_counts.items():
        pct = count / len(v8_good) * 100
        mean_r2 = v8_good[v8_good['kinetics_source'] == source]['r2_score'].mean()
        print(f'  {source:<25} {count:4d} ({pct:5.1f}%)  R2={mean_r2:.4f}')

# Check for new metrics
print(f'\n{"="*80}')
print('NEW METRICS IN v8')
print('='*80)
new_metrics = ['hurst_exponent', 'baseline_drift']
for metric in new_metrics:
    if metric in v8_good.columns:
        mean_val = v8_good[metric].mean()
        median_val = v8_good[metric].median()
        print(f'  {metric:<20} mean={mean_val:.4f}, median={median_val:.4f}')
    else:
        print(f'  {metric:<20} NOT FOUND')
