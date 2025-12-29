"""
Investigate R2 score distribution for FOF_F05_1D
"""
import pandas as pd
import numpy as np

session = 'FOF_F05_1D'

# Load v8 metrics
v8_path = f'data/capcan_validation_99_v8/capcan_artifacts_{session}/metrics_init.csv'
v8 = pd.read_csv(v8_path)

print(f'Session: {session}')
print('='*80)

# All neurons
print(f'\nAll neurons ({len(v8)}):')
print(f'  R2 score range: [{v8["r2_score"].min():.4f}, {v8["r2_score"].max():.4f}]')
print(f'  R2 score mean: {v8["r2_score"].mean():.4f}')
print(f'  R2 score median: {v8["r2_score"].median():.4f}')

# Distribution
print(f'\nR2 distribution (all neurons):')
print(f'  R2 > 0.5: {(v8["r2_score"] > 0.5).sum()} neurons')
print(f'  R2 > 0.0: {(v8["r2_score"] > 0.0).sum()} neurons')
print(f'  R2 < 0.0: {(v8["r2_score"] < 0.0).sum()} neurons')

# Good neurons filter
v8_good = v8[v8['t_off'] > -1]
print(f'\nGood neurons (t_off > -1): {len(v8_good)} / {len(v8)}')
print(f'  R2 score range: [{v8_good["r2_score"].min():.4f}, {v8_good["r2_score"].max():.4f}]')
print(f'  R2 score mean: {v8_good["r2_score"].mean():.4f}')
print(f'  R2 score median: {v8_good["r2_score"].median():.4f}')

print(f'\nR2 distribution (good neurons):')
print(f'  R2 > 0.5: {(v8_good["r2_score"] > 0.5).sum()} neurons ({(v8_good["r2_score"] > 0.5).sum()/len(v8_good)*100:.1f}%)')
print(f'  R2 > 0.0: {(v8_good["r2_score"] > 0.0).sum()} neurons ({(v8_good["r2_score"] > 0.0).sum()/len(v8_good)*100:.1f}%)')
print(f'  R2 < 0.0: {(v8_good["r2_score"] < 0.0).sum()} neurons ({(v8_good["r2_score"] < 0.0).sum()/len(v8_good)*100:.1f}%)')

# Check t_off values
print(f'\nt_off distribution:')
print(f'  t_off = -1: {(v8["t_off"] == -1).sum()} neurons (no events detected)')
print(f'  t_off > -1: {(v8["t_off"] > -1).sum()} neurons (events detected)')
print(f'  t_off > 0: {(v8["t_off"] > 0).sum()} neurons')

# Check if there's correlation between t_off and R2
print(f'\nCorrelation:')
print(f'  t_off vs r2_score: {v8["t_off"].corr(v8["r2_score"]):.4f}')

# Sample some neurons with negative R2 but t_off > -1
negative_r2_good = v8_good[v8_good['r2_score'] < 0].head(5)
if len(negative_r2_good) > 0:
    print(f'\nSample neurons with negative R2 but t_off > -1:')
    print(negative_r2_good[['component_idx', 't_off', 'r2_score', 'event_r2_score', 'events_per_min', 'kinetics_source']].to_string())
