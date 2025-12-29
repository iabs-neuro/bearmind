"""
Analyze correlation data characteristics to understand Pearson vs Spearman.
"""
import pandas as pd
import numpy as np

# Load full dataset
df = pd.read_csv('ml/results/corr/correlation_pairs_full.csv')

print('=' * 80)
print('CORRELATION DATA ANALYSIS')
print('=' * 80)
print()

# Basic stats
print('Dataset size:', len(df), 'pairs')
print('Distance range:', f'{df.distance.min():.2f} - {df.distance.max():.2f} pixels')
print()

# Correlation comparison
print('PEARSON vs SPEARMAN:')
print(f'  Pearson  - Mean: {df.pearson_corr.mean():.3f}, Std: {df.pearson_corr.std():.3f}')
print(f'  Spearman - Mean: {df.spearman_corr.mean():.3f}, Std: {df.spearman_corr.std():.3f}')
print(f'  Difference: Spearman is {df.spearman_corr.mean() - df.pearson_corr.mean():.3f} higher on average')
print()

# Correlation of correlations
from scipy.stats import spearmanr
corr_of_corr, _ = spearmanr(df.pearson_corr, df.spearman_corr)
print(f'Correlation between Pearson and Spearman: {corr_of_corr:.3f}')
print()

# Threshold analysis
pearson_high = df[df.pearson_corr >= 0.6]
spearman_high_528 = df[df.spearman_corr >= 0.528]
spearman_high_419 = df[df.spearman_corr >= 0.419]

print('THRESHOLD IMPACT:')
print(f'  Pearson >= 0.6:     {len(pearson_high):5d} pairs ({100*len(pearson_high)/len(df):.1f}%)')
print(f'  Spearman >= 0.528:  {len(spearman_high_528):5d} pairs ({100*len(spearman_high_528)/len(df):.1f}%)')
print(f'  Spearman >= 0.419:  {len(spearman_high_419):5d} pairs ({100*len(spearman_high_419)/len(df):.1f}%)')
print()

# What does Spearman catch that Pearson doesn't?
spearman_only_528 = df[(df.spearman_corr >= 0.528) & (df.pearson_corr < 0.6)]
print(f'Spearman 0.528 catches {len(spearman_only_528)} pairs that Pearson 0.6 misses:')
print(f'  Mean distance: {spearman_only_528.distance.mean():.2f} pixels')
print(f'  Mean Pearson: {spearman_only_528.pearson_corr.mean():.3f}')
print(f'  Mean Spearman: {spearman_only_528.spearman_corr.mean():.3f}')
print(f'  Spearman/Pearson ratio: {spearman_only_528.spearman_corr.mean() / spearman_only_528.pearson_corr.mean():.2f}')
print()

# Non-linearity analysis
print('NON-LINEARITY INDICATORS:')
# For highly correlated pairs, check if Spearman >> Pearson
high_corr = df[df.pearson_corr >= 0.4]
nonlinear = high_corr[high_corr.spearman_corr - high_corr.pearson_corr > 0.1]
print(f'  Pairs with Pearson >= 0.4: {len(high_corr)}')
print(f'  Of these, Spearman > Pearson + 0.1: {len(nonlinear)} ({100*len(nonlinear)/len(high_corr):.1f}%)')
print(f'  -> Suggests significant non-linearity in {100*len(nonlinear)/len(high_corr):.1f}% of correlated pairs')
print()

# Very close pairs (likely duplicates)
very_close = df[df.distance < 3]
print(f'VERY CLOSE PAIRS (< 3 pixels, likely duplicates): {len(very_close)} pairs')
if len(very_close) > 0:
    print(f'  Mean Pearson: {very_close.pearson_corr.mean():.3f}')
    print(f'  Mean Spearman: {very_close.spearman_corr.mean():.3f}')
    print(f'  Pearson >= 0.6: {len(very_close[very_close.pearson_corr >= 0.6])} ({100*len(very_close[very_close.pearson_corr >= 0.6])/len(very_close):.1f}%)')
    print(f'  Spearman >= 0.528: {len(very_close[very_close.spearman_corr >= 0.528])} ({100*len(very_close[very_close.spearman_corr >= 0.528])/len(very_close):.1f}%)')
print()

# Distance stratification
print('CORRELATION BY DISTANCE:')
for d_min, d_max in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]:
    subset = df[(df.distance >= d_min) & (df.distance < d_max)]
    if len(subset) > 0:
        print(f'  {d_min}-{d_max} pixels ({len(subset):5d} pairs):')
        print(f'    Pearson: {subset.pearson_corr.mean():.3f}, Spearman: {subset.spearman_corr.mean():.3f}')

print()
print('=' * 80)
