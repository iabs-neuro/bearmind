"""
Compare different methods of computing trace_range:
1. Raw range: max - min
2. Percentile range: p99 - p1 (robust to outliers)
3. IQR range: p75 - p25 (very robust)
4. Trimmed range: p95 - p5 (moderate robustness)
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('COMPARISON: DIFFERENT trace_range COMPUTATION METHODS')
print('='*80)

# Load existing dataset
dataset_path = 'ml/results/training_dataset_v9_corrected_iter7.csv'
print(f'\nLoading dataset: {dataset_path}')
df = pd.read_csv(dataset_path)
print(f'Dataset: {len(df):,} neurons')

# Find estimates files and compute different range metrics
print('\n' + '='*80)
print('COMPUTING MULTIPLE RANGE METRICS')
print('='*80)

unique_sessions = df['session_name'].unique()
print(f'\nProcessing {len(unique_sessions)} sessions...')

# Storage for all range variants
range_metrics = {
    'session_name': [],
    'component_idx': [],
    'range_raw': [],          # max - min
    'range_p99_p1': [],       # p99 - p1
    'range_p95_p5': [],       # p95 - p5
    'range_iqr': [],          # p75 - p25
    'range_p90_p10': [],      # p90 - p10
}

sessions_processed = 0

for session in unique_sessions[:20]:  # Limit to first 20 for speed
    # Try to find processed estimates
    processed_path = Path(f'output/inspection_artifacts_{session}/{session}_processed.pickle')
    if not processed_path.exists():
        raw_path = Path(f'data/raw_compressed/{session}_estimates.pickle')
        if not raw_path.exists():
            continue
        estimates_path = raw_path
    else:
        estimates_path = processed_path

    try:
        with open(estimates_path, 'rb') as f:
            est = pickle.load(f)

        session_df = df[df['session_name'] == session]

        for idx, row in session_df.iterrows():
            comp_idx = int(row['component_idx'])

            if comp_idx >= est.C.shape[0]:
                continue

            trace = est.C[comp_idx, :].copy()

            # Compute all range variants
            range_metrics['session_name'].append(session)
            range_metrics['component_idx'].append(comp_idx)
            range_metrics['range_raw'].append(np.max(trace) - np.min(trace))
            range_metrics['range_p99_p1'].append(np.percentile(trace, 99) - np.percentile(trace, 1))
            range_metrics['range_p95_p5'].append(np.percentile(trace, 95) - np.percentile(trace, 5))
            range_metrics['range_iqr'].append(np.percentile(trace, 75) - np.percentile(trace, 25))
            range_metrics['range_p90_p10'].append(np.percentile(trace, 90) - np.percentile(trace, 10))

        sessions_processed += 1
        if sessions_processed % 5 == 0:
            print(f'  Processed {sessions_processed} sessions')

    except Exception as e:
        print(f'  Error processing {session}: {e}')
        continue

print(f'\nSuccessfully processed {sessions_processed} sessions')
print(f'Computed range metrics for {len(range_metrics["range_raw"]):,} neurons')

# Convert to DataFrame
metrics_df = pd.DataFrame(range_metrics)

# Merge with ground truth
merged_df = df.merge(metrics_df, on=['session_name', 'component_idx'], how='inner')
print(f'\nMerged dataset: {len(merged_df):,} neurons')

# Save
merged_df.to_csv('ml/results/dataset_with_range_variants.csv', index=False)
print(f'Saved to: ml/results/dataset_with_range_variants.csv')

# ANALYSIS 1: Correlation between methods
print('\n' + '='*80)
print('ANALYSIS 1: CORRELATION BETWEEN RANGE METHODS')
print('='*80)

range_cols = ['range_raw', 'range_p99_p1', 'range_p95_p5', 'range_iqr', 'range_p90_p10']

print('\nPearson correlation matrix:')
corr_matrix = merged_df[range_cols].corr()
print(corr_matrix.round(3))

# ANALYSIS 2: Discriminative power (AUC)
print('\n' + '='*80)
print('ANALYSIS 2: DISCRIMINATIVE POWER (AUC)')
print('='*80)

y = merged_df['ground_truth'].values
keep_mask = y == 1
delete_mask = y == 0

print(f'\n{"Method":<20} {"AUC":<10} {"Mean KEEP":<15} {"Mean DELETE":<15} {"Effect size"}')
print('-'*80)

results = []

for col in range_cols:
    values = merged_df[col].values
    valid_mask = ~np.isnan(values)

    if valid_mask.sum() < 100:
        continue

    # Compute AUC
    auc_score = roc_auc_score(y[valid_mask], values[valid_mask])
    auc_score = max(auc_score, 1 - auc_score)  # Take max for discriminative power

    # Compute means
    keep_vals = merged_df.loc[keep_mask, col].dropna()
    delete_vals = merged_df.loc[delete_mask, col].dropna()
    mean_keep = np.mean(keep_vals)
    mean_delete = np.mean(delete_vals)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(keep_vals)**2 + np.std(delete_vals)**2) / 2)
    if pooled_std > 0:
        cohens_d = (mean_keep - mean_delete) / pooled_std
    else:
        cohens_d = 0.0

    print(f'{col:<20} {auc_score:<10.4f} {mean_keep:<15.2f} {mean_delete:<15.2f} {cohens_d:+.3f}')

    results.append({
        'method': col,
        'auc': auc_score,
        'mean_keep': mean_keep,
        'mean_delete': mean_delete,
        'cohens_d': cohens_d
    })

# ANALYSIS 3: Statistical comparison
print('\n' + '='*80)
print('ANALYSIS 3: DISTRIBUTION COMPARISON (KEEP vs DELETE)')
print('='*80)

for col in range_cols:
    keep_vals = merged_df.loc[keep_mask, col].dropna()
    delete_vals = merged_df.loc[delete_mask, col].dropna()

    # Mann-Whitney U test
    stat, p_val = stats.mannwhitneyu(keep_vals, delete_vals, alternative='two-sided')

    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))

    print(f'\n{col}:')
    print(f'  KEEP:   mean={np.mean(keep_vals):8.2f}, median={np.median(keep_vals):8.2f}, std={np.std(keep_vals):8.2f}')
    print(f'  DELETE: mean={np.mean(delete_vals):8.2f}, median={np.median(delete_vals):8.2f}, std={np.std(delete_vals):8.2f}')
    print(f'  p-value: {p_val:.2e} {sig}')

# ANALYSIS 4: Outlier sensitivity
print('\n' + '='*80)
print('ANALYSIS 4: OUTLIER SENSITIVITY ANALYSIS')
print('='*80)

print('\nRatio of range methods to IQR (robustness metric):')
for col in ['range_raw', 'range_p99_p1', 'range_p95_p5', 'range_p90_p10']:
    ratio = merged_df[col] / merged_df['range_iqr']
    print(f'  {col:20s}: mean ratio = {ratio.mean():.2f} ± {ratio.std():.2f}')

# Higher ratio = more sensitive to outliers

# VISUALIZATION
print('\n' + '='*80)
print('GENERATING VISUALIZATIONS')
print('='*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot distributions for each method
for idx, col in enumerate(range_cols):
    ax = axes[idx // 3, idx % 3]

    keep_data = merged_df.loc[keep_mask, col].dropna()
    delete_data = merged_df.loc[delete_mask, col].dropna()

    ax.hist(keep_data, bins=50, alpha=0.6, label=f'KEEP (n={len(keep_data)})',
            color='green', density=True)
    ax.hist(delete_data, bins=50, alpha=0.6, label=f'DELETE (n={len(delete_data)})',
            color='red', density=True)

    # Get AUC for this method
    method_result = [r for r in results if r['method'] == col][0]
    auc_val = method_result['auc']

    ax.set_xlabel(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(f'{col}\nAUC={auc_val:.4f}', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Remove empty subplot
axes[1, 2].axis('off')

plt.tight_layout()
output_path = 'ml/results/range_methods_comparison.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f'\nSaved visualization: {output_path}')

# RANKING
print('\n' + '='*80)
print('FINAL RANKING BY AUC')
print('='*80)

results_sorted = sorted(results, key=lambda x: x['auc'], reverse=True)

print(f'\n{"Rank":<6} {"Method":<20} {"AUC":<10} {"Effect size":<15} {"Interpretation"}')
print('-'*80)

for rank, res in enumerate(results_sorted, 1):
    interpretation = "BEST" if rank == 1 else ("GOOD" if res['auc'] > 0.70 else "MODERATE")
    print(f'{rank:<6} {res["method"]:<20} {res["auc"]:<10.4f} {res["cohens_d"]:<+15.3f} {interpretation}')

# RECOMMENDATION
print('\n' + '='*80)
print('RECOMMENDATION')
print('='*80)

best = results_sorted[0]
second = results_sorted[1]

auc_diff = best['auc'] - second['auc']

print(f'\nBest method: {best["method"]} (AUC={best["auc"]:.4f})')
print(f'Second best: {second["method"]} (AUC={second["auc"]:.4f})')
print(f'Difference: {auc_diff:.4f} ({auc_diff*100:.1f}% relative improvement)')

if best['method'] == 'range_raw':
    print('\nINTERPRETATION:')
    print('  Raw range (max-min) is most discriminative.')
    print('  The extreme values (outliers) contain useful information.')
    print('  RECOMMENDATION: Use raw range.')
elif best['method'] in ['range_p99_p1', 'range_p95_p5']:
    print('\nINTERPRETATION:')
    print('  Percentile-based range is more discriminative than raw.')
    print('  Extreme outliers are confounding the signal.')
    print(f'  RECOMMENDATION: Use {best["method"]}.')
else:
    print('\nINTERPRETATION:')
    print('  Conservative range metric is most discriminative.')
    print('  Focus on typical variation, not extremes.')
    print(f'  RECOMMENDATION: Use {best["method"]}.')

if auc_diff < 0.01:
    print('\nNOTE: Difference is small (<0.01 AUC). Methods are comparable.')
    print('      Consider using the most robust method (p99-p1) for stability.')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
