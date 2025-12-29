"""
Deep analysis of simple trace statistics discriminative power.

Investigates whether basic features like trace mean, median, std, etc.
have strong discriminative power for neuron quality classification.
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('ANALYSIS: DISCRIMINATIVE POWER OF SIMPLE TRACE FEATURES')
print('='*80)

# Load dataset
dataset_path = 'ml/results/training_dataset_v9_corrected_iter7.csv'
print(f'\nLoading dataset: {dataset_path}')
df = pd.read_csv(dataset_path)
print(f'Dataset: {len(df):,} neurons')
print(f'KEEP: {(df["ground_truth"]==1).sum():,} ({(df["ground_truth"]==1).mean()*100:.1f}%)')
print(f'DELETE: {(df["ground_truth"]==0).sum():,} ({(df["ground_truth"]==0).mean()*100:.1f}%)')

# Load estimates to compute trace statistics
print('\n' + '='*80)
print('COMPUTING TRACE STATISTICS FROM RAW DATA')
print('='*80)

# Get unique sessions
unique_sessions = df['session_name'].unique()
print(f'\nFound {len(unique_sessions)} unique sessions')

# Dictionary to store trace stats for each neuron
trace_stats = {
    'session_name': [],
    'component_idx': [],
    'trace_mean': [],
    'trace_median': [],
    'trace_std': [],
    'trace_mad': [],  # Median Absolute Deviation
    'trace_iqr': [],  # Interquartile range
    'trace_min': [],
    'trace_max': [],
    'trace_range': [],
    'trace_cv': [],  # Coefficient of variation
}

print('\nProcessing sessions...')
sessions_processed = 0
sessions_found = 0

for session in unique_sessions[:20]:  # Limit to first 20 sessions for speed
    # Try to find processed estimates file
    processed_path = Path(f'output/inspection_artifacts_{session}/{session}_processed.pickle')

    if not processed_path.exists():
        # Try raw estimates
        raw_path = Path(f'data/raw_compressed/{session}_estimates.pickle')
        if not raw_path.exists():
            continue
        estimates_path = raw_path
    else:
        estimates_path = processed_path

    sessions_found += 1

    try:
        with open(estimates_path, 'rb') as f:
            est = pickle.load(f)

        # Get component indices from this session in dataset
        session_df = df[df['session_name'] == session]

        # For each component in this session
        for idx, row in session_df.iterrows():
            comp_idx = int(row['component_idx'])

            # Get trace (C matrix, row = component)
            if comp_idx >= est.C.shape[0]:
                continue

            trace = est.C[comp_idx, :].copy()

            # Compute statistics
            trace_stats['session_name'].append(session)
            trace_stats['component_idx'].append(comp_idx)
            trace_stats['trace_mean'].append(np.mean(trace))
            trace_stats['trace_median'].append(np.median(trace))
            trace_stats['trace_std'].append(np.std(trace))
            trace_stats['trace_mad'].append(np.median(np.abs(trace - np.median(trace))))
            trace_stats['trace_iqr'].append(np.percentile(trace, 75) - np.percentile(trace, 25))
            trace_stats['trace_min'].append(np.min(trace))
            trace_stats['trace_max'].append(np.max(trace))
            trace_stats['trace_range'].append(np.max(trace) - np.min(trace))

            # Coefficient of variation (avoid division by zero)
            if np.abs(np.mean(trace)) > 1e-10:
                trace_stats['trace_cv'].append(np.std(trace) / np.abs(np.mean(trace)))
            else:
                trace_stats['trace_cv'].append(np.nan)

        sessions_processed += 1
        if sessions_processed % 5 == 0:
            print(f'  Processed {sessions_processed}/{sessions_found} sessions found')

    except Exception as e:
        print(f'  Error processing {session}: {e}')
        continue

print(f'\nSuccessfully processed {sessions_processed} sessions')
print(f'Computed stats for {len(trace_stats["trace_mean"]):,} neurons')

# Convert to DataFrame
stats_df = pd.DataFrame(trace_stats)

# Merge with ground truth
merged_df = df.merge(stats_df, on=['session_name', 'component_idx'], how='inner')
print(f'\nMerged dataset: {len(merged_df):,} neurons with trace statistics')

# Save merged dataset
merged_df.to_csv('ml/results/dataset_with_trace_stats.csv', index=False)
print(f'Saved to: ml/results/dataset_with_trace_stats.csv')

# ANALYSIS 1: Distribution comparison
print('\n' + '='*80)
print('ANALYSIS 1: DISTRIBUTION COMPARISON (KEEP vs DELETE)')
print('='*80)

keep_mask = merged_df['ground_truth'] == 1
delete_mask = merged_df['ground_truth'] == 0

simple_features = ['trace_mean', 'trace_median', 'trace_std', 'trace_mad',
                   'trace_iqr', 'trace_range', 'trace_cv']

print('\nStatistical comparison:')
print(f'{"Feature":<20} {"KEEP mean":<15} {"DELETE mean":<15} {"Effect size":<12} {"p-value":<12} {"Significant?"}')
print('-'*100)

effect_sizes = {}
p_values = {}

for feat in simple_features:
    if feat not in merged_df.columns:
        continue

    keep_vals = merged_df.loc[keep_mask, feat].dropna()
    delete_vals = merged_df.loc[delete_mask, feat].dropna()

    # Mann-Whitney U test (non-parametric)
    stat, p_val = stats.mannwhitneyu(keep_vals, delete_vals, alternative='two-sided')

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(keep_vals)**2 + np.std(delete_vals)**2) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(keep_vals) - np.mean(delete_vals)) / pooled_std
    else:
        cohens_d = 0.0

    effect_sizes[feat] = cohens_d
    p_values[feat] = p_val

    sig = "YES" if p_val < 0.001 else "NO"
    print(f'{feat:<20} {np.mean(keep_vals):<15.4f} {np.mean(delete_vals):<15.4f} '
          f'{cohens_d:<12.3f} {p_val:<12.2e} {sig}')

# ANALYSIS 2: Individual feature discriminative power (AUC)
print('\n' + '='*80)
print('ANALYSIS 2: INDIVIDUAL FEATURE AUC (DISCRIMINATIVE POWER)')
print('='*80)

y = merged_df['ground_truth'].values
aucs = {}

print(f'\n{"Feature":<20} {"AUC":<10} {"Baseline AUC":<15} {"Improvement"}')
print('-'*70)

# Baseline: random classifier
baseline_auc = 0.5

for feat in simple_features:
    if feat not in merged_df.columns:
        continue

    # Get feature values (handle NaN)
    feat_vals = merged_df[feat].values
    valid_mask = ~np.isnan(feat_vals)

    if valid_mask.sum() < 100:
        continue

    try:
        # Compute AUC
        auc_score = roc_auc_score(y[valid_mask], feat_vals[valid_mask])

        # AUC can be < 0.5 if feature is inversely correlated
        # Take max(auc, 1-auc) to get discriminative power
        auc_score = max(auc_score, 1 - auc_score)

        aucs[feat] = auc_score
        improvement = (auc_score - baseline_auc) / baseline_auc * 100

        print(f'{feat:<20} {auc_score:<10.4f} {baseline_auc:<15.4f} {improvement:+.1f}%')
    except Exception as e:
        print(f'{feat:<20} ERROR: {e}')

# ANALYSIS 3: Compare to existing features
print('\n' + '='*80)
print('ANALYSIS 3: COMPARISON TO EXISTING COMPLEX FEATURES')
print('='*80)

existing_features = ['baseline', 'caiman_snr', 'event_snr', 'snr_recon',
                     'r2_score', 'event_r2_score', 'noise_level']

print(f'\n{"Feature":<20} {"AUC":<10} {"Feature Type"}')
print('-'*70)

all_feature_aucs = {}

for feat in existing_features:
    if feat not in merged_df.columns:
        continue

    feat_vals = merged_df[feat].values
    valid_mask = ~np.isnan(feat_vals)

    if valid_mask.sum() < 100:
        continue

    try:
        auc_score = roc_auc_score(y[valid_mask], feat_vals[valid_mask])
        auc_score = max(auc_score, 1 - auc_score)
        all_feature_aucs[feat] = auc_score

        print(f'{feat:<20} {auc_score:<10.4f} {"Existing (complex)"}')
    except:
        pass

# Add simple features
for feat, auc_score in aucs.items():
    all_feature_aucs[feat] = auc_score
    print(f'{feat:<20} {auc_score:<10.4f} {"Simple (new)"}')

# Rank all features
print('\n' + '='*80)
print('RANKING: ALL FEATURES BY DISCRIMINATIVE POWER')
print('='*80)

sorted_features = sorted(all_feature_aucs.items(), key=lambda x: x[1], reverse=True)

print(f'\n{"Rank":<6} {"Feature":<25} {"AUC":<10} {"Type"}')
print('-'*70)

for rank, (feat, auc_score) in enumerate(sorted_features, 1):
    feat_type = "Simple" if feat in simple_features else "Existing"
    print(f'{rank:<6} {feat:<25} {auc_score:<10.4f} {feat_type}')

# ANALYSIS 4: Session-specific effects
print('\n' + '='*80)
print('ANALYSIS 4: SESSION-SPECIFIC NORMALIZATION EFFECTS')
print('='*80)

# Check variance in trace_mean across sessions
session_means = merged_df.groupby('session_name')['trace_mean'].mean()
print(f'\nTrace mean across sessions:')
print(f'  Min session mean: {session_means.min():.6f}')
print(f'  Max session mean: {session_means.max():.6f}')
print(f'  Ratio: {session_means.max() / session_means.min():.2f}x')
print(f'  Coefficient of variation: {session_means.std() / session_means.mean():.3f}')

# Check if within-session discrimination is good
print('\n' + '='*80)
print('ANALYSIS 5: WITHIN-SESSION DISCRIMINATIVE POWER')
print('='*80)

within_session_aucs = []

for session in merged_df['session_name'].unique():
    session_data = merged_df[merged_df['session_name'] == session]

    # Need at least some positives and negatives
    if session_data['ground_truth'].sum() < 5 or (1-session_data['ground_truth']).sum() < 5:
        continue

    try:
        y_session = session_data['ground_truth'].values
        feat_vals = session_data['trace_mean'].values
        valid_mask = ~np.isnan(feat_vals)

        if valid_mask.sum() < 10:
            continue

        auc_score = roc_auc_score(y_session[valid_mask], feat_vals[valid_mask])
        auc_score = max(auc_score, 1 - auc_score)
        within_session_aucs.append(auc_score)
    except:
        pass

if len(within_session_aucs) > 0:
    print(f'\nWithin-session AUC for trace_mean:')
    print(f'  Mean: {np.mean(within_session_aucs):.4f}')
    print(f'  Median: {np.median(within_session_aucs):.4f}')
    print(f'  Std: {np.std(within_session_aucs):.4f}')
    print(f'  Min: {np.min(within_session_aucs):.4f}')
    print(f'  Max: {np.max(within_session_aucs):.4f}')

# VISUALIZATION
print('\n' + '='*80)
print('GENERATING VISUALIZATIONS')
print('='*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Distribution comparison for trace_mean
ax = axes[0, 0]
keep_data = merged_df.loc[keep_mask, 'trace_mean'].dropna()
delete_data = merged_df.loc[delete_mask, 'trace_mean'].dropna()

ax.hist(keep_data, bins=50, alpha=0.6, label=f'KEEP (n={len(keep_data)})', color='green', density=True)
ax.hist(delete_data, bins=50, alpha=0.6, label=f'DELETE (n={len(delete_data)})', color='red', density=True)
ax.set_xlabel('Trace Mean', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12, fontweight='bold')
ax.set_title(f'Trace Mean Distribution\nAUC={aucs.get("trace_mean", 0):.4f}', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Distribution comparison for trace_median
ax = axes[0, 1]
keep_data = merged_df.loc[keep_mask, 'trace_median'].dropna()
delete_data = merged_df.loc[delete_mask, 'trace_median'].dropna()

ax.hist(keep_data, bins=50, alpha=0.6, label=f'KEEP (n={len(keep_data)})', color='green', density=True)
ax.hist(delete_data, bins=50, alpha=0.6, label=f'DELETE (n={len(delete_data)})', color='red', density=True)
ax.set_xlabel('Trace Median', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12, fontweight='bold')
ax.set_title(f'Trace Median Distribution\nAUC={aucs.get("trace_median", 0):.4f}', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Distribution comparison for trace_std
ax = axes[0, 2]
keep_data = merged_df.loc[keep_mask, 'trace_std'].dropna()
delete_data = merged_df.loc[delete_mask, 'trace_std'].dropna()

ax.hist(keep_data, bins=50, alpha=0.6, label=f'KEEP (n={len(keep_data)})', color='green', density=True)
ax.hist(delete_data, bins=50, alpha=0.6, label=f'DELETE (n={len(delete_data)})', color='red', density=True)
ax.set_xlabel('Trace Std', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12, fontweight='bold')
ax.set_title(f'Trace Std Distribution\nAUC={aucs.get("trace_std", 0):.4f}', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Effect sizes
ax = axes[1, 0]
features_sorted = sorted(effect_sizes.items(), key=lambda x: abs(x[1]), reverse=True)
feat_names = [f[0].replace('trace_', '') for f in features_sorted]
effect_vals = [f[1] for f in features_sorted]

colors = ['green' if e > 0 else 'red' for e in effect_vals]
ax.barh(feat_names, effect_vals, color=colors, alpha=0.7)
ax.set_xlabel('Effect Size (Cohen\'s d)', fontsize=12, fontweight='bold')
ax.set_title('Effect Sizes: KEEP vs DELETE', fontsize=13, fontweight='bold')
ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

# Plot 5: AUC comparison
ax = axes[1, 1]
top_features = sorted_features[:10]
feat_names = [f[0] for f in top_features]
auc_vals = [f[1] for f in top_features]
colors_list = ['skyblue' if f in simple_features else 'orange' for f in feat_names]

ax.barh(feat_names, auc_vals, color=colors_list, alpha=0.7)
ax.set_xlabel('AUC', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Features by AUC\n(Blue=Simple, Orange=Existing)', fontsize=13, fontweight='bold')
ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='Random')
ax.set_xlim([0.5, 1.0])
ax.grid(True, alpha=0.3, axis='x')

# Plot 6: Within-session AUC distribution
if len(within_session_aucs) > 0:
    ax = axes[1, 2]
    ax.hist(within_session_aucs, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(within_session_aucs), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(within_session_aucs):.3f}')
    ax.set_xlabel('Within-Session AUC', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Within-Session Discriminative Power\n(trace_mean)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    axes[1, 2].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=14)
    axes[1, 2].set_title('Within-Session AUC', fontsize=13, fontweight='bold')

plt.tight_layout()
output_path = 'ml/results/simple_trace_features_analysis.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f'\nSaved visualization: {output_path}')

# SUMMARY
print('\n' + '='*80)
print('SUMMARY AND RECOMMENDATIONS')
print('='*80)

print('\nKEY FINDINGS:')

# Find best simple feature
best_simple = max([(f, aucs[f]) for f in simple_features if f in aucs], key=lambda x: x[1])
print(f'\n1. Best simple feature: {best_simple[0]} (AUC={best_simple[1]:.4f})')

# Compare to best existing
best_existing = max([(f, auc) for f, auc in all_feature_aucs.items() if f not in simple_features], key=lambda x: x[1])
print(f'2. Best existing feature: {best_existing[0]} (AUC={best_existing[1]:.4f})')

# Rank of best simple feature
simple_rank = [f for f, _ in sorted_features].index(best_simple[0]) + 1
print(f'3. Rank of best simple feature: #{simple_rank} out of {len(sorted_features)} total features')

# Effect size analysis
max_effect = max(effect_sizes.items(), key=lambda x: abs(x[1]))
print(f'4. Largest effect size: {max_effect[0]} (Cohen\'s d={max_effect[1]:.3f})')

if abs(max_effect[1]) > 0.8:
    print('   Interpretation: LARGE effect (strong discrimination)')
elif abs(max_effect[1]) > 0.5:
    print('   Interpretation: MEDIUM effect (moderate discrimination)')
else:
    print('   Interpretation: SMALL effect (weak discrimination)')

# Session variability
if len(session_means) > 0:
    cv = session_means.std() / session_means.mean()
    if cv > 0.5:
        print(f'\n5. Session variability: HIGH (CV={cv:.3f})')
        print('   WARNING: Raw trace statistics may not generalize across sessions')
        print('   RECOMMENDATION: Use session-normalized features or relative metrics')
    else:
        print(f'\n5. Session variability: LOW (CV={cv:.3f})')
        print('   GOOD: Raw trace statistics relatively stable across sessions')

print('\n' + '='*80)
print('CONCLUSION')
print('='*80)

if best_simple[1] > 0.75:
    print('\nSimple trace features show STRONG discriminative power!')
    print('RECOMMENDATION: Consider adding these features to the model.')
elif best_simple[1] > 0.65:
    print('\nSimple trace features show MODERATE discriminative power.')
    print('RECOMMENDATION: Worth testing if they improve model performance.')
else:
    print('\nSimple trace features show WEAK discriminative power.')
    print('RECOMMENDATION: Current complex features likely sufficient.')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
