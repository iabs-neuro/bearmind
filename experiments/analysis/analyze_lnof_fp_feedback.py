"""
Deep analysis of LNOF False Positive feedback neurons.

Investigate what distinguishes neurons that experts marked as FP
(model said KEEP, expert said DELETE) from correctly classified neurons.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

print('='*80)
print('LNOF FALSE POSITIVE FEEDBACK ANALYSIS')
print('='*80)

# Load datasets
print('\nLoading datasets...')
lnof = pd.read_csv('LNOF_dataset_from_processed.csv')
v9 = pd.read_csv('ml/results/training_dataset_v9.csv')

print(f'LNOF dataset: {len(lnof)} neurons')
print(f'v9 dataset: {len(v9)} neurons')

# Filter to LNOF only in v9
v9_lnof = v9[v9['experiment'] == 'LNOF'].copy()
print(f'LNOF in v9: {len(v9_lnof)} neurons')

# Identify feedback categories
print('\n' + '='*80)
print('FEEDBACK CATEGORIES')
print('='*80)

# In LNOF dataset, 'delete' column shows final ground truth after feedback
# If decision='delete' but delete=0, that's a False Negative (FN) correction
# If decision='ok' but delete=1, that's a False Positive (FP) correction

# Actually, let me check if there's a feedback file
feedback_files = list(Path('data/LNOF').glob('**/LNOF_*_feedback.csv'))
print(f'\nFound {len(feedback_files)} feedback files')

# Collect all feedback
all_feedback = []
for fb_file in feedback_files:
    try:
        fb_df = pd.read_csv(fb_file)
        session_name = fb_file.stem.replace('_feedback', '')
        fb_df['session_name'] = session_name
        all_feedback.append(fb_df)
        print(f'  {session_name}: {len(fb_df)} feedback entries')
    except Exception as e:
        print(f'  [ERROR] {fb_file.name}: {e}')

if all_feedback:
    feedback_df = pd.concat(all_feedback, ignore_index=True)
    print(f'\nTotal feedback entries: {len(feedback_df)}')
    print(f'\nFeedback type distribution:')
    print(feedback_df['feedback_type'].value_counts())
else:
    print('\n[WARNING] No feedback files found')
    feedback_df = pd.DataFrame()

# Analyze LNOF dataset directly
print('\n' + '='*80)
print('LNOF DATASET ANALYSIS')
print('='*80)

# Check for decision column
if 'decision' in lnof.columns:
    print('\nDecision distribution:')
    print(lnof['decision'].value_counts())

# Check delete column
if 'delete' in lnof.columns:
    print('\nGround truth (delete) distribution:')
    delete_dist = lnof['delete'].value_counts()
    print(f'  KEEP (delete=0): {delete_dist.get(0, 0)}')
    print(f'  DELETE (delete=1): {delete_dist.get(1, 0)}')

# If we have both decision and delete, we can infer feedback
if 'decision' in lnof.columns and 'delete' in lnof.columns:
    # Map decision to initial prediction
    lnof['initial_prediction'] = lnof['decision'].map({'ok': 0, 'delete': 1})

    # Compare with final ground truth
    lnof['was_corrected'] = lnof['initial_prediction'] != lnof['delete']
    lnof['correction_type'] = 'none'

    # FP: model said ok (0), expert said delete (1)
    fp_mask = (lnof['initial_prediction'] == 0) & (lnof['delete'] == 1)
    lnof.loc[fp_mask, 'correction_type'] = 'FP'

    # FN: model said delete (1), expert said ok (0)
    fn_mask = (lnof['initial_prediction'] == 1) & (lnof['delete'] == 0)
    lnof.loc[fn_mask, 'correction_type'] = 'FN'

    print('\n' + '='*80)
    print('CORRECTIONS IDENTIFIED')
    print('='*80)
    print(f'\nTotal corrections: {lnof["was_corrected"].sum()}')
    print(f'  False Positives (FP): {fp_mask.sum()} - model said KEEP, expert said DELETE')
    print(f'  False Negatives (FN): {fn_mask.sum()} - model said DELETE, expert said KEEP')
    print(f'  Correct: {(~lnof["was_corrected"]).sum()}')

# Focus on FP neurons
fp_neurons = lnof[fp_mask].copy()
correct_keep = lnof[(lnof['delete'] == 0) & (~lnof['was_corrected'])].copy()

print('\n' + '='*80)
print('FALSE POSITIVE CHARACTERISTICS')
print('='*80)

print(f'\nAnalyzing {len(fp_neurons)} FP neurons vs {len(correct_keep)} correctly kept neurons')

# Define features to analyze
feature_groups = {
    'CaImAn Quality': ['caiman_snr', 'caiman_r_score'],
    'Spatial Features': ['area', 'circularity', 'convexity', 'aspect_ratio',
                        'eccentricity', 'edge_distance', 'footprint_compactness'],
    'Signal Features': ['event_snr', 'events_fraction', 'events_per_min',
                       'kinetics_opt', 't_rise', 't_off'],
    'Reconstruction Quality': ['r2_score', 'event_r2_score', 'nmae', 'nrmse', 'snr_recon'],
    'Trace Statistics': ['baseline', 'noise_level', 'tau_decay',
                        'trace_skewness', 'trace_kurtosis', 'bimodality']
}

# Statistical comparison
results = []

for group_name, features in feature_groups.items():
    print(f'\n{group_name}:')
    print('-' * 80)

    for feature in features:
        if feature not in lnof.columns:
            continue

        fp_vals = fp_neurons[feature].dropna()
        correct_vals = correct_keep[feature].dropna()

        if len(fp_vals) < 5 or len(correct_vals) < 5:
            continue

        # Statistical test
        stat, pval = stats.mannwhitneyu(fp_vals, correct_vals, alternative='two-sided')

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((fp_vals.std()**2 + correct_vals.std()**2) / 2)
        cohens_d = (fp_vals.mean() - correct_vals.mean()) / pooled_std if pooled_std > 0 else 0

        # Determine significance
        sig = ''
        if pval < 0.001:
            sig = '***'
        elif pval < 0.01:
            sig = '**'
        elif pval < 0.05:
            sig = '*'

        results.append({
            'feature': feature,
            'group': group_name,
            'fp_mean': fp_vals.mean(),
            'fp_median': fp_vals.median(),
            'correct_mean': correct_vals.mean(),
            'correct_median': correct_vals.median(),
            'cohens_d': cohens_d,
            'p_value': pval,
            'significant': sig
        })

        print(f'  {feature:25s} | FP: {fp_vals.mean():8.3f} ± {fp_vals.std():6.3f} | '
              f'Correct: {correct_vals.mean():8.3f} ± {correct_vals.std():6.3f} | '
              f'd={cohens_d:6.3f} {sig}')

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('p_value')

# Save detailed results
output_path = 'ml/results/lnof_fp_analysis.csv'
results_df.to_csv(output_path, index=False)
print(f'\n\nDetailed results saved: {output_path}')

# Summary of most significant differences
print('\n' + '='*80)
print('TOP 10 DISTINGUISHING FEATURES (by p-value)')
print('='*80)

top_features = results_df.head(10)
for idx, row in top_features.iterrows():
    direction = 'LOWER' if row['cohens_d'] < 0 else 'HIGHER'
    print(f"\n{row['feature']} ({row['group']}):")
    print(f"  FP neurons have {direction} values (d={row['cohens_d']:.3f})")
    print(f"  FP: {row['fp_mean']:.3f} (median: {row['fp_median']:.3f})")
    print(f"  Correct: {row['correct_mean']:.3f} (median: {row['correct_median']:.3f})")
    print(f"  p-value: {row['p_value']:.2e} {row['significant']}")

# Session distribution
print('\n' + '='*80)
print('SESSION DISTRIBUTION OF FP NEURONS')
print('='*80)

if 'session_name' in fp_neurons.columns:
    fp_by_session = fp_neurons['session_name'].value_counts()
    total_by_session = lnof['session_name'].value_counts()

    fp_rate_by_session = (fp_by_session / total_by_session * 100).sort_values(ascending=False)

    print(f'\nTop 10 sessions by FP rate:')
    for session, rate in fp_rate_by_session.head(10).items():
        n_fp = fp_by_session[session]
        n_total = total_by_session[session]
        print(f'  {session:20s}: {n_fp:3d}/{n_total:3d} ({rate:5.1f}%)')

# Generate visualizations
print('\n' + '='*80)
print('GENERATING VISUALIZATIONS')
print('='*80)

# Plot top distinguishing features
top_n = 8
top_features = results_df.head(top_n)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, (_, row) in enumerate(top_features.iterrows()):
    if idx >= len(axes):
        break

    ax = axes[idx]
    feature = row['feature']

    fp_vals = fp_neurons[feature].dropna()
    correct_vals = correct_keep[feature].dropna()

    # Box plot
    ax.boxplot([fp_vals, correct_vals], labels=['FP', 'Correct KEEP'])
    ax.set_title(f"{feature}\n(d={row['cohens_d']:.2f}, p={row['p_value']:.2e})",
                fontsize=9)
    ax.set_ylabel(feature)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = 'output/lnof_fp_feature_comparison.png'
Path('output').mkdir(exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f'\nSaved: {plot_path}')
plt.close()

# Summary report
print('\n' + '='*80)
print('SUMMARY')
print('='*80)

print(f'''
LNOF False Positive Analysis Complete

Dataset Statistics:
- Total LNOF neurons: {len(lnof):,}
- False Positives (FP): {len(fp_neurons):,} ({100*len(fp_neurons)/len(lnof):.2f}%)
- Correctly kept: {len(correct_keep):,}
- False Negatives (FN): {fn_mask.sum():,}

Key Findings:
- {len(results_df[results_df['p_value'] < 0.001])} features with highly significant differences (p < 0.001)
- {len(results_df[results_df['p_value'] < 0.05])} features with significant differences (p < 0.05)
- Top distinguishing feature: {results_df.iloc[0]['feature']} (d={results_df.iloc[0]['cohens_d']:.3f})

Output Files:
- Detailed analysis: {output_path}
- Feature comparison plots: {plot_path}
''')

print('='*80)
