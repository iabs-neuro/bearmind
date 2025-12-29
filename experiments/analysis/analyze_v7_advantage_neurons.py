"""
Deep analysis of neurons where v7 is correct but v8 makes mistakes.

Focus: Understanding what v7 "sees" that v8 misses, and how to improve v8.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print('='*80)
print('ANALYZING v7 ADVANTAGE NEURONS: Where v8 Can Be Improved')
print('='*80)

# Load datasets
v7 = pd.read_csv('ml/results/training_dataset_v7.csv')
v8 = pd.read_csv('ml/results/training_dataset_v8.csv')

# Load models
with open('production_models/ebm_v7.pkl', 'rb') as f:
    v7_model = pickle.load(f)
with open('production_models/ebm_v8.pkl', 'rb') as f:
    v8_model = pickle.load(f)

# Get predictions
exclude_cols = {'ground_truth', 'session', 'experiment', 'component_idx', 'center',
                'distance_to_gt', 'is_corner_artifact', 'corr_groups'}

v7_features = [c for c in v7.columns if c not in exclude_cols and c in v7_model.feature_names_in_]
v8_features = [c for c in v8.columns if c not in exclude_cols and c in v8_model.feature_names_in_]

X_v7 = v7[v7_features].copy()
y_v7 = v7['ground_truth'].values

X_v8 = v8[v8_features].copy()
y_v8 = v8['ground_truth'].values

v7_proba = v7_model.predict_proba(X_v7)[:, 1]
v7_pred = (v7_proba >= 0.75).astype(int)

v8_proba = v8_model.predict_proba(X_v8)[:, 1]
v8_pred = (v8_proba >= 0.75).astype(int)

# Create neuron IDs
v7['neuron_id'] = v7['session'] + '_' + v7['component_idx'].astype(str)
v8['neuron_id'] = v8['session'] + '_' + v8['component_idx'].astype(str)

v7['prediction'] = v7_pred
v7['proba'] = v7_proba
v7['correct'] = (v7_pred == y_v7)

v8['prediction'] = v8_pred
v8['proba'] = v8_proba
v8['correct'] = (v8_pred == y_v8)

# Identify v7 advantage neurons
comparison = v7[['neuron_id', 'ground_truth']].merge(
    v7[['neuron_id', 'prediction', 'proba', 'correct']].rename(
        columns={'prediction': 'v7_pred', 'proba': 'v7_proba', 'correct': 'v7_correct'}
    ),
    on='neuron_id'
).merge(
    v8[['neuron_id', 'prediction', 'proba', 'correct']].rename(
        columns={'prediction': 'v8_pred', 'proba': 'v8_proba', 'correct': 'v8_correct'}
    ),
    on='neuron_id'
)

v7_advantage = comparison[(comparison['v7_correct']) & (~comparison['v8_correct'])].copy()

print(f'\nv7 advantage neurons: {len(v7_advantage):,}')
print(f'  v7 correct, v8 wrong: {len(v7_advantage):,} ({len(v7_advantage)/len(comparison)*100:.2f}%)')

# Merge full feature data
# Drop columns that will conflict
v7_features_to_merge = [col for col in v7.columns if col not in ['prediction', 'proba', 'correct', 'neuron_id', 'ground_truth']]
v8_features_to_merge = [col for col in v8.columns if col not in ['prediction', 'proba', 'correct', 'neuron_id', 'ground_truth']]

v7_advantage = v7_advantage.merge(
    v7[['neuron_id'] + v7_features_to_merge],
    on='neuron_id', suffixes=('', '_v7')
).merge(
    v8[['neuron_id'] + v8_features_to_merge],
    on='neuron_id', suffixes=('_v7', '_v8')
)

# Clean up column names
for col in v7_advantage.columns:
    if col.endswith('_v7_v7'):
        v7_advantage.rename(columns={col: col.replace('_v7_v7', '_v7')}, inplace=True)
    elif col.endswith('_v8_v8'):
        v7_advantage.rename(columns={col: col.replace('_v8_v8', '_v8')}, inplace=True)

print(f'\nError breakdown:')
v8_fp = ((v7_advantage['v8_pred'] == 1) & (v7_advantage['ground_truth'] == 0)).sum()
v8_fn = ((v7_advantage['v8_pred'] == 0) & (v7_advantage['ground_truth'] == 1)).sum()
print(f'  v8 False Positives (wrongly kept bad neurons): {v8_fp:,} ({v8_fp/len(v7_advantage)*100:.1f}%)')
print(f'  v8 False Negatives (wrongly deleted good neurons): {v8_fn:,} ({v8_fn/len(v7_advantage)*100:.1f}%)')

# Analyze by error type
print(f'\n{"="*80}')
print('ANALYSIS BY ERROR TYPE')
print('='*80)

v7_adv_fp = v7_advantage[v7_advantage['v8_pred'] == 1]  # v8 false positives
v7_adv_fn = v7_advantage[v7_advantage['v8_pred'] == 0]  # v8 false negatives

print(f'\nv8 FALSE POSITIVES (wrongly kept bad neurons): {len(v7_adv_fp):,}')
print(f'  Ground truth: all should be DELETE (0)')
print(f'  v7 correctly predicted: DELETE')
print(f'  v8 incorrectly predicted: KEEP')

print(f'\nv8 FALSE NEGATIVES (wrongly deleted good neurons): {len(v7_adv_fn):,}')
print(f'  Ground truth: all should be KEEP (1)')
print(f'  v7 correctly predicted: KEEP')
print(f'  v8 incorrectly predicted: DELETE')

# Event detection comparison
print(f'\n{"="*80}')
print('EVENT DETECTION COMPARISON')
print('='*80)

v7_advantage['v7_has_events'] = v7_advantage['t_off_v7'] > -1
v7_advantage['v8_has_events'] = v7_advantage['t_off_v8'] > -1

print(f'\nEvent detection on v7 advantage neurons:')
print(f'  v7: {v7_advantage["v7_has_events"].sum():,}/{len(v7_advantage):,} ({v7_advantage["v7_has_events"].mean()*100:.1f}%)')
print(f'  v8: {v7_advantage["v8_has_events"].sum():,}/{len(v7_advantage):,} ({v7_advantage["v8_has_events"].mean()*100:.1f}%)')

print(f'\nFor v8 FALSE POSITIVES:')
print(f'  v7 event detection: {v7_adv_fp["v7_has_events"].mean()*100:.1f}%')
print(f'  v8 event detection: {v7_adv_fp["v8_has_events"].mean()*100:.1f}%')

print(f'\nFor v8 FALSE NEGATIVES:')
print(f'  v7 event detection: {v7_adv_fn["v7_has_events"].mean()*100:.1f}%')
print(f'  v8 event detection: {v7_adv_fn["v8_has_events"].mean()*100:.1f}%')

# Kinetics tier analysis
if 'kinetics_source_v8' in v7_advantage.columns:
    print(f'\n{"="*80}')
    print('KINETICS TIER ANALYSIS (v8)')
    print('='*80)

    tier_dist = v7_advantage['kinetics_source_v8'].value_counts()
    print(f'\nKinetics tier distribution:')
    for tier, count in tier_dist.items():
        print(f'  {tier:<25} {count:,} ({count/len(v7_advantage)*100:.1f}%)')

    print(f'\nBy error type:')
    print(f'\nFalse Positives:')
    fp_tiers = v7_adv_fp['kinetics_source_v8'].value_counts()
    for tier, count in fp_tiers.items():
        print(f'  {tier:<25} {count:,} ({count/len(v7_adv_fp)*100:.1f}%)')

    print(f'\nFalse Negatives:')
    fn_tiers = v7_adv_fn['kinetics_source_v8'].value_counts()
    for tier, count in fn_tiers.items():
        print(f'  {tier:<25} {count:,} ({count/len(v7_adv_fn)*100:.1f}%)')

# Feature comparison: Common features
print(f'\n{"="*80}')
print('FEATURE COMPARISON: v7 vs v8 (Common Features)')
print('='*80)

common_features = [f for f in v7_features if f in v8_features and f not in exclude_cols]
print(f'\nCommon features: {len(common_features)}')

# New features in v8
new_features_v8 = [f for f in v8_features if f not in v7_features and f not in exclude_cols]
print(f'New features in v8: {new_features_v8}')

# Compare feature values for common features
print(f'\n{"="*80}')
print('FEATURE VALUE DIFFERENCES: Where v8 data differs from v7')
print('='*80)

feature_diffs = []
for feat in common_features:
    v7_col = f'{feat}_v7' if f'{feat}_v7' in v7_advantage.columns else feat
    v8_col = f'{feat}_v8' if f'{feat}_v8' in v7_advantage.columns else feat

    if v7_col in v7_advantage.columns and v8_col in v7_advantage.columns:
        v7_vals = v7_advantage[v7_col].values
        v8_vals = v7_advantage[v8_col].values

        # Remove NaNs
        mask = ~(np.isnan(v7_vals) | np.isnan(v8_vals))
        if mask.sum() > 0:
            v7_vals_clean = v7_vals[mask]
            v8_vals_clean = v8_vals[mask]

            # Statistical test
            if len(v7_vals_clean) > 10:
                stat, pval = stats.ttest_rel(v7_vals_clean, v8_vals_clean)
                mean_diff = np.mean(v8_vals_clean - v7_vals_clean)
                rel_diff = mean_diff / (np.abs(np.mean(v7_vals_clean)) + 1e-10) * 100

                feature_diffs.append({
                    'feature': feat,
                    'v7_mean': np.mean(v7_vals_clean),
                    'v8_mean': np.mean(v8_vals_clean),
                    'mean_diff': mean_diff,
                    'rel_diff_pct': rel_diff,
                    'pval': pval,
                    'significant': pval < 0.001
                })

if feature_diffs:
    diff_df = pd.DataFrame(feature_diffs).sort_values('pval')
    print(f'\nTop 15 features with SIGNIFICANT differences (p < 0.001):')
    print(f'\n{"Feature":<25} {"v7 mean":<12} {"v8 mean":<12} {"Diff":<12} {"% Diff":<10} {"p-value"}')
    print('-'*95)

    sig_diffs = diff_df[diff_df['significant']].head(15)
    for _, row in sig_diffs.iterrows():
        print(f'{row["feature"]:<25} {row["v7_mean"]:<12.4f} {row["v8_mean"]:<12.4f} '
              f'{row["mean_diff"]:<12.4f} {row["rel_diff_pct"]:<10.1f} {row["pval"]:.2e}')

    if len(sig_diffs) > 0:
        print(f'\nKEY INSIGHT: v8 has different feature values than v7 on these neurons!')
        print(f'This could be due to:')
        print(f'  1. Hybrid kinetics producing different kinetics parameters')
        print(f'  2. Wavelet n=3 producing different event metrics')
        print(f'  3. Numerical differences in reconstruction')

# Analyze new features in v8
print(f'\n{"="*80}')
print('NEW v8 FEATURES ANALYSIS')
print('='*80)

if new_features_v8:
    print(f'\nAnalyzing how new features behave on v7 advantage neurons:')

    for feat in new_features_v8:
        feat_col = f'{feat}_v8' if f'{feat}_v8' in v7_advantage.columns else feat
        if feat_col in v7_advantage.columns:
            vals = v7_advantage[feat_col].dropna()
            if len(vals) > 0:
                print(f'\n{feat}:')
                print(f'  Mean: {vals.mean():.4f}, Std: {vals.std():.4f}')
                print(f'  Range: [{vals.min():.4f}, {vals.max():.4f}]')

                # Compare to overall v8 distribution
                v8_overall = v8[feat].dropna()
                overall_mean = v8_overall.mean()
                diff = vals.mean() - overall_mean
                print(f'  Overall v8 mean: {overall_mean:.4f}')
                print(f'  Difference: {diff:+.4f} ({diff/overall_mean*100:+.1f}%)')

                # Statistical test
                if len(vals) > 30 and len(v8_overall) > 30:
                    stat, pval = stats.ttest_ind(vals, v8_overall.sample(min(len(vals)*2, len(v8_overall))))
                    if pval < 0.001:
                        print(f'  SIGNIFICANT difference (p={pval:.2e})')

# Model confidence analysis
print(f'\n{"="*80}')
print('MODEL CONFIDENCE ANALYSIS')
print('='*80)

print(f'\nProbability distribution:')
print(f'  v7 probability: {v7_advantage["v7_proba"].mean():.3f} ± {v7_advantage["v7_proba"].std():.3f}')
print(f'  v8 probability: {v7_advantage["v8_proba"].mean():.3f} ± {v7_advantage["v8_proba"].std():.3f}')

print(f'\nBy error type:')
print(f'\nFalse Positives (v8 wrongly kept):')
print(f'  v7 proba: {v7_adv_fp["v7_proba"].mean():.3f} (correctly predicted DELETE)')
print(f'  v8 proba: {v7_adv_fp["v8_proba"].mean():.3f} (incorrectly predicted KEEP)')

print(f'\nFalse Negatives (v8 wrongly deleted):')
print(f'  v7 proba: {v7_adv_fn["v7_proba"].mean():.3f} (correctly predicted KEEP)')
print(f'  v8 proba: {v7_adv_fn["v8_proba"].mean():.3f} (incorrectly predicted DELETE)')

# Probability delta analysis
v7_advantage['proba_delta'] = v7_advantage['v8_proba'] - v7_advantage['v7_proba']

print(f'\nProbability shift (v8 - v7):')
print(f'  Overall: {v7_advantage["proba_delta"].mean():.3f} ± {v7_advantage["proba_delta"].std():.3f}')
print(f'  False Positives: {v7_adv_fp["proba_delta"].mean():+.3f} (v8 shifted UP)')
print(f'  False Negatives: {v7_adv_fn["proba_delta"].mean():+.3f} (v8 shifted DOWN)')

# Session analysis
print(f'\n{"="*80}')
print('SESSION ANALYSIS')
print('='*80)

session_counts = v7_advantage['session'].value_counts().head(10)
print(f'\nTop 10 sessions with most v7 advantage neurons:')
print(f'\n{"Session":<20} {"Count":<10} {"% of Total"}')
print('-'*45)
for session, count in session_counts.items():
    print(f'{session:<20} {count:<10} {count/len(v7_advantage)*100:.1f}%')

# Are certain sessions problematic for v8?
total_per_session = comparison.groupby('session').size()
v7_adv_per_session = v7_advantage.groupby('session').size()
session_v7_adv_pct = (v7_adv_per_session / total_per_session * 100).sort_values(ascending=False)

print(f'\nSessions with highest % v7 advantage:')
print(f'\n{"Session":<20} {"v7 Adv %":<12} {"Count"}')
print('-'*45)
for session, pct in session_v7_adv_pct.head(10).items():
    count = v7_adv_per_session[session]
    print(f'{session:<20} {pct:<12.1f} {count}')

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Error type breakdown
ax1 = axes[0, 0]
error_types = ['False Positive\n(wrongly KEEP)', 'False Negative\n(wrongly DELETE)']
error_counts = [len(v7_adv_fp), len(v7_adv_fn)]
colors = ['#FF6B6B', '#4ECDC4']
bars = ax1.bar(error_types, error_counts, color=colors, alpha=0.7)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('v8 Error Types on v7 Advantage Neurons', fontsize=12, fontweight='bold')
for bar, count in zip(bars, error_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 20,
             f'{count:,}\n({count/len(v7_advantage)*100:.1f}%)',
             ha='center', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Event detection comparison
ax2 = axes[0, 1]
categories = ['All v7 Adv', 'False Positive', 'False Negative']
v7_evt = [v7_advantage['v7_has_events'].mean()*100,
          v7_adv_fp['v7_has_events'].mean()*100,
          v7_adv_fn['v7_has_events'].mean()*100]
v8_evt = [v7_advantage['v8_has_events'].mean()*100,
          v7_adv_fp['v8_has_events'].mean()*100,
          v7_adv_fn['v8_has_events'].mean()*100]

x = np.arange(len(categories))
width = 0.35
ax2.bar(x - width/2, v7_evt, width, label='v7', color='orange', alpha=0.7)
ax2.bar(x + width/2, v8_evt, width, label='v8', color='green', alpha=0.7)
ax2.set_ylabel('Event Detection Success %', fontsize=11)
ax2.set_title('Event Detection: v8 Better but Still Wrong', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=9)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Kinetics tier (v8)
ax3 = axes[0, 2]
if 'kinetics_source_v8' in v7_advantage.columns:
    tier_counts = v7_advantage['kinetics_source_v8'].value_counts()
    colors_tier = ['steelblue', 'skyblue', 'lightblue', 'yellow', 'red', 'gray']
    ax3.pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%',
            colors=colors_tier[:len(tier_counts)], startangle=90)
    ax3.set_title(f'v8 Kinetics Tiers\n({len(v7_advantage):,} neurons)',
                  fontsize=12, fontweight='bold')

# Plot 4: Probability comparison
ax4 = axes[1, 0]
ax4.scatter(v7_advantage['v7_proba'], v7_advantage['v8_proba'],
            c=v7_advantage['ground_truth'], cmap='RdYlGn',
            alpha=0.4, s=20, edgecolors='none')
ax4.axhline(0.75, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax4.axvline(0.75, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)
ax4.set_xlabel('v7 Probability (CORRECT)', fontsize=11)
ax4.set_ylabel('v8 Probability (WRONG)', fontsize=11)
ax4.set_title('Model Confidence: v7 vs v8', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax4)
cbar.set_label('Ground Truth', fontsize=9)

# Plot 5: Probability delta by error type
ax5 = axes[1, 1]
data_to_plot = [v7_adv_fp['proba_delta'].dropna(), v7_adv_fn['proba_delta'].dropna()]
bp = ax5.boxplot(data_to_plot, labels=['False Positive', 'False Negative'],
                  patch_artist=True)
bp['boxes'][0].set_facecolor('#FF6B6B')
bp['boxes'][1].set_facecolor('#4ECDC4')
ax5.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax5.set_ylabel('Probability Shift (v8 - v7)', fontsize=11)
ax5.set_title('How v8 Probability Changed from v7', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# Plot 6: Top sessions with v7 advantage
ax6 = axes[1, 2]
top_sessions = session_v7_adv_pct.head(10)
y_pos = np.arange(len(top_sessions))
ax6.barh(y_pos, top_sessions.values, color='steelblue', alpha=0.7)
ax6.set_yticks(y_pos)
ax6.set_yticklabels(top_sessions.index, fontsize=8)
ax6.set_xlabel('% v7 Advantage', fontsize=11)
ax6.set_title('Sessions Where v8 Struggles Most', fontsize=12, fontweight='bold')
ax6.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('ml/results/v7_advantage_deep_analysis.png', dpi=150, bbox_inches='tight')
print(f'\n{"="*80}')
print(f'Plot saved to: ml/results/v7_advantage_deep_analysis.png')

# Save detailed data
v7_advantage.to_csv('ml/results/v7_advantage_neurons_detailed.csv', index=False)
print(f'Detailed data saved to: ml/results/v7_advantage_neurons_detailed.csv')

# Summary and recommendations
print(f'\n{"="*80}')
print('SUMMARY: Why v8 Fails Where v7 Succeeds')
print('='*80)

print(f'\n1. v8 MAKES MORE FALSE NEGATIVES ({len(v7_adv_fn):,}, {len(v7_adv_fn)/len(v7_advantage)*100:.1f}%)')
print(f'   Problem: v8 WRONGLY DELETES good neurons that v7 correctly keeps')
print(f'   v8 probability too low: {v7_adv_fn["v8_proba"].mean():.3f} (below 0.75 threshold)')
print(f'   v8 shifted DOWN by: {v7_adv_fn["proba_delta"].mean():.3f}')

print(f'\n2. v8 MAKES FEWER FALSE POSITIVES ({len(v7_adv_fp):,}, {len(v7_adv_fp)/len(v7_advantage)*100:.1f}%)')
print(f'   Problem: v8 WRONGLY KEEPS bad neurons that v7 correctly deletes')
print(f'   v8 probability too high: {v7_adv_fp["v8_proba"].mean():.3f} (above 0.75 threshold)')
print(f'   v8 shifted UP by: {v7_adv_fp["proba_delta"].mean():.3f}')

print(f'\n3. EVENT DETECTION PARADOX')
print(f'   v8 has BETTER event detection ({v7_advantage["v8_has_events"].mean()*100:.1f}% vs {v7_advantage["v7_has_events"].mean()*100:.1f}%)')
print(f'   But v8 still predicts WRONG while v7 predicts RIGHT')
print(f'   => Better events do NOT guarantee better predictions')

print(f'\n4. POSSIBLE ROOT CAUSES')
if len(sig_diffs) > 0:
    print(f'   a) Feature value differences: {len(sig_diffs)} features differ significantly')
    print(f'      Top differences: {", ".join(sig_diffs.head(3)["feature"].tolist())}')
    print(f'      => Hybrid kinetics/wavelet n=3 produce different feature values')

if new_features_v8:
    print(f'   b) New v8 features may be misleading: {new_features_v8}')
    print(f'      => Model may over-rely on new features that are noisy')

print(f'   c) Model hyperparameters:')
print(f'      v7: bins=256, leaves=5')
print(f'      v8: bins=1024, leaves=3')
print(f'      => v8 is more complex but less flexible (fewer leaves)')

print(f'\n{"="*80}')
print('RECOMMENDATIONS TO IMPROVE v8')
print('='*80)

print(f'\n1. RETRAIN with adjusted hyperparameters:')
print(f'   - Increase max_leaves from 3 to 5 (match v7)')
print(f'   - This may help v8 better use the richer feature space')

print(f'\n2. FEATURE ENGINEERING:')
print(f'   - Investigate why event features differ between v7 and v8')
print(f'   - Consider weighted average of v7 and v8 event metrics')
print(f'   - Add interaction terms between new and old features')

print(f'\n3. THRESHOLD TUNING:')
print(f'   - Current 0.75 threshold may not be optimal for v8')
print(f'   - Run threshold sensitivity analysis')
print(f'   - Consider different thresholds for different kinetics tiers')

print(f'\n4. ENSEMBLE APPROACH:')
print(f'   - Combine v7 and v8 predictions (e.g., average probabilities)')
print(f'   - Use v7 when v8 uncertainty is high')
print(f'   - This would capture strengths of both models')

print(f'\n5. ANALYZE SPECIFIC PROBLEMATIC SESSIONS:')
print(f'   - Focus on top sessions where v8 struggles')
print(f'   - Understand if there are systematic data quality issues')
