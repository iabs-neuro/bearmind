"""
Deep analysis of prediction mismatches between v7 and v8 models.

Identifies where models disagree and analyzes characteristics of:
- v7 wrong, v8 correct (v8 advantage)
- v7 correct, v8 wrong (v7 advantage)
- Both wrong (hard cases)
- Both correct (easy cases)
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

print('='*80)
print('ANALYZING PREDICTION MISMATCHES: v7 vs v8')
print('='*80)

# Load datasets
v7 = pd.read_csv('ml/results/training_dataset_v7.csv')
v8 = pd.read_csv('ml/results/training_dataset_v8.csv')

print(f'\nDatasets: {len(v7):,} neurons each')

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

# Predictions at threshold 0.75
v7_proba = v7_model.predict_proba(X_v7)[:, 1]
v7_pred = (v7_proba >= 0.75).astype(int)

v8_proba = v8_model.predict_proba(X_v8)[:, 1]
v8_pred = (v8_proba >= 0.75).astype(int)

# Add to dataframes
v7['prediction'] = v7_pred
v7['proba'] = v7_proba
v7['correct'] = (v7_pred == y_v7)

v8['prediction'] = v8_pred
v8['proba'] = v8_proba
v8['correct'] = (v8_pred == y_v8)

# Create neuron ID for matching
v7['neuron_id'] = v7['session'] + '_' + v7['component_idx'].astype(str)
v8['neuron_id'] = v8['session'] + '_' + v8['component_idx'].astype(str)

print(f'\nOverall accuracy:')
print(f'  v7: {v7["correct"].mean()*100:.2f}%')
print(f'  v8: {v8["correct"].mean()*100:.2f}%')

# Categorize neurons
print(f'\n{"="*80}')
print('CATEGORIZING NEURONS BY PREDICTION OUTCOMES')
print('='*80)

# Merge datasets for comparison
comparison = v7[['neuron_id', 'session', 'experiment', 'ground_truth']].merge(
    v7[['neuron_id', 'prediction', 'proba', 'correct', 't_off']].rename(
        columns={'prediction': 'v7_pred', 'proba': 'v7_proba',
                 'correct': 'v7_correct', 't_off': 'v7_t_off'}
    ),
    on='neuron_id'
).merge(
    v8[['neuron_id', 'prediction', 'proba', 'correct', 't_off', 'kinetics_source']].rename(
        columns={'prediction': 'v8_pred', 'proba': 'v8_proba',
                 'correct': 'v8_correct', 't_off': 'v8_t_off',
                 'kinetics_source': 'v8_kinetics_source'}
    ),
    on='neuron_id'
)

print(f'\nMatched neurons: {len(comparison):,}')

# Categorize
comparison['category'] = 'both_correct'
comparison.loc[(~comparison['v7_correct']) & (~comparison['v8_correct']), 'category'] = 'both_wrong'
comparison.loc[(~comparison['v7_correct']) & (comparison['v8_correct']), 'category'] = 'v8_advantage'
comparison.loc[(comparison['v7_correct']) & (~comparison['v8_correct']), 'category'] = 'v7_advantage'

# Category counts
category_counts = comparison['category'].value_counts()

print(f'\n{"Category":<20} {"Count":<10} {"% Total":<10} {"Description"}')
print('-'*80)
print(f'{"both_correct":<20} {category_counts.get("both_correct", 0):<10,} {category_counts.get("both_correct", 0)/len(comparison)*100:<10.2f} Both models correct')
print(f'{"both_wrong":<20} {category_counts.get("both_wrong", 0):<10,} {category_counts.get("both_wrong", 0)/len(comparison)*100:<10.2f} Both models wrong')
print(f'{"v8_advantage":<20} {category_counts.get("v8_advantage", 0):<10,} {category_counts.get("v8_advantage", 0)/len(comparison)*100:<10.2f} v7 wrong, v8 correct')
print(f'{"v7_advantage":<20} {category_counts.get("v7_advantage", 0):<10,} {category_counts.get("v7_advantage", 0)/len(comparison)*100:<10.2f} v7 correct, v8 wrong')

# Net advantage
v8_net_advantage = category_counts.get('v8_advantage', 0) - category_counts.get('v7_advantage', 0)
print(f'\nNet v8 advantage: {v8_net_advantage:+,} neurons ({v8_net_advantage/len(comparison)*100:+.2f}%)')

# Analyze mismatches by ground truth
print(f'\n{"="*80}')
print('MISMATCH ANALYSIS BY GROUND TRUTH')
print('='*80)

for category in ['v8_advantage', 'v7_advantage', 'both_wrong']:
    subset = comparison[comparison['category'] == category]
    if len(subset) > 0:
        print(f'\n{category.upper().replace("_", " ")} ({len(subset):,} neurons):')
        gt_dist = subset['ground_truth'].value_counts()
        keep_pct = gt_dist.get(1, 0) / len(subset) * 100
        delete_pct = gt_dist.get(0, 0) / len(subset) * 100
        print(f'  Ground truth: KEEP={gt_dist.get(1, 0):,} ({keep_pct:.1f}%), DELETE={gt_dist.get(0, 0):,} ({delete_pct:.1f}%)')

        # Prediction distribution
        if category == 'v8_advantage':
            v7_keep_preds = (subset['v7_pred'] == 1).sum()
            v8_keep_preds = (subset['v8_pred'] == 1).sum()
            print(f'  v7 predicted KEEP: {v7_keep_preds:,} ({v7_keep_preds/len(subset)*100:.1f}%)')
            print(f'  v8 predicted KEEP: {v8_keep_preds:,} ({v8_keep_preds/len(subset)*100:.1f}%)')
        elif category == 'v7_advantage':
            v7_keep_preds = (subset['v7_pred'] == 1).sum()
            v8_keep_preds = (subset['v8_pred'] == 1).sum()
            print(f'  v7 predicted KEEP: {v7_keep_preds:,} ({v7_keep_preds/len(subset)*100:.1f}%)')
            print(f'  v8 predicted KEEP: {v8_keep_preds:,} ({v8_keep_preds/len(subset)*100:.1f}%)')

# Analyze mismatches by event detection
print(f'\n{"="*80}')
print('MISMATCH ANALYSIS BY EVENT DETECTION')
print('='*80)

comparison['v7_has_events'] = comparison['v7_t_off'] > -1
comparison['v8_has_events'] = comparison['v8_t_off'] > -1

for category in ['v8_advantage', 'v7_advantage', 'both_wrong']:
    subset = comparison[comparison['category'] == category]
    if len(subset) > 0:
        print(f'\n{category.upper().replace("_", " ")}:')
        v7_evt_pct = subset['v7_has_events'].mean() * 100
        v8_evt_pct = subset['v8_has_events'].mean() * 100
        print(f'  v7 event detection: {subset["v7_has_events"].sum():,}/{len(subset):,} ({v7_evt_pct:.1f}%)')
        print(f'  v8 event detection: {subset["v8_has_events"].sum():,}/{len(subset):,} ({v8_evt_pct:.1f}%)')

        # Cross-tabulation
        both_yes = ((subset['v7_has_events']) & (subset['v8_has_events'])).sum()
        both_no = ((~subset['v7_has_events']) & (~subset['v8_has_events'])).sum()
        v7_only = ((subset['v7_has_events']) & (~subset['v8_has_events'])).sum()
        v8_only = ((~subset['v7_has_events']) & (subset['v8_has_events'])).sum()

        print(f'  Both detected: {both_yes:,}, Both failed: {both_no:,}')
        print(f'  v7 only: {v7_only:,}, v8 only: {v8_only:,}')

# Analyze v8 advantage by kinetics tier
if 'v8_kinetics_source' in comparison.columns:
    print(f'\n{"="*80}')
    print('V8 ADVANTAGE BY KINETICS TIER')
    print('='*80)

    v8_adv = comparison[comparison['category'] == 'v8_advantage']

    if len(v8_adv) > 0:
        tier_dist = v8_adv['v8_kinetics_source'].value_counts()
        print(f'\nWhere v8 corrects v7 mistakes ({len(v8_adv):,} neurons):')
        print(f'\n{"Kinetics Tier":<25} {"Count":<10} {"% of v8 advantage"}')
        print('-'*50)
        for tier, count in tier_dist.items():
            print(f'{tier:<25} {count:<10,} {count/len(v8_adv)*100:.1f}%')

        # Compare to overall tier distribution
        print(f'\nCompare to overall v8 tier distribution:')
        overall_tier_dist = comparison['v8_kinetics_source'].value_counts(normalize=True) * 100
        v8_adv_tier_dist = tier_dist / len(v8_adv) * 100

        print(f'\n{"Tier":<25} {"Overall %":<12} {"v8 Adv %":<12} {"Enrichment"}')
        print('-'*65)
        for tier in overall_tier_dist.index:
            overall_pct = overall_tier_dist.get(tier, 0)
            adv_pct = v8_adv_tier_dist.get(tier, 0)
            enrichment = adv_pct / overall_pct if overall_pct > 0 else 0
            marker = ' [ENRICHED]' if enrichment > 1.2 else ''
            print(f'{tier:<25} {overall_pct:<12.1f} {adv_pct:<12.1f} {enrichment:.2f}x{marker}')

# Analyze error types
print(f'\n{"="*80}')
print('ERROR TYPE ANALYSIS')
print('='*80)

def analyze_errors(subset, name):
    """Analyze false positives and false negatives."""
    if len(subset) == 0:
        return

    print(f'\n{name} ({len(subset):,} neurons):')

    # For v8_advantage: analyze v7 errors
    # For v7_advantage: analyze v8 errors
    # For both_wrong: analyze both

    if name == 'V8 ADVANTAGE (v7 errors)':
        fp = ((subset['v7_pred'] == 1) & (subset['ground_truth'] == 0)).sum()
        fn = ((subset['v7_pred'] == 0) & (subset['ground_truth'] == 1)).sum()
        print(f'  v7 false positives: {fp:,} ({fp/len(subset)*100:.1f}%)')
        print(f'  v7 false negatives: {fn:,} ({fn/len(subset)*100:.1f}%)')
        print(f'  v8 fixed all {len(subset):,} errors')

    elif name == 'V7 ADVANTAGE (v8 errors)':
        fp = ((subset['v8_pred'] == 1) & (subset['ground_truth'] == 0)).sum()
        fn = ((subset['v8_pred'] == 0) & (subset['ground_truth'] == 1)).sum()
        print(f'  v8 false positives: {fp:,} ({fp/len(subset)*100:.1f}%)')
        print(f'  v8 false negatives: {fn:,} ({fn/len(subset)*100:.1f}%)')
        print(f'  v7 fixed all {len(subset):,} errors')

    elif name == 'BOTH WRONG (hard cases)':
        # Check if errors are same type
        v7_fp = ((subset['v7_pred'] == 1) & (subset['ground_truth'] == 0)).sum()
        v7_fn = ((subset['v7_pred'] == 0) & (subset['ground_truth'] == 1)).sum()
        v8_fp = ((subset['v8_pred'] == 1) & (subset['ground_truth'] == 0)).sum()
        v8_fn = ((subset['v8_pred'] == 0) & (subset['ground_truth'] == 1)).sum()

        print(f'  v7: FP={v7_fp:,} ({v7_fp/len(subset)*100:.1f}%), FN={v7_fn:,} ({v7_fn/len(subset)*100:.1f}%)')
        print(f'  v8: FP={v8_fp:,} ({v8_fp/len(subset)*100:.1f}%), FN={v8_fn:,} ({v8_fn/len(subset)*100:.1f}%)')

        # Same error type?
        same_error = ((subset['v7_pred'] == subset['v8_pred'])).sum()
        print(f'  Same prediction (both make identical error): {same_error:,} ({same_error/len(subset)*100:.1f}%)')

analyze_errors(comparison[comparison['category'] == 'v8_advantage'], 'V8 ADVANTAGE (v7 errors)')
analyze_errors(comparison[comparison['category'] == 'v7_advantage'], 'V7 ADVANTAGE (v8 errors)')
analyze_errors(comparison[comparison['category'] == 'both_wrong'], 'BOTH WRONG (hard cases)')

# Probability analysis for mismatches
print(f'\n{"="*80}')
print('PROBABILITY CONFIDENCE ANALYSIS')
print('='*80)

for category in ['v8_advantage', 'v7_advantage', 'both_wrong']:
    subset = comparison[comparison['category'] == category]
    if len(subset) > 0:
        print(f'\n{category.upper().replace("_", " ")}:')
        print(f'  v7 probability: {subset["v7_proba"].mean():.3f} ± {subset["v7_proba"].std():.3f} (range: {subset["v7_proba"].min():.3f}-{subset["v7_proba"].max():.3f})')
        print(f'  v8 probability: {subset["v8_proba"].mean():.3f} ± {subset["v8_proba"].std():.3f} (range: {subset["v8_proba"].min():.3f}-{subset["v8_proba"].max():.3f})')

        # How many are near threshold?
        v7_near_thresh = ((subset['v7_proba'] > 0.70) & (subset['v7_proba'] < 0.80)).sum()
        v8_near_thresh = ((subset['v8_proba'] > 0.70) & (subset['v8_proba'] < 0.80)).sum()
        print(f'  Near threshold (0.70-0.80): v7={v7_near_thresh:,} ({v7_near_thresh/len(subset)*100:.1f}%), v8={v8_near_thresh:,} ({v8_near_thresh/len(subset)*100:.1f}%)')

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Category distribution
ax1 = axes[0, 0]
categories = ['both_correct', 'both_wrong', 'v8_advantage', 'v7_advantage']
counts = [category_counts.get(cat, 0) for cat in categories]
colors = ['green', 'red', 'blue', 'orange']
ax1.bar(categories, counts, color=colors, alpha=0.7)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('Prediction Agreement Categories', fontsize=12, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
for i, (cat, count) in enumerate(zip(categories, counts)):
    ax1.text(i, count + 100, f'{count:,}\n({count/len(comparison)*100:.1f}%)',
             ha='center', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Event detection by category
ax2 = axes[0, 1]
event_data = []
for cat in ['v8_advantage', 'v7_advantage', 'both_wrong']:
    subset = comparison[comparison['category'] == cat]
    if len(subset) > 0:
        event_data.append({
            'category': cat.replace('_', '\n'),
            'v7_events': subset['v7_has_events'].mean() * 100,
            'v8_events': subset['v8_has_events'].mean() * 100
        })

if event_data:
    event_df = pd.DataFrame(event_data)
    x = np.arange(len(event_df))
    width = 0.35
    ax2.bar(x - width/2, event_df['v7_events'], width, label='v7', color='orange', alpha=0.7)
    ax2.bar(x + width/2, event_df['v8_events'], width, label='v8', color='green', alpha=0.7)
    ax2.set_ylabel('Event Detection Success %', fontsize=11)
    ax2.set_title('Event Detection by Mismatch Category', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(event_df['category'], fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

# Plot 3: Kinetics tier for v8 advantage
ax3 = axes[0, 2]
v8_adv = comparison[comparison['category'] == 'v8_advantage']
if len(v8_adv) > 0 and 'v8_kinetics_source' in v8_adv.columns:
    tier_counts = v8_adv['v8_kinetics_source'].value_counts()
    colors_tier = ['steelblue', 'skyblue', 'lightblue', 'yellow', 'red', 'gray']
    ax3.pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%',
            colors=colors_tier[:len(tier_counts)], startangle=90)
    ax3.set_title(f'v8 Advantage by Kinetics Tier\n({len(v8_adv):,} neurons)',
                  fontsize=12, fontweight='bold')
else:
    ax3.text(0.5, 0.5, 'No v8 advantage cases', ha='center', va='center')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

# Plot 4: Ground truth distribution by category
ax4 = axes[1, 0]
gt_by_cat = []
for cat in ['both_correct', 'both_wrong', 'v8_advantage', 'v7_advantage']:
    subset = comparison[comparison['category'] == cat]
    if len(subset) > 0:
        keep_pct = (subset['ground_truth'] == 1).mean() * 100
        gt_by_cat.append({'category': cat.replace('_', '\n'), 'keep_pct': keep_pct})

if gt_by_cat:
    gt_df = pd.DataFrame(gt_by_cat)
    colors_gt = ['green', 'red', 'blue', 'orange']
    ax4.bar(range(len(gt_df)), gt_df['keep_pct'], color=colors_gt[:len(gt_df)], alpha=0.7)
    ax4.axhline(80.8, color='black', linestyle='--', alpha=0.5, label='Overall (80.8%)')
    ax4.set_ylabel('% Ground Truth KEEP', fontsize=11)
    ax4.set_title('Ground Truth Distribution by Category', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(gt_df)))
    ax4.set_xticklabels(gt_df['category'], fontsize=9)
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)

# Plot 5: Probability scatter for mismatches
ax5 = axes[1, 1]
v8_adv_subset = comparison[comparison['category'] == 'v8_advantage'].sample(min(1000, len(v8_adv)))
v7_adv_subset = comparison[comparison['category'] == 'v7_advantage'].sample(min(1000, len(comparison[comparison['category'] == 'v7_advantage'])))

if len(v8_adv_subset) > 0:
    ax5.scatter(v8_adv_subset['v7_proba'], v8_adv_subset['v8_proba'],
                c='blue', alpha=0.3, s=20, label='v8 advantage')
if len(v7_adv_subset) > 0:
    ax5.scatter(v7_adv_subset['v7_proba'], v7_adv_subset['v8_proba'],
                c='orange', alpha=0.3, s=20, label='v7 advantage')

ax5.axhline(0.75, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax5.axvline(0.75, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)
ax5.set_xlabel('v7 Probability', fontsize=11)
ax5.set_ylabel('v8 Probability', fontsize=11)
ax5.set_title('Probability Comparison (Mismatches)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)

# Plot 6: Confusion matrix comparison
ax6 = axes[1, 2]
conf_data = []
for cat in ['both_correct', 'both_wrong', 'v8_advantage', 'v7_advantage']:
    count = category_counts.get(cat, 0)
    pct = count / len(comparison) * 100
    conf_data.append([cat.replace('_', ' '), count, f'{pct:.2f}%'])

table = ax6.table(cellText=conf_data, colLabels=['Category', 'Count', '% Total'],
                  cellLoc='left', loc='center',
                  colWidths=[0.5, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color code rows
for i in range(1, 5):
    if i == 1:  # both_correct
        table[(i, 0)].set_facecolor('#90EE90')
    elif i == 2:  # both_wrong
        table[(i, 0)].set_facecolor('#FFB6C6')
    elif i == 3:  # v8_advantage
        table[(i, 0)].set_facecolor('#87CEEB')
    elif i == 4:  # v7_advantage
        table[(i, 0)].set_facecolor('#FFD700')

ax6.axis('off')
ax6.set_title('Prediction Summary', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('ml/results/v7_v8_mismatch_analysis.png', dpi=150, bbox_inches='tight')
print(f'\n{"="*80}')
print(f'Plot saved to: ml/results/v7_v8_mismatch_analysis.png')

# Save detailed mismatch data
mismatch_summary = comparison[comparison['category'].isin(['v8_advantage', 'v7_advantage', 'both_wrong'])]
mismatch_summary.to_csv('ml/results/v7_v8_mismatches.csv', index=False)
print(f'Mismatch details saved to: ml/results/v7_v8_mismatches.csv')

print(f'\n{"="*80}')
print('SUMMARY')
print('='*80)

print(f'\nPrediction agreement:')
print(f'  Both correct: {category_counts.get("both_correct", 0):,} ({category_counts.get("both_correct", 0)/len(comparison)*100:.1f}%)')
print(f'  Disagree: {category_counts.get("v8_advantage", 0) + category_counts.get("v7_advantage", 0):,} ({(category_counts.get("v8_advantage", 0) + category_counts.get("v7_advantage", 0))/len(comparison)*100:.2f}%)')
print(f'  Both wrong: {category_counts.get("both_wrong", 0):,} ({category_counts.get("both_wrong", 0)/len(comparison)*100:.1f}%)')

print(f'\nNet advantage:')
print(f'  v8 fixes {category_counts.get("v8_advantage", 0):,} cases v7 gets wrong')
print(f'  v7 fixes {category_counts.get("v7_advantage", 0):,} cases v8 gets wrong')
print(f'  Net v8 advantage: {v8_net_advantage:+,} neurons ({v8_net_advantage/len(comparison)*100:+.3f}%)')

if abs(v8_net_advantage) < 100:
    print(f'\nCONCLUSION: Models have VIRTUALLY IDENTICAL error patterns')
    print(f'  Disagreement on only {(category_counts.get("v8_advantage", 0) + category_counts.get("v7_advantage", 0))/len(comparison)*100:.2f}% of neurons')
    print(f'  Net difference is negligible ({abs(v8_net_advantage)} neurons)')
else:
    winner = 'v8' if v8_net_advantage > 0 else 'v7'
    print(f'\nCONCLUSION: {winner.upper()} has slight advantage on edge cases')
