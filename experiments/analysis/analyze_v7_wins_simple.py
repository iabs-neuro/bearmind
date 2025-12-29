"""
Simplified analysis of where v7 beats v8.
Focus on actionable insights for improving v8.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('='*80)
print('WHERE V7 BEATS V8: Actionable Insights')
print('='*80)

# Load data
v7 = pd.read_csv('ml/results/training_dataset_v7.csv')
v8 = pd.read_csv('ml/results/training_dataset_v8.csv')

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

# Add to datasets
v7['v7_pred'] = v7_pred
v7['v7_proba'] = v7_proba
v7['v7_correct'] = (v7_pred == y_v7)

v8['v8_pred'] = v8_pred
v8['v8_proba'] = v8_proba
v8['v8_correct'] = (v8_pred == y_v8)

# Match neurons
v7['neuron_id'] = v7['session'] + '_' + v7['component_idx'].astype(str)
v8['neuron_id'] = v8['session'] + '_' + v8['component_idx'].astype(str)

# Find v7 advantage cases
v7_wins = v7[v7['v7_correct']].copy()
v8_results = v8[['neuron_id', 'v8_pred', 'v8_proba', 'v8_correct']].copy()

merged = v7_wins.merge(v8_results, on='neuron_id', how='inner')
v7_advantage = merged[~merged['v8_correct']].copy()

print(f'\nv7 advantage neurons: {len(v7_advantage):,} ({len(v7_advantage)/len(v7)*100:.2f}%)')

# Error types
v8_fp = (v7_advantage['v8_pred'] == 1) & (v7_advantage['ground_truth'] == 0)
v8_fn = (v7_advantage['v8_pred'] == 0) & (v7_advantage['ground_truth'] == 1)

print(f'\nv8 error breakdown:')
print(f'  False Positives (wrongly KEEP bad): {v8_fp.sum():,} ({v8_fp.mean()*100:.1f}%)')
print(f'  False Negatives (wrongly DELETE good): {v8_fn.sum():,} ({v8_fn.mean()*100:.1f}%)')

# Event detection
print(f'\n{"="*80}')
print('EVENT DETECTION')
print('='*80)

v7_adv_has_evt = (v7_advantage['t_off'] > -1).sum()
print(f'\nv7 event detection on these neurons: {v7_adv_has_evt:,}/{len(v7_advantage):,} ({v7_adv_has_evt/len(v7_advantage)*100:.1f}%)')

# Kinetics tier (v8)
if 'kinetics_source' in v8.columns:
    v8_kin = v8[['neuron_id', 'kinetics_source', 't_off']].copy()
    v8_kin.rename(columns={'t_off': 't_off_v8'}, inplace=True)
    v7_adv_with_kin = v7_advantage.merge(v8_kin, on='neuron_id', how='left')

    print(f'\nv8 kinetics tiers on v7 advantage neurons:')
    if 'kinetics_source' in v7_adv_with_kin.columns:
        kin_dist = v7_adv_with_kin['kinetics_source'].value_counts()
        for tier, count in kin_dist.items():
            print(f'  {tier:<25} {count:,} ({count/len(v7_adv_with_kin)*100:.1f}%)')

        if 't_off_v8' in v7_adv_with_kin.columns:
            v8_evt = (v7_adv_with_kin['t_off_v8'] > -1).sum()
            print(f'\nv8 event detection on these neurons: {v8_evt:,}/{len(v7_adv_with_kin):,} ({v8_evt/len(v7_adv_with_kin)*100:.1f}%)')

# Probability analysis
print(f'\n{"="*80}')
print('PROBABILITY ANALYSIS')
print('='*80)

v7_adv_fp = v7_advantage[v8_fp]
v7_adv_fn = v7_advantage[v8_fn]

print(f'\nFalse Positives (v8 wrongly KEEP):')
print(f'  v7 proba: {v7_adv_fp["v7_proba"].mean():.3f} (correct: predicted DELETE)')
print(f'  v8 proba: {v7_adv_fp["v8_proba"].mean():.3f} (wrong: predicted KEEP)')
print(f'  v8 shift: {(v7_adv_fp["v8_proba"] - v7_adv_fp["v7_proba"]).mean():+.3f} (TOO HIGH)')

print(f'\nFalse Negatives (v8 wrongly DELETE):')
print(f'  v7 proba: {v7_adv_fn["v7_proba"].mean():.3f} (correct: predicted KEEP)')
print(f'  v8 proba: {v7_adv_fn["v8_proba"].mean():.3f} (wrong: predicted DELETE)')
print(f'  v8 shift: {(v7_adv_fn["v8_proba"] - v7_adv_fn["v7_proba"]).mean():+.3f} (TOO LOW)')

# New features in v8
print(f'\n{"="*80}')
print('NEW v8 FEATURES')
print('='*80)

new_v8_features = [f for f in v8_features if f not in v7_features]
print(f'\nNew features in v8: {new_v8_features}')

if len(new_v8_features) > 0 and 'kinetics_source' in v7_adv_with_kin.columns:
    v7_adv_full = v7_adv_with_kin.merge(v8[['neuron_id'] + new_v8_features], on='neuron_id', how='left')

    for feat in new_v8_features:
        if feat in v7_adv_full.columns:
            vals = v7_adv_full[feat].dropna()
            overall = v8[feat].dropna()
            if len(vals) > 0 and len(overall) > 0:
                print(f'\n{feat}:')
                print(f'  v7 advantage mean: {vals.mean():.4f}')
                print(f'  Overall v8 mean: {overall.mean():.4f}')
                print(f'  Difference: {vals.mean() - overall.mean():+.4f}')

# Sessions
print(f'\n{"="*80}')
print('PROBLEMATIC SESSIONS FOR v8')
print('='*80)

session_counts = v7_advantage.groupby('session').size().sort_values(ascending=False).head(10)
print(f'\nTop 10 sessions where v8 loses to v7:')
for session, count in session_counts.items():
    total = len(v7[v7['session'] == session])
    print(f'  {session:<20} {count:,}/{total:,} ({count/total*100:.1f}%)')

# Key insights
print(f'\n{"="*80}')
print('KEY INSIGHTS FOR IMPROVING v8')
print('='*80)

print(f'\n1. PRIMARY ISSUE: v8 makes MORE FALSE NEGATIVES')
print(f'   - Deletes {v8_fn.sum():,} good neurons that v7 correctly keeps')
print(f'   - v8 probability TOO LOW by {abs((v7_adv_fn["v8_proba"] - v7_adv_fn["v7_proba"]).mean()):.3f}')
print(f'   - This is the biggest improvement opportunity')

print(f'\n2. SECONDARY ISSUE: v8 makes some FALSE POSITIVES')
print(f'   - Keeps {v8_fp.sum():,} bad neurons that v7 correctly deletes')
print(f'   - v8 probability TOO HIGH by {(v7_adv_fp["v8_proba"] - v7_adv_fp["v7_proba"]).mean():.3f}')

print(f'\n3. EVENT DETECTION PARADOX')
if 'kinetics_source' in v7_adv_with_kin.columns:
    print(f'   - v8 detects events on {v8_evt/len(v7_adv_with_kin)*100:.1f}% vs v7 on {v7_adv_has_evt/len(v7_advantage)*100:.1f}%')
    print(f'   - Better events do NOT help prediction')
    print(f'   - Model may be over-weighting unreliable event features')

print(f'\n4. HYPERPARAMETER DIFFERENCE')
print(f'   - v7: max_bins=256, max_leaves=5')
print(f'   - v8: max_bins=1024, max_leaves=3')
print(f'   - v8 has MORE bins but FEWER leaves')
print(f'   - Reducing flexibility may hurt performance')

print(f'\n{"="*80}')
print('RECOMMENDATIONS')
print('='*80)

print(f'\n1. INCREASE max_leaves from 3 to 5')
print(f'   - Match v7 tree flexibility')
print(f'   - Allow model to capture more complex patterns')

print(f'\n2. RECALIBRATE threshold')
print(f'   - Current 0.75 may not be optimal for v8')
print(f'   - v8 probabilities shifted vs v7')
print(f'   - Try threshold 0.72-0.73 to reduce false negatives')

print(f'\n3. FEATURE ENGINEERING')
print(f'   - Blend v7 and v8 event features')
print(f'   - Down-weight unreliable event metrics')
print(f'   - Add interaction terms')

print(f'\n4. ENSEMBLE v7 + v8')
print(f'   - Average probabilities')
print(f'   - Use v7 when v8 uncertain')
print(f'   - Capture strengths of both')

# Save
v7_advantage.to_csv('ml/results/v7_wins_detailed.csv', index=False)
print(f'\nDetailed data saved to: ml/results/v7_wins_detailed.csv')

# Simple visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Error types
ax1 = axes[0]
error_types = ['False Positive\n(wrongly KEEP)', 'False Negative\n(wrongly DELETE)']
error_counts = [v8_fp.sum(), v8_fn.sum()]
colors = ['#FF6B6B', '#4ECDC4']
ax1.bar(error_types, error_counts, color=colors, alpha=0.7)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('v8 Errors Where v7 Succeeds', fontsize=12, fontweight='bold')
for i, (typ, count) in enumerate(zip(error_types, error_counts)):
    ax1.text(i, count + 20, f'{count:,}\n({count/len(v7_advantage)*100:.1f}%)',
             ha='center', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Probability shift
ax2 = axes[1]
ax2.scatter(v7_advantage['v7_proba'], v7_advantage['v8_proba'],
            c=v7_advantage['ground_truth'], cmap='RdYlGn', alpha=0.3, s=20)
ax2.axhline(0.75, color='green', linestyle='--', alpha=0.5)
ax2.axvline(0.75, color='green', linestyle='--', alpha=0.5)
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax2.set_xlabel('v7 Probability (CORRECT)', fontsize=11)
ax2.set_ylabel('v8 Probability (WRONG)', fontsize=11)
ax2.set_title('Probability Comparison', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Probability distribution
ax3 = axes[2]
ax3.hist([v7_adv_fp['v8_proba'], v7_adv_fn['v8_proba']],
         bins=20, label=['False Positive', 'False Negative'],
         color=colors, alpha=0.7, stacked=False)
ax3.axvline(0.75, color='green', linestyle='--', linewidth=2, label='Threshold')
ax3.set_xlabel('v8 Probability', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('v8 Probability Distribution', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('ml/results/v7_wins_analysis.png', dpi=150, bbox_inches='tight')
print(f'Plot saved to: ml/results/v7_wins_analysis.png')

print(f'\n{"="*80}')
print(f'NEXT STEP: Retrain v8 with max_leaves=5 and see if performance improves')
print('='*80)
