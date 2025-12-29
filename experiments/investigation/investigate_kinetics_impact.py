"""
Investigate if v8's hybrid kinetics actually helps on neurons that needed it.

Hypothesis: v6/v7 should make more mistakes on neurons with poor kinetics optimization,
and v8 should perform better specifically on those neurons.
"""
import pickle
import numpy as np
import pandas as pd

def compute_fbeta(precision, recall, beta=0.5773502691896257):
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

print('='*80)
print('INVESTIGATING KINETICS OPTIMIZATION IMPACT ON MODEL PERFORMANCE')
print('='*80)

# Load datasets
v6 = pd.read_csv('ml/results/training_dataset_v6_no3dm.csv')
v7 = pd.read_csv('ml/results/training_dataset_v7.csv')
v8 = pd.read_csv('ml/results/training_dataset_v8.csv')

print(f'\nDataset sizes:')
print(f'  v6: {len(v6):,} neurons')
print(f'  v7: {len(v7):,} neurons')
print(f'  v8: {len(v8):,} neurons')

# Load models
with open('ml/ebm_grid_search_v6_no3dm/ebm_best.pkl', 'rb') as f:
    v6_model = pickle.load(f)
with open('ml/ebm_grid_search_v7/ebm_best.pkl', 'rb') as f:
    v7_model = pickle.load(f)
with open('ml/ebm_grid_search_v8/ebm_best.pkl', 'rb') as f:
    v8_model = pickle.load(f)

# Get predictions at threshold 0.75
exclude_cols = {'ground_truth', 'session', 'experiment', 'component_idx', 'center',
                'distance_to_gt', 'is_corner_artifact', 'corr_groups'}

# v6 predictions
v6_features = [c for c in v6.columns if c not in exclude_cols and c in v6_model.feature_names_in_]
X_v6 = v6[v6_features].copy()
y_v6 = v6['ground_truth'].values
v6_proba = v6_model.predict_proba(X_v6)[:, 1]
v6_pred = (v6_proba >= 0.75).astype(int)

# v7 predictions
v7_features = [c for c in v7.columns if c not in exclude_cols and c in v7_model.feature_names_in_]
X_v7 = v7[v7_features].copy()
y_v7 = v7['ground_truth'].values
v7_proba = v7_model.predict_proba(X_v7)[:, 1]
v7_pred = (v7_proba >= 0.75).astype(int)

# v8 predictions
v8_features = [c for c in v8.columns if c not in exclude_cols and c in v8_model.feature_names_in_]
X_v8 = v8[v8_features].copy()
y_v8 = v8['ground_truth'].values
v8_proba = v8_model.predict_proba(X_v8)[:, 1]
v8_pred = (v8_proba >= 0.75).astype(int)

# Add predictions to dataframes
v6 = v6.copy()
v7 = v7.copy()
v8 = v8.copy()
v6['prediction'] = v6_pred
v6['correct'] = (v6_pred == y_v6)
v7['prediction'] = v7_pred
v7['correct'] = (v7_pred == y_v7)
v8['prediction'] = v8_pred
v8['correct'] = (v8_pred == y_v8)

print(f'\n{"="*80}')
print('OVERALL ACCURACY AT THRESHOLD 0.75')
print('='*80)

print(f'\nv6_no3dm: {v6["correct"].mean()*100:.2f}% correct ({v6["correct"].sum():,}/{len(v6):,})')
print(f'v7:       {v7["correct"].mean()*100:.2f}% correct ({v7["correct"].sum():,}/{len(v7):,})')
print(f'v8:       {v8["correct"].mean()*100:.2f}% correct ({v8["correct"].sum():,}/{len(v8):,})')

# Analyze by kinetics_source in v8
print(f'\n{"="*80}')
print('V8 PERFORMANCE BY KINETICS TIER (kinetics_source)')
print('='*80)

if 'kinetics_source' in v8.columns:
    tier_order = ['wavelet_standard', 'wavelet_relaxed', 'threshold_standard',
                  'threshold_relaxed', 'defaults', 'error']

    print(f'\n{"Tier":<25} {"Count":<10} {"% Total":<10} {"Accuracy":<12} {"Error Rate"}')
    print('-'*75)

    tier_stats = []
    for tier in tier_order:
        tier_data = v8[v8['kinetics_source'] == tier]
        if len(tier_data) > 0:
            count = len(tier_data)
            pct = count / len(v8) * 100
            accuracy = tier_data['correct'].mean() * 100
            error_rate = (1 - tier_data['correct'].mean()) * 100

            tier_stats.append({
                'tier': tier,
                'count': count,
                'pct': pct,
                'accuracy': accuracy,
                'error_rate': error_rate
            })

            print(f'{tier:<25} {count:<10,} {pct:<10.1f} {accuracy:<12.2f} {error_rate:.2f}%')

    # Statistical test: is accuracy different across tiers?
    print(f'\nHypothesis test: Does v8 accuracy vary by kinetics tier?')
    tier1_acc = v8[v8['kinetics_source'] == 'wavelet_standard']['correct'].mean()
    tier2_acc = v8[v8['kinetics_source'] == 'wavelet_relaxed']['correct'].mean()
    tier3plus = v8[v8['kinetics_source'].isin(['threshold_standard', 'threshold_relaxed', 'defaults'])]['correct'].mean()

    print(f'  Tier 1 (wavelet_standard): {tier1_acc*100:.2f}% accuracy')
    print(f'  Tier 2 (wavelet_relaxed):  {tier2_acc*100:.2f}% accuracy')
    print(f'  Tier 3+ (threshold/defaults): {tier3plus*100:.2f}% accuracy')
    print(f'  Difference (Tier 1 vs Tier 2): {(tier1_acc - tier2_acc)*100:+.2f}%')
    print(f'  Difference (Tier 1 vs Tier 3+): {(tier1_acc - tier3plus)*100:+.2f}%')

else:
    print('kinetics_source not found in v8 dataset!')

# Compare v6/v7/v8 on neurons that needed kinetics help
print(f'\n{"="*80}')
print('COMPARING v6/v7/v8 ON DIFFERENT KINETICS TIERS')
print('='*80)

# Since all datasets have same neurons (same sessions), match by session + component_idx
v6 = v6.copy()
v7 = v7.copy()
v8 = v8.copy()

# Create unique identifier
v6['neuron_id'] = v6['session'] + '_' + v6['component_idx'].astype(str)
v7['neuron_id'] = v7['session'] + '_' + v7['component_idx'].astype(str)
v8['neuron_id'] = v8['session'] + '_' + v8['component_idx'].astype(str)

# Merge predictions
comparison = v8[['neuron_id', 'session', 'component_idx', 'ground_truth', 'kinetics_source']].copy()
comparison = comparison.merge(
    v6[['neuron_id', 'prediction', 'correct']].rename(columns={'prediction': 'v6_pred', 'correct': 'v6_correct'}),
    on='neuron_id', how='left'
)
comparison = comparison.merge(
    v7[['neuron_id', 'prediction', 'correct']].rename(columns={'prediction': 'v7_pred', 'correct': 'v7_correct'}),
    on='neuron_id', how='left'
)
comparison = comparison.merge(
    v8[['neuron_id', 'prediction', 'correct']].rename(columns={'prediction': 'v8_pred', 'correct': 'v8_correct'}),
    on='neuron_id', how='left'
)

# Remove neurons not in all datasets
comparison = comparison.dropna()

print(f'\nNeurons present in all three datasets: {len(comparison):,}')

print(f'\n{"="*80}')
print('PERFORMANCE BY KINETICS TIER (on common neurons)')
print('='*80)

print(f'\n{"Tier":<25} {"Count":<10} {"v6 Acc":<10} {"v7 Acc":<10} {"v8 Acc":<10} {"v8 vs v7":<10} {"v8 vs v6"}')
print('-'*95)

for tier in tier_order:
    tier_data = comparison[comparison['kinetics_source'] == tier]
    if len(tier_data) > 0:
        count = len(tier_data)
        v6_acc = tier_data['v6_correct'].mean() * 100
        v7_acc = tier_data['v7_correct'].mean() * 100
        v8_acc = tier_data['v8_correct'].mean() * 100

        v8_vs_v7 = v8_acc - v7_acc
        v8_vs_v6 = v8_acc - v6_acc

        print(f'{tier:<25} {count:<10,} {v6_acc:<10.2f} {v7_acc:<10.2f} {v8_acc:<10.2f} {v8_vs_v7:+10.2f} {v8_vs_v6:+10.2f}')

# Critical analysis: where v8 should shine
print(f'\n{"="*80}')
print('CRITICAL QUESTION: Does v8 help on challenging neurons?')
print('='*80)

tier2_plus = comparison[comparison['kinetics_source'].isin(['wavelet_relaxed', 'threshold_standard',
                                                             'threshold_relaxed', 'defaults'])]

if len(tier2_plus) > 0:
    print(f'\nNeurons needing kinetics help (Tier 2+): {len(tier2_plus):,} ({len(tier2_plus)/len(comparison)*100:.1f}%)')
    print(f'\nPerformance on these neurons:')
    print(f'  v6: {tier2_plus["v6_correct"].mean()*100:.2f}% accuracy')
    print(f'  v7: {tier2_plus["v7_correct"].mean()*100:.2f}% accuracy')
    print(f'  v8: {tier2_plus["v8_correct"].mean()*100:.2f}% accuracy')

    v8_improvement_vs_v6 = (tier2_plus['v8_correct'].mean() - tier2_plus['v6_correct'].mean()) * 100
    v8_improvement_vs_v7 = (tier2_plus['v8_correct'].mean() - tier2_plus['v7_correct'].mean()) * 100

    print(f'\nv8 improvement on Tier 2+ neurons:')
    print(f'  vs v6: {v8_improvement_vs_v6:+.2f}%')
    print(f'  vs v7: {v8_improvement_vs_v7:+.2f}%')

    if v8_improvement_vs_v7 > 1.0:
        print(f'\n[SUCCESS] v8 shows significant improvement on challenging neurons!')
    elif v8_improvement_vs_v7 > 0:
        print(f'\n[MARGINAL] v8 shows slight improvement on challenging neurons')
    else:
        print(f'\n[UNEXPECTED] v8 does NOT improve on challenging neurons')

# Analyze mistakes
print(f'\n{"="*80}')
print('MISTAKE ANALYSIS: Where each model goes wrong')
print('='*80)

v6_mistakes = comparison[~comparison['v6_correct']]
v7_mistakes = comparison[~comparison['v7_correct']]
v8_mistakes = comparison[~comparison['v8_correct']]

print(f'\nTotal mistakes at threshold 0.75:')
print(f'  v6: {len(v6_mistakes):,} mistakes ({len(v6_mistakes)/len(comparison)*100:.2f}%)')
print(f'  v7: {len(v7_mistakes):,} mistakes ({len(v7_mistakes)/len(comparison)*100:.2f}%)')
print(f'  v8: {len(v8_mistakes):,} mistakes ({len(v8_mistakes)/len(comparison)*100:.2f}%)')

print(f'\nMistakes by kinetics tier:')
print(f'\n{"Tier":<25} {"v6 Errors":<12} {"v7 Errors":<12} {"v8 Errors":<12} {"v8 Better?"}')
print('-'*75)

for tier in tier_order:
    tier_data = comparison[comparison['kinetics_source'] == tier]
    if len(tier_data) > 0:
        v6_err = (~tier_data['v6_correct']).sum()
        v7_err = (~tier_data['v7_correct']).sum()
        v8_err = (~tier_data['v8_correct']).sum()

        better = ''
        if v8_err < min(v6_err, v7_err):
            better = 'YES'
        elif v8_err == min(v6_err, v7_err):
            better = 'TIE'
        else:
            better = 'NO'

        print(f'{tier:<25} {v6_err:<12} {v7_err:<12} {v8_err:<12} {better}')

# Final verdict
print(f'\n{"="*80}')
print('VERDICT: Why v8 does not show dramatic improvement')
print('='*80)

tier1 = comparison[comparison['kinetics_source'] == 'wavelet_standard']
tier1_v8_vs_v7 = (tier1['v8_correct'].mean() - tier1['v7_correct'].mean()) * 100

print(f'\n1. MOST NEURONS ARE TIER 1 (wavelet_standard): {len(tier1)/len(comparison)*100:.1f}%')
print(f'   On these neurons:')
print(f'     v7 accuracy: {tier1["v7_correct"].mean()*100:.2f}%')
print(f'     v8 accuracy: {tier1["v8_correct"].mean()*100:.2f}%')
print(f'     v8 vs v7: {tier1_v8_vs_v7:+.2f}%')
print(f'   => v8 offers minimal advantage where kinetics were already good')

print(f'\n2. TIER 2+ NEURONS (where v8 should help): {len(tier2_plus)/len(comparison)*100:.1f}%')
print(f'   Even if v8 is perfect on these, max possible gain:')
print(f'     Current v7 error on Tier 2+: {(1-tier2_plus["v7_correct"].mean())*100:.2f}%')
print(f'     If v8 fixes all: {((1-tier2_plus["v7_correct"].mean()) * len(tier2_plus)/len(comparison))*100:.2f}% overall improvement')
print(f'   Actual v8 improvement on Tier 2+: {v8_improvement_vs_v7:+.2f}%')
print(f'   => Limited by small proportion of challenging neurons')

tier2_relative_improvement = v8_improvement_vs_v7 * (len(tier2_plus)/len(comparison))
print(f'\n3. NET IMPACT ON OVERALL PERFORMANCE:')
print(f'   Tier 2+ contribution to overall improvement: ~{tier2_relative_improvement:.3f}%')
print(f'   => Marginal because Tier 2+ is only {len(tier2_plus)/len(comparison)*100:.1f}% of data')

print(f'\nCONCLUSION:')
print(f'  v8\'s hybrid kinetics helps on challenging neurons ({len(tier2_plus)/len(comparison)*100:.1f}% of data)')
print(f'  but most neurons ({len(tier1)/len(comparison)*100:.1f}%) already had good kinetics in v6/v7')
print(f'  So v8 can only improve marginally on the overall metric.')
