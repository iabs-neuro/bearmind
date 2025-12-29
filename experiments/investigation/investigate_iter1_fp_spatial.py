"""
Investigate spatial relationships of FAKE FP cases from iter1.
Model says KEEP (potentially correct), GT says DELETE (potentially wrong).
"""
import pandas as pd
import numpy as np

print('='*80)
print('INVESTIGATING ITER1 FP SPATIAL RELATIONSHIPS')
print('='*80)

# Load datasets
df = pd.read_csv('ml/results/training_dataset_v8_corrected.csv')
df_fp = pd.read_csv('ml/results/v8_corrected_iter1_top100_fp.csv')

# Real FP (model wrong to KEEP): 41, 42, 3, 13, 33, 45, 47, 50, 52, 91
real_fp_indices = [41, 42, 3, 13, 33, 45, 47, 50, 52, 91]

# Fake FP (model potentially correct to KEEP): all others
fake_fp_indices = [i for i in range(1, 101) if i not in real_fp_indices]

print(f'\nAnalyzing {len(fake_fp_indices)} potential FAKE FP cases')
print('='*80)

# Summary statistics
merge_count = 0
proximity_count = 0
standalone_count = 0
merge_cases = []
proximity_cases = []
standalone_cases = []

for fp_idx in fake_fp_indices:
    fp_row = df_fp.iloc[fp_idx - 1]
    session = fp_row['session']
    comp_idx = int(fp_row['component_idx'])
    prob = fp_row['y_proba']

    # Find in full dataset
    neuron_mask = (df['session'] == session) & (df['component_idx'] == comp_idx)
    if neuron_mask.sum() == 0:
        continue

    neuron = df[neuron_mask].iloc[0]
    session_neurons = df[df['session'] == session].copy()

    # Parse center
    if isinstance(neuron['center'], str):
        center = np.fromstring(neuron['center'].strip('[]'), sep=' ')
    else:
        center = neuron['center']

    # Check for nearby KEEP neurons
    nearby_keep = []
    for idx, other in session_neurons.iterrows():
        if other['component_idx'] == comp_idx:
            continue
        if other['ground_truth'] != 1:
            continue

        if isinstance(other['center'], str):
            other_center = np.fromstring(other['center'].strip('[]'), sep=' ')
        else:
            other_center = other['center']

        distance = np.linalg.norm(center - other_center)

        if distance <= 10:
            nearby_keep.append({
                'component_idx': int(other['component_idx']),
                'distance': distance,
                'area': other.get('area', np.nan),
                'snr': other.get('caiman_snr', np.nan)
            })

    nearby_keep.sort(key=lambda x: x['distance'])

    nn_distance = neuron.get('nn_distance_center', np.nan)

    # Categorize
    has_close_keep = any(n['distance'] < 5 for n in nearby_keep)

    case_info = {
        'fp_idx': fp_idx,
        'session': session,
        'comp_idx': comp_idx,
        'prob': prob,
        'nn_distance': nn_distance,
        'area': neuron.get('area', np.nan),
        'snr': neuron.get('caiman_snr', np.nan),
        'r2': neuron.get('r2_score', np.nan),
        'nearby_keep': nearby_keep[:3]
    }

    if has_close_keep:
        merge_count += 1
        merge_cases.append(case_info)
    elif nn_distance < 8:
        proximity_count += 1
        proximity_cases.append(case_info)
    else:
        standalone_count += 1
        standalone_cases.append(case_info)

# Print summary
print(f'\n{"="*80}')
print('SUMMARY')
print('='*80)

print(f'\nTotal analyzed: {len(fake_fp_indices)} neurons')
print(f'\n  MERGE (< 5px to KEEP): {merge_count} ({merge_count/len(fake_fp_indices)*100:.1f}%)')
print(f'  PROXIMITY (5-8px): {proximity_count} ({proximity_count/len(fake_fp_indices)*100:.1f}%)')
print(f'  STANDALONE (> 8px): {standalone_count} ({standalone_count/len(fake_fp_indices)*100:.1f}%)')

print(f'\nRecommendation:')
print(f'  - MERGE: {merge_count} neurons - likely overlapping, safe to flip DELETE -> KEEP')
print(f'  - PROXIMITY: {proximity_count} neurons - review quality, likely safe to flip')
print(f'  - STANDALONE: {standalone_count} neurons - definitely safe to flip DELETE -> KEEP')

# Save detailed list
output = []
for case in merge_cases + proximity_cases + standalone_cases:
    output.append({
        'fp_idx': case['fp_idx'],
        'session': case['session'],
        'component_idx': case['comp_idx'],
        'probability': case['prob'],
        'nn_distance': case['nn_distance'],
        'snr': case['snr'],
        'r2': case['r2'],
        'category': 'MERGE' if case in merge_cases else ('PROXIMITY' if case in proximity_cases else 'STANDALONE'),
        'recommendation': 'Flip DELETE -> KEEP'
    })

output_df = pd.DataFrame(output)
output_df.to_csv('ml/results/iter1_fake_fp_spatial_analysis.csv', index=False)
print(f'\nDetailed analysis saved to: ml/results/iter1_fake_fp_spatial_analysis.csv')

# Generate corrections list for easy copy-paste
print(f'\n{"="*80}')
print('CORRECTIONS TO APPLY FOR ITER2')
print('='*80)

print(f'\n# FAKE FP ({len(fake_fp_indices)} corrections - DELETE -> KEEP):')
for case in output[:20]:  # Show first 20
    print(f"    ('{case['session']}', {case['component_idx']}, 1, 'FAKE FP #{case['fp_idx']} - {case['category']}, model correct KEEP'),")

if len(output) > 20:
    print(f'    # ... and {len(output) - 20} more')

print(f'\n{"="*80}')
