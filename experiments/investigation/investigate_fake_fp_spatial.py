"""
Investigate spatial relationships of FAKE FP cases.
These are neurons where model says KEEP (correct) but GT says DELETE (wrong).
Check if they were mislabeled or merged with nearby neurons.
"""
import pandas as pd
import numpy as np

print('='*80)
print('INVESTIGATING FAKE FP SPATIAL RELATIONSHIPS')
print('Model says KEEP (correct), GT says DELETE (wrong)')
print('='*80)

# Load datasets
df = pd.read_csv('ml/results/training_dataset_v8.csv')
df_fp = pd.read_csv('ml/results/v8_top100_false_positives.csv')

# Real FP (model wrong): 1,6,33,35,38,40,47,50,59,61,97,100
real_fp_indices = [1, 6, 33, 35, 38, 40, 47, 50, 59, 61, 97, 100]

# Fake FP (model correct, GT wrong): all others
fake_fp_indices = [i for i in range(1, 101) if i not in real_fp_indices]

print(f'\nAnalyzing {len(fake_fp_indices)} FAKE FP cases (model correct to KEEP)')
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

    # Check for nearby KEEP neurons (potential merge targets)
    nearby_keep = []
    for idx, other in session_neurons.iterrows():
        if other['component_idx'] == comp_idx:
            continue
        if other['ground_truth'] != 1:  # Only check KEEP neurons
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
        'nearby_keep': nearby_keep[:3]  # Top 3 closest KEEP neurons
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

# Print detailed results
print(f'\n{"="*80}')
print(f'MERGE CANDIDATES (< 5px to KEEP neuron): {merge_count}')
print('='*80)

for case in merge_cases[:10]:  # Show first 10
    print(f'\nFP #{case["fp_idx"]}: {case["session"]} component {case["comp_idx"]}')
    print(f'  Prob: {case["prob"]:.3f}, SNR: {case["snr"]:.2f}, R2: {case["r2"]:.3f}')
    print(f'  Nearby KEEP neurons:')
    for n in case['nearby_keep']:
        print(f'    Comp {n["component_idx"]}: {n["distance"]:.2f}px, SNR={n["snr"]:.2f}')
    print(f'  LIKELY: Merged/overlapping with good neuron - GT mislabeled as DELETE')

if len(merge_cases) > 10:
    print(f'\n... and {len(merge_cases) - 10} more merge cases')

print(f'\n{"="*80}')
print(f'PROXIMITY CANDIDATES (5-8px to nearest): {proximity_count}')
print('='*80)

for case in proximity_cases[:10]:  # Show first 10
    print(f'\nFP #{case["fp_idx"]}: {case["session"]} component {case["comp_idx"]}')
    print(f'  Prob: {case["prob"]:.3f}, SNR: {case["snr"]:.2f}, NN: {case["nn_distance"]:.2f}px')
    if case['nearby_keep']:
        print(f'  Nearby KEEP: Comp {case["nearby_keep"][0]["component_idx"]} at {case["nearby_keep"][0]["distance"]:.2f}px')

if len(proximity_cases) > 10:
    print(f'\n... and {len(proximity_cases) - 10} more proximity cases')

print(f'\n{"="*80}')
print(f'STANDALONE (> 8px): {standalone_count}')
print('='*80)

for case in standalone_cases[:10]:  # Show first 10
    print(f'\nFP #{case["fp_idx"]}: {case["session"]} component {case["comp_idx"]}')
    print(f'  Prob: {case["prob"]:.3f}, SNR: {case["snr"]:.2f}, NN: {case["nn_distance"]:.2f}px')
    print(f'  LIKELY: True good neuron, GT incorrectly labeled DELETE')

if len(standalone_cases) > 10:
    print(f'\n... and {len(standalone_cases) - 10} more standalone cases')

# Overall summary
print(f'\n{"="*80}')
print('SUMMARY: FAKE FP Spatial Analysis')
print('='*80)

print(f'\nTotal analyzed: {len(fake_fp_indices)} neurons (model correct to KEEP, GT wrong)')
print(f'\n  MERGE (< 5px to KEEP): {merge_count} ({merge_count/len(fake_fp_indices)*100:.1f}%)')
print(f'    - Likely merged/overlapping with good neurons')
print(f'    - GT probably matched to wrong neuron or artifact')
print(f'\n  PROXIMITY (5-8px): {proximity_count} ({proximity_count/len(fake_fp_indices)*100:.1f}%)')
print(f'    - Near other neurons but not overlapping')
print(f'    - May be real neurons or edge cases')
print(f'\n  STANDALONE (> 8px): {standalone_count} ({standalone_count/len(fake_fp_indices)*100:.1f}%)')
print(f'    - Isolated, clearly good neurons')
print(f'    - GT definitely wrong - these ARE real neurons')

print(f'\nRecommendation for corrections:')
print(f'  - MERGE cases: Likely GT matching error - SAFE to flip DELETE -> KEEP')
print(f'  - PROXIMITY cases: Review quality metrics - likely safe to flip')
print(f'  - STANDALONE cases: Definitely GT error - SAFE to flip DELETE -> KEEP')

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
output_df.to_csv('ml/results/fake_fp_spatial_analysis.csv', index=False)
print(f'\nDetailed analysis saved to: ml/results/fake_fp_spatial_analysis.csv')

print(f'\n{"="*80}')
