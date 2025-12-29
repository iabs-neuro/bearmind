"""
Investigate spatial relationships of real FP errors.
Check if they're merged with nearby neurons or just mislabeled artifacts.
"""
import pandas as pd
import numpy as np

print('='*80)
print('INVESTIGATING FP SPATIAL RELATIONSHIPS')
print('='*80)

# Load datasets
df = pd.read_csv('ml/results/training_dataset_v8.csv')
df_fp = pd.read_csv('ml/results/v8_top100_false_positives.csv')

# Real FP indices (model wrong to KEEP, should DELETE)
real_fp_indices = [1, 6, 33, 35, 38, 40, 47, 50, 59, 61, 97, 100]

print(f'\nAnalyzing {len(real_fp_indices)} REAL FP errors')
print('='*80)

for fp_idx in real_fp_indices:
    fp_row = df_fp.iloc[fp_idx - 1]
    session = fp_row['session']
    comp_idx = int(fp_row['component_idx'])
    prob = fp_row['y_proba']

    # Find in full dataset
    neuron_mask = (df['session'] == session) & (df['component_idx'] == comp_idx)
    if neuron_mask.sum() == 0:
        print(f'\nFP #{fp_idx}: {session} component {comp_idx} - NOT FOUND')
        continue

    neuron = df[neuron_mask].iloc[0]

    # Get all neurons from same session
    session_neurons = df[df['session'] == session].copy()

    # Parse center coordinates
    if isinstance(neuron['center'], str):
        center = np.fromstring(neuron['center'].strip('[]'), sep=' ')
    else:
        center = neuron['center']

    print(f'\n{"="*80}')
    print(f'FP #{fp_idx}: {session} component {comp_idx}')
    print(f'Probability: {prob:.3f} (model says KEEP, should DELETE)')
    print('='*80)

    # Spatial metrics
    nn_distance = neuron.get('nn_distance_center', np.nan)
    area = neuron.get('area', np.nan)
    edge_distance = neuron.get('edge_distance', np.nan)

    print(f'\nNeuron properties:')
    print(f'  Center: ({center[0]:.1f}, {center[1]:.1f})')
    print(f'  Area: {area:.1f} pixels')
    print(f'  Edge distance: {edge_distance:.3f}')
    print(f'  NN distance (center): {nn_distance:.2f} pixels')

    # Find nearby neurons (within 10 pixels)
    nearby_neurons = []
    for idx, other in session_neurons.iterrows():
        if other['component_idx'] == comp_idx:
            continue  # Skip self

        # Parse other center
        if isinstance(other['center'], str):
            other_center = np.fromstring(other['center'].strip('[]'), sep=' ')
        else:
            other_center = other['center']

        # Calculate distance
        distance = np.linalg.norm(center - other_center)

        if distance <= 10:
            nearby_neurons.append({
                'component_idx': int(other['component_idx']),
                'distance': distance,
                'area': other.get('area', np.nan),
                'ground_truth': int(other['ground_truth']),
                'center': other_center
            })

    # Sort by distance
    nearby_neurons.sort(key=lambda x: x['distance'])

    print(f'\nNearby neurons (within 10 pixels): {len(nearby_neurons)}')
    if nearby_neurons:
        print(f'{"Comp":<8} {"Distance":<10} {"Area":<8} {"GT":<8} {"Center"}')
        print('-'*60)
        for n in nearby_neurons[:5]:  # Top 5 closest
            gt_str = 'KEEP' if n['ground_truth'] == 1 else 'DELETE'
            print(f'{n["component_idx"]:<8} {n["distance"]:<10.2f} {n["area"]:<8.1f} {gt_str:<8} ({n["center"][0]:.1f}, {n["center"][1]:.1f})')

    # Check for potential merge
    merge_candidates = [n for n in nearby_neurons if n['distance'] < 5]

    if merge_candidates:
        print(f'\nPOTENTIAL MERGE: {len(merge_candidates)} neurons within 5 pixels')
        print(f'  This neuron may be overlapping/merged with:')
        for n in merge_candidates:
            gt_str = 'KEEP' if n['ground_truth'] == 1 else 'DELETE'
            print(f'    Component {n["component_idx"]} (d={n["distance"]:.2f}px, GT={gt_str})')
    else:
        print(f'\nNO MERGE: Nearest neuron is {nn_distance:.2f} pixels away')
        print(f'  Likely a standalone artifact, not merged')

    # Quality metrics
    print(f'\nQuality metrics:')
    print(f'  SNR: {neuron.get("caiman_snr", np.nan):.2f}')
    print(f'  R2: {neuron.get("r2_score", np.nan):.3f}')
    print(f'  Kurtosis: {neuron.get("trace_kurtosis", np.nan):.2f}')
    print(f'  Events/min: {neuron.get("events_per_min", np.nan):.2f}')

    # Decision
    if merge_candidates:
        print(f'\nRECOMMENDATION: Review for MERGE with nearby neurons')
    elif nn_distance < 8:
        print(f'\nRECOMMENDATION: Check for PROXIMITY to neuron {nn_distance:.1f}px away')
    else:
        print(f'\nRECOMMENDATION: Likely STANDALONE ARTIFACT - safe to relabel DELETE')

print(f'\n{"="*80}')
print('SUMMARY')
print('='*80)

# Aggregate analysis
merge_count = 0
proximity_count = 0
standalone_count = 0

for fp_idx in real_fp_indices:
    fp_row = df_fp.iloc[fp_idx - 1]
    session = fp_row['session']
    comp_idx = int(fp_row['component_idx'])

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

    # Check for nearby neurons
    has_merge = False
    for idx, other in session_neurons.iterrows():
        if other['component_idx'] == comp_idx:
            continue

        if isinstance(other['center'], str):
            other_center = np.fromstring(other['center'].strip('[]'), sep=' ')
        else:
            other_center = other['center']

        distance = np.linalg.norm(center - other_center)

        if distance < 5:
            has_merge = True
            break

    nn_distance = neuron.get('nn_distance_center', np.nan)

    if has_merge:
        merge_count += 1
    elif nn_distance < 8:
        proximity_count += 1
    else:
        standalone_count += 1

print(f'\nSpatial relationship breakdown ({len(real_fp_indices)} real FP):')
print(f'  Potential MERGE (< 5px): {merge_count} neurons - REVIEW CAREFULLY')
print(f'  PROXIMITY (5-8px): {proximity_count} neurons - CHECK OVERLAP')
print(f'  STANDALONE (> 8px): {standalone_count} neurons - LIKELY ARTIFACTS')

print(f'\nNext steps:')
print(f'  1. For MERGE cases: Check if overlapping with good neurons')
print(f'  2. For PROXIMITY cases: Verify no significant overlap')
print(f'  3. For STANDALONE cases: Safe to relabel as DELETE')

print(f'\n{"="*80}')
