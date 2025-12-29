"""
Spatial analysis of FAKE FP neurons for iteration 5.
Identifies MERGE cases (< 5px to KEEP neuron) to exclude from corrections.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# FAKE FP indices (model correct to KEEP, but GT says DELETE)
real_fp = [16,18,19,30,31,40,48,69]
fake_fp_indices = [i for i in range(1, 101) if i not in real_fp]

print('='*80)
print('SPATIAL ANALYSIS: FAKE FP (ITER5)')
print('='*80)
print(f'\nAnalyzing {len(fake_fp_indices)} FAKE FP neurons')
print('Goal: Exclude MERGE cases (< 5px to KEEP neuron)')

# Load FP error report
df_fp = pd.read_csv('ml/results/v8_corrected_iter4_top100_fp.csv')

# Load full dataset to check labels
df_full = pd.read_csv('ml/results/training_dataset_v8_corrected_iter4.csv')

results = []

for fp_idx in fake_fp_indices:
    fp_row = df_fp.iloc[fp_idx - 1]
    session = fp_row['session']
    comp_idx = int(fp_row['component_idx'])

    # Parse center as numpy array
    if isinstance(fp_row['center'], str):
        center = np.fromstring(fp_row['center'].strip('[]'), sep=' ')
    else:
        center = fp_row['center']

    # Get all neurons from this session
    session_neurons = df_full[df_full['session'] == session].copy()

    # Find KEEP neurons in this session
    keep_neurons = session_neurons[session_neurons['ground_truth'] == 1].copy()

    # Calculate distances to all KEEP neurons
    min_distance = np.inf
    closest_keep_idx = None

    for idx, keep_row in keep_neurons.iterrows():
        if int(keep_row['component_idx']) == comp_idx:
            continue  # Skip self

        # Parse keep center
        if isinstance(keep_row['center'], str):
            keep_center = np.fromstring(keep_row['center'].strip('[]'), sep=' ')
        else:
            keep_center = keep_row['center']

        distance = np.sqrt((center[0] - keep_center[0])**2 + (center[1] - keep_center[1])**2)

        if distance < min_distance:
            min_distance = distance
            closest_keep_idx = int(keep_row['component_idx'])

    # Categorize
    if min_distance < 5:
        category = 'MERGE'
    elif min_distance < 8:
        category = 'PROXIMITY'
    else:
        category = 'STANDALONE'

    results.append({
        'fp_idx': fp_idx,
        'session': session,
        'component_idx': comp_idx,
        'center': center,
        'min_distance_to_keep': min_distance,
        'closest_keep_idx': closest_keep_idx,
        'category': category
    })

df_results = pd.DataFrame(results)

# Summary
print(f'\n{"="*80}')
print('SPATIAL ANALYSIS RESULTS')
print('='*80)

merge_count = (df_results['category'] == 'MERGE').sum()
proximity_count = (df_results['category'] == 'PROXIMITY').sum()
standalone_count = (df_results['category'] == 'STANDALONE').sum()

print(f'\nMERGE (< 5px to KEEP): {merge_count}')
print(f'PROXIMITY (5-8px): {proximity_count}')
print(f'STANDALONE (> 8px): {standalone_count}')
print(f'Total FAKE FP: {len(df_results)}')

print(f'\n{"="*80}')
print('CORRECTION DECISION')
print('='*80)

df_to_flip = df_results[df_results['category'] != 'MERGE']
print(f'\nFAKE FP to flip (excluding MERGE): {len(df_to_flip)}')
print(f'  PROXIMITY: {(df_to_flip["category"] == "PROXIMITY").sum()}')
print(f'  STANDALONE: {(df_to_flip["category"] == "STANDALONE").sum()}')

print(f'\nMERGE cases to EXCLUDE: {merge_count}')
print('  Reason: These are real duplicates/overlaps, GT is correct to DELETE')

# Save results
output_path = 'ml/results/iter5_fake_fp_spatial_analysis.csv'
df_results.to_csv(output_path, index=False)
print(f'\nResults saved to: {output_path}')

print(f'\n{"="*80}')
print('FINAL ITER5 CORRECTION COUNTS')
print('='*80)
print(f'FAKE FN (KEEP→DELETE): 80')
print(f'FAKE FP (DELETE→KEEP): {len(df_to_flip)} (excluding {merge_count} MERGE)')
print(f'Total corrections: {80 + len(df_to_flip)}')
