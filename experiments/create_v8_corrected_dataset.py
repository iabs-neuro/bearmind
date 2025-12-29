"""
Create v8_corrected dataset with manual ground truth corrections.
Allows fixing mislabeled neurons based on expert review.
"""
import pandas as pd
import numpy as np

print('='*80)
print('CREATING v8_corrected DATASET')
print('='*80)

# Load original v8 dataset
df = pd.read_csv('ml/results/training_dataset_v8.csv')
print(f'\nLoaded v8 dataset: {len(df):,} neurons')
print(f'Original labels: KEEP={df["ground_truth"].sum():,}, DELETE={len(df) - df["ground_truth"].sum():,}')

# Create copy for corrections
df_corrected = df.copy()

# Define corrections: (session, component_idx, new_label, reason)
# new_label: 1=KEEP, 0=DELETE
corrections = [
    # Example format:
    # ('NOF_H27_4D', 28, 0, 'FN #3 - noisy, negative R2, actually bad'),
    # ('RFC_F30_3D', 242, 0, 'FN #6 - sparse, minimal activity, actually bad'),

    # ADD YOUR CORRECTIONS HERE
    # After reviewing visualizations, you can add entries like:
    # ('session_name', component_idx, new_label, 'reason'),
]

print(f'\n{"="*80}')
print('APPLYING CORRECTIONS')
print('='*80)

if len(corrections) == 0:
    print('\nNo corrections specified yet.')
    print('Add corrections to the "corrections" list in this script.')
    print('\nFormat: (session, component_idx, new_label, reason)')
    print('  session: session name (e.g., "NOF_H27_4D")')
    print('  component_idx: component index (integer)')
    print('  new_label: 1 for KEEP, 0 for DELETE')
    print('  reason: explanation for correction')
else:
    print(f'\nApplying {len(corrections)} corrections:')
    print('-'*80)

    corrections_applied = 0
    corrections_failed = 0

    for session, comp_idx, new_label, reason in corrections:
        # Find neuron
        mask = (df_corrected['session'] == session) & (df_corrected['component_idx'] == comp_idx)

        if mask.sum() == 0:
            print(f'[NOT FOUND] {session} component {comp_idx}')
            corrections_failed += 1
            continue

        if mask.sum() > 1:
            print(f'[DUPLICATE] {session} component {comp_idx} - {mask.sum()} matches')
            corrections_failed += 1
            continue

        # Get current label
        old_label = df_corrected.loc[mask, 'ground_truth'].values[0]

        if old_label == new_label:
            print(f'[NO CHANGE] {session} comp {comp_idx}: already {new_label} - {reason}')
        else:
            # Apply correction
            df_corrected.loc[mask, 'ground_truth'] = new_label
            old_str = 'KEEP' if old_label == 1 else 'DELETE'
            new_str = 'KEEP' if new_label == 1 else 'DELETE'
            print(f'[CORRECTED] {session} comp {comp_idx}: {old_str} -> {new_str} - {reason}')
            corrections_applied += 1

    print(f'\n{"="*80}')
    print(f'Summary: {corrections_applied} applied, {corrections_failed} failed, {len(corrections) - corrections_applied - corrections_failed} no change')

# Show label changes
original_keep = df['ground_truth'].sum()
corrected_keep = df_corrected['ground_truth'].sum()
label_change = corrected_keep - original_keep

print(f'\n{"="*80}')
print('DATASET STATISTICS')
print('='*80)

print(f'\nOriginal v8:')
print(f'  KEEP:   {original_keep:,} ({original_keep/len(df)*100:.2f}%)')
print(f'  DELETE: {len(df) - original_keep:,} ({(len(df) - original_keep)/len(df)*100:.2f}%)')

print(f'\nCorrected v8:')
print(f'  KEEP:   {corrected_keep:,} ({corrected_keep/len(df_corrected)*100:.2f}%)')
print(f'  DELETE: {len(df_corrected) - corrected_keep:,} ({(len(df_corrected) - corrected_keep)/len(df_corrected)*100:.2f}%)')

print(f'\nNet change: {label_change:+,} KEEP labels')
if label_change > 0:
    print(f'  ({label_change} neurons changed from DELETE to KEEP)')
elif label_change < 0:
    print(f'  ({abs(label_change)} neurons changed from KEEP to DELETE)')
else:
    print(f'  (No net change)')

# Save corrected dataset
output_path = 'ml/results/training_dataset_v8_corrected.csv'
df_corrected.to_csv(output_path, index=False)
print(f'\n{"="*80}')
print(f'Corrected dataset saved to: {output_path}')
print('='*80)

# Create correction log
if len(corrections) > 0:
    log_df = pd.DataFrame(corrections, columns=['session', 'component_idx', 'new_label', 'reason'])
    log_path = 'ml/results/v8_corrections_log.csv'
    log_df.to_csv(log_path, index=False)
    print(f'Correction log saved to: {log_path}')

print(f'\nNext steps:')
print(f'  1. Add more corrections to this script and re-run')
print(f'  2. When ready, retrain model on v8_corrected dataset')
print(f'  3. Compare v8_corrected model to original v8')

print(f'\n{"="*80}')
