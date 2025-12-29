"""
Apply iter3 corrections to v8_corrected_iter2 dataset based on user feedback.
User reviewed iter2 errors and identified:
- Real FN (13): #14, 18, 27, 29, 31, 32, 34, 41, 43, 67, 73, 81, 95 (model genuinely wrong)
- FAKE FN (87): All others (model correct to DELETE, GT wrong)
- Real FP (6): #5, 9, 12, 18, 53, 71 (model genuinely wrong)
- FAKE FP (94): All others (model correct to KEEP, GT wrong)
"""
import pandas as pd
import numpy as np

print('='*80)
print('APPLYING ITER3 CORRECTIONS TO v8_corrected_iter2')
print('='*80)

# Load v8_corrected_iter2 dataset
df = pd.read_csv('ml/results/training_dataset_v8_corrected_iter2.csv')
print(f'\nLoaded: {len(df):,} neurons')
print(f'Original KEEP: {df["ground_truth"].sum():,} ({df["ground_truth"].mean()*100:.2f}%)')

# Load iter2 error reports
df_fn = pd.read_csv('ml/results/v8_corrected_iter2_top100_fn.csv')
df_fp_spatial = pd.read_csv('ml/results/iter2_fake_fp_spatial_analysis.csv')

# FAKE FN corrections (87): Model correct to DELETE, GT wrong (KEEP→DELETE)
# Real FN are: 14, 18, 27, 29, 31, 32, 34, 41, 43, 67, 73, 81, 95
real_fn_indices = [14, 18, 27, 29, 31, 32, 34, 41, 43, 67, 73, 81, 95]
fake_fn_indices = [i for i in range(1, 101) if i not in real_fn_indices]

print(f'\n{"="*80}')
print(f'FAKE FN CORRECTIONS (model correct to DELETE)')
print('='*80)
print(f'Total FAKE FN: {len(fake_fn_indices)}')

corrections_fn = []
for fn_idx in fake_fn_indices:
    fn_row = df_fn.iloc[fn_idx - 1]
    session = fn_row['session']
    comp_idx = int(fn_row['component_idx'])
    corrections_fn.append((session, comp_idx, 0, f'FAKE FN #{fn_idx} - model correct DELETE'))

# FAKE FP corrections (94): Model correct to KEEP, GT wrong (DELETE→KEEP)
print(f'\n{"="*80}')
print(f'FAKE FP CORRECTIONS (model correct to KEEP)')
print('='*80)
print(f'Total FAKE FP: {len(df_fp_spatial)}')

corrections_fp = []
for idx, row in df_fp_spatial.iterrows():
    fp_idx = int(row['fp_idx'])
    session = row['session']
    comp_idx = int(row['component_idx'])
    category = row['category']
    corrections_fp.append((session, comp_idx, 1, f'FAKE FP #{fp_idx} - {category}, model correct KEEP'))

# Combine all corrections
all_corrections = corrections_fn + corrections_fp
print(f'\n{"="*80}')
print('SUMMARY')
print('='*80)
print(f'Total corrections: {len(all_corrections)}')
print(f'  FAKE FN (KEEP→DELETE): {len(corrections_fn)}')
print(f'  FAKE FP (DELETE→KEEP): {len(corrections_fp)}')

# Apply corrections
n_applied = 0
n_not_found = 0
correction_log = []

for session, comp_idx, new_label, reason in all_corrections:
    mask = (df['session'] == session) & (df['component_idx'] == comp_idx)
    if mask.sum() == 0:
        n_not_found += 1
        correction_log.append({
            'session': session,
            'component_idx': comp_idx,
            'new_label': new_label,
            'reason': reason,
            'status': 'NOT FOUND'
        })
        continue

    old_label = df.loc[mask, 'ground_truth'].values[0]
    df.loc[mask, 'ground_truth'] = new_label
    n_applied += 1
    correction_log.append({
        'session': session,
        'component_idx': comp_idx,
        'old_label': old_label,
        'new_label': new_label,
        'reason': reason,
        'status': 'APPLIED'
    })

print(f'\nCorrections applied: {n_applied}')
print(f'Corrections not found: {n_not_found}')

# Save corrected dataset
output_path = 'ml/results/training_dataset_v8_corrected_iter3.csv'
df.to_csv(output_path, index=False)

print(f'\nNew KEEP: {df["ground_truth"].sum():,} ({df["ground_truth"].mean()*100:.2f}%)')
print(f'Net change: {df["ground_truth"].sum() - 34210:+,} labels')

# Save correction log
log_df = pd.DataFrame(correction_log)
log_df.to_csv('ml/results/iter3_corrections_log.csv', index=False)

print(f'\nFiles saved:')
print(f'  Dataset: {output_path}')
print(f'  Log: ml/results/iter3_corrections_log.csv')

print(f'\n{"="*80}')
print('NEXT STEP: Retrain model with seed=45')
print('='*80)
print(f'Command: conda run -n bearmind python retrain_v8_corrected_iter3.py')
print(f'\n{"="*80}')
