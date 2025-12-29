"""
Save CORRECTED expert classifications for v9_iter2.
"""
import pandas as pd
import json

# CORRECTED expert classifications from manual review
REAL_FP = [6, 7, 16, 22, 27, 35, 40, 51, 54, 72, 73, 77, 88, 92]
REAL_FN = [12, 13, 20, 29, 31, 54, 95]

print('='*80)
print('V9 ITERATION 2 - CORRECTED EXPERT CLASSIFICATIONS')
print('='*80)

print(f'\nReal FP (model wrong, GT correct DELETE): {len(REAL_FP)}')
print(f'  Indices: {sorted(REAL_FP)}')

print(f'\nReal FN (model wrong, GT correct KEEP): {len(REAL_FN)}')
print(f'  Indices: {sorted(REAL_FN)}')

# Calculate FAKE cases
print(f'\nFake FP (model correct KEEP, GT wrong DELETE): {100 - len(REAL_FP)}')
print(f'  Will flip to KEEP (excluding MERGE cases)')

print(f'\nFake FN (model correct DELETE, GT wrong KEEP): {100 - len(REAL_FN)}')
print(f'  Will flip to DELETE')

# Load spatial analysis to check MERGE overlap
df_spatial = pd.read_csv('ml/ebm_v9_iter2/fp_spatial_analysis.csv')
merge_indices = set(df_spatial[df_spatial['category'] == 'MERGE']['fp_idx'].values)

print('\n' + '='*80)
print('SPATIAL ANALYSIS CHECK')
print('='*80)

print(f'\nMERGE cases (< 5px, exclude from corrections): {len(merge_indices)}')
print(f'  Indices: {sorted(merge_indices)}')

# Check overlap between REAL_FP and MERGE
overlap = set(REAL_FP) & merge_indices
if overlap:
    print(f'\nWARNING: {len(overlap)} REAL_FP are also MERGE cases:')
    print(f'  {sorted(overlap)}')
    print('  These are true duplicates (GT is correct), no conflict.')

# Save classification summary
summary = {
    'real_fp_indices': sorted(REAL_FP),
    'real_fn_indices': sorted(REAL_FN),
    'merge_indices': [int(x) for x in sorted(merge_indices)],
    'n_real_fp': len(REAL_FP),
    'n_real_fn': len(REAL_FN),
    'n_fake_fp': 100 - len(REAL_FP),
    'n_fake_fn': 100 - len(REAL_FN),
    'n_merge': len(merge_indices),
    'n_fake_fp_to_correct': 100 - len(REAL_FP) - len(merge_indices - set(REAL_FP))
}

with open('ml/ebm_v9_iter2/expert_classifications.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('\n' + '='*80)
print('CORRECTION PLAN')
print('='*80)

fake_fp_indices = [i for i in range(1, 101) if i not in REAL_FP and i not in merge_indices]
fake_fn_indices = [i for i in range(1, 101) if i not in REAL_FN]

print(f'\nFake FP to flip from DELETE to KEEP: {len(fake_fp_indices)}')
print(f'  (Excluded {len(merge_indices - set(REAL_FP))} MERGE cases that are not in REAL_FP)')

print(f'\nFake FN to flip from KEEP to DELETE: {len(fake_fn_indices)}')

print(f'\nTotal labels to correct: {len(fake_fp_indices) + len(fake_fn_indices)}')

print(f'\nSaved to: ml/ebm_v9_iter2/expert_classifications.json')

print('\nNext step:')
print('  python ml/apply_corrections.py \\')
print('    --dataset ml/results/training_dataset_v9_corrected_iter1.csv \\')
print('    --fp-errors ml/ebm_v9_iter2/top100_fp.csv \\')
print('    --fn-errors ml/ebm_v9_iter2/top100_fn.csv \\')
print(f'    --real-fp {",".join(map(str, sorted(REAL_FP)))} \\')
print(f'    --real-fn {",".join(map(str, sorted(REAL_FN)))} \\')
print('    --fp-spatial ml/ebm_v9_iter2/fp_spatial_analysis.csv \\')
print('    --output ml/results/training_dataset_v9_corrected_iter2.csv')
