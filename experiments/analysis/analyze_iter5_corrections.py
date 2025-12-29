"""
Analyze iteration 5 corrections based on user feedback.
"""

# Real model errors (model genuinely wrong)
real_fn = [11,17,23,26,27,32,36,39,42,45,49,54,56,66,75,84,90,95,97,98]
real_fp = [16,18,19,30,31,40,48,69]

# FAKE errors (model correct, GT wrong)
fake_fn = [i for i in range(1, 101) if i not in real_fn]
fake_fp = [i for i in range(1, 101) if i not in real_fp]

print('='*80)
print('ITERATION 5 ERROR CLASSIFICATION')
print('='*80)

print(f'\nReal FN (model genuinely wrong to DELETE): {len(real_fn)}')
print(f'  Indices: {real_fn}')

print(f'\nFAKE FN (model correct to DELETE, GT wrong): {len(fake_fn)}')
print(f'  First 20: {fake_fn[:20]}')
print(f'  Total: {len(fake_fn)}')

print(f'\nReal FP (model genuinely wrong to KEEP): {len(real_fp)}')
print(f'  Indices: {real_fp}')

print(f'\nFAKE FP (model correct to KEEP, GT wrong): {len(fake_fp)}')
print(f'  First 20: {fake_fp[:20]}')
print(f'  Total: {len(fake_fp)}')

print(f'\n{"="*80}')
print('CORRECTION SUMMARY (before MERGE exclusion)')
print('='*80)
print(f'FAKE FN corrections (KEEP→DELETE): {len(fake_fn)}')
print(f'FAKE FP corrections (DELETE→KEEP): {len(fake_fp)} (pending spatial analysis)')
print(f'Total corrections: {len(fake_fn) + len(fake_fp)}')

print(f'\n{"="*80}')
print('NEXT STEP: Run spatial analysis on FAKE FP to exclude MERGE cases')
print('='*80)
