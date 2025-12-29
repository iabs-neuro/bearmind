"""
Compare error rates between iter2 and iter3 to investigate FP increase.
"""

# Iter2 (seed=44)
test_size_2 = 10413
fp_2 = 567
fn_2 = 1180
total_errors_2 = fp_2 + fn_2

# Iter3 (seed=45)
test_size_3 = 12124
fp_3 = 832
fn_3 = 772
total_errors_3 = fp_3 + fn_3

print('='*80)
print('ERROR RATE COMPARISON: ITER2 vs ITER3')
print('='*80)

print(f'\nITER2 (seed=44, test_size={test_size_2:,}):')
print(f'  FP rate:    {fp_2/test_size_2*100:6.2f}% ({fp_2:,} / {test_size_2:,})')
print(f'  FN rate:    {fn_2/test_size_2*100:6.2f}% ({fn_2:,} / {test_size_2:,})')
print(f'  Total rate: {total_errors_2/test_size_2*100:6.2f}% ({total_errors_2:,} / {test_size_2:,})')

print(f'\nITER3 (seed=45, test_size={test_size_3:,}):')
print(f'  FP rate:    {fp_3/test_size_3*100:6.2f}% ({fp_3:,} / {test_size_3:,})')
print(f'  FN rate:    {fn_3/test_size_3*100:6.2f}% ({fn_3:,} / {test_size_3:,})')
print(f'  Total rate: {total_errors_3/test_size_3*100:6.2f}% ({total_errors_3:,} / {test_size_3:,})')

print(f'\n{"="*80}')
print('CHANGE (iter3 - iter2):')
print('='*80)
print(f'  FP rate:    {(fp_3/test_size_3 - fp_2/test_size_2)*100:+.2f} percentage points')
print(f'  FN rate:    {(fn_3/test_size_3 - fn_2/test_size_2)*100:+.2f} percentage points')
print(f'  Total rate: {(total_errors_3/test_size_3 - total_errors_2/test_size_2)*100:+.2f} percentage points')

print(f'\n{"="*80}')
print('ANALYSIS')
print('='*80)

fp_rate_increase = (fp_3/test_size_3 - fp_2/test_size_2)*100
fn_rate_decrease = (fn_2/test_size_2 - fn_3/test_size_3)*100
total_rate_decrease = (total_errors_2/test_size_2 - total_errors_3/test_size_3)*100

print(f'\n1. TEST SET SIZE CHANGE:')
print(f'   Iter3 has {test_size_3 - test_size_2:,} more test neurons (+{(test_size_3/test_size_2 - 1)*100:.1f}%)')
print(f'   Different random seed = different train/test split')

print(f'\n2. FP RATE INCREASED by {fp_rate_increase:.2f} percentage points')
print(f'   This means the model is being more permissive (says KEEP more often)')
print(f'   Possible causes:')
print(f'   - 90 FAKE FP corrections (DELETE→KEEP) taught model to be less strict')
print(f'   - Seed=45 may have harder test sessions')
print(f'   - Random variation in train/test split')

print(f'\n3. FN RATE DECREASED by {fn_rate_decrease:.2f} percentage points')
print(f'   This is GOOD - fewer false rejections of valid neurons')
print(f'   Recall improved significantly')

print(f'\n4. OVERALL ERROR RATE DECREASED by {total_rate_decrease:.2f} percentage points')
print(f'   Despite FP increase, total errors went down')

print(f'\n5. F-BETA IMPROVED: 0.9102 → 0.9150 (+0.48%)')
print(f'   F-beta (β=0.577) balances precision vs recall')
print(f'   The improvement shows the trade-off favors the model')

print(f'\n{"="*80}')
print('RECOMMENDATION')
print('='*80)

print(f'\nThe FP increase is concerning but may be due to:')
print(f'  1. Random variation from different train/test split (seed change)')
print(f'  2. Model learned to be more permissive (higher recall) from corrections')
print(f'  3. Seed=45 may have selected harder sessions for test set')

print(f'\nOPTIONS:')
print(f'  A) ACCEPT iter3: Overall error rate improved, F-beta improved')
print(f'  B) RETRAIN iter3 with seed=44 (same as iter2) for fair comparison')
print(f'  C) INVESTIGATE which sessions are in iter3 test set vs iter2')
print(f'  D) STOP iterations here - diminishing returns')

print(f'\n{"="*80}')
