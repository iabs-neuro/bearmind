"""
Compare iter4 vs iter5 performance on same test set (seed=45).
"""

print('='*80)
print('ITERATION 4 vs ITERATION 5 COMPARISON')
print('='*80)
print('Same seed (45) = Same train/test split = Fair comparison')

# Iter4 results
iter4_fbeta = 0.9241
iter4_auc = 0.9357
iter4_precision = 0.9227
iter4_recall = 0.9285
iter4_fp = 745
iter4_fn = 684
iter4_total_errors = iter4_fp + iter4_fn
iter4_test_size = 12124

# Iter5 results
iter5_fbeta = 0.9263
iter5_auc = 0.9387
iter5_precision = 0.9254
iter5_recall = 0.9292
iter5_fp = 718
iter5_fn = 678
iter5_total_errors = iter5_fp + iter5_fn
iter5_test_size = 12124

print('\n' + '='*80)
print('ITER4 RESULTS (before 156 corrections)')
print('='*80)
print(f'Test F-beta:     {iter4_fbeta:.4f}')
print(f'Test AUC:        {iter4_auc:.4f}')
print(f'Test Precision:  {iter4_precision:.4f}')
print(f'Test Recall:     {iter4_recall:.4f}')
print(f'False Positives: {iter4_fp:,}')
print(f'False Negatives: {iter4_fn:,}')
print(f'Total Errors:    {iter4_total_errors:,}')

print('\n' + '='*80)
print('ITER5 RESULTS (after 156 corrections)')
print('='*80)
print(f'Test F-beta:     {iter5_fbeta:.4f}')
print(f'Test AUC:        {iter5_auc:.4f}')
print(f'Test Precision:  {iter5_precision:.4f}')
print(f'Test Recall:     {iter5_recall:.4f}')
print(f'False Positives: {iter5_fp:,}')
print(f'False Negatives: {iter5_fn:,}')
print(f'Total Errors:    {iter5_total_errors:,}')

print('\n' + '='*80)
print('IMPROVEMENT (iter5 - iter4)')
print('='*80)
print(f'F-beta:          {iter5_fbeta - iter4_fbeta:+.4f} ({(iter5_fbeta/iter4_fbeta - 1)*100:+.2f}%)')
print(f'AUC:             {iter5_auc - iter4_auc:+.4f} ({(iter5_auc/iter4_auc - 1)*100:+.2f}%)')
print(f'Precision:       {iter5_precision - iter4_precision:+.4f} ({(iter5_precision/iter4_precision - 1)*100:+.2f}%)')
print(f'Recall:          {iter5_recall - iter4_recall:+.4f} ({(iter5_recall/iter4_recall - 1)*100:+.2f}%)')
print(f'False Positives: {iter5_fp - iter4_fp:+,} ({(iter5_fp/iter4_fp - 1)*100:+.2f}%)')
print(f'False Negatives: {iter5_fn - iter4_fn:+,} ({(iter5_fn/iter4_fn - 1)*100:+.2f}%)')
print(f'Total Errors:    {iter5_total_errors - iter4_total_errors:+,} ({(iter5_total_errors/iter4_total_errors - 1)*100:+.2f}%)')

print('\n' + '='*80)
print('SUMMARY')
print('='*80)
print(f'\n156 corrections applied (80 FAKE FN + 76 FAKE FP)')
print(f'\nResults on SAME test set (seed=45):')
print(f'  - F-beta improved by {(iter5_fbeta/iter4_fbeta - 1)*100:.2f}%')
print(f'  - AUC improved by {(iter5_auc/iter4_auc - 1)*100:.2f}%')
print(f'  - Total errors reduced by {abs(iter5_total_errors - iter4_total_errors)} (-{abs((iter5_total_errors/iter4_total_errors - 1)*100):.1f}%)')
print(f'  - FP reduced by {abs(iter5_fp - iter4_fp)} (-{abs((iter5_fp/iter4_fp - 1)*100):.1f}%)')
print(f'  - FN reduced by {abs(iter5_fn - iter4_fn)} (-{abs((iter5_fn/iter4_fn - 1)*100):.1f}%)')

print('\n' + '='*80)
print('CUMULATIVE PROGRESS (ITER3 → ITER5)')
print('='*80)

# Iter3 baseline (seed=45)
iter3_fbeta = 0.9151
iter3_total_errors = 1602

print(f'\nIter3 → Iter5 (329 total corrections):')
print(f'  - F-beta: {iter3_fbeta:.4f} → {iter5_fbeta:.4f} (+{(iter5_fbeta/iter3_fbeta - 1)*100:.2f}%)')
print(f'  - Total errors: {iter3_total_errors:,} → {iter5_total_errors:,} (-{abs(iter5_total_errors - iter3_total_errors)} errors, -{abs((iter5_total_errors/iter3_total_errors - 1)*100):.1f}%)')

print('\n' + '='*80)
print('CONCLUSION')
print('='*80)
print('\nIteration 5 shows continued improvement, though with diminishing returns.')
print('Approaching point where further iterations may yield minimal gains.')
print('\nConsider deploying iter5 model or continue to iteration 6 if desired.')
print('='*80)
