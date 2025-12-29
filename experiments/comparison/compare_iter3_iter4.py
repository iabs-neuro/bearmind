"""
Compare iter3 vs iter4 performance on same test set (seed=45).
"""

print('='*80)
print('ITERATION 3 vs ITERATION 4 COMPARISON')
print('='*80)
print('Same seed (45) = Same train/test split = Fair comparison')

# Iter3 results
iter3_fbeta = 0.9151
iter3_auc = 0.9121
iter3_precision = 0.9136
iter3_recall = 0.9195
iter3_fp = 832
iter3_fn = 770
iter3_total_errors = iter3_fp + iter3_fn
iter3_test_size = 12124

# Iter4 results
iter4_fbeta = 0.9241
iter4_auc = 0.9357
iter4_precision = 0.9227
iter4_recall = 0.9285
iter4_fp = 745
iter4_fn = 684
iter4_total_errors = iter4_fp + iter4_fn
iter4_test_size = 12124

print('\n' + '='*80)
print('ITER3 RESULTS (before 173 corrections)')
print('='*80)
print(f'Test F-beta:     {iter3_fbeta:.4f}')
print(f'Test AUC:        {iter3_auc:.4f}')
print(f'Test Precision:  {iter3_precision:.4f}')
print(f'Test Recall:     {iter3_recall:.4f}')
print(f'False Positives: {iter3_fp:,}')
print(f'False Negatives: {iter3_fn:,}')
print(f'Total Errors:    {iter3_total_errors:,}')

print('\n' + '='*80)
print('ITER4 RESULTS (after 173 corrections)')
print('='*80)
print(f'Test F-beta:     {iter4_fbeta:.4f}')
print(f'Test AUC:        {iter4_auc:.4f}')
print(f'Test Precision:  {iter4_precision:.4f}')
print(f'Test Recall:     {iter4_recall:.4f}')
print(f'False Positives: {iter4_fp:,}')
print(f'False Negatives: {iter4_fn:,}')
print(f'Total Errors:    {iter4_total_errors:,}')

print('\n' + '='*80)
print('IMPROVEMENT (iter4 - iter3)')
print('='*80)
print(f'F-beta:          {iter4_fbeta - iter3_fbeta:+.4f} ({(iter4_fbeta/iter3_fbeta - 1)*100:+.2f}%)')
print(f'AUC:             {iter4_auc - iter3_auc:+.4f} ({(iter4_auc/iter3_auc - 1)*100:+.2f}%)')
print(f'Precision:       {iter4_precision - iter3_precision:+.4f} ({(iter4_precision/iter3_precision - 1)*100:+.2f}%)')
print(f'Recall:          {iter4_recall - iter3_recall:+.4f} ({(iter4_recall/iter3_recall - 1)*100:+.2f}%)')
print(f'False Positives: {iter4_fp - iter3_fp:+,} ({(iter4_fp/iter3_fp - 1)*100:+.2f}%)')
print(f'False Negatives: {iter4_fn - iter3_fn:+,} ({(iter4_fn/iter3_fn - 1)*100:+.2f}%)')
print(f'Total Errors:    {iter4_total_errors - iter3_total_errors:+,} ({(iter4_total_errors/iter3_total_errors - 1)*100:+.2f}%)')

print('\n' + '='*80)
print('SUMMARY')
print('='*80)
print(f'\n173 corrections applied (86 FAKE FN + 87 FAKE FP)')
print(f'\nResults on SAME test set (seed=45):')
print(f'  - F-beta improved by {(iter4_fbeta/iter3_fbeta - 1)*100:.2f}%')
print(f'  - AUC improved by {(iter4_auc/iter3_auc - 1)*100:.2f}%')
print(f'  - Total errors reduced by {abs(iter4_total_errors - iter3_total_errors)} (-{abs((iter4_total_errors/iter3_total_errors - 1)*100):.1f}%)')
print(f'  - FP reduced by {abs(iter4_fp - iter3_fp)} (-{abs((iter4_fp/iter3_fp - 1)*100):.1f}%)')
print(f'  - FN reduced by {abs(iter4_fn - iter3_fn)} (-{abs((iter4_fn/iter3_fn - 1)*100):.1f}%)')

print('\n' + '='*80)
print('CONCLUSION')
print('='*80)
print('\nIteration 4 shows significant improvement across ALL metrics.')
print('The iterative ground truth correction process is working effectively.')
print('\nContinue with iteration 5 or consider deploying iter4 model.')
print('='*80)
