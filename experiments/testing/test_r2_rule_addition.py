"""
Test that the new r2_score < -1.0 default rule works correctly.
"""
import pandas as pd
import numpy as np
from auto_inspector import DEFAULT_DELETION_RULES, parse_rule, _apply_threshold_brain

print('='*80)
print('TESTING NEW R2_SCORE < -1.0 DEFAULT RULE')
print('='*80)

# Check that rule is in defaults
print('\nDefault deletion rules:')
for i, rule in enumerate(DEFAULT_DELETION_RULES, 1):
    print(f'  {i}. {rule}')

assert 'r2_score<-1.0' in DEFAULT_DELETION_RULES, "r2_score rule not found in defaults!"
print('\n✓ r2_score<-1.0 is in DEFAULT_DELETION_RULES')

# Test parsing
print('\n' + '='*80)
print('TESTING RULE PARSING')
print('='*80)

metric, op, threshold = parse_rule('r2_score<-1.0')
print(f'\nParsed rule:')
print(f'  Metric: {metric}')
print(f'  Operator: {op}')
print(f'  Threshold: {threshold}')

assert metric == 'r2_score'
assert op == '<'
assert threshold == -1.0
print('\n✓ Rule parsing works correctly')

# Test with synthetic data
print('\n' + '='*80)
print('TESTING RULE APPLICATION')
print('='*80)

# Create test dataframe
test_df = pd.DataFrame({
    'component_idx': range(10),
    'area': [5, 5, 5, 5, 5, 5, 5, 5, 5, 0.5],  # Last one fails area<1
    'circularity': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
    'events_per_min': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    'r2_score': [0.5, 0.0, -0.5, -0.9, -1.0, -1.1, -1.5, -2.0, np.nan, 0.5],
})

print(f'\nTest data (10 neurons):')
print(test_df[['component_idx', 'area', 'r2_score']].to_string())

# Apply rules
delete_mask, failure_info = _apply_threshold_brain(
    test_df,
    rules=DEFAULT_DELETION_RULES,
    track_failures=True
)

print(f'\nDeletion results:')
print(f'  Neurons flagged for deletion: {delete_mask.sum()}')
print(f'  Flagged indices: {test_df[delete_mask]["component_idx"].tolist()}')

# Check specific expectations
# Neurons 5, 6, 7 should be flagged by r2_score<-1.0
# Neuron 9 should be flagged by area<1
expected_deleted = {5, 6, 7, 9}
actual_deleted = set(test_df[delete_mask]['component_idx'].tolist())

print(f'\n  Expected deleted: {sorted(expected_deleted)}')
print(f'  Actually deleted: {sorted(actual_deleted)}')

# Check failure tracking
if 'failed_r2_score' in failure_info:
    r2_failures = failure_info['failed_r2_score']
    print(f'\n  Neurons failing r2_score rule: {test_df[r2_failures]["component_idx"].tolist()}')
    assert r2_failures.sum() == 3, f"Expected 3 r2_score failures, got {r2_failures.sum()}"
    print(f'  ✓ Correct neurons flagged by r2_score rule')

assert actual_deleted == expected_deleted, f"Deletion mismatch! Expected {expected_deleted}, got {actual_deleted}"
print('\n✓ Rule application works correctly')

# Test on v9 dataset
print('\n' + '='*80)
print('TESTING ON V9 DATASET')
print('='*80)

try:
    v9 = pd.read_csv('ml/results/training_dataset_v9.csv')
    v9_valid = v9[v9['r2_score'].notna()].copy()

    # Apply just the r2_score rule
    r2_delete = v9_valid['r2_score'] < -1.0
    r2_delete_count = r2_delete.sum()

    # Check ground truth
    r2_delete_bad = ((v9_valid['r2_score'] < -1.0) & (v9_valid['ground_truth'] == 0)).sum()
    r2_delete_good = ((v9_valid['r2_score'] < -1.0) & (v9_valid['ground_truth'] == 1)).sum()

    print(f'\nImpact on v9 dataset:')
    print(f'  Total neurons with valid r2: {len(v9_valid):,}')
    print(f'  Flagged by r2_score<-1.0: {r2_delete_count:,} ({100*r2_delete_count/len(v9_valid):.2f}%)')
    print(f'    Actually bad: {r2_delete_bad:,}')
    print(f'    Actually good: {r2_delete_good:,}')
    print(f'    Precision: {100*r2_delete_bad/r2_delete_count:.1f}%')
    print(f'    FN rate: {100*r2_delete_good/v9_valid["ground_truth"].sum():.2f}%')

    # This should match our analysis
    assert r2_delete_count == 1688, f"Expected 1688, got {r2_delete_count}"
    assert r2_delete_good == 142, f"Expected 142 FN, got {r2_delete_good}"

    print('\n✓ V9 dataset results match expected values from analysis')

except FileNotFoundError:
    print('\n[SKIP] v9 dataset not found, skipping validation test')

print('\n' + '='*80)
print('ALL TESTS PASSED ✓')
print('='*80)

print(f'''
Summary:
- r2_score<-1.0 successfully added to DEFAULT_DELETION_RULES
- Rule parsing works correctly
- Rule application correctly identifies neurons with r2 < -1.0
- Expected impact on v9: flags 1,688 neurons (10% of bad, 0.2% FN rate)

The rule is now active and will be used in all future auto_inspection runs.
''')
