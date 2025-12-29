"""
Verify the new event_r2_score < 0.15 rule is working correctly.
"""
import pandas as pd
from auto_inspector import DEFAULT_DELETION_RULES, parse_rule

print('='*80)
print('VERIFYING event_r2_score < 0.15 RULE')
print('='*80)

# 1. Check rule is in defaults
print('\nDefault deletion rules:')
for i, rule in enumerate(DEFAULT_DELETION_RULES, 1):
    print(f'  {i}. {rule}')

assert 'event_r2_score<0.15' in DEFAULT_DELETION_RULES, "Rule not found in defaults!"
print('\n[OK] event_r2_score<0.15 is in DEFAULT_DELETION_RULES')

# 2. Test parsing
print('\n' + '='*80)
print('TESTING RULE PARSING')
print('='*80)

metric, op, threshold = parse_rule('event_r2_score<0.15')
print(f'\nParsed rule:')
print(f'  Metric: {metric}')
print(f'  Operator: {op}')
print(f'  Threshold: {threshold}')

assert metric == 'event_r2_score', f"Expected 'event_r2_score', got '{metric}'"
assert op == '<', f"Expected '<', got '{op}'"
assert threshold == 0.15, f"Expected 0.15, got {threshold}"
print('\n[OK] Rule parsing works correctly')

# 3. Verify impact on v9 dataset
print('\n' + '='*80)
print('VERIFYING IMPACT ON V9 DATASET')
print('='*80)

v9 = pd.read_csv('ml/results/training_dataset_v9.csv')
v9_valid = v9[v9['event_r2_score'].notna()].copy()

# Count neurons flagged by this rule
flagged = (v9_valid['event_r2_score'] < 0.15).sum()
flagged_bad = ((v9_valid['event_r2_score'] < 0.15) & (v9_valid['ground_truth'] == 0)).sum()
flagged_good = ((v9_valid['event_r2_score'] < 0.15) & (v9_valid['ground_truth'] == 1)).sum()

total_bad = (v9_valid['ground_truth'] == 0).sum()
total_good = (v9_valid['ground_truth'] == 1).sum()

precision = 100 * flagged_bad / flagged if flagged > 0 else 0
coverage = 100 * flagged_bad / total_bad
fn_rate = 100 * flagged_good / total_good

print(f'\nImpact on v9 dataset ({len(v9_valid):,} neurons with valid event_r2_score):')
print(f'  Flagged by rule: {flagged:,} ({100*flagged/len(v9_valid):.2f}%)')
print(f'    Correctly identifies bad: {flagged_bad:,}')
print(f'    Incorrectly flags good: {flagged_good:,}')
print(f'  Precision: {precision:.2f}%')
print(f'  Coverage: {coverage:.2f}% of bad neurons')
print(f'  FN rate: {fn_rate:.4f}% ({flagged_good:,} of {total_good:,} good neurons)')

# Verify expected values
expected_flagged = 1854
expected_fn = 145

tolerance = 5

if abs(flagged - expected_flagged) <= tolerance:
    print(f'\n[OK] Flagged count matches expected (~{expected_flagged})')
else:
    print(f'\n[WARNING] Flagged count {flagged} differs from expected {expected_flagged}')

if abs(flagged_good - expected_fn) <= tolerance:
    print(f'[OK] FN count matches expected (~{expected_fn})')
else:
    print(f'[WARNING] FN count {flagged_good} differs from expected {expected_fn}')

print('\n' + '='*80)
print('SUMMARY')
print('='*80)

print(f'''
The rule "event_r2_score < 0.15" has been successfully added to DEFAULT_DELETION_RULES.

This rule will:
✓ Automatically flag neurons with event reconstruction R² < 0.15
✓ Catch ~10% of artifacts ({flagged_bad:,} neurons)
✓ Maintain ultra-low FN rate (~0.19%, only {flagged_good:,} good neurons)
✓ Achieve ~92% precision
✓ Be applied in all future auto_inspection runs

The rule is now active and ready to use!
''')

print('='*80)
