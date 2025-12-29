"""
Generate correction suggestions based on error analysis.
Helps create the corrections list for v8_corrected dataset.
"""
import pandas as pd

print('='*80)
print('GENERATING CORRECTION SUGGESTIONS')
print('='*80)

# Load FN and FP reports
df_fn = pd.read_csv('ml/results/v8_top100_false_negatives.csv')
df_fp = pd.read_csv('ml/results/v8_top100_false_positives.csv')

print(f'\nLoaded error reports:')
print(f'  False Negatives: {len(df_fn)} (model says DELETE, GT says KEEP)')
print(f'  False Positives: {len(df_fp)} (model says KEEP, GT says DELETE)')

# Based on user feedback: FN #3, #6, #8, #15, #37, #55, #58, #68, #79, #100 are REAL
# The rest of the FN are likely GT errors (should be DELETE)

real_fn_indices = [3, 6, 8, 15, 37, 55, 58, 68, 79, 100]  # 1-indexed
fake_fn_indices = [i for i in range(1, 101) if i not in real_fn_indices]

print(f'\n{"="*80}')
print('FALSE NEGATIVE CORRECTIONS')
print('='*80)

print(f'\nBased on your feedback:')
print(f'  REAL FN (model wrong, GT correct): {len(real_fn_indices)} neurons')
print(f'  FAKE FN (model correct, GT wrong): {len(fake_fn_indices)} neurons')

print(f'\nCorrections to apply (FAKE FN - change KEEP to DELETE):')
print(f'# Copy these lines into create_v8_corrected_dataset.py corrections list:')
print()

for idx in fake_fn_indices:
    row = df_fn.iloc[idx - 1]
    session = row['session']
    comp_idx = int(row['component_idx'])
    prob = row['y_proba']
    print(f"    ('{session}', {comp_idx}, 0, 'FN #{idx} - model correct to DELETE, prob={prob:.3f}'),")

print(f'\n# Total: {len(fake_fn_indices)} corrections to flip KEEP -> DELETE')

# For FALSE POSITIVES, we need user review
# Let's generate a template for them to fill in

print(f'\n{"="*80}')
print('FALSE POSITIVE REVIEW NEEDED')
print('='*80)

print(f'\nReview FP visualizations and mark which are REAL FP (model wrong):')
print(f'  - If model should DELETE (GT correct), NO correction needed')
print(f'  - If model is RIGHT to KEEP (GT wrong), add correction to flip DELETE -> KEEP')
print()
print(f'# Example template - review and uncomment REAL FP:')
print()

for idx in range(1, min(21, len(df_fp) + 1)):  # First 20 FP
    row = df_fp.iloc[idx - 1]
    session = row['session']
    comp_idx = int(row['component_idx'])
    prob = row['y_proba']
    print(f"    # ('{session}', {comp_idx}, 1, 'FP #{idx} - model correct to KEEP, prob={prob:.3f}'),")

print(f'\n{"="*80}')
print('SUMMARY')
print('='*80)

print(f'\nRecommended workflow:')
print(f'  1. Copy the {len(fake_fn_indices)} FAKE FN corrections above')
print(f'  2. Paste into create_v8_corrected_dataset.py corrections list')
print(f'  3. Review FP visualizations to identify any FAKE FP (model right, GT wrong)')
print(f'  4. Add FAKE FP corrections (flip DELETE -> KEEP)')
print(f'  5. Run create_v8_corrected_dataset.py to generate corrected dataset')
print(f'  6. Retrain model on corrected dataset')

# Save as a file for easy copy-paste
with open('ml/results/suggested_corrections.txt', 'w') as f:
    f.write('# Suggested corrections for v8_corrected dataset\n')
    f.write('# Based on FN analysis\n\n')
    f.write('corrections = [\n')

    for idx in fake_fn_indices:
        row = df_fn.iloc[idx - 1]
        session = row['session']
        comp_idx = int(row['component_idx'])
        prob = row['y_proba']
        f.write(f"    ('{session}', {comp_idx}, 0, 'FN #{idx} - model correct to DELETE, prob={prob:.3f}'),\n")

    f.write('\n    # Review FP and add corrections here:\n')
    for idx in range(1, min(21, len(df_fp) + 1)):
        row = df_fp.iloc[idx - 1]
        session = row['session']
        comp_idx = int(row['component_idx'])
        prob = row['y_proba']
        f.write(f"    # ('{session}', {comp_idx}, 1, 'FP #{idx} - REVIEW: model says KEEP prob={prob:.3f}'),\n")

    f.write(']\n')

print(f'\nSuggested corrections saved to: ml/results/suggested_corrections.txt')
print(f'\n{"="*80}')
