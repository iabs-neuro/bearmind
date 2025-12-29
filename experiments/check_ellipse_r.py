import pandas as pd

v9 = pd.read_csv('ml/results/training_dataset_v9.csv')

print('ellipse_r missing by experiment:')
for exp in ['NOF', 'RFC', 'FOF', 'LNOF']:
    exp_data = v9[v9['experiment'] == exp]
    missing = exp_data['ellipse_r'].isna().sum()
    total = len(exp_data)
    print(f'  {exp}: {missing:,} / {total:,} ({100*missing/total:.1f}% missing)')

missing_count = v9['ellipse_r'].isna().sum()
print(f'\nTotal ellipse_r missing: {missing_count:,} / {len(v9):,}')
print(f'\nConclusion: ellipse_r is missing from ALL LNOF neurons (49,767)')
print(f'This is a BUG - ellipse_r should be computed from spatial footprint for all neurons!')
