import pandas as pd

v9 = pd.read_csv('ml/results/training_dataset_v9.csv')

print('Checking for half_crossing_rate column...\n')

if 'half_crossing_rate' in v9.columns:
    print('half_crossing_rate column EXISTS')
    print('\nhalf_crossing_rate missing by experiment:')
    for exp in ['NOF', 'RFC', 'FOF', 'LNOF']:
        exp_data = v9[v9['experiment'] == exp]
        missing = exp_data['half_crossing_rate'].isna().sum()
        total = len(exp_data)
        print(f'  {exp}: {missing:,} / {total:,} ({100*missing/total:.1f}% missing)')

    total_missing = v9['half_crossing_rate'].isna().sum()
    print(f'\nTotal half_crossing_rate missing: {total_missing:,} / {len(v9):,}')
else:
    print('half_crossing_rate column does NOT exist in v9 dataset')
    print('This column needs to be added!')
