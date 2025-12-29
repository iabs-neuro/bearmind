from pathlib import Path

session = 'LNOF_J01_1D'
lnof_dir = Path('data/LNOF')

print(f'Looking for: {session}')
print(f'Glob pattern: inspection_artifacts_{session}_*')

matches = list(lnof_dir.glob(f'inspection_artifacts_{session}_*'))
print(f'Found {len(matches)} matches:')
for m in matches:
    print(f'  {m}')

    processed = m / f'{session}_processed.pickle'
    print(f'    Checking: {processed}')
    print(f'    Exists: {processed.exists()}')

    estimates = m / f'{session}_estimates.pickle'
    print(f'    Checking: {estimates}')
    print(f'    Exists: {estimates.exists()}')
