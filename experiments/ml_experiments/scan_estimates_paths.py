"""
Scan filesystem to find where estimates files are actually located.
"""
import pandas as pd
from pathlib import Path
from collections import defaultdict

print('='*80)
print('SCANNING FOR ESTIMATES FILES')
print('='*80)

# Load dataset to get session names
dataset_path = 'ml/results/training_dataset_v9_corrected_iter7.csv'
print(f'\nLoading: {dataset_path}')
df = pd.read_csv(dataset_path)

unique_sessions = df['session_name'].unique()
print(f'Total sessions in dataset: {len(unique_sessions)}')

# Get experiment types
experiments = df.groupby('session_name')['experiment'].first()
exp_counts = experiments.value_counts()

print('\nExperiment breakdown:')
for exp, count in exp_counts.items():
    n_sessions = (experiments == exp).sum()
    print(f'  {exp}: {n_sessions} sessions')

# Define search paths
def get_search_paths(session):
    """Generate all possible paths to check."""
    exp_type = session.split('_')[0]

    paths = []

    # LNOF specific (in timestamped folders with timestamped filenames)
    if exp_type == 'LNOF':
        lnof_dir = Path('data/LNOF')
        if lnof_dir.exists():
            for artifact_dir in lnof_dir.glob(f'inspection_artifacts_{session}_*'):
                # Files are named {session}_{timestamp}_processed.pickle
                paths.extend(list(artifact_dir.glob(f'{session}_*_processed.pickle')))
                paths.extend(list(artifact_dir.glob(f'{session}_*_estimates.pickle')))

    # data/capcan_validation_99_v8 (in capcan_artifacts folders)
    capcan_dir = Path('data/capcan_validation_99_v8')
    if capcan_dir.exists():
        artifact_dir = capcan_dir / f'capcan_artifacts_{session}'
        if artifact_dir.exists():
            paths.append(artifact_dir / f'{session}_processed.pickle')
            paths.append(artifact_dir / f'{session}_estimates.pickle')

    # output folder
    paths.append(Path(f'output/inspection_artifacts_{session}/{session}_processed.pickle'))

    # raw_compressed (with glob to handle parameter suffixes)
    raw_dir = Path('data/raw_compressed')
    if raw_dir.exists():
        patterns = [
            f'{session}_estimates*.pickle',
            f'{session}_raw*.pickle',
            f'{session}_*.pickle',
        ]
        for pattern in patterns:
            files = list(raw_dir.glob(pattern))
            if files:
                paths.append(files[0])
                break

    return paths

# Scan for files
print('\n' + '='*80)
print('SCANNING FILESYSTEM')
print('='*80)

found_by_location = defaultdict(int)
found_sessions = []
missing_sessions = []

print('\nChecking paths for each session...')

for session in unique_sessions[:10]:  # Sample first 10
    found = False
    paths = get_search_paths(session)

    for path in paths:
        if path.exists():
            location = str(path.parent)
            found_by_location[location] += 1
            found_sessions.append(session)
            found = True
            print(f'  FOUND: {session} -> {path}')
            break

    if not found:
        missing_sessions.append(session)
        print(f'  MISSING: {session}')

# Now scan entire dataset
print('\n' + '='*80)
print('FULL SCAN')
print('='*80)

found_by_location_full = defaultdict(int)
found_sessions_full = []
missing_sessions_full = []

for session in unique_sessions:
    found = False
    paths = get_search_paths(session)

    for path in paths:
        if path.exists():
            location = str(path.parent)
            found_by_location_full[location] += 1
            found_sessions_full.append(session)
            found = True
            break

    if not found:
        missing_sessions_full.append(session)

# Summary
print(f'\nSessions found: {len(found_sessions_full)} / {len(unique_sessions)} ({len(found_sessions_full)/len(unique_sessions)*100:.1f}%)')
print(f'Sessions missing: {len(missing_sessions_full)}')

print('\nFiles found by location:')
for location, count in sorted(found_by_location_full.items(), key=lambda x: x[1], reverse=True):
    print(f'  {location}: {count} files')

# Breakdown by experiment
print('\n' + '='*80)
print('BREAKDOWN BY EXPERIMENT')
print('='*80)

found_sessions_set = set(found_sessions_full)
for exp in exp_counts.index:
    exp_sessions = experiments[experiments == exp].index
    found_count = len(set(exp_sessions) & found_sessions_set)
    total_count = len(exp_sessions)
    pct = found_count / total_count * 100 if total_count > 0 else 0
    print(f'{exp:8s}: {found_count:3d} / {total_count:3d} found ({pct:5.1f}%)')

# Sample of missing sessions
if missing_sessions_full:
    print('\n' + '='*80)
    print('SAMPLE OF MISSING SESSIONS')
    print('='*80)
    for session in missing_sessions_full[:20]:
        exp = experiments[session]
        print(f'  {session} ({exp})')

# Estimate neurons coverage
print('\n' + '='*80)
print('NEURON COVERAGE ESTIMATE')
print('='*80)

neurons_in_found_sessions = df[df['session_name'].isin(found_sessions_full)].shape[0]
print(f'Neurons in found sessions: {neurons_in_found_sessions:,} / {len(df):,} ({neurons_in_found_sessions/len(df)*100:.1f}%)')

print('\n' + '='*80)
print('SCAN COMPLETE')
print('='*80)
print('\nReady to run full computation with discovered paths.')
