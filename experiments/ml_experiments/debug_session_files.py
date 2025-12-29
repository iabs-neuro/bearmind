"""
Debug script to understand session name to file mapping.
"""
import pandas as pd
from pathlib import Path
import re

# Load top100 FP to see session names
df_fp = pd.read_csv('ml/ebm_v9_iter1/top100_fp.csv')

print("Session names in top100_fp.csv:")
print(df_fp['session_name'].unique()[:10])

print("\nRaw compressed files:")
raw_dir = Path('data/raw_compressed')
pickle_files = sorted(raw_dir.glob('*.pickle'))

print(f"Total pickle files: {len(pickle_files)}")
print("\nFirst 10 files:")
for f in pickle_files[:10]:
    print(f"  {f.name}")

# Try to understand mapping
print("\nTrying to match sessions to files:")
for session in df_fp['session_name'].unique()[:5]:
    print(f"\nSession: {session}")

    # Try exact match
    exact = list(raw_dir.glob(f'{session}*.pickle'))
    if exact:
        print(f"  EXACT MATCH: {exact[0].name}")
        continue

    # Try without trailing _1D/_2D etc
    base = session.rsplit('_', 1)[0] if '_' in session else session
    base_match = list(raw_dir.glob(f'{base}*.pickle'))
    if base_match:
        print(f"  BASE MATCH ({base}): {[f.name for f in base_match[:3]]}")
        continue

    # Try pattern matching
    parts = session.split('_')
    for i in range(len(parts), 0, -1):
        pattern = '_'.join(parts[:i])
        pattern_match = list(raw_dir.glob(f'{pattern}*.pickle'))
        if pattern_match:
            print(f"  PATTERN MATCH ({pattern}): {[f.name for f in pattern_match[:3]]}")
            break
    else:
        print(f"  NO MATCH FOUND")

print("\nChecking specific missing sessions:")
missing = ['LNOF_J61_2D', 'LNOF_J52_1D', 'LNOF_J01_2D']
for session in missing:
    matches = list(raw_dir.glob(f'*{session}*.pickle'))
    if matches:
        print(f"{session}: FOUND {len(matches)} matches")
        for m in matches[:2]:
            print(f"  {m.name}")
    else:
        # Try searching for components
        parts = session.split('_')
        for part in parts:
            partial = list(raw_dir.glob(f'*{part}*.pickle'))
            if partial:
                print(f"{session}: Partial match on '{part}': {len(partial)} files")
                break
        else:
            print(f"{session}: NO MATCHES")
