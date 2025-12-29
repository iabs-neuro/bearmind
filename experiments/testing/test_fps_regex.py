"""Demonstrate the FPS lookup bug for LNOF sessions."""
import re
import pandas as pd

# The current regex pattern from ae_launch.py:49
pattern = r'([A-Z0-9]{3}_[A-Z]\d+_\d[A-Z])'

# Test session
session = 'LNOF_J53_3D'

# Show what the regex extracts
match = re.search(pattern, session)
if match:
    extracted_key = match.group(1)
    print(f"Input session: {session}")
    print(f"Regex pattern: {pattern}")
    print(f"Extracted key: {extracted_key}")
    print()

    # Load CSV and check
    df = pd.read_csv('fps_data.csv', sep=';')

    # Check if extracted key exists
    extracted_lookup = df[df['Filename'] == extracted_key]
    correct_lookup = df[df['Filename'] == session]

    print(f"Lookup with extracted key '{extracted_key}':")
    if len(extracted_lookup) > 0:
        print(f"  Found: {extracted_lookup['FPS'].values[0]} fps")
    else:
        print(f"  NOT FOUND -> returns default 30 fps")

    print()
    print(f"Correct lookup with '{session}':")
    if len(correct_lookup) > 0:
        print(f"  Found: {correct_lookup['FPS'].values[0]} fps (rounds to 20)")
    else:
        print(f"  NOT FOUND")

    print()
    print("BUG ANALYSIS:")
    print(f"  - Pattern requires exactly 3 chars: [A-Z0-9]{{3}}")
    print(f"  - LNOF has 4 chars, so pattern matches starting from 'N'")
    print(f"  - Extracts 'NOF_J53_3D' instead of 'LNOF_J53_3D'")
    print(f"  - Lookup fails, returns default 30 instead of correct 20")
