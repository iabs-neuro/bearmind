"""Test the proposed fix for FPS lookup regex."""
import re
import pandas as pd

# Current BROKEN pattern
pattern_old = r'([A-Z0-9]{3}_[A-Z]\d+_\d[A-Z])'

# FIXED pattern (supports 3-4 char codes)
pattern_new = r'([A-Z0-9]{3,4}_[A-Z]\d+_\d[A-Z])'

# Test cases from actual data
test_cases = [
    'NOF_H01_1D',    # 3-char code
    'FOF_F05_1D',    # 3-char code
    'RFC_F01_1D',    # 3-char code
    '3DM_D17_1D',    # 3-char code (numeric)
    'LNOF_J53_3D',   # 4-char code (BUG CASE)
    'LNOF_J01_1D',   # 4-char code
]

print("=" * 80)
print("FPS REGEX FIX VERIFICATION")
print("=" * 80)
print()

# Load CSV
df = pd.read_csv('fps_data.csv', sep=';')

for session in test_cases:
    # Old pattern
    match_old = re.search(pattern_old, session)
    key_old = match_old.group(1) if match_old else 'NO MATCH'

    # New pattern
    match_new = re.search(pattern_new, session)
    key_new = match_new.group(1) if match_new else 'NO MATCH'

    # Lookup
    actual_fps = df[df['Filename'] == session]['FPS'].values[0] if session in df['Filename'].values else None

    print(f"Session: {session}")
    print(f"  Old pattern: {key_old}")
    print(f"  New pattern: {key_new}")
    print(f"  Actual FPS in CSV: {actual_fps}")

    if key_old != key_new:
        print(f"  [FIX APPLIED] Old extracted wrong key!")

    if key_new == session:
        print(f"  [SUCCESS] New pattern extracts correct key")
    else:
        print(f"  [ERROR] Pattern still broken!")

    print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("Fixed pattern supports both 3-char (NOF, FOF, RFC, 3DM) and 4-char (LNOF) codes")
print("All test cases should show [SUCCESS] with new pattern")
