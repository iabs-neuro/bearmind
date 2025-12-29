"""Test FPS lookup for LNOF_J53_3D to verify the bug."""
from ae_launch import get_fps_from_table

session = 'LNOF_J53_3D'
result = get_fps_from_table(session)

print(f"Session: {session}")
print(f"Result: {result} fps")
print(f"Expected: 20 fps (from 19.76 in CSV)")
print()
if result == 30:
    print("BUG CONFIRMED: Returns default 30 instead of 20")
    print("Root cause: Regex extracts 'NOF_J53_3D' instead of 'LNOF_J53_3D'")
elif result == 20:
    print("WORKING: Correctly returns 20")
else:
    print(f"UNEXPECTED: Got {result}, expected 20")
