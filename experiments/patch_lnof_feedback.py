"""
One-time patch to replace older feedback files with latest versions for LNOF_J19_3D and LNOF_J23_3D.
"""
from pathlib import Path
import shutil

LNOF_DIR = Path('data/LNOF')

# Sessions that need feedback update
SESSIONS_TO_PATCH = ['LNOF_J19_3D', 'LNOF_J23_3D']

print('=' * 80)
print('PATCHING LNOF FEEDBACK FILES')
print('=' * 80)
print()

for session_name in SESSIONS_TO_PATCH:
    print(f'Processing {session_name}...')

    # Find latest feedback file in LNOF directory
    feedback_pattern = f'inspection_artifacts_{session_name}__feedback_*.csv'
    feedback_files = list(LNOF_DIR.glob(feedback_pattern))

    if not feedback_files:
        print(f'  [WARNING] No feedback file found for {session_name}')
        continue

    # Sort to get latest (last one chronologically)
    feedback_files_sorted = sorted(feedback_files, key=lambda x: x.name)
    latest_feedback = feedback_files_sorted[-1]

    print(f'  Found latest: {latest_feedback.name}')

    # Find inspection_artifacts folder
    folder_pattern = f'inspection_artifacts_{session_name}_*'
    folders = list(LNOF_DIR.glob(folder_pattern))

    if not folders:
        print(f'  [WARNING] No inspection_artifacts folder found for {session_name}')
        continue

    folder = folders[0]
    target = folder / f'{session_name}_feedback.csv'

    # Replace old feedback with new one
    if target.exists():
        target.unlink()
        print(f'  [REMOVED] Old feedback from folder')

    shutil.copy(str(latest_feedback), str(target))
    print(f'  [COPIED] Latest feedback -> {target.name}')

    # Remove the source file from LNOF directory
    latest_feedback.unlink()
    print(f'  [CLEANED] Removed source file from LNOF directory')

    print()

print('=' * 80)
print('PATCH COMPLETE')
print('=' * 80)
