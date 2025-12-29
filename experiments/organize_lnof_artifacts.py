"""
Organize LNOF inspection artifacts:
1. Move feedback CSV files into their respective session folders
2. Rename all files inside folders to include session name
"""
import os
import re
import shutil
from pathlib import Path

LNOF_DIR = Path('data/LNOF')

def extract_session_name(folder_or_file):
    """Extract session name like LNOF_J01_1D from folder/file name."""
    match = re.search(r'(LNOF_J\d+_\dD)', folder_or_file)
    if match:
        return match.group(1)
    return None

def organize_artifacts():
    """Organize LNOF artifacts."""
    print('=' * 80)
    print('ORGANIZING LNOF INSPECTION ARTIFACTS')
    print('=' * 80)
    print()

    # Get all session folders
    session_folders = sorted([f for f in LNOF_DIR.iterdir()
                             if f.is_dir() and f.name.startswith('inspection_artifacts_LNOF_')])

    print(f'Found {len(session_folders)} session folders')
    print()

    moved_count = 0
    renamed_count = 0

    for folder in session_folders:
        session_name = extract_session_name(folder.name)
        if not session_name:
            print(f'[WARNING] Could not extract session name from: {folder.name}')
            continue

        print(f'Processing {session_name}...')

        # 1. Find and move feedback CSV file into folder
        feedback_pattern = f'inspection_artifacts_{session_name}__feedback_*.csv'
        feedback_files = list(LNOF_DIR.glob(feedback_pattern))

        if feedback_files:
            # If multiple feedback files, keep the LATEST one (by filename timestamp)
            # Feedback files have format: *__feedback_DD-MM-YYYY HH-MM-SS.csv
            # Sort by name (which includes timestamp) to get chronological order
            feedback_files_sorted = sorted(feedback_files, key=lambda x: x.name)
            latest_feedback = feedback_files_sorted[-1]  # Last = latest

            target = folder / f'{session_name}_feedback.csv'

            # Remove existing feedback if present (might be older)
            if target.exists():
                target.unlink()
                print(f'  [REPLACED] Old feedback with newer version')

            # Move latest feedback to folder
            shutil.move(str(latest_feedback), str(target))
            print(f'  [MOVED] {latest_feedback.name} -> {target.name}')
            moved_count += 1

            # Remove any older duplicate feedback files
            for old_feedback in feedback_files_sorted[:-1]:
                if old_feedback.exists():
                    old_feedback.unlink()
                    print(f'  [REMOVED] {old_feedback.name} (older duplicate)')

        # 2. Rename files inside folder to include session name
        for file in folder.iterdir():
            if file.is_file():
                filename = file.name

                # Skip if already has session name
                if session_name in filename:
                    continue

                # Determine new name based on file type
                if filename == 'FBD.npy':
                    new_name = f'{session_name}_FBD.npy'
                elif filename == 'FCD.npy':
                    new_name = f'{session_name}_FCD.npy'
                elif filename == 'match_mtx.npy':
                    new_name = f'{session_name}_match_mtx.npy'
                elif filename == 'match_mtx_crop.npy':
                    new_name = f'{session_name}_match_mtx_crop.npy'
                elif filename == 'metrics_with_decisions.csv':
                    new_name = f'{session_name}_metrics_with_decisions.csv'
                elif filename == 'rejected_neurons.csv':
                    new_name = f'{session_name}_rejected_neurons.csv'
                else:
                    # For any other file, prepend session name
                    new_name = f'{session_name}_{filename}'

                new_path = file.parent / new_name
                if not new_path.exists():
                    file.rename(new_path)
                    print(f'  [RENAMED] {filename} -> {new_name}')
                    renamed_count += 1

    print()
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'Feedback files moved: {moved_count}')
    print(f'Files renamed: {renamed_count}')
    print()

if __name__ == '__main__':
    organize_artifacts()
