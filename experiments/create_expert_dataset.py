"""
Create expert-labeled dataset from inspection artifacts.

For each experiment (e.g., LNOF):
1. Concatenate all session metrics_with_decisions.csv
2. Apply expert feedback corrections:
   - FP (False Positive) = model should DELETE (delete=1)
   - FN (False Negative) = model should KEEP (delete=0)
3. Create <exp>_dataset.csv with corrected decisions
4. Create <exp>_feedback.csv with all expert feedback

Sessions without feedback = all model decisions are correct.
"""
import pandas as pd
from pathlib import Path
import argparse
import re

def extract_session_name(folder_name):
    """Extract session name like LNOF_J01_1D from folder name."""
    match = re.search(r'([A-Z0-9]+_[A-Z]\d+_\dD)', folder_name)
    return match.group(1) if match else None

def create_expert_dataset(experiment, data_path=None):
    """
    Create expert-labeled dataset for an experiment.

    Args:
        experiment: Experiment code (e.g., 'LNOF', 'NOF', 'FOF')
        data_path: Path to data folder (default: data/<experiment>)

    Returns:
        Tuple of (dataset_df, feedback_df)
    """
    if data_path is None:
        data_path = Path('data') / experiment
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    print('=' * 80)
    print(f'CREATING EXPERT DATASET FOR {experiment}')
    print('=' * 80)
    print(f'Data path: {data_path}')
    print()

    # Find all inspection_artifacts folders
    artifact_folders = sorted([f for f in data_path.iterdir()
                              if f.is_dir() and f.name.startswith(f'inspection_artifacts_{experiment}')])

    print(f'Found {len(artifact_folders)} session folders')
    print()

    all_sessions = []
    all_feedback = []

    sessions_with_feedback = 0
    sessions_without_feedback = 0
    total_corrections = 0
    fp_corrections = 0
    fn_corrections = 0

    for folder in artifact_folders:
        session_name = extract_session_name(folder.name)
        if not session_name:
            print(f'[WARNING] Could not extract session name from: {folder.name}')
            continue

        # Load metrics_with_decisions.csv
        metrics_pattern = f'{session_name}_metrics_with_decisions.csv'
        metrics_file = folder / metrics_pattern

        if not metrics_file.exists():
            print(f'[WARNING] Metrics file not found for {session_name}: {metrics_file}')
            continue

        df = pd.read_csv(metrics_file)

        # Add session identifier
        df['session_name'] = session_name

        # Check for feedback file
        feedback_file = folder / f'{session_name}_feedback.csv'

        if feedback_file.exists():
            sessions_with_feedback += 1
            feedback_df = pd.read_csv(feedback_file)

            # Add session name to feedback if not present
            if 'session_name' not in feedback_df.columns:
                feedback_df['session_name'] = session_name

            all_feedback.append(feedback_df)

            # Apply expert corrections
            session_corrections = 0
            session_fp = 0
            session_fn = 0

            # Determine which column to use for neuron index in metrics df
            # Could be 'component_idx', 'idx', or 'neuron_idx'
            idx_col = None
            for col_name in ['component_idx', 'idx', 'neuron_idx']:
                if col_name in df.columns:
                    idx_col = col_name
                    break

            if idx_col is None:
                print(f'[WARNING] Could not find index column in {session_name}')
                continue

            for _, fb_row in feedback_df.iterrows():
                # Get neuron index from feedback (could be 'neuron_idx' or 'idx')
                neuron_idx = fb_row.get('neuron_idx', fb_row.get('idx', None))
                if neuron_idx is None:
                    continue

                feedback_type = fb_row['feedback_type']

                # Find neuron in metrics
                mask = df[idx_col] == neuron_idx

                if mask.any():
                    if feedback_type == 'FP':
                        # False Positive: model should DELETE
                        df.loc[mask, 'delete'] = 1
                        session_fp += 1
                        session_corrections += 1
                    elif feedback_type == 'FN':
                        # False Negative: model should KEEP
                        df.loc[mask, 'delete'] = 0
                        session_fn += 1
                        session_corrections += 1

            total_corrections += session_corrections
            fp_corrections += session_fp
            fn_corrections += session_fn

            print(f'[{session_name}] Feedback: {len(feedback_df)} neurons '
                  f'({session_fp} FP, {session_fn} FN)')
        else:
            sessions_without_feedback += 1
            print(f'[{session_name}] No feedback (model decisions correct)')

        all_sessions.append(df)

    print()
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'Total sessions processed: {len(all_sessions)}')
    print(f'  With feedback: {sessions_with_feedback}')
    print(f'  Without feedback: {sessions_without_feedback}')
    print()
    print(f'Expert corrections applied: {total_corrections}')
    print(f'  FP corrections (model -> DELETE): {fp_corrections}')
    print(f'  FN corrections (model -> KEEP): {fn_corrections}')
    print()

    # Concatenate all sessions
    if all_sessions:
        dataset_df = pd.concat(all_sessions, ignore_index=True)
        print(f'Final dataset: {len(dataset_df)} neurons across {len(all_sessions)} sessions')
    else:
        dataset_df = pd.DataFrame()
        print('[ERROR] No sessions were processed')

    # Concatenate all feedback
    if all_feedback:
        feedback_df = pd.concat(all_feedback, ignore_index=True)
        print(f'Final feedback: {len(feedback_df)} expert-labeled neurons')
    else:
        feedback_df = pd.DataFrame()
        print('No feedback data collected')

    print()

    return dataset_df, feedback_df

def main():
    parser = argparse.ArgumentParser(
        description='Create expert-labeled dataset from inspection artifacts'
    )
    parser.add_argument(
        'experiment',
        type=str,
        help='Experiment code (e.g., LNOF, NOF, FOF, RFC)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to data folder (default: data/<experiment>)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for dataset files (default: current directory)'
    )

    args = parser.parse_args()

    # Create dataset
    dataset_df, feedback_df = create_expert_dataset(args.experiment, args.data_path)

    # Save dataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    dataset_file = output_dir / f'{args.experiment}_dataset.csv'
    feedback_file = output_dir / f'{args.experiment}_feedback.csv'

    if not dataset_df.empty:
        dataset_df.to_csv(dataset_file, index=False)
        print(f'Saved dataset: {dataset_file}')
        print(f'  Total neurons: {len(dataset_df)}')
        print(f'  Sessions: {dataset_df["session_name"].nunique()}')
        print(f'  Columns: {len(dataset_df.columns)}')

        # Summary statistics
        if 'delete' in dataset_df.columns:
            keep_count = (dataset_df['delete'] == 0).sum()
            delete_count = (dataset_df['delete'] == 1).sum()
            print(f'  Decisions: KEEP={keep_count}, DELETE={delete_count}')

    if not feedback_df.empty:
        feedback_df.to_csv(feedback_file, index=False)
        print()
        print(f'Saved feedback: {feedback_file}')
        print(f'  Total feedback: {len(feedback_df)}')
        print(f'  Sessions: {feedback_df["session_name"].nunique()}')

        # Feedback breakdown
        if 'feedback_type' in feedback_df.columns:
            fp_count = (feedback_df['feedback_type'] == 'FP').sum()
            fn_count = (feedback_df['feedback_type'] == 'FN').sum()
            print(f'  FP (model -> DELETE): {fp_count}')
            print(f'  FN (model -> KEEP): {fn_count}')

    print()
    print('=' * 80)
    print('COMPLETE')
    print('=' * 80)

if __name__ == '__main__':
    main()
