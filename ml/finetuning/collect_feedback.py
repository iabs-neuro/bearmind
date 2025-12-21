"""
Feedback collection module.

Aggregates feedback CSVs from output/ folder into unified dataset.
Handles multiple feedback files per session, experiment filtering, and date ranges.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import warnings

from .utils import (
    extract_session_name,
    parse_feedback_timestamp,
    get_experiment_from_session,
    validate_feedback_columns,
    standardize_feedback_columns,
    create_y_true_labels,
)


def find_feedback_files(
    output_dir='output',
    experiment_filter=None,
    date_range=None,
    verbose=True
) -> List[Path]:
    """
    Scan output/ directory for feedback CSV files.

    Args:
        output_dir: Root directory to search (default: 'output')
        experiment_filter: Filter by experiment ID(s)
            - None: all experiments
            - str: single experiment (e.g., 'NOF')
            - list: multiple experiments (e.g., ['NOF', 'RFC'])
        date_range: Filter by date range
            - None: all dates
            - tuple: (start_datetime, end_datetime)
        verbose: Print progress messages

    Returns:
        List of Path objects to feedback CSV files, sorted by timestamp
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        if verbose:
            print(f"WARNING: Output directory not found: {output_dir}")
        return []

    # Find all *_feedback_*.csv files
    feedback_files = list(output_path.rglob('*_feedback_*.csv'))

    if verbose:
        print(f"Found {len(feedback_files)} feedback files in {output_dir}")

    # Filter by experiment if requested
    if experiment_filter is not None:
        if isinstance(experiment_filter, str):
            experiment_filter = [experiment_filter]

        filtered_files = []
        for file_path in feedback_files:
            session_name = extract_session_name(str(file_path))
            experiment = get_experiment_from_session(session_name)

            if experiment in experiment_filter:
                filtered_files.append(file_path)

        feedback_files = filtered_files

        if verbose:
            print(f"After experiment filter ({experiment_filter}): {len(feedback_files)} files")

    # Filter by date range if requested
    if date_range is not None:
        start_date, end_date = date_range
        filtered_files = []

        for file_path in feedback_files:
            timestamp = parse_feedback_timestamp(file_path.name)

            if timestamp is not None:
                if start_date <= timestamp <= end_date:
                    filtered_files.append(file_path)

        feedback_files = filtered_files

        if verbose:
            print(f"After date range filter: {len(feedback_files)} files")

    # Sort by timestamp (oldest to newest)
    files_with_timestamps = []
    for file_path in feedback_files:
        timestamp = parse_feedback_timestamp(file_path.name)
        files_with_timestamps.append((file_path, timestamp))

    # Sort (None timestamps go to end)
    files_with_timestamps.sort(key=lambda x: x[1] if x[1] is not None else datetime.max)

    sorted_files = [f[0] for f in files_with_timestamps]

    return sorted_files


def load_and_merge_feedback(
    feedback_files: List[Path],
    handle_duplicates='latest',
    verbose=True
) -> pd.DataFrame:
    """
    Load feedback CSVs and merge into single DataFrame.

    Args:
        feedback_files: List of Path objects to feedback CSV files
        handle_duplicates: How to handle multiple feedback files for same session
            - 'latest': Keep feedback file with most recent timestamp (DEFAULT)
              Use when: Users refined their judgment over time
              Trade-off: May discard valuable early feedback if user changed mind

            - 'largest': Keep feedback file with most neuron entries
              Use when: Users incrementally added feedback across multiple sessions
              Trade-off: May use stale feedback if user revisited and corrected

        verbose: Print progress messages

    Returns:
        DataFrame with columns:
          - neuron_idx, session_name, feedback_type, timestamp,
            ml_keep_probability, delete, + all available feature columns
          - y_true: Binary labels (1=KEEP for FN, 0=DELETE for FP)
    """
    if not feedback_files:
        if verbose:
            print("WARNING: No feedback files to load")
        return pd.DataFrame()

    # Group files by session
    session_files = {}
    for file_path in feedback_files:
        session_name = extract_session_name(str(file_path))

        if not session_name:
            if verbose:
                print(f"WARNING: Could not extract session name from {file_path.name}, skipping")
            continue

        if session_name not in session_files:
            session_files[session_name] = []

        session_files[session_name].append(file_path)

    if verbose:
        print(f"Found feedback for {len(session_files)} unique sessions")

    # Select one file per session based on handle_duplicates strategy
    selected_files = {}

    for session_name, files in session_files.items():
        if len(files) == 1:
            selected_files[session_name] = files[0]
            continue

        if handle_duplicates == 'latest':
            # Sort by timestamp, take last
            files_with_ts = [(f, parse_feedback_timestamp(f.name)) for f in files]
            files_with_ts.sort(key=lambda x: x[1] if x[1] is not None else datetime.min)
            selected_files[session_name] = files_with_ts[-1][0]

            if verbose:
                print(f"Session {session_name}: Selected latest of {len(files)} files ({files_with_ts[-1][0].name})")

        elif handle_duplicates == 'largest':
            # Load each file, count rows, take largest
            file_sizes = []
            for f in files:
                try:
                    df_temp = pd.read_csv(f)
                    file_sizes.append((f, len(df_temp)))
                except Exception as e:
                    if verbose:
                        print(f"WARNING: Could not read {f.name}: {e}")
                    file_sizes.append((f, 0))

            file_sizes.sort(key=lambda x: x[1], reverse=True)
            selected_files[session_name] = file_sizes[0][0]

            if verbose:
                print(f"Session {session_name}: Selected largest of {len(files)} files ({file_sizes[0][1]} rows, {file_sizes[0][0].name})")

        else:
            raise ValueError(f"Unknown handle_duplicates mode: {handle_duplicates}. Use 'latest' or 'largest'.")

    # Load selected files
    feedback_dfs = []

    for session_name, file_path in selected_files.items():
        try:
            df = pd.read_csv(file_path)

            # Standardize column names (handle legacy formats)
            df = standardize_feedback_columns(df)

            # Validate required columns
            is_valid, missing = validate_feedback_columns(df)
            if not is_valid:
                if verbose:
                    print(f"WARNING: {file_path.name} missing required columns: {missing}, skipping")
                continue

            # Add session_name if not present
            if 'session_name' not in df.columns:
                df['session_name'] = session_name

            feedback_dfs.append(df)

            if verbose:
                print(f"Loaded {len(df)} feedback samples from {file_path.name}")

        except Exception as e:
            if verbose:
                print(f"ERROR loading {file_path.name}: {e}")
            continue

    if not feedback_dfs:
        if verbose:
            print("WARNING: No valid feedback data loaded")
        return pd.DataFrame()

    # Merge all dataframes
    feedback_merged = pd.concat(feedback_dfs, ignore_index=True)

    if verbose:
        print(f"\nTotal feedback samples loaded: {len(feedback_merged)}")

    # Within each session, handle per-neuron duplicates (keep latest timestamp)
    if 'timestamp' in feedback_merged.columns:
        # Convert timestamp strings to datetime for sorting
        def parse_ts(ts_str):
            try:
                return datetime.strptime(ts_str, '%d-%m-%Y %H:%M:%S')
            except:
                return datetime.min

        feedback_merged['timestamp_dt'] = feedback_merged['timestamp'].apply(parse_ts)
        feedback_merged = feedback_merged.sort_values('timestamp_dt')

        # Keep last (latest) entry for each (session_name, neuron_idx)
        feedback_merged = feedback_merged.drop_duplicates(
            subset=['session_name', 'neuron_idx'],
            keep='last'
        )

        feedback_merged = feedback_merged.drop('timestamp_dt', axis=1)

        if verbose:
            print(f"After removing per-neuron duplicates: {len(feedback_merged)} samples")

    # Create y_true labels (1=KEEP for FN, 0=DELETE for FP)
    feedback_merged = create_y_true_labels(feedback_merged)

    # Report summary
    if verbose:
        print(f"\nFeedback Summary:")
        print(f"  FP count: {(feedback_merged['feedback_type'] == 'FP').sum()}")
        print(f"  FN count: {(feedback_merged['feedback_type'] == 'FN').sum()}")
        print(f"  Unique sessions: {feedback_merged['session_name'].nunique()}")

    return feedback_merged


def collect_feedback(
    output_dir='output',
    experiment_filter=None,
    date_range=None,
    handle_duplicates='latest',
    verbose=True
) -> pd.DataFrame:
    """
    Main entry point: Find and merge feedback CSVs.

    Args:
        output_dir: Root directory to search
        experiment_filter: Filter by experiment ID(s)
        date_range: Filter by date range (tuple of datetimes)
        handle_duplicates: 'latest' or 'largest'
        verbose: Print progress

    Returns:
        Merged feedback DataFrame
    """
    # Find feedback files
    feedback_files = find_feedback_files(
        output_dir=output_dir,
        experiment_filter=experiment_filter,
        date_range=date_range,
        verbose=verbose
    )

    if not feedback_files:
        return pd.DataFrame()

    # Load and merge
    feedback_df = load_and_merge_feedback(
        feedback_files=feedback_files,
        handle_duplicates=handle_duplicates,
        verbose=verbose
    )

    return feedback_df


# Command-line interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Collect feedback CSVs from output/ folder')
    parser.add_argument('--output-dir', default='output', help='Output directory to scan')
    parser.add_argument('--experiment-filter', nargs='+', help='Experiment IDs to include (e.g., NOF RFC)')
    parser.add_argument('--handle-duplicates', default='latest', choices=['latest', 'largest'],
                        help='How to handle multiple feedback files per session')
    parser.add_argument('--save-to', help='Save merged feedback to CSV file')

    args = parser.parse_args()

    # Collect feedback
    feedback_df = collect_feedback(
        output_dir=args.output_dir,
        experiment_filter=args.experiment_filter,
        handle_duplicates=args.handle_duplicates,
        verbose=True
    )

    # Save if requested
    if args.save_to and not feedback_df.empty:
        feedback_df.to_csv(args.save_to, index=False)
        print(f"\nSaved {len(feedback_df)} feedback samples to {args.save_to}")

    # Display sample
    if not feedback_df.empty:
        print(f"\nSample of collected feedback:")
        print(feedback_df[['neuron_idx', 'session_name', 'feedback_type', 'ml_keep_probability', 'y_true']].head(10))
