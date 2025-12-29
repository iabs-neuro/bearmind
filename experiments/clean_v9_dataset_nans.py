"""
Clean v9 dataset by handling NaNs in event-based metrics.

This script:
1. Identifies neurons with NaN event-based metrics (227 neurons, 0.25% of dataset)
2. Replaces NaNs with zeros for all event-based metrics
3. Optionally removes these neurons from the dataset
4. Saves cleaned dataset as training_dataset_v9_cleaned.csv

Background:
- 177 neurons had complete DRIADA processing failures (kinetics_source='error')
- 50 neurons had partial failures (event detection OK, reconstruction metrics failed)
- These NaNs come from auto_inspector.py exception handling when processing fails
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Event-based metrics that can have NaNs
EVENT_METRICS = [
    'event_r2_score', 'event_snr', 'events_fraction', 'events_per_min',
    'kinetics_opt', 't_off', 't_rise', 'nmae', 'nrmse', 'r2_score', 'snr_recon'
]


def clean_v9_dataset(remove_failed=False, input_path='ml/results/training_dataset_v9.csv',
                     output_path='ml/results/training_dataset_v9_cleaned.csv'):
    """
    Clean v9 dataset by handling NaN event metrics.

    Args:
        remove_failed: If True, remove neurons with NaN event metrics.
                      If False, replace NaNs with zeros and keep neurons.
        input_path: Path to input v9 dataset
        output_path: Path to save cleaned dataset
    """
    print('='*80)
    print('CLEANING V9 DATASET - NaN HANDLING')
    print('='*80)

    # Load dataset
    print(f'\nLoading: {input_path}')
    df = pd.read_csv(input_path)
    print(f'  Total neurons: {len(df):,}')
    print(f'  Total columns: {len(df.columns)}')

    # Identify neurons with NaN event metrics
    has_nan_mask = df[EVENT_METRICS].isna().any(axis=1)
    nan_count = has_nan_mask.sum()

    print(f'\n' + '='*80)
    print('NaN DETECTION')
    print('='*80)
    print(f'Neurons with NaN event metrics: {nan_count} ({100*nan_count/len(df):.2f}%)')

    # Breakdown
    complete_failures = df['kinetics_source'] == 'error'
    partial_failures = has_nan_mask & ~complete_failures

    print(f'\nBreakdown:')
    print(f'  Complete failures (all metrics NaN): {complete_failures.sum()}')
    print(f'  Partial failures (some metrics NaN): {partial_failures.sum()}')

    # Ground truth distribution
    print(f'\nGround truth of NaN neurons:')
    print(f'  KEEP (ground_truth=1): {df[has_nan_mask]["ground_truth"].sum()}')
    print(f'  DELETE (ground_truth=0): {(1-df[has_nan_mask]["ground_truth"]).sum()}')

    # Process dataset
    print(f'\n' + '='*80)
    if remove_failed:
        print('MODE: REMOVE FAILED NEURONS')
    else:
        print('MODE: REPLACE NaN WITH ZEROS')
    print('='*80)

    if remove_failed:
        # Remove neurons with NaN event metrics
        df_cleaned = df[~has_nan_mask].copy()
        removed_count = nan_count

        print(f'\nRemoving {removed_count} neurons with NaN event metrics...')
        print(f'  Original size: {len(df):,}')
        print(f'  Cleaned size: {len(df_cleaned):,}')
        print(f'  Removed: {removed_count} ({100*removed_count/len(df):.2f}%)')

    else:
        # Replace NaNs with zeros
        df_cleaned = df.copy()

        print(f'\nReplacing NaNs with zeros for event-based metrics...')
        nan_counts_before = df_cleaned[EVENT_METRICS].isna().sum()

        for metric in EVENT_METRICS:
            nan_before = df_cleaned[metric].isna().sum()
            if nan_before > 0:
                df_cleaned[metric] = df_cleaned[metric].fillna(0)
                print(f'  {metric}: replaced {nan_before} NaNs with 0')

        # Verify no more NaNs in event metrics
        nan_counts_after = df_cleaned[EVENT_METRICS].isna().sum()
        total_nans_after = nan_counts_after.sum()

        if total_nans_after == 0:
            print(f'\n[SUCCESS] All NaNs in event metrics replaced with zeros')
        else:
            print(f'\n[WARNING] Still have {total_nans_after} NaNs in event metrics')

    # Quality check
    print(f'\n' + '='*80)
    print('QUALITY CHECK')
    print('='*80)

    # Check for remaining NaNs in event metrics
    remaining_nans = df_cleaned[EVENT_METRICS].isna().sum().sum()
    print(f'\nRemaining NaNs in event metrics: {remaining_nans}')

    # Check ground truth distribution
    gt_dist = df_cleaned['ground_truth'].value_counts()
    keep_count = gt_dist.get(1, 0)
    delete_count = gt_dist.get(0, 0)
    keep_pct = 100 * keep_count / len(df_cleaned)

    print(f'\nGround truth distribution:')
    print(f'  KEEP (ground_truth=1): {keep_count:,} ({keep_pct:.2f}%)')
    print(f'  DELETE (ground_truth=0): {delete_count:,} ({100-keep_pct:.2f}%)')

    # Session count
    print(f'\nTotal sessions: {df_cleaned["session_name"].nunique()}')

    # Save
    print(f'\n' + '='*80)
    print('SAVING')
    print('='*80)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df_cleaned.to_csv(output_path, index=False)

    print(f'\nSaved: {output_path}')
    print(f'  Total neurons: {len(df_cleaned):,}')
    print(f'  Total sessions: {df_cleaned["session_name"].nunique()}')
    print(f'  Total columns: {len(df_cleaned.columns)}')

    # Summary
    print(f'\n' + '='*80)
    print('SUMMARY')
    print('='*80)

    if remove_failed:
        print(f'\nRemoved {removed_count} neurons with failed event metrics')
        print(f'Cleaned dataset: {len(df_cleaned):,} neurons ({100*len(df_cleaned)/len(df):.2f}% of original)')
    else:
        print(f'\nReplaced NaNs with zeros in {nan_count} neurons')
        print(f'All {len(df_cleaned):,} neurons retained')

    print(f'\nDataset ready for training: {output_path}')
    print('\n' + '='*80)
    print('COMPLETE')
    print('='*80)

    return df_cleaned


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clean v9 dataset by handling NaN event metrics'
    )
    parser.add_argument(
        '--remove-failed',
        action='store_true',
        help='Remove neurons with NaN event metrics instead of replacing with zeros'
    )
    parser.add_argument(
        '--input',
        default='ml/results/training_dataset_v9.csv',
        help='Input dataset path (default: ml/results/training_dataset_v9.csv)'
    )
    parser.add_argument(
        '--output',
        default='ml/results/training_dataset_v9_cleaned.csv',
        help='Output dataset path (default: ml/results/training_dataset_v9_cleaned.csv)'
    )

    args = parser.parse_args()

    clean_v9_dataset(
        remove_failed=args.remove_failed,
        input_path=args.input,
        output_path=args.output
    )
