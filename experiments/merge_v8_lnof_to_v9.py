"""
Merge v8_iter5 and LNOF datasets to create v9 training dataset.

This script:
1. Loads v8_iter5 (42,475 neurons, 99 sessions) and LNOF_from_processed (49,767 neurons, 88 sessions)
2. Aligns schemas (adds missing columns with appropriate defaults)
3. Concatenates into unified v9 dataset
4. Runs quality checks
5. Generates summary report
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse

# Expected feature columns (35 features)
EXPECTED_FEATURES = [
    'area', 'aspect_ratio', 'baseline', 'baseline_drift', 'bimodality',
    'caiman_r_score', 'caiman_snr', 'circularity', 'convexity', 'eccentricity',
    'edge_distance', 'ellipse_r', 'event_r2_score', 'event_snr', 'events_fraction',
    'events_per_min', 'footprint_compactness', 'hurst_exponent', 'kinetics_opt',
    'local_density', 'max_edge', 'mean_time_at_peak', 'nmae', 'nn_distance_center',
    'noise_level', 'nrmse', 'peak_amplitude_cv', 'r2_score', 'snr_recon',
    't_off', 't_rise', 'tau_decay', 'trace_kurtosis', 'trace_skewness',
    'half_crossing_rate', 'kinetics_source'
]

def merge_datasets(test_mode=False):
    """Merge v8_iter5 and LNOF datasets."""
    print('='*80)
    print('MERGING v8_iter5 AND LNOF TO CREATE v9')
    print('='*80)
    if test_mode:
        print('[TEST MODE] Using subset of sessions')
    print()

    # Step 1: Load datasets
    print('STEP 1: LOADING DATASETS')
    print('-'*80)

    v8_path = 'ml/results/training_dataset_v8_corrected_iter5.csv'
    lnof_path = 'LNOF_dataset_from_processed.csv'

    v8_df = pd.read_csv(v8_path)
    lnof_df = pd.read_csv(lnof_path)

    print(f'v8_iter5:')
    print(f'  Neurons: {len(v8_df):,}')
    print(f'  Sessions: {v8_df["session"].nunique()}')
    print(f'  Columns: {len(v8_df.columns)}')

    print(f'\nLNOF:')
    print(f'  Neurons: {len(lnof_df):,}')
    print(f'  Sessions: {lnof_df["session_name"].nunique()}')
    print(f'  Columns: {len(lnof_df.columns)}')

    # Test mode: use subset
    if test_mode:
        v8_sessions = list(v8_df['session'].unique()[:5])
        lnof_sessions = list(lnof_df['session_name'].unique()[:5])

        v8_df = v8_df[v8_df['session'].isin(v8_sessions)]
        lnof_df = lnof_df[lnof_df['session_name'].isin(lnof_sessions)]

        print(f'\n[TEST MODE] Using:')
        print(f'  v8: {len(v8_df)} neurons from {len(v8_sessions)} sessions')
        print(f'  LNOF: {len(lnof_df)} neurons from {len(lnof_sessions)} sessions')

    # Step 2: Verify no session overlap
    print('\n' + '='*80)
    print('STEP 2: VERIFY NO SESSION OVERLAP')
    print('-'*80)

    v8_sessions = set(v8_df['session'].unique())
    lnof_sessions = set(lnof_df['session_name'].unique())

    overlap = v8_sessions & lnof_sessions
    if overlap:
        raise ValueError(f'Session overlap detected: {overlap}')

    print(f'[OK] No overlap (v8: {len(v8_sessions)} sessions, LNOF: {len(lnof_sessions)} sessions)')

    # Step 3: Schema alignment
    print('\n' + '='*80)
    print('STEP 3: SCHEMA ALIGNMENT')
    print('-'*80)

    v8_aligned = v8_df.copy()
    lnof_aligned = lnof_df.copy()

    # Add missing columns to v8
    print('\nAligning v8_iter5 schema:')
    v8_aligned['session_name'] = v8_aligned['session']
    v8_aligned['delete'] = 1 - v8_aligned['ground_truth']
    v8_aligned['merge'] = 0
    v8_aligned['failed_corner_artifact'] = v8_aligned['is_corner_artifact']
    v8_aligned['failed_area'] = 0
    v8_aligned['failed_circularity'] = 0
    v8_aligned['ml_keep_probability'] = np.nan
    v8_aligned['decision'] = v8_aligned['ground_truth'].map({1: 'ok', 0: 'delete'})
    print('  Added: session_name, delete, merge, failed_*, ml_keep_probability, decision')

    # Add missing columns to LNOF
    print('\nAligning LNOF schema:')
    lnof_aligned['ground_truth'] = 1 - lnof_aligned['delete']
    lnof_aligned['distance_to_gt'] = np.nan
    lnof_aligned['is_corner_artifact'] = lnof_aligned.get('failed_corner_artifact', 0)
    lnof_aligned['ellipse_r'] = np.nan
    lnof_aligned['experiment'] = 'LNOF'
    lnof_aligned['session'] = lnof_aligned['session_name']
    # Only add half_crossing_rate if missing (preserve existing values)
    if 'half_crossing_rate' not in lnof_aligned.columns:
        lnof_aligned['half_crossing_rate'] = np.nan
    print('  Added: ground_truth, distance_to_gt, is_corner_artifact, ellipse_r, experiment, session')
    if 'half_crossing_rate' in lnof_aligned.columns:
        hcr_coverage = 100 * (1 - lnof_aligned['half_crossing_rate'].isna().sum() / len(lnof_aligned))
        print(f'  Preserved: half_crossing_rate ({hcr_coverage:.1f}% coverage)')

    # Step 4: Feature validation
    print('\n' + '='*80)
    print('STEP 4: FEATURE VALIDATION')
    print('-'*80)

    # Check v8
    missing_v8 = set(EXPECTED_FEATURES) - set(v8_aligned.columns)
    if missing_v8:
        print(f'[WARNING] v8 missing features: {missing_v8}')
    else:
        print('[OK] v8 has all expected features')

    # Check LNOF
    missing_lnof = set(EXPECTED_FEATURES) - set(lnof_aligned.columns)
    if missing_lnof:
        print(f'[INFO] LNOF missing features (will be added): {missing_lnof}')
    else:
        print('[OK] LNOF has all expected features')

    # Check for excessive NaN
    print('\nNaN check:')
    for col in EXPECTED_FEATURES:
        if col in v8_aligned.columns:
            v8_nan_pct = 100 * v8_aligned[col].isna().mean()
            if v8_nan_pct > 50:
                print(f'  [ERROR] v8: {col} has {v8_nan_pct:.1f}% NaN')
            elif v8_nan_pct > 10:
                print(f'  [WARNING] v8: {col} has {v8_nan_pct:.1f}% NaN')

        if col in lnof_aligned.columns:
            lnof_nan_pct = 100 * lnof_aligned[col].isna().mean()
            if lnof_nan_pct > 50:
                print(f'  [ERROR] LNOF: {col} has {lnof_nan_pct:.1f}% NaN')
            elif lnof_nan_pct > 10:
                print(f'  [WARNING] LNOF: {col} has {lnof_nan_pct:.1f}% NaN')

    # Step 5: Column harmonization
    print('\n' + '='*80)
    print('STEP 5: COLUMN HARMONIZATION')
    print('-'*80)

    # Get union of columns
    all_columns = sorted(set(v8_aligned.columns) | set(lnof_aligned.columns))

    # Add missing columns
    for col in all_columns:
        if col not in v8_aligned.columns:
            v8_aligned[col] = np.nan
            print(f'Added to v8: {col}')
        if col not in lnof_aligned.columns:
            lnof_aligned[col] = np.nan
            print(f'Added to LNOF: {col}')

    # Reorder columns to match
    v8_aligned = v8_aligned[all_columns]
    lnof_aligned = lnof_aligned[all_columns]

    print(f'\n[OK] Both datasets now have {len(all_columns)} columns')

    # Step 6: Concatenation
    print('\n' + '='*80)
    print('STEP 6: CONCATENATION')
    print('-'*80)

    v9_df = pd.concat([v8_aligned, lnof_aligned], ignore_index=True)

    expected_count = len(v8_df) + len(lnof_df)
    print(f'v9 dataset created:')
    print(f'  Total neurons: {len(v9_df):,}')
    print(f'  Expected: {expected_count:,}')
    print(f'  Match: {len(v9_df) == expected_count}')

    # Step 7: Quality checks
    print('\n' + '='*80)
    print('STEP 7: QUALITY CHECKS')
    print('-'*80)

    # 1. Check for duplicates
    duplicates = v9_df.duplicated(subset=['component_idx', 'session_name'])
    if duplicates.any():
        print(f'[ERROR] Found {duplicates.sum()} duplicate neurons')
    else:
        print('[OK] No duplicates found')

    # 2. Verify ground_truth distribution
    gt_dist = v9_df['ground_truth'].value_counts()
    keep_count = gt_dist.get(1, 0)
    delete_count = gt_dist.get(0, 0)
    keep_pct = 100 * keep_count / len(v9_df)

    print(f'\nGround truth distribution:')
    print(f'  KEEP (ground_truth=1): {keep_count:,} ({keep_pct:.2f}%)')
    print(f'  DELETE (ground_truth=0): {delete_count:,} ({100-keep_pct:.2f}%)')

    if 75 < keep_pct < 85:
        print('[OK] Distribution within expected range (75-85% KEEP)')
    else:
        print(f'[WARNING] Distribution outside expected range: {keep_pct:.2f}% KEEP')

    # 3. Verify session counts
    total_sessions = v9_df['session_name'].nunique()
    expected_sessions = v8_df['session'].nunique() + lnof_df['session_name'].nunique()
    print(f'\nSession count:')
    print(f'  Total: {total_sessions}')
    print(f'  Expected: {expected_sessions}')
    print(f'  Match: {total_sessions == expected_sessions}')

    # 4. Verify experiment distribution
    exp_dist = v9_df['experiment'].value_counts()
    print(f'\nExperiment distribution:')
    for exp, count in exp_dist.items():
        print(f'  {exp}: {count:,} neurons')

    # 5. Verify ground_truth vs delete consistency
    inconsistent = v9_df[v9_df['ground_truth'] != (1 - v9_df['delete'])]
    if len(inconsistent) > 0:
        print(f'\n[ERROR] {len(inconsistent)} neurons have inconsistent ground_truth/delete')
    else:
        print('\n[OK] ground_truth and delete columns are consistent')

    # Step 8: Save
    print('\n' + '='*80)
    print('STEP 8: SAVE')
    print('-'*80)

    if test_mode:
        output_path = 'ml/results/training_dataset_v9_test.csv'
    else:
        output_path = 'ml/results/training_dataset_v9.csv'

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    v9_df.to_csv(output_path, index=False)

    print(f'Saved: {output_path}')
    print(f'  Total neurons: {len(v9_df):,}')
    print(f'  Total sessions: {v9_df["session_name"].nunique()}')
    print(f'  Total columns: {len(v9_df.columns)}')

    # Step 9: Generate summary
    print('\n' + '='*80)
    print('STEP 9: GENERATE SUMMARY')
    print('-'*80)

    summary = {
        'Total Neurons': int(len(v9_df)),
        'Total Sessions': int(v9_df['session_name'].nunique()),
        'Total Columns': int(len(v9_df.columns)),
        'Experiments': {exp: int(count) for exp, count in exp_dist.items()},
        'Ground Truth Distribution': {
            'KEEP': int(keep_count),
            'DELETE': int(delete_count),
            'KEEP %': f"{keep_pct:.2f}%"
        },
        'Source Breakdown': {
            'v8_iter5': int(len(v8_df)),
            'LNOF': int(len(lnof_df))
        },
        'Feature Columns': len([c for c in EXPECTED_FEATURES if c in v9_df.columns])
    }

    if test_mode:
        summary_path = 'ml/results/v9_dataset_summary_test.json'
    else:
        summary_path = 'ml/results/v9_dataset_summary.json'

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Saved summary: {summary_path}')

    print('\n' + '='*80)
    print('v9 DATASET SUMMARY')
    print('='*80)
    print(json.dumps(summary, indent=2))

    print('\n' + '='*80)
    print('COMPLETE')
    print('='*80)

    return v9_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge v8_iter5 and LNOF datasets to create v9')
    parser.add_argument('--test-mode', action='store_true', help='Test on subset of sessions')
    args = parser.parse_args()

    merge_datasets(test_mode=args.test_mode)
