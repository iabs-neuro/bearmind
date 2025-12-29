"""
Universal label correction application script.

Applies expert corrections to dataset based on error review.
Supports both manual index lists and correction CSV files.

Usage:
    # Using manual index lists
    python ml/apply_corrections.py \\
        --dataset ml/results/training_dataset_v9.csv \\
        --fp-errors ml/ebm_v9_iter1/top100_fp.csv \\
        --fn-errors ml/ebm_v9_iter1/top100_fn.csv \\
        --real-fp 1,5,12,23 \\
        --real-fn 3,8,15,42 \\
        --output ml/results/training_dataset_v9_corrected_iter1.csv

    # Using correction CSV file
    python ml/apply_corrections.py \\
        --dataset ml/results/training_dataset_v9.csv \\
        --corrections ml/ebm_v9_iter1/manual_corrections.csv \\
        --output ml/results/training_dataset_v9_corrected_iter1.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_index_list(index_str):
    """Parse comma-separated index string into list of integers."""
    if not index_str:
        return []
    return [int(x.strip()) for x in index_str.split(',')]


def apply_corrections(
    dataset_csv,
    output_csv,
    fp_errors_csv=None,
    fn_errors_csv=None,
    real_fp_indices=None,
    real_fn_indices=None,
    corrections_csv=None,
    fp_spatial_csv=None,
    log_csv=None
):
    """
    Apply label corrections to dataset.

    Parameters
    ----------
    dataset_csv : str
        Path to input dataset CSV
    output_csv : str
        Path for corrected output dataset CSV
    fp_errors_csv : str, optional
        Path to FP error CSV
    fn_errors_csv : str, optional
        Path to FN error CSV
    real_fp_indices : list of int, optional
        Indices of REAL FP errors (model wrong, keep GT=0)
    real_fn_indices : list of int, optional
        Indices of REAL FN errors (model wrong, keep GT=1)
    corrections_csv : str, optional
        Path to CSV with manual corrections (alternative to index lists)
    fp_spatial_csv : str, optional
        Path to FP spatial analysis CSV (to exclude MERGE cases)
    log_csv : str, optional
        Path for correction log CSV
    """
    print('='*80)
    print('APPLYING LABEL CORRECTIONS')
    print('='*80)
    print(f'Input dataset: {dataset_csv}')
    print(f'Output dataset: {output_csv}')

    # Load dataset
    df = pd.read_csv(dataset_csv)
    print(f'\nLoaded: {len(df):,} neurons')
    print(f'Original KEEP: {df["ground_truth"].sum():,} ({df["ground_truth"].mean()*100:.2f}%)')

    # Determine session column
    if 'session_name' in df.columns:
        session_col = 'session_name'
    elif 'session' in df.columns:
        session_col = 'session'
    else:
        raise ValueError('Dataset must have "session_name" or "session" column')

    corrections = []

    # Mode 1: Use corrections CSV
    if corrections_csv:
        print(f'\n{"="*80}')
        print('LOADING CORRECTIONS FROM CSV')
        print('='*80)

        df_corr = pd.read_csv(corrections_csv)
        print(f'Loaded {len(df_corr)} corrections from {corrections_csv}')

        required_cols = ['session', 'component_idx', 'new_label', 'reason']
        missing = [c for c in required_cols if c not in df_corr.columns]
        if missing:
            raise ValueError(f'Corrections CSV missing columns: {missing}')

        for idx, row in df_corr.iterrows():
            corrections.append((
                row['session'],
                int(row['component_idx']),
                int(row['new_label']),
                row['reason']
            ))

    # Mode 2: Use manual index lists
    else:
        print(f'\n{"="*80}')
        print('PROCESSING MANUAL INDEX LISTS')
        print('='*80)

        # Process FN errors
        if fn_errors_csv and real_fn_indices is not None:
            df_fn = pd.read_csv(fn_errors_csv)
            print(f'\nFN errors: {len(df_fn)} total, {len(real_fn_indices)} REAL')

            # FAKE FN: model correct (DELETE), GT wrong (KEEP) -> flip to DELETE
            fake_fn_indices = [i for i in range(1, len(df_fn)+1) if i not in real_fn_indices]
            print(f'  FAKE FN: {len(fake_fn_indices)} (model correct DELETE, GT wrong KEEP -> DELETE)')

            for fn_idx in fake_fn_indices:
                fn_row = df_fn.iloc[fn_idx - 1]
                session = fn_row[session_col]
                comp_idx = int(fn_row['component_idx'])
                corrections.append((
                    session,
                    comp_idx,
                    0,  # Change to DELETE
                    f'FAKE FN #{fn_idx} - model correct DELETE'
                ))

        # Process FP errors
        if fp_errors_csv and real_fp_indices is not None:
            df_fp = pd.read_csv(fp_errors_csv)
            print(f'\nFP errors: {len(df_fp)} total, {len(real_fp_indices)} REAL')

            # Load spatial analysis to exclude MERGE cases
            merge_indices = set()
            if fp_spatial_csv:
                df_spatial = pd.read_csv(fp_spatial_csv)
                merge_indices = set(df_spatial[df_spatial['category'] == 'MERGE']['fp_idx'].values)
                print(f'  MERGE cases (excluded): {len(merge_indices)}')

            # FAKE FP: model correct (KEEP), GT wrong (DELETE) -> flip to KEEP
            # But exclude MERGE cases (those are real duplicates)
            fake_fp_indices = [
                i for i in range(1, len(df_fp)+1)
                if i not in real_fp_indices and i not in merge_indices
            ]
            print(f'  FAKE FP: {len(fake_fp_indices)} (model correct KEEP, GT wrong DELETE -> KEEP)')

            for fp_idx in fake_fp_indices:
                fp_row = df_fp.iloc[fp_idx - 1]
                session = fp_row[session_col]
                comp_idx = int(fp_row['component_idx'])
                corrections.append((
                    session,
                    comp_idx,
                    1,  # Change to KEEP
                    f'FAKE FP #{fp_idx} - model correct KEEP'
                ))

    # Apply corrections
    print(f'\n{"="*80}')
    print('APPLYING CORRECTIONS')
    print('='*80)
    print(f'Total corrections to apply: {len(corrections)}')

    n_applied = 0
    n_not_found = 0
    correction_log = []

    for session, comp_idx, new_label, reason in corrections:
        mask = (df[session_col] == session) & (df['component_idx'] == comp_idx)

        if mask.sum() == 0:
            n_not_found += 1
            correction_log.append({
                'session': session,
                'component_idx': comp_idx,
                'new_label': new_label,
                'reason': reason,
                'status': 'NOT_FOUND'
            })
            print(f'Warning: Neuron not found: {session} comp {comp_idx}')
            continue

        old_label = df.loc[mask, 'ground_truth'].values[0]
        df.loc[mask, 'ground_truth'] = new_label
        n_applied += 1

        correction_log.append({
            'session': session,
            'component_idx': comp_idx,
            'old_label': old_label,
            'new_label': new_label,
            'reason': reason,
            'status': 'APPLIED'
        })

    print(f'\nCorrections applied: {n_applied}')
    print(f'Corrections not found: {n_not_found}')

    # Save corrected dataset
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    new_keep = df['ground_truth'].sum()
    old_keep = pd.read_csv(dataset_csv)['ground_truth'].sum()

    print(f'\nNew KEEP: {new_keep:,} ({new_keep/len(df)*100:.2f}%)')
    print(f'Net change: {new_keep - old_keep:+,} labels')

    # Save correction log
    if log_csv or n_applied > 0:
        if log_csv is None:
            log_csv = output_path.parent / f'{output_path.stem}_log.csv'

        log_df = pd.DataFrame(correction_log)
        log_df.to_csv(log_csv, index=False)
        print(f'\nCorrection log saved to: {log_csv}')

    print(f'\nCorrected dataset saved to: {output_csv}')

    print('\n' + '='*80)
    print('CORRECTION COMPLETE')
    print('='*80)
    print('\nNext steps:')
    print(f'1. Retrain model: python ml/retrain_iter.py --dataset {output_csv} --output ...')
    print('2. Compare performance to previous iteration')

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Universal label correction application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Using manual index lists with spatial analysis:
  python ml/apply_corrections.py \\
      --dataset ml/results/training_dataset_v9.csv \\
      --fp-errors ml/ebm_v9_iter1/top100_fp.csv \\
      --fn-errors ml/ebm_v9_iter1/top100_fn.csv \\
      --fp-spatial ml/ebm_v9_iter1/fp_spatial_analysis.csv \\
      --real-fp 1,5,12,23 \\
      --real-fn 3,8,15,42 \\
      --output ml/results/training_dataset_v9_corrected_iter1.csv

  # Using corrections CSV:
  python ml/apply_corrections.py \\
      --dataset ml/results/training_dataset_v9.csv \\
      --corrections corrections.csv \\
      --output ml/results/training_dataset_v9_corrected_iter1.csv
        """
    )

    parser.add_argument('--dataset', required=True, help='Input dataset CSV')
    parser.add_argument('--output', required=True, help='Output corrected dataset CSV')

    # Mode 1: Manual index lists
    parser.add_argument('--fp-errors', default=None, help='FP error CSV')
    parser.add_argument('--fn-errors', default=None, help='FN error CSV')
    parser.add_argument('--real-fp', default=None,
                       help='Comma-separated indices of REAL FP (model wrong)')
    parser.add_argument('--real-fn', default=None,
                       help='Comma-separated indices of REAL FN (model wrong)')
    parser.add_argument('--fp-spatial', default=None,
                       help='FP spatial analysis CSV (to exclude MERGE cases)')

    # Mode 2: Corrections CSV
    parser.add_argument('--corrections', default=None,
                       help='Corrections CSV (columns: session, component_idx, new_label, reason)')

    parser.add_argument('--log', default=None, help='Correction log output CSV')

    args = parser.parse_args()

    # Validate inputs
    if args.corrections:
        # CSV mode
        if args.fp_errors or args.fn_errors or args.real_fp or args.real_fn:
            parser.error('Cannot use --corrections with manual index lists')
    else:
        # Manual mode - need at least one error type
        if not (args.fp_errors or args.fn_errors):
            parser.error('Must specify either --corrections CSV or at least one of --fp-errors/--fn-errors')

    # Parse index lists
    real_fp = parse_index_list(args.real_fp) if args.real_fp else None
    real_fn = parse_index_list(args.real_fn) if args.real_fn else None

    apply_corrections(
        dataset_csv=args.dataset,
        output_csv=args.output,
        fp_errors_csv=args.fp_errors,
        fn_errors_csv=args.fn_errors,
        real_fp_indices=real_fp,
        real_fn_indices=real_fn,
        corrections_csv=args.corrections,
        fp_spatial_csv=args.fp_spatial,
        log_csv=args.log
    )
