"""
Batch Auto-Inspection with Corner Artifact Detection

Process all 64 sessions from data/4.1_EstimatesRaw and create
capcan_artifacts_<session_name> folders for each.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import time

from auto_inspector import estimates_to_metrics, metrics_to_decision
from ae_utils import save_auto_inspection_outputs


def process_single_session(session_file, output_base='./'):
    """
    Process a single session through the complete auto-inspection pipeline.

    Parameters:
        session_file: Path to estimates pickle file
        output_base: Base directory for outputs

    Returns:
        dict with processing results and statistics
    """
    session_name = session_file.stem.replace('_estimates', '')

    print(f'\n{"="*80}')
    print(f'Processing: {session_name}')
    print(f'{"="*80}')

    try:
        # Load estimates
        print(f'[1/4] Loading estimates...')
        with open(session_file, 'rb') as f:
            est = pickle.load(f)

        n_components = len(est.idx_components)
        print(f'  Loaded {n_components} neurons')

        # Extract metrics with corner detection
        print(f'[2/4] Extracting metrics with corner artifact detection...')
        t1 = time.time()
        metrics_df, match_mtx, FCD, FBD, corner_info, _, _ = estimates_to_metrics(
            est,
            fps=20,
            include_event_based=True,
            include_heavy=False,
            detect_corner_artifacts_flag=True
        )
        t2 = time.time()
        metrics_time = t2 - t1
        print(f'  Completed in {metrics_time:.1f}s ({metrics_time/n_components:.2f}s per neuron)')

        # Make decisions with criteria tracking
        print(f'[3/4] Making decisions with criteria tracking...')
        decision_df = metrics_to_decision(
            metrics_df.copy(),
            match_mtx,
            FCD,
            FBD,
            track_criteria_failures=True
        )

        n_rejected = (decision_df['delete'] == 1).sum()
        n_kept = len(decision_df) - n_rejected
        n_corner = (decision_df.get('is_corner_artifact', 0) == 1).sum()

        print(f'  Rejected: {n_rejected} ({n_rejected/len(decision_df)*100:.1f}%)')
        print(f'  Kept: {n_kept} ({n_kept/len(decision_df)*100:.1f}%)')
        print(f'  Corner artifacts: {n_corner} ({n_corner/len(decision_df)*100:.1f}%)')

        # Save outputs
        print(f'[4/4] Saving outputs...')
        output_folder = save_auto_inspection_outputs(
            session_name=session_name,
            metrics_df=metrics_df,
            decision_df=decision_df,
            corner_info=corner_info,
            base_path=output_base
        )

        # Collect statistics
        stats = {
            'session_name': session_name,
            'n_total': len(decision_df),
            'n_rejected': n_rejected,
            'n_kept': n_kept,
            'n_corner': n_corner,
            'rejection_rate': n_rejected/len(decision_df)*100,
            'corner_rate': n_corner/len(decision_df)*100,
            'processing_time': metrics_time,
            'output_folder': str(output_folder),
            'status': 'SUCCESS'
        }

        # Count criteria failures
        failure_cols = [col for col in decision_df.columns if col.startswith('failed_')]
        rejected_df = decision_df[decision_df['delete'] == 1]
        for col in failure_cols:
            criterion_name = col.replace('failed_', '')
            stats[f'n_{criterion_name}'] = rejected_df[col].sum() if len(rejected_df) > 0 else 0

        print(f'  SUCCESS: {session_name}')
        return stats

    except Exception as e:
        print(f'  ERROR: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()

        return {
            'session_name': session_name,
            'status': 'FAILED',
            'error': str(e)
        }


def batch_process_all_sessions(data_dir='data/4.1_EstimatesRaw', output_base='./'):
    """
    Process all sessions in the data directory.

    Parameters:
        data_dir: Directory containing estimates pickle files
        output_base: Base directory for outputs

    Returns:
        DataFrame with statistics for all sessions
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f'ERROR: Data directory not found: {data_dir}')
        return None

    # Get all session files
    session_files = sorted(data_path.glob('*_estimates.pickle'))

    if len(session_files) == 0:
        print(f'ERROR: No session files found in {data_dir}')
        return None

    print('='*80)
    print('BATCH AUTO-INSPECTION WITH CORNER ARTIFACT DETECTION')
    print('='*80)
    print(f'\nFound {len(session_files)} sessions to process')
    print(f'Output base directory: {output_base}')
    print()

    # Process all sessions
    all_stats = []
    start_time = time.time()

    for idx, session_file in enumerate(session_files, 1):
        print(f'\n[{idx}/{len(session_files)}]')
        stats = process_single_session(session_file, output_base)
        all_stats.append(stats)

    total_time = time.time() - start_time

    # Create summary dataframe
    stats_df = pd.DataFrame(all_stats)

    # Save summary
    summary_path = Path(output_base) / 'batch_autoinspect_summary.csv'
    stats_df.to_csv(summary_path, index=False)

    # Print overall summary
    print('\n' + '='*80)
    print('BATCH PROCESSING SUMMARY')
    print('='*80)

    n_success = (stats_df['status'] == 'SUCCESS').sum()
    n_failed = (stats_df['status'] == 'FAILED').sum()

    print(f'\nProcessed: {len(session_files)} sessions')
    print(f'Success: {n_success}')
    print(f'Failed: {n_failed}')
    print(f'Total time: {total_time/60:.1f} minutes')
    print(f'Average time per session: {total_time/len(session_files):.1f} seconds')

    if n_success > 0:
        success_df = stats_df[stats_df['status'] == 'SUCCESS']

        print(f'\n--- OVERALL STATISTICS (across {n_success} successful sessions) ---')
        print(f'Total neurons: {success_df["n_total"].sum()}')
        print(f'Total rejected: {success_df["n_rejected"].sum()} ({success_df["n_rejected"].sum()/success_df["n_total"].sum()*100:.1f}%)')
        print(f'Total kept: {success_df["n_kept"].sum()} ({success_df["n_kept"].sum()/success_df["n_total"].sum()*100:.1f}%)')
        print(f'Total corner artifacts: {success_df["n_corner"].sum()} ({success_df["n_corner"].sum()/success_df["n_total"].sum()*100:.1f}%)')

        print(f'\n--- AVERAGE PER SESSION ---')
        print(f'Neurons per session: {success_df["n_total"].mean():.1f} +/- {success_df["n_total"].std():.1f}')
        print(f'Rejection rate: {success_df["rejection_rate"].mean():.1f}% +/- {success_df["rejection_rate"].std():.1f}%')
        print(f'Corner artifact rate: {success_df["corner_rate"].mean():.1f}% +/- {success_df["corner_rate"].std():.1f}%')

        # Criteria breakdown
        print(f'\n--- REJECTION CRITERIA BREAKDOWN (total across all sessions) ---')
        failure_criteria = [col for col in success_df.columns if col.startswith('n_') and col not in ['n_total', 'n_rejected', 'n_kept', 'n_corner']]
        for col in failure_criteria:
            criterion_name = col.replace('n_', '').replace('_', ' ').title()
            total_failed = success_df[col].sum()
            pct_of_rejected = total_failed / success_df['n_rejected'].sum() * 100 if success_df['n_rejected'].sum() > 0 else 0
            print(f'  {criterion_name}: {total_failed} neurons ({pct_of_rejected:.1f}% of all rejected)')

    if n_failed > 0:
        print(f'\n--- FAILED SESSIONS ---')
        failed_df = stats_df[stats_df['status'] == 'FAILED']
        for _, row in failed_df.iterrows():
            print(f'  {row["session_name"]}: {row.get("error", "Unknown error")}')

    print(f'\nSummary saved to: {summary_path}')
    print('='*80)

    return stats_df


if __name__ == '__main__':
    stats_df = batch_process_all_sessions()
