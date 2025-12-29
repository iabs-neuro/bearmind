"""
Universal FP spatial analysis script.

Analyzes spatial relationships of FP errors to identify MERGE cases.
MERGE cases (< 5px to KEEP neuron) should be excluded from label corrections
as they are likely true duplicates.

Usage:
    python ml/spatial_analysis_fp.py --fp ml/ebm_v9_iter1/top100_fp.csv --dataset ml/results/training_dataset_v9.csv --output ml/ebm_v9_iter1/fp_spatial_analysis.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def analyze_fp_spatial(fp_csv, dataset_csv, output_csv, merge_threshold=5, proximity_threshold=8):
    """
    Analyze spatial relationships of FP errors.

    Parameters
    ----------
    fp_csv : str
        Path to FP error CSV
    dataset_csv : str
        Path to full dataset CSV
    output_csv : str
        Output path for spatial analysis results
    merge_threshold : float
        Distance threshold for MERGE classification (pixels)
    proximity_threshold : float
        Distance threshold for PROXIMITY classification (pixels)
    """
    print('='*80)
    print('FP SPATIAL ANALYSIS')
    print('='*80)
    print(f'FP errors: {fp_csv}')
    print(f'Dataset: {dataset_csv}')
    print(f'MERGE threshold: < {merge_threshold}px')
    print(f'PROXIMITY threshold: < {proximity_threshold}px')

    # Load data
    df_fp = pd.read_csv(fp_csv)
    df_full = pd.read_csv(dataset_csv)

    print(f'\nLoaded {len(df_fp)} FP errors')
    print(f'Full dataset: {len(df_full)} neurons')

    # Determine session column
    if 'session_name' in df_fp.columns:
        session_col = 'session_name'
    elif 'session' in df_fp.columns:
        session_col = 'session'
    else:
        raise ValueError('FP CSV must have "session_name" or "session" column')

    results = []

    for fp_idx, fp_row in df_fp.iterrows():
        session = fp_row[session_col]
        comp_idx = int(fp_row['component_idx'])

        # Parse center coordinates
        center_raw = fp_row['center']
        if isinstance(center_raw, str):
            # Handle string format: "[x y]" or "x y"
            center_str = center_raw.strip('[]')
            center = np.fromstring(center_str, sep=' ')
        elif isinstance(center_raw, (list, np.ndarray)):
            center = np.array(center_raw)
        else:
            print(f'Warning: Could not parse center for FP #{fp_idx+1}')
            results.append({
                'fp_idx': fp_idx + 1,
                'session': session,
                'component_idx': comp_idx,
                'center': None,
                'min_distance_to_keep': np.nan,
                'closest_keep_idx': None,
                'category': 'UNKNOWN'
            })
            continue

        # Get all neurons from this session
        session_neurons = df_full[df_full[session_col] == session].copy()

        # Find KEEP neurons in this session
        keep_neurons = session_neurons[session_neurons['ground_truth'] == 1].copy()

        # Calculate distances to all KEEP neurons
        min_distance = np.inf
        closest_keep_idx = None

        for idx, keep_row in keep_neurons.iterrows():
            keep_comp_idx = int(keep_row['component_idx'])
            if keep_comp_idx == comp_idx:
                continue  # Skip self

            # Parse keep center
            keep_center_raw = keep_row['center']
            if isinstance(keep_center_raw, str):
                keep_center_str = keep_center_raw.strip('[]')
                keep_center = np.fromstring(keep_center_str, sep=' ')
            elif isinstance(keep_center_raw, (list, np.ndarray)):
                keep_center = np.array(keep_center_raw)
            else:
                continue

            # Calculate Euclidean distance
            distance = np.sqrt(np.sum((center - keep_center)**2))

            if distance < min_distance:
                min_distance = distance
                closest_keep_idx = keep_comp_idx

        # Categorize based on distance
        if min_distance < merge_threshold:
            category = 'MERGE'
        elif min_distance < proximity_threshold:
            category = 'PROXIMITY'
        else:
            category = 'STANDALONE'

        results.append({
            'fp_idx': fp_idx + 1,
            'session': session,
            'component_idx': comp_idx,
            'center': center.tolist() if isinstance(center, np.ndarray) else center,
            'min_distance_to_keep': min_distance,
            'closest_keep_idx': closest_keep_idx,
            'category': category
        })

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Summary statistics
    print('\n' + '='*80)
    print('SPATIAL ANALYSIS RESULTS')
    print('='*80)

    n_merge = (df_results['category'] == 'MERGE').sum()
    n_proximity = (df_results['category'] == 'PROXIMITY').sum()
    n_standalone = (df_results['category'] == 'STANDALONE').sum()
    n_unknown = (df_results['category'] == 'UNKNOWN').sum()

    print(f'\nCategory distribution:')
    print(f'  MERGE (< {merge_threshold}px):      {n_merge:>4} ({100*n_merge/len(df_results):.1f}%)')
    print(f'  PROXIMITY ({merge_threshold}-{proximity_threshold}px): {n_proximity:>4} ({100*n_proximity/len(df_results):.1f}%)')
    print(f'  STANDALONE (> {proximity_threshold}px):  {n_standalone:>4} ({100*n_standalone/len(df_results):.1f}%)')
    if n_unknown > 0:
        print(f'  UNKNOWN:          {n_unknown:>4} ({100*n_unknown/len(df_results):.1f}%)')

    # Distance statistics
    valid_distances = df_results['min_distance_to_keep'].dropna()
    if len(valid_distances) > 0:
        print(f'\nDistance statistics (px):')
        print(f'  Min:    {valid_distances.min():.2f}')
        print(f'  Median: {valid_distances.median():.2f}')
        print(f'  Mean:   {valid_distances.mean():.2f}')
        print(f'  Max:    {valid_distances.max():.2f}')

    # Save results
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)

    print(f'\nResults saved to: {output_csv}')

    print('\n' + '='*80)
    print('INTERPRETATION & NEXT STEPS')
    print('='*80)

    print(f'\nMERGE cases ({n_merge} neurons):')
    print('  These FP are < 5px from a KEEP neuron - likely true duplicates.')
    print('  RECOMMENDATION: EXCLUDE from label corrections (GT is correct: DELETE)')

    print(f'\nPROXIMITY cases ({n_proximity} neurons):')
    print('  These FP are 5-8px from a KEEP neuron - review carefully.')
    print('  Could be duplicates or distinct neurons.')

    print(f'\nSTANDALONE cases ({n_standalone} neurons):')
    print('  These FP are > 8px from any KEEP neuron - spatially independent.')
    print('  If model predicted KEEP, likely FAKE FP (GT wrong).')

    print('\nNext steps:')
    print('1. Review visualizations for PROXIMITY and STANDALONE cases')
    print('2. Classify each FP as REAL or FAKE')
    print('3. For FAKE FP (excluding MERGE): prepare correction list')
    print('4. Apply corrections using ml/apply_corrections.py')

    return df_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Universal FP spatial analysis')
    parser.add_argument('--fp', required=True, help='Path to FP error CSV')
    parser.add_argument('--dataset', required=True, help='Path to full dataset CSV')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--merge-threshold', type=float, default=5.0,
                       help='Distance threshold for MERGE (default: 5px)')
    parser.add_argument('--proximity-threshold', type=float, default=8.0,
                       help='Distance threshold for PROXIMITY (default: 8px)')

    args = parser.parse_args()

    analyze_fp_spatial(
        fp_csv=args.fp,
        dataset_csv=args.dataset,
        output_csv=args.output,
        merge_threshold=args.merge_threshold,
        proximity_threshold=args.proximity_threshold
    )
