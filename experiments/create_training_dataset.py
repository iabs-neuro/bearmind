"""
Create merged training dataset with all neuronal metrics and ground truth labels.

Combines all sessions from capcan_validation_127 into a single dataframe with:
- All features used for ML
- Ground truth labels (matched to GT by distance)
- Session and experiment identifiers
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path


def load_session_data(session_dir):
    """Load raw and ground truth metrics for a session."""
    try:
        raw_metrics = os.path.join(session_dir, "metrics_init.csv")
        gt_metrics = os.path.join(session_dir, "metrics_gt.csv")

        df_raw = pd.read_csv(raw_metrics)
        df_gt = pd.read_csv(gt_metrics)

        # Parse center column if stored as string
        if df_raw['center'].dtype == 'object':
            df_raw['center'] = df_raw['center'].apply(
                lambda x: np.fromstring(x.strip('[]'), sep=' ')
            )
        if df_gt['center'].dtype == 'object':
            df_gt['center'] = df_gt['center'].apply(
                lambda x: np.fromstring(x.strip('[]'), sep=' ')
            )

        return df_raw, df_gt
    except Exception as e:
        print(f"  ERROR loading {os.path.basename(session_dir)}: {e}")
        return None, None


def create_training_dataset(
    artifacts_dir="data/capcan_validation_127",
    output_path="ml/results/training_dataset_full.csv",
    max_distance=3
):
    """
    Create complete training dataset from all sessions.

    Args:
        artifacts_dir: Directory containing capcan_artifacts_* subdirectories
        output_path: Where to save the merged CSV
        max_distance: Maximum distance (pixels) for GT matching

    Returns:
        DataFrame with all neurons and features
    """
    print("="*80)
    print("CREATING TRAINING DATASET")
    print("="*80)

    # Find all session directories
    artifacts_path = Path(artifacts_dir)
    session_dirs = sorted([
        d for d in artifacts_path.iterdir()
        if d.is_dir() and d.name.startswith('capcan_artifacts_')
    ])

    print(f"\nFound {len(session_dirs)} sessions in {artifacts_dir}")

    # Collect data from all sessions
    all_data = []
    total_neurons = 0
    total_keep = 0
    total_delete = 0

    for session_dir in session_dirs:
        # Extract session name and experiment ID
        session_name = session_dir.name.replace('capcan_artifacts_', '')
        experiment_id = session_name.split('_')[0]

        # Load session data
        df_raw, df_gt = load_session_data(session_dir)
        if df_raw is None:
            continue

        # Filter out corner artifacts
        if 'is_corner_artifact' in df_raw.columns:
            df_raw = df_raw[df_raw['is_corner_artifact'] == 0].copy()

        # Create ground truth labels by matching to GT neurons
        raw_centers = np.array(df_raw['center'].tolist())
        gt_centers = np.array(df_gt['center'].tolist())

        labels = np.zeros(len(df_raw), dtype=int)
        distances = np.full(len(df_raw), np.inf)

        for i, raw_center in enumerate(raw_centers):
            dists = np.linalg.norm(gt_centers - raw_center, axis=1)
            min_dist = dists.min()
            distances[i] = min_dist

            if min_dist <= max_distance:
                labels[i] = 1  # KEEP

        # Add metadata columns
        df_raw['session'] = session_name
        df_raw['experiment'] = experiment_id
        df_raw['ground_truth'] = labels
        df_raw['distance_to_gt'] = distances

        all_data.append(df_raw)

        # Track statistics
        n_keep = (labels == 1).sum()
        n_delete = (labels == 0).sum()
        total_neurons += len(df_raw)
        total_keep += n_keep
        total_delete += n_delete

        print(f"  {session_name}: {len(df_raw)} neurons "
              f"(KEEP: {n_keep}, DELETE: {n_delete})")

    # Merge all sessions
    df_all = pd.concat(all_data, ignore_index=True)

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_path, index=False)

    # Print summary
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"Total neurons: {total_neurons:,}")
    print(f"  KEEP (ground_truth=1): {total_keep:,} ({total_keep/total_neurons*100:.1f}%)")
    print(f"  DELETE (ground_truth=0): {total_delete:,} ({total_delete/total_neurons*100:.1f}%)")
    print(f"\nSessions: {df_all['session'].nunique()}")
    print(f"Experiments: {sorted(df_all['experiment'].unique())}")
    print(f"\nExperiment distribution:")
    for exp in sorted(df_all['experiment'].unique()):
        exp_count = (df_all['experiment'] == exp).sum()
        print(f"  {exp}: {exp_count:,} neurons")

    print(f"\nColumns in dataset ({len(df_all.columns)}):")
    print(f"  {list(df_all.columns)}")

    print(f"\nSaved to: {output_path.resolve()}")
    print("="*80)

    return df_all


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create training dataset from capcan validation artifacts"
    )
    parser.add_argument(
        "--artifacts-dir",
        default="data/capcan_validation_127",
        help="Directory containing capcan_artifacts_* subdirectories"
    )
    parser.add_argument(
        "--output",
        default="ml/results/training_dataset_full.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=3.0,
        help="Maximum distance (pixels) for ground truth matching"
    )

    args = parser.parse_args()

    df = create_training_dataset(
        artifacts_dir=args.artifacts_dir,
        output_path=args.output,
        max_distance=args.max_distance
    )
