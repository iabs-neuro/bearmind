"""
Merge v7 validation data into single dataset (threshold method).

This creates a dataset equivalent to v6 but using v7 data where
event detection was done with threshold method instead of wavelet.
"""
import sys
from pathlib import Path

# Add parent directory to path to import create_training_dataset
sys.path.insert(0, str(Path(__file__).parents[2]))

from create_training_dataset import create_training_dataset

if __name__ == "__main__":
    print("[INFO] Merging v7 validation data (threshold method)")
    print("[INFO] Using capcan_validation_127_v7 artifacts")

    # Get project root directory
    project_root = Path(__file__).parents[2]

    df_v7 = create_training_dataset(
        artifacts_dir=str(project_root / "data" / "capcan_validation_127_v7"),
        output_path=str(project_root / "analysis_event_method_comparison_2025_12_15" / "data" / "training_dataset_v7_merged.csv"),
        max_distance=3.0
    )

    print(f"\n[SUCCESS] Created v7 merged dataset: {len(df_v7)} neurons")
    print(f"[INFO] Saved to: analysis_event_method_comparison_2025_12_15/data/training_dataset_v7_merged.csv")
