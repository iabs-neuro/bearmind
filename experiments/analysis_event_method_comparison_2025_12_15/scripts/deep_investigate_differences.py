"""
Deep investigation: Why are v6 and v7 metrics identical?

Check individual session files to understand if event detection differs.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parents[2]


def compare_session_files(session_name='3DM_D17_1D'):
    """Compare actual session files between v6 and v7."""

    print("="*80)
    print(f"DEEP INVESTIGATION: Session {session_name}")
    print("="*80)

    v6_dir = project_root / "data" / "capcan_validation_127_v6" / f"capcan_artifacts_{session_name}"
    v7_dir = project_root / "data" / "capcan_validation_127_v7" / f"capcan_artifacts_{session_name}"

    print(f"\nv6 dir: {v6_dir}")
    print(f"v7 dir: {v7_dir}")

    # Check if directories exist
    if not v6_dir.exists():
        print(f"ERROR: v6 directory not found")
        return
    if not v7_dir.exists():
        print(f"ERROR: v7 directory not found")
        return

    # Load metrics_init.csv from both
    v6_metrics = pd.read_csv(v6_dir / "metrics_init.csv")
    v7_metrics = pd.read_csv(v7_dir / "metrics_init.csv")

    print(f"\nv6 metrics shape: {v6_metrics.shape}")
    print(f"v7 metrics shape: {v7_metrics.shape}")

    # Check columns
    print(f"\nColumns in v6: {v6_metrics.columns.tolist()}")
    print(f"\nColumns in v7: {v7_metrics.columns.tolist()}")

    # Check if they're identical
    common_cols = set(v6_metrics.columns) & set(v7_metrics.columns)

    print(f"\n\nCOLUMN-BY-COLUMN COMPARISON:")
    print("-" * 80)

    differences_found = False

    for col in sorted(common_cols):
        if col == 'center':  # Skip center column (array comparison is complex)
            continue

        v6_vals = v6_metrics[col].values
        v7_vals = v7_metrics[col].values

        # Check if identical
        if len(v6_vals) == len(v7_vals):
            # Handle NaN
            v6_clean = pd.Series(v6_vals).fillna(-999)
            v7_clean = pd.Series(v7_vals).fillna(-999)

            if np.allclose(v6_clean, v7_clean, rtol=1e-10, atol=1e-10):
                status = "IDENTICAL"
            else:
                status = "DIFFERENT"
                differences_found = True

                # Calculate difference statistics
                diff = np.abs(v6_clean - v7_clean)
                max_diff = diff.max()
                mean_diff = diff.mean()
                n_diff = (diff > 1e-10).sum()

                print(f"\n{col}:")
                print(f"  Status: {status}")
                print(f"  Max difference: {max_diff:.10f}")
                print(f"  Mean difference: {mean_diff:.10f}")
                print(f"  N different: {n_diff}/{len(diff)}")

                # Show some examples
                if n_diff > 0:
                    diff_indices = np.where(diff > 1e-10)[0][:5]
                    print(f"  Examples (first {min(5, len(diff_indices))} differences):")
                    for idx in diff_indices:
                        print(f"    neuron {idx}: v6={v6_vals[idx]:.10f}, v7={v7_vals[idx]:.10f}, diff={diff[idx]:.10f}")
        else:
            print(f"\n{col}: DIFFERENT LENGTHS (v6={len(v6_vals)}, v7={len(v7_vals)})")
            differences_found = True

    if not differences_found:
        print("\n" + "="*80)
        print("CRITICAL FINDING: ALL METRICS ARE IDENTICAL!")
        print("="*80)
        print("\nThis means:")
        print("1. The wavelet vs threshold distinction does NOT affect these metrics")
        print("2. Event-based metrics may be calculated from the same source")
        print("3. Need to check WHERE the wavelet/threshold difference actually occurs")

    return differences_found


def check_what_differs():
    """Check validation_summary.txt to see what actually differs."""

    print("\n\n" + "="*80)
    print("CHECKING VALIDATION SUMMARY FILES")
    print("="*80)

    session_name = '3DM_D17_1D'

    v6_summary = project_root / "data" / "capcan_validation_127_v6" / f"capcan_artifacts_{session_name}" / "validation_summary.txt"
    v7_summary = project_root / "data" / "capcan_validation_127_v7" / f"capcan_artifacts_{session_name}" / "validation_summary.txt"

    if v6_summary.exists() and v7_summary.exists():
        print(f"\nv6 validation_summary.txt:")
        print("-" * 80)
        with open(v6_summary, 'r') as f:
            v6_text = f.read()
            print(v6_text)

        print(f"\n\nv7 validation_summary.txt:")
        print("-" * 80)
        with open(v7_summary, 'r') as f:
            v7_text = f.read()
            print(v7_text)

        if v6_text == v7_text:
            print("\n[RESULT] validation_summary.txt files are IDENTICAL")
        else:
            print("\n[RESULT] validation_summary.txt files are DIFFERENT")
    else:
        print("  [ERROR] Summary files not found")


def check_decisions_file():
    """Check decisions_with_criteria.csv to see auto-inspection decisions."""

    print("\n\n" + "="*80)
    print("CHECKING AUTO-INSPECTION DECISIONS")
    print("="*80)

    session_name = '3DM_D17_1D'

    v6_decisions = project_root / "data" / "capcan_validation_127_v6" / f"capcan_artifacts_{session_name}" / "decisions_with_criteria.csv"
    v7_decisions = project_root / "data" / "capcan_validation_127_v7" / f"capcan_artifacts_{session_name}" / "decisions_with_criteria.csv"

    if v6_decisions.exists() and v7_decisions.exists():
        df_v6_dec = pd.read_csv(v6_decisions)
        df_v7_dec = pd.read_csv(v7_decisions)

        print(f"\nv6 decisions shape: {df_v6_dec.shape}")
        print(f"v7 decisions shape: {df_v7_dec.shape}")

        print(f"\nv6 columns: {df_v6_dec.columns.tolist()}")
        print(f"v7 columns: {df_v7_dec.columns.tolist()}")

        # Compare decisions
        if 'decision' in df_v6_dec.columns and 'decision' in df_v7_dec.columns:
            v6_keep = (df_v6_dec['decision'] == 'keep').sum()
            v6_delete = (df_v6_dec['decision'] == 'delete').sum()
            v7_keep = (df_v7_dec['decision'] == 'keep').sum()
            v7_delete = (df_v7_dec['decision'] == 'delete').sum()

            print(f"\nv6 decisions: KEEP={v6_keep}, DELETE={v6_delete}")
            print(f"v7 decisions: KEEP={v7_keep}, DELETE={v7_delete}")

            if v6_keep == v7_keep and v6_delete == v7_delete:
                print("\n[RESULT] Decision counts are IDENTICAL")

                # Check if actual decisions match neuron-by-neuron
                if len(df_v6_dec) == len(df_v7_dec):
                    decisions_match = (df_v6_dec['decision'] == df_v7_dec['decision']).all()
                    print(f"[RESULT] Neuron-by-neuron decisions match: {decisions_match}")
            else:
                print("\n[RESULT] Decision counts are DIFFERENT")
    else:
        print("  [ERROR] Decision files not found")


def main():
    """Run deep investigation."""

    # Compare session files
    differences_found = compare_session_files('3DM_D17_1D')

    # Check validation summaries
    check_what_differs()

    # Check auto-inspection decisions
    check_decisions_file()

    print("\n\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if not differences_found:
        print("""
The analysis reveals that v6 (wavelet) and v7 (threshold) produce IDENTICAL metrics
in the metrics_init.csv files. This means:

HYPOTHESIS: The event detection method (wavelet vs threshold) is used ONLY for
auto-inspection decision-making, NOT for metric calculation.

The event-based metrics (event_r2_score, events_per_min, etc.) are likely calculated
using a consistent method regardless of the wavelet/threshold setting.

The wavelet vs threshold distinction probably affects:
1. Which events are used for auto-inspection thresholds
2. How event-based criteria are evaluated
3. The final KEEP/DELETE decisions

But NOT:
- The actual event detection used for metrics
- The quality metrics themselves

RECOMMENDATION: Check the auto_inspector.py code to understand where the
wavelet/threshold distinction is actually applied.
""")


if __name__ == "__main__":
    main()
