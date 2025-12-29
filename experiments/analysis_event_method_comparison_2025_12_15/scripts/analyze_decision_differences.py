"""
Analyze the REAL difference: auto-inspection decision criteria.

Key finding: v6 uses event-based rejection criteria, v7 does not!
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parents[2]
sns.set_style("whitegrid")


def analyze_all_sessions():
    """Analyze decision differences across all 127 sessions."""

    print("="*80)
    print("ANALYZING AUTO-INSPECTION DECISIONS: V6 vs V7")
    print("="*80)

    v6_dir = project_root / "data" / "capcan_validation_127_v6"
    v7_dir = project_root / "data" / "capcan_validation_127_v7"

    session_dirs = sorted([d for d in v6_dir.iterdir() if d.is_dir() and d.name.startswith('capcan_artifacts_')])

    results = []

    for session_dir in session_dirs:
        session_name = session_dir.name.replace('capcan_artifacts_', '')

        # Load decision files
        v6_dec_path = v6_dir / f"capcan_artifacts_{session_name}" / "decisions_with_criteria.csv"
        v7_dec_path = v7_dir / f"capcan_artifacts_{session_name}" / "decisions_with_criteria.csv"

        if not v6_dec_path.exists() or not v7_dec_path.exists():
            print(f"  [SKIP] {session_name}: missing decision files")
            continue

        df_v6 = pd.read_csv(v6_dec_path)
        df_v7 = pd.read_csv(v7_dec_path)

        # Count decisions
        v6_delete = (df_v6['decision'] == 'delete').sum() if 'decision' in df_v6.columns else 0
        v7_delete = (df_v7['decision'] == 'delete').sum() if 'decision' in df_v7.columns else 0

        # Count by failure reason (v6)
        v6_failures = {}
        for col in df_v6.columns:
            if col.startswith('failed_'):
                reason = col.replace('failed_', '')
                v6_failures[reason] = df_v6[col].sum() if df_v6[col].dtype in [int, bool] else 0

        # Count by failure reason (v7)
        v7_failures = {}
        for col in df_v7.columns:
            if col.startswith('failed_'):
                reason = col.replace('failed_', '')
                v7_failures[reason] = df_v7[col].sum() if df_v7[col].dtype in [int, bool] else 0

        results.append({
            'session': session_name,
            'n_neurons': len(df_v6),
            'v6_deleted': v6_delete,
            'v7_deleted': v7_delete,
            'diff_deleted': v6_delete - v7_delete,
            **{f'v6_failed_{k}': v for k, v in v6_failures.items()},
            **{f'v7_failed_{k}': v for k, v in v7_failures.items()}
        })

    df_results = pd.DataFrame(results)

    # Save raw results
    output_path = project_root / "analysis_event_method_comparison_2025_12_15" / "data" / "decision_comparison_all_sessions.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n[SAVED] {output_path}")

    return df_results


def summarize_results(df_results):
    """Summarize the decision differences."""

    print("\n" + "="*80)
    print("SUMMARY: AUTO-INSPECTION DECISION DIFFERENCES")
    print("="*80)

    total_neurons = df_results['n_neurons'].sum()
    total_v6_deleted = df_results['v6_deleted'].sum()
    total_v7_deleted = df_results['v7_deleted'].sum()
    total_diff = df_results['diff_deleted'].sum()

    print(f"\nTotal neurons analyzed: {total_neurons:,}")
    print(f"\nv6 (wavelet) - Total deleted: {total_v6_deleted:,} ({total_v6_deleted/total_neurons*100:.2f}%)")
    print(f"v7 (threshold) - Total deleted: {total_v7_deleted:,} ({total_v7_deleted/total_neurons*100:.2f}%)")
    print(f"\nDifference: {total_diff:,} FEWER deletions with v7")
    print(f"Reduction: {total_diff/total_v6_deleted*100:.2f}% fewer deletions")

    # Breakdown by failure type
    print("\n" + "-"*80)
    print("V6 REJECTION BREAKDOWN:")
    print("-"*80)

    v6_failure_cols = [col for col in df_results.columns if col.startswith('v6_failed_')]
    for col in sorted(v6_failure_cols):
        reason = col.replace('v6_failed_', '')
        count = df_results[col].sum()
        pct = count / total_v6_deleted * 100 if total_v6_deleted > 0 else 0
        print(f"  {reason:20s}: {count:6,} neurons ({pct:5.1f}% of v6 deletions)")

    print("\n" + "-"*80)
    print("V7 REJECTION BREAKDOWN:")
    print("-"*80)

    v7_failure_cols = [col for col in df_results.columns if col.startswith('v7_failed_')]
    for col in sorted(v7_failure_cols):
        reason = col.replace('v7_failed_', '')
        count = df_results[col].sum()
        pct = count / total_v7_deleted * 100 if total_v7_deleted > 0 else 0
        print(f"  {reason:20s}: {count:6,} neurons ({pct:5.1f}% of v7 deletions)")

    # Identify which criteria are MISSING in v7
    v6_reasons = {col.replace('v6_failed_', '') for col in v6_failure_cols}
    v7_reasons = {col.replace('v7_failed_', '') for col in v7_failure_cols}

    missing_in_v7 = v6_reasons - v7_reasons

    print("\n" + "="*80)
    print("CRITICAL FINDING:")
    print("="*80)
    print(f"\nCriteria used ONLY in v6 (wavelet), NOT in v7 (threshold):")
    for reason in sorted(missing_in_v7):
        if f'v6_failed_{reason}' in df_results.columns:
            count = df_results[f'v6_failed_{reason}'].sum()
            print(f"  - {reason:20s}: {count:6,} neurons rejected")

    print(f"\nTOTAL IMPACT: These event-based criteria rejected {total_diff:,} neurons in v6")
    print(f"              that were ACCEPTED in v7")

    return {
        'total_neurons': total_neurons,
        'v6_deleted': total_v6_deleted,
        'v7_deleted': total_v7_deleted,
        'difference': total_diff,
        'missing_criteria': sorted(missing_in_v7)
    }


def create_visualizations(df_results, summary, output_dir):
    """Create visualizations of decision differences."""

    print("\n" + "="*80)
    print("GENERATING DECISION COMPARISON VISUALIZATIONS")
    print("="*80)

    output_dir.mkdir(exist_ok=True)

    # 1. Overall deletion counts
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    data = pd.DataFrame({
        'Method': ['v6 (wavelet)', 'v7 (threshold)'],
        'Deleted': [summary['v6_deleted'], summary['v7_deleted']],
        'Kept': [summary['total_neurons'] - summary['v6_deleted'],
                 summary['total_neurons'] - summary['v7_deleted']]
    })

    x = np.arange(len(data))
    width = 0.35

    bars1 = ax.bar(x, data['Kept'], width, label='KEPT', alpha=0.8, color='green')
    bars2 = ax.bar(x, data['Deleted'], width, bottom=data['Kept'], label='DELETED', alpha=0.8, color='red')

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Number of Neurons', fontsize=12)
    ax.set_title(f'Auto-Inspection Decisions: Wavelet vs Threshold (n={summary["total_neurons"]:,})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(data['Method'])
    ax.legend()

    # Add counts on bars
    for i, (kept, deleted) in enumerate(zip(data['Kept'], data['Deleted'])):
        ax.text(i, kept/2, f'{kept:,}\n({kept/summary["total_neurons"]*100:.1f}%)',
                ha='center', va='center', fontweight='bold', fontsize=10)
        ax.text(i, kept + deleted/2, f'{deleted:,}\n({deleted/summary["total_neurons"]*100:.1f}%)',
                ha='center', va='center', fontweight='bold', fontsize=10, color='white')

    plt.tight_layout()
    output_path = output_dir / "decision_counts_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [SAVED] {output_path}")
    plt.close()

    # 2. Per-session difference histogram
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.hist(df_results['diff_deleted'], bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(df_results['diff_deleted'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean = {df_results["diff_deleted"].mean():.1f}')
    ax.axvline(df_results['diff_deleted'].median(), color='orange', linestyle='--',
               linewidth=2, label=f'Median = {df_results["diff_deleted"].median():.1f}')

    ax.set_xlabel('Difference in deletions (v6 - v7)', fontsize=12)
    ax.set_ylabel('Number of sessions', fontsize=12)
    ax.set_title('Per-Session Deletion Differences (positive = v6 deleted more)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "per_session_differences.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [SAVED] {output_path}")
    plt.close()

    # 3. Rejection criteria breakdown
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # v6 criteria
    v6_failure_cols = [col for col in df_results.columns if col.startswith('v6_failed_')]
    v6_data = {col.replace('v6_failed_', ''): df_results[col].sum() for col in v6_failure_cols}
    v6_data = dict(sorted(v6_data.items(), key=lambda x: x[1], reverse=True))

    axes[0].barh(list(v6_data.keys()), list(v6_data.values()), alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Number of neurons rejected', fontsize=11)
    axes[0].set_title('v6 (wavelet): Rejection Criteria', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='x')

    # v7 criteria
    v7_failure_cols = [col for col in df_results.columns if col.startswith('v7_failed_')]
    v7_data = {col.replace('v7_failed_', ''): df_results[col].sum() for col in v7_failure_cols}
    v7_data = dict(sorted(v7_data.items(), key=lambda x: x[1], reverse=True))

    axes[1].barh(list(v7_data.keys()), list(v7_data.values()), alpha=0.7, color='coral')
    axes[1].set_xlabel('Number of neurons rejected', fontsize=11)
    axes[1].set_title('v7 (threshold): Rejection Criteria', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='x')

    plt.tight_layout()
    output_path = output_dir / "rejection_criteria_breakdown.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [SAVED] {output_path}")
    plt.close()


def main():
    """Run decision analysis."""

    # Analyze all sessions
    df_results = analyze_all_sessions()

    # Summarize
    summary = summarize_results(df_results)

    # Create visualizations
    output_dir = project_root / "analysis_event_method_comparison_2025_12_15" / "plots"
    create_visualizations(df_results, summary, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
