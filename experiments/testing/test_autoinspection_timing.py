"""
Direct timing test of autoinspection as used in production.
"""
import time
import numpy as np
from pathlib import Path


def test_actual_autoinspection():
    """Test timing of actual autoinspection on small dataset."""
    from tkinter.filedialog import askopenfilename
    from ae_launch import run_auto_inspection

    print("=" * 80)
    print("AUTOINSPECTION TIMING TEST")
    print("=" * 80)
    print()
    print("This test runs autoinspection and measures parallelization efficiency.")
    print("Please select a small estimates file (100-200 neurons recommended).")
    print()

    # Manual file selection
    fname = askopenfilename(
        title='Select estimates file for timing test',
        initialdir='C:\\Users\\User\\PycharmProjects\\bearmind\\data\\raw_compressed',
        filetypes=[('Estimates files', '*estimates.pickle')]
    )

    if not fname:
        print("No file selected. Aborting.")
        return

    print(f"Selected: {Path(fname).name}")
    print()
    print("Running autoinspection...")
    print("-" * 80)

    t1 = time.time()

    result = run_auto_inspection(
        fname,
        fps=None,
        session_name=None,
        comps_to_select=None,
        cthr=0.35,
        corr_thr=0.6,
        num_sessions=1,
        match_threshold=3,
        sf=None,
        ef=None,
        ds=1,
        include_event_based=True,
        include_heavy=False,  # Faster test
        detect_corner_artifacts=False,  # Skip for speed
        corner_artifact_params=None,
        event_method='wavelet',
        n_iter=2,
        correlation_method='spearman',
        hybrid_kinetics=True,
        brain='ml',
        ml_model_path="production_models/ebm_v8_corrected_iter5.pkl",
        ml_threshold=0.75,
        deletion_rules=None,
        pxlthr_distance_boundary=5,
        d_snr_thr=10,
        enable_merge=False,  # Skip merge for speed
        track_criteria_failures=False,
        save_artifacts=False,  # Don't save
        artifacts_path='./output',
        save_estimates=False,
        save_matrices=False,
        save_corner_detection=False,
        compress_estimates=False,
        verbose=True
    )

    t2 = time.time()
    total_time = t2 - t1

    est = result['estimates']
    n_neurons = len(est.idx_components)

    print()
    print("=" * 80)
    print("TIMING RESULTS")
    print("=" * 80)
    print(f"Neurons processed: {n_neurons}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Time per neuron: {total_time/n_neurons:.3f}s")
    print()

    # Rough estimation of parallelization efficiency
    # Based on diagnostic: sequential is ~0.2s/neuron, parallel should be ~0.05s/neuron
    time_per_neuron = total_time / n_neurons

    print("EFFICIENCY ESTIMATE:")
    if time_per_neuron > 0.8:
        print(f"  [WARNING] {time_per_neuron:.3f}s/neuron is SLOW")
        print("  Parallelization likely NOT working.")
        print("  Expected: ~0.1-0.2s/neuron with parallelization")
    elif time_per_neuron > 0.3:
        print(f"  [OK] {time_per_neuron:.3f}s/neuron is moderate")
        print("  Parallelization may be working but not optimally.")
    else:
        print(f"  [GOOD] {time_per_neuron:.3f}s/neuron is fast")
        print("  Parallelization appears to be working.")

    print("=" * 80)


if __name__ == '__main__':
    test_actual_autoinspection()
