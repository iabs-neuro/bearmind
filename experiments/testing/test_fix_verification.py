"""
Direct test to verify the joblib backend fix improves performance.

Tests the SAME estimates file before and after the fix.
"""
import time
import pickle
import numpy as np
from pathlib import Path


def test_with_current_code(estimates_path):
    """Test with the CURRENT code (after fix applied)."""
    from auto_inspector import get_multineuron_metrics

    print("[TEST 1] Testing WITH explicit backend='loky' (current code)")
    print("-" * 70)

    # Load estimates
    with open(estimates_path, 'rb') as f:
        est = pickle.load(f)

    # Get traces
    n_neurons = len(est.idx_components)
    traces = []
    for tr in est.C[est.idx_components]:
        tr_min, tr_max = tr.min(), tr.max()
        if tr_max > tr_min:
            normalized = (tr - tr_min) / (tr_max - tr_min)
        else:
            normalized = np.zeros_like(tr)
        traces.append(normalized)

    traces = np.array(traces)
    fps = 30

    print(f"  Loaded {n_neurons} neurons, {traces.shape[1]} frames")
    print(f"  Computing event-based metrics with wavelet...")

    t1 = time.time()
    metrics, recons = get_multineuron_metrics(
        traces, fps=fps,
        include_heavy=False,
        event_method='wavelet',
        n_iter=2,
        hybrid_kinetics=True
    )
    t2 = time.time()

    time_taken = t2 - t1
    time_per_neuron = time_taken / n_neurons

    print(f"  Time: {time_taken:.2f}s ({time_per_neuron:.3f}s/neuron)")
    print()

    return time_per_neuron, n_neurons


def test_without_backend_spec(estimates_path):
    """Test WITHOUT explicit backend (remove the backend='loky' temporarily)."""
    import sys
    from pathlib import Path
    import importlib

    print("[TEST 2] Testing WITHOUT explicit backend (simulating old code)")
    print("-" * 70)

    # Load auto_inspector and monkey-patch to remove backend specification
    import auto_inspector
    from joblib import Parallel, delayed

    # Save original function
    original_func = auto_inspector.get_multineuron_metrics

    # Create patched version without backend='loky'
    def get_multineuron_metrics_no_backend(traces, fps=30, include_heavy=False, event_method='threshold', n_iter=2, hybrid_kinetics=True):
        all_metrics = {}
        reconstructions = {}
        n = traces.shape[0]
        # NO backend specification (like old code)
        metrics_res = Parallel(n_jobs=-1)(
            delayed(auto_inspector.get_single_neuron_metrics)(traces[i], fps=fps, include_heavy=include_heavy, event_method=event_method, n_iter=n_iter, hybrid_kinetics=hybrid_kinetics)
            for i in range(traces.shape[0])
        )

        for metric in metrics_res[0].keys():
            if metric == 'reconstruction':
                for i in range(n):
                    rec = metrics_res[i].get('reconstruction')
                    if rec is not None:
                        reconstructions[i] = rec
            else:
                all_metrics[metric] = [metrics_res[i][metric] for i in range(n)]

        return all_metrics, reconstructions

    # Load estimates
    with open(estimates_path, 'rb') as f:
        est = pickle.load(f)

    # Get traces
    n_neurons = len(est.idx_components)
    traces = []
    for tr in est.C[est.idx_components]:
        tr_min, tr_max = tr.min(), tr.max()
        if tr_max > tr_min:
            normalized = (tr - tr_min) / (tr_max - tr_min)
        else:
            normalized = np.zeros_like(tr)
        traces.append(normalized)

    traces = np.array(traces)
    fps = 30

    print(f"  Loaded {n_neurons} neurons, {traces.shape[1]} frames")
    print(f"  Computing event-based metrics with wavelet...")

    t1 = time.time()
    metrics, recons = get_multineuron_metrics_no_backend(
        traces, fps=fps,
        include_heavy=False,
        event_method='wavelet',
        n_iter=2,
        hybrid_kinetics=True
    )
    t2 = time.time()

    time_taken = t2 - t1
    time_per_neuron = time_taken / n_neurons

    print(f"  Time: {time_taken:.2f}s ({time_per_neuron:.3f}s/neuron)")
    print()

    return time_per_neuron, n_neurons


if __name__ == '__main__':
    from tkinter.filedialog import askopenfilename

    print("=" * 80)
    print("JOBLIB BACKEND FIX VERIFICATION")
    print("=" * 80)
    print()
    print("This test compares performance BEFORE and AFTER the fix.")
    print("Please select a small-medium estimates file (100-250 neurons).")
    print()

    fname = askopenfilename(
        title='Select estimates file for testing',
        initialdir='C:\\Users\\User\\PycharmProjects\\bearmind\\data\\raw_compressed',
        filetypes=[('Estimates files', '*estimates.pickle')]
    )

    if not fname:
        print("No file selected. Aborting.")
    else:
        print(f"Selected: {Path(fname).name}")
        print()

        # Test with current code (with backend='loky')
        time_with_fix, n = test_with_current_code(fname)

        # Test without backend specification
        time_without_fix, _ = test_without_backend_spec(fname)

        # Compare
        print("=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"Neurons: {n}")
        print()
        print(f"WITH backend='loky' (NEW):    {time_with_fix:.3f}s/neuron")
        print(f"WITHOUT backend spec (OLD):   {time_without_fix:.3f}s/neuron")
        print()

        improvement = time_without_fix / time_with_fix
        print(f"Speedup from fix: {improvement:.2f}x")
        print()

        if improvement > 1.5:
            print("[SUCCESS] Fix provides significant speedup!")
            print(f"  Batch processing will be ~{improvement:.1f}x faster.")
        elif improvement > 1.1:
            print("[OK] Fix provides moderate speedup.")
        else:
            print("[INCONCLUSIVE] No significant improvement detected.")
            print("  Possible reasons:")
            print("    - Default backend already using loky")
            print("    - Test environment differs from production")
            print("    - Small sample size")

        print("=" * 80)
