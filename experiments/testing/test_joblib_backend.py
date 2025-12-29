"""
Test different joblib backends to diagnose parallelization issue.
"""
import numpy as np
import time
from joblib import Parallel, delayed


def test_backend(backend_name):
    """Test parallelization with a specific backend."""
    print(f"\nTesting backend: '{backend_name}'")
    print("-" * 60)

    try:
        from auto_inspector import get_multineuron_metrics

        # Create dummy traces
        n_neurons = 20
        n_frames = 5000
        fps = 30

        traces = np.random.rand(n_neurons, n_frames)
        for i in range(n_neurons):
            tr = traces[i]
            tr_min, tr_max = tr.min(), tr.max()
            if tr_max > tr_min:
                traces[i] = (tr - tr_min) / (tr_max - tr_min)

        print(f"Test data: {n_neurons} neurons, {n_frames} frames")

        # Manually call with specified backend
        t1 = time.time()

        if backend_name == 'default':
            # Use default from auto_inspector (n_jobs=-1 with no backend specified)
            metrics, recons = get_multineuron_metrics(
                traces, fps=fps, include_heavy=False,
                event_method='wavelet', n_iter=2, hybrid_kinetics=True
            )
        else:
            # Test by directly modifying the Parallel call
            from auto_inspector import get_single_neuron_metrics

            metrics_res = Parallel(n_jobs=-1, backend=backend_name, verbose=10)(
                delayed(get_single_neuron_metrics)(
                    traces[i], fps=fps, include_heavy=False,
                    event_method='wavelet', n_iter=2, hybrid_kinetics=True
                )
                for i in range(n_neurons)
            )

            # Process results
            all_metrics = {}
            recons = {}
            for metric in metrics_res[0].keys():
                if metric == 'reconstruction':
                    for i in range(n_neurons):
                        rec = metrics_res[i].get('reconstruction')
                        if rec is not None:
                            recons[i] = rec
                else:
                    all_metrics[metric] = [metrics_res[i][metric] for i in range(n_neurons)]

        t2 = time.time()
        total_time = t2 - t1

        print(f"Time: {total_time:.2f}s ({total_time/n_neurons:.3f}s/neuron)")

        # Expected sequential time
        expected_sequential = total_time if backend_name == 'sequential' else None
        if expected_sequential:
            print(f"This is the sequential baseline for comparison.")

        return total_time

    except Exception as e:
        print(f"ERROR with backend '{backend_name}': {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("=" * 80)
    print("JOBLIB BACKEND DIAGNOSTIC")
    print("=" * 80)
    print()
    print("Testing different backends to identify parallelization issue:")
    print("  - 'default': What auto_inspector.py currently uses")
    print("  - 'loky': Process-based (default on Windows, requires pickling)")
    print("  - 'threading': Thread-based (limited by GIL for CPU work)")
    print("  - 'sequential': No parallelization (baseline)")
    print()

    results = {}

    # Test each backend
    backends = ['default', 'loky', 'threading', 'sequential']

    for backend in backends:
        time_taken = test_backend(backend)
        if time_taken is not None:
            results[backend] = time_taken

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        baseline = results.get('sequential', results.get('default', 0))
        for backend, time_val in results.items():
            speedup = baseline / time_val if time_val > 0 else 0
            print(f"  {backend:15s}: {time_val:6.2f}s (speedup: {speedup:.2f}x)")

        print()
        print("RECOMMENDATION:")

        if results.get('default', 0) > baseline * 0.7:
            print("  [ISSUE CONFIRMED] Default backend is running sequentially!")

            if results.get('loky', 0) > baseline * 0.7:
                print("  'loky' backend ALSO slow -> pickling issue likely")
                print("  CAUSE: Neuron or wavelet detection functions not picklable")
                print("  FIX: Refactor to avoid unpicklable objects or use threading")

            if results.get('threading', 0) < baseline * 0.7:
                print("  'threading' backend IS faster -> suggests GIL not main bottleneck")
                print("  FIX: Switch auto_inspector.py to use backend='threading'")

        else:
            print("  Default backend is working fine.")
    else:
        print("  All backends failed. Check error messages above.")

    print("=" * 80)
