"""
Diagnostic script to test if joblib parallelization is working.

Tests the parallelization performance compared to sequential processing.
"""
import numpy as np
import time
from joblib import Parallel, delayed
import psutil
import os


def dummy_heavy_task(x, sleep_time=0.1):
    """Simulate heavy computation similar to get_single_neuron_metrics."""
    # Simulate CPU-bound work (similar to wavelet transforms, event detection)
    result = 0
    for _ in range(1000):
        result += np.sum(np.sin(x) ** 2 + np.cos(x) ** 2)

    # Small sleep to simulate I/O or other blocking operations
    time.sleep(sleep_time)

    return {'mean': np.mean(x), 'std': np.std(x), 'result': result}


def test_parallelization(n_tasks=100, task_size=1000, sleep_time=0.1):
    """
    Test parallelization with varying number of jobs.

    Args:
        n_tasks: Number of tasks to run (like number of neurons)
        task_size: Size of each task (like trace length)
        sleep_time: Artificial delay per task (seconds)

    Returns:
        dict: Results with timings and core info
    """
    # Generate dummy data
    data = [np.random.rand(task_size) for _ in range(n_tasks)]

    print("=" * 80)
    print("JOBLIB PARALLELIZATION DIAGNOSTIC")
    print("=" * 80)
    print(f"System info:")
    print(f"  CPU cores (physical): {psutil.cpu_count(logical=False)}")
    print(f"  CPU cores (logical):  {psutil.cpu_count(logical=True)}")
    print(f"  Process ID: {os.getpid()}")
    print()
    print(f"Test configuration:")
    print(f"  Tasks: {n_tasks}")
    print(f"  Task size: {task_size}")
    print(f"  Sleep per task: {sleep_time}s")
    print()

    results = {}

    # Test 1: Sequential (n_jobs=1)
    print(f"[1/4] Testing SEQUENTIAL processing (n_jobs=1)...")
    t1 = time.time()
    _ = Parallel(n_jobs=1)(
        delayed(dummy_heavy_task)(data[i], sleep_time=sleep_time)
        for i in range(n_tasks)
    )
    t2 = time.time()
    sequential_time = t2 - t1
    results['sequential'] = sequential_time
    print(f"  Time: {sequential_time:.2f}s ({sequential_time/n_tasks:.3f}s/task)")
    print()

    # Test 2: Parallel with n_jobs=-1 (all cores)
    print(f"[2/4] Testing PARALLEL processing (n_jobs=-1, all cores)...")
    t1 = time.time()
    _ = Parallel(n_jobs=-1)(
        delayed(dummy_heavy_task)(data[i], sleep_time=sleep_time)
        for i in range(n_tasks)
    )
    t2 = time.time()
    parallel_time = t2 - t1
    results['parallel_all'] = parallel_time
    speedup = sequential_time / parallel_time
    print(f"  Time: {parallel_time:.2f}s ({parallel_time/n_tasks:.3f}s/task)")
    print(f"  Speedup: {speedup:.2f}x")
    print()

    # Test 3: Parallel with n_jobs=2
    print(f"[3/4] Testing PARALLEL processing (n_jobs=2)...")
    t1 = time.time()
    _ = Parallel(n_jobs=2)(
        delayed(dummy_heavy_task)(data[i], sleep_time=sleep_time)
        for i in range(n_tasks)
    )
    t2 = time.time()
    parallel_2_time = t2 - t1
    results['parallel_2'] = parallel_2_time
    speedup_2 = sequential_time / parallel_2_time
    print(f"  Time: {parallel_2_time:.2f}s ({parallel_2_time/n_tasks:.3f}s/task)")
    print(f"  Speedup: {speedup_2:.2f}x")
    print()

    # Test 4: Parallel with n_jobs=4
    print(f"[4/4] Testing PARALLEL processing (n_jobs=4)...")
    t1 = time.time()
    _ = Parallel(n_jobs=4)(
        delayed(dummy_heavy_task)(data[i], sleep_time=sleep_time)
        for i in range(n_tasks)
    )
    t2 = time.time()
    parallel_4_time = t2 - t1
    results['parallel_4'] = parallel_4_time
    speedup_4 = sequential_time / parallel_4_time
    print(f"  Time: {parallel_4_time:.2f}s ({parallel_4_time/n_tasks:.3f}s/task)")
    print(f"  Speedup: {speedup_4:.2f}x")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Sequential time:     {sequential_time:.2f}s")
    print(f"Parallel time (-1):  {parallel_time:.2f}s (speedup: {speedup:.2f}x)")
    print(f"Parallel time (2):   {parallel_2_time:.2f}s (speedup: {speedup_2:.2f}x)")
    print(f"Parallel time (4):   {parallel_4_time:.2f}s (speedup: {speedup_4:.2f}x)")
    print()

    # Diagnostic interpretation
    print("INTERPRETATION:")
    if speedup < 1.5:
        print("  [WARNING] Minimal speedup detected!")
        print("  Parallelization may NOT be working effectively.")
        print("  Possible causes:")
        print("    - Joblib backend issue")
        print("    - Windows process spawning overhead")
        print("    - GIL contention (if using threading)")
        print("    - Small task size (overhead dominates)")
    elif speedup < 3:
        print("  [OK] Moderate speedup detected.")
        print("  Parallelization is working but not optimal.")
        print(f"  Expected speedup on {psutil.cpu_count(logical=False)} physical cores: ~{psutil.cpu_count(logical=False)}x")
        print("  Consider investigating overhead or task granularity.")
    else:
        print("  [SUCCESS] Good speedup detected!")
        print("  Parallelization is working effectively.")
    print()

    if speedup_2 > speedup:
        print("  [ANOMALY] n_jobs=2 faster than n_jobs=-1!")
        print("  This suggests overhead from spawning too many processes.")
        print("  Consider using n_jobs=2 or n_jobs=4 instead of n_jobs=-1.")

    print("=" * 80)

    return results


def test_with_actual_autoinspector():
    """Test the actual get_multineuron_metrics function if available."""
    try:
        from auto_inspector import get_multineuron_metrics

        print("\n" + "=" * 80)
        print("TESTING ACTUAL auto_inspector.get_multineuron_metrics()")
        print("=" * 80)

        # Create dummy traces similar to real data
        n_neurons = 50
        n_frames = 10000
        fps = 30

        traces = np.random.rand(n_neurons, n_frames)
        # Normalize like real code does
        for i in range(n_neurons):
            tr = traces[i]
            tr_min, tr_max = tr.min(), tr.max()
            if tr_max > tr_min:
                traces[i] = (tr - tr_min) / (tr_max - tr_min)

        print(f"Test data: {n_neurons} neurons, {n_frames} frames, {fps} fps")
        print()

        # Test with include_heavy=False (faster, just event detection)
        print("[1/2] Testing with include_heavy=False (wavelet event detection only)...")
        t1 = time.time()
        metrics, recons = get_multineuron_metrics(
            traces,
            fps=fps,
            include_heavy=False,
            event_method='wavelet',
            n_iter=2,
            hybrid_kinetics=True
        )
        t2 = time.time()
        total_time = t2 - t1
        print(f"  Time: {total_time:.2f}s ({total_time/n_neurons:.3f}s/neuron)")
        print(f"  Metrics computed: {len(metrics)} types")
        print()

        # Expected time if sequential
        time_per_neuron = total_time / n_neurons
        expected_sequential = time_per_neuron * n_neurons
        n_cores = psutil.cpu_count(logical=False)
        expected_parallel = expected_sequential / n_cores

        print("Estimated parallelization efficiency:")
        print(f"  Time per neuron: {time_per_neuron:.3f}s")
        print(f"  Expected if purely sequential: {expected_sequential:.2f}s")
        print(f"  Expected if perfectly parallel ({n_cores} cores): {expected_parallel:.2f}s")
        print(f"  Actual time: {total_time:.2f}s")

        if total_time < expected_sequential * 0.7:
            print("  [SUCCESS] Parallelization appears to be working!")
        else:
            print("  [WARNING] Parallelization may not be working effectively.")

        print("=" * 80)

    except ImportError as e:
        print(f"Could not import auto_inspector: {e}")
        print("Skipping actual function test.")


if __name__ == '__main__':
    # Test 1: Generic joblib parallelization test
    test_parallelization(n_tasks=50, task_size=1000, sleep_time=0.05)

    # Test 2: Actual auto_inspector function (if available)
    test_with_actual_autoinspector()
