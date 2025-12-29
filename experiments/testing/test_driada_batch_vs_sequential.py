"""
Test if driada batch processing is faster than one-by-one Neuron objects.
"""
import numpy as np
import time
from driada.experiment.neuron import Neuron
from driada.experiment.wavelet_event_detection import extract_wvt_events, WVT_EVENT_DETECTION_PARAMS

# Test data: 100 neurons, 5000 frames each
n_neurons = 100
n_frames = 5000
fps = 30

print("=" * 80)
print("DRIADA WORKFLOW EFFICIENCY TEST")
print("=" * 80)
print(f"Test data: {n_neurons} neurons × {n_frames} frames")
print()

# Generate realistic traces (not pure random - add baseline + events)
traces = []
for i in range(n_neurons):
    baseline = 0.1 + np.random.rand() * 0.2
    noise = np.random.randn(n_frames) * 0.05
    trace = baseline + noise

    # Add some events
    n_events = np.random.randint(10, 30)
    for _ in range(n_events):
        event_time = np.random.randint(0, n_frames - 100)
        event_amp = 0.3 + np.random.rand() * 0.5
        event_width = np.random.randint(20, 60)

        # Gaussian-like event
        x = np.arange(event_width)
        event_shape = event_amp * np.exp(-((x - event_width/2) ** 2) / (2 * (event_width/6) ** 2))
        trace[event_time:event_time + event_width] += event_shape

    # Normalize
    trace_min, trace_max = trace.min(), trace.max()
    if trace_max > trace_min:
        trace = (trace - trace_min) / (trace_max - trace_min)

    traces.append(trace)

traces = np.array(traces)

# Method 1: One-by-one (current autoinspection approach)
print("[1/2] Testing ONE-BY-ONE approach (current autoinspection)...")
print("      Creating Neuron objects and calling reconstruct_spikes individually...")

t1 = time.time()
neurons_sequential = []
for i in range(n_neurons):
    neuron = Neuron(cell_id="", ca=traces[i], sp=None, fps=fps)
    neuron.reconstruct_spikes(method='wavelet', iterative=False, create_event_regions=False, fps=fps)
    neurons_sequential.append(neuron)
t2 = time.time()

sequential_time = t2 - t1
print(f"      Time: {sequential_time:.2f}s ({sequential_time/n_neurons:.3f}s/neuron)")
print()

# Method 2: Batch (like notebook does)
print("[2/2] Testing BATCH approach (extract_wvt_events on all traces)...")
print("      Processing all traces at once...")

t3 = time.time()
st_evinds, end_evinds, ridges = extract_wvt_events(traces, WVT_EVENT_DETECTION_PARAMS)
t4 = time.time()

batch_time = t4 - t3
print(f"      Time: {batch_time:.2f}s ({batch_time/n_neurons:.3f}s/neuron)")
print()

# Comparison
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"One-by-one (Neuron objects):  {sequential_time:.2f}s ({sequential_time/n_neurons:.3f}s/neuron)")
print(f"Batch (extract_wvt_events):   {batch_time:.2f}s ({batch_time/n_neurons:.3f}s/neuron)")
print()

if batch_time < sequential_time:
    speedup = sequential_time / batch_time
    print(f"SPEEDUP: {speedup:.2f}x faster with batch processing")
    print()
    print("RECOMMENDATION: Refactor autoinspection to use batch wavelet detection")
    print("instead of creating Neuron objects one-by-one in parallel.")
else:
    slowdown = batch_time / sequential_time
    print(f"RESULT: Batch is {slowdown:.2f}x SLOWER")
    print("Current approach is already optimal.")

print()

# Verify results are similar
print("VERIFICATION: Checking event detection consistency...")
n_events_sequential = sum([len(neuron.sp) if hasattr(neuron, 'sp') and neuron.sp is not None else 0
                          for neuron in neurons_sequential])
n_events_batch = sum([len(end_evinds[i]) for i in range(n_neurons)])

print(f"  Sequential: {n_events_sequential} total events detected")
print(f"  Batch: {n_events_batch} total events detected")
print(f"  Difference: {abs(n_events_sequential - n_events_batch)} events")

print("=" * 80)
