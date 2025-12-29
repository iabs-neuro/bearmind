"""
Profile where the overhead comes from in one-by-one vs batch processing.
"""
import numpy as np
import time
from driada.experiment.neuron import Neuron
from driada.experiment.wavelet_event_detection import extract_wvt_events, WVT_EVENT_DETECTION_PARAMS

# Test data
n_neurons = 50
traces = np.random.rand(n_neurons, 5000)

print("=" * 80)
print("OVERHEAD PROFILING: Where does the 7x speedup come from?")
print("=" * 80)
print()

# Test 1: Just creating Neuron objects (no wavelet detection)
print("[1/5] Creating Neuron objects only (no detection)...")
t1 = time.time()
neurons = []
for i in range(n_neurons):
    neuron = Neuron(cell_id="", ca=traces[i], sp=None, fps=30)
    neurons.append(neuron)
t2 = time.time()
print(f"  Time: {t2-t1:.3f}s ({(t2-t1)/n_neurons*1000:.1f}ms/neuron)")
print()

# Test 2: Creating Neuron + calling reconstruct_spikes
print("[2/5] Creating Neuron + reconstruct_spikes (wavelet)...")
t1 = time.time()
neurons = []
for i in range(n_neurons):
    neuron = Neuron(cell_id="", ca=traces[i], sp=None, fps=30)
    neuron.reconstruct_spikes(method='wavelet', iterative=False, fps=30)
    neurons.append(neuron)
t2 = time.time()
sequential_total = t2 - t1
print(f"  Time: {sequential_total:.3f}s ({sequential_total/n_neurons*1000:.1f}ms/neuron)")
print()

# Test 3: Batch wavelet detection
print("[3/5] Batch wavelet detection (extract_wvt_events)...")
t1 = time.time()
st_events, end_events, ridges = extract_wvt_events(traces, WVT_EVENT_DETECTION_PARAMS)
t2 = time.time()
batch_total = t2 - t1
print(f"  Time: {batch_total:.3f}s ({batch_total/n_neurons*1000:.1f}ms/neuron)")
print()

# Test 4: Just the wavelet detection call on pre-created Neurons
print("[4/5] Wavelet detection on pre-created Neurons...")
# Reuse neurons from test 1
t1 = time.time()
for neuron in neurons:
    neuron.reconstruct_spikes(method='wavelet', iterative=False, fps=30)
t2 = time.time()
detection_only = t2 - t1
print(f"  Time: {detection_only:.3f}s ({detection_only/n_neurons*1000:.1f}ms/neuron)")
print()

# Test 5: Neuron creation overhead estimation
creation_overhead = sequential_total - detection_only
print("[5/5] Analysis of overhead...")
print(f"  Neuron creation overhead: {creation_overhead:.3f}s ({creation_overhead/n_neurons*1000:.1f}ms/neuron)")
print(f"  Wavelet detection time: {detection_only:.3f}s ({detection_only/n_neurons*1000:.1f}ms/neuron)")
print()

# Summary
print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print(f"Sequential (Neuron objects): {sequential_total:.3f}s")
print(f"  - Neuron creation:         {creation_overhead:.3f}s ({creation_overhead/sequential_total*100:.1f}%)")
print(f"  - Wavelet detection:       {detection_only:.3f}s ({detection_only/sequential_total*100:.1f}%)")
print()
print(f"Batch (extract_wvt_events):  {batch_total:.3f}s")
print(f"Speedup: {sequential_total/batch_total:.2f}x")
print()

if batch_total < detection_only:
    print("KEY FINDING: Batch is faster than even just wavelet detection on pre-created Neurons!")
    print("This means extract_wvt_events() has internal optimizations (vectorization, etc.)")
    optimization_factor = detection_only / batch_total
    print(f"Internal optimization: {optimization_factor:.2f}x faster")
else:
    overhead_savings = sequential_total - detection_only
    print("KEY FINDING: Most speedup comes from avoiding Neuron object creation overhead")
    print(f"Overhead savings: {overhead_savings:.3f}s ({overhead_savings/(sequential_total-batch_total)*100:.1f}% of speedup)")

print()

# Check if extract_wvt_events uses different parameters
print("Parameter comparison:")
print(f"  WVT_EVENT_DETECTION_PARAMS: {WVT_EVENT_DETECTION_PARAMS}")
print("=" * 80)
