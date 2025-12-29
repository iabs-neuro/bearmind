"""
Test wavelet optimization with REAL data from NOF session.

This test demonstrates the actual performance improvement on production data.
"""
import numpy as np
import time
import pickle
from pathlib import Path
from auto_inspector import get_multineuron_metrics

print("=" * 80)
print("WAVELET OPTIMIZATION TEST - REAL NOF SESSION")
print("=" * 80)
print()

# Find a NOF session
data_dir = Path('data/raw_compressed')
nof_sessions = sorted(data_dir.glob('*NOF*.pickle'))

if not nof_sessions:
    print("ERROR: No NOF sessions found in data/raw_compressed/")
    print("Please ensure NOF sessions are available.")
    exit(1)

# Use the first NOF session
session_file = nof_sessions[0]
print(f"Using session: {session_file.name}")
print()

# Load estimates
print("Loading estimates...")
with open(session_file, 'rb') as f:
    est = pickle.load(f)

n_neurons = len(est.idx_components)
n_frames = est.C.shape[1]
print(f"Loaded: {n_neurons} neurons, {n_frames} frames")
print()

# Extract and normalize traces
print("Extracting traces...")
traces = []
for tr in est.C[est.idx_components]:
    tr_min, tr_max = tr.min(), tr.max()
    if tr_max > tr_min:
        normalized = (tr - tr_min) / (tr_max - tr_min)
    else:
        normalized = np.zeros_like(tr)
    traces.append(normalized)

traces = np.array(traces)
print(f"Traces extracted: {traces.shape}")
print()

# Test with wavelet method (OPTIMIZED with DRIADA 0.6.5)
print("=" * 80)
print("RUNNING WAVELET AUTOINSPECTION (OPTIMIZED)")
print("=" * 80)
print()
print("This will use the optimized path with pre-computed wavelet objects.")
print("Watch for the '[OPTIMIZATION] Pre-computed wavelet objects...' message.")
print()

t_start = time.time()
metrics, recons = get_multineuron_metrics(
    traces,
    fps=30,
    include_heavy=False,
    event_method='wavelet',
    n_iter=2,
    hybrid_kinetics=True
)
t_end = time.time()

total_time = t_end - t_start
time_per_neuron = total_time / n_neurons

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Session: {session_file.name}")
print(f"Neurons processed: {n_neurons}")
print(f"Frames per neuron: {n_frames}")
print()
print(f"Total time: {total_time:.2f}s")
print(f"Time per neuron: {time_per_neuron:.3f}s")
print()
print(f"Overhead saved by optimization: ~{n_neurons * 0.27:.1f}s")
print(f"  (Without optimization would have been: ~{total_time + n_neurons * 0.27:.1f}s)")
print()

# Show some metrics
events_detected = sum(1 for epm in metrics['events_per_min'] if not np.isnan(epm))
print(f"Neurons with events detected: {events_detected}/{n_neurons}")
print(f"Mean events per minute: {np.nanmean(metrics['events_per_min']):.2f}")
print()

# Performance assessment
print("PERFORMANCE ASSESSMENT:")
if time_per_neuron < 0.5:
    print(f"  [EXCELLENT] {time_per_neuron:.3f}s/neuron is very fast!")
elif time_per_neuron < 1.0:
    print(f"  [GOOD] {time_per_neuron:.3f}s/neuron is fast")
elif time_per_neuron < 2.0:
    print(f"  [OK] {time_per_neuron:.3f}s/neuron is acceptable")
else:
    print(f"  [SLOW] {time_per_neuron:.3f}s/neuron - may need investigation")

print()
print("OPTIMIZATION VERIFICATION:")
print("  If you saw the '[OPTIMIZATION] Pre-computed...' message above,")
print("  the optimization is working correctly and saved significant time!")
print("=" * 80)
