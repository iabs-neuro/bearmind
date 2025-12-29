import numpy as np
from scipy.stats import spearmanr

# Test with many neurons
trace_data = np.random.rand(10, 100)  # 10 neurons, 100 timepoints
print('trace_data shape:', trace_data.shape)
print()

# Test axis=1 (what the code uses)
print('Testing spearmanr with axis=1:')
result = spearmanr(trace_data, axis=1)
print('  Result type:', type(result))
if isinstance(result, tuple):
    print('  Tuple length:', len(result))
    print('  First element type:', type(result[0]))
    print('  First element:', result[0])
    if hasattr(result[0], 'shape'):
        print('  First element shape:', result[0].shape)
else:
    print('  Result:', result)
print()

# Test axis=0 (potentially correct for neuron-neuron correlation)
print('Testing spearmanr with axis=0:')
result2 = spearmanr(trace_data, axis=0)
print('  Result type:', type(result2))
if isinstance(result2, tuple):
    print('  Tuple length:', len(result2))
    print('  First element type:', type(result2[0]))
    if hasattr(result2[0], 'shape'):
        print('  First element shape:', result2[0].shape)
else:
    print('  Result:', result2)
print()

# Test with no axis (default behavior)
print('Testing spearmanr with no axis (default):')
result3 = spearmanr(trace_data.T)  # Transpose to get (timepoints, neurons)
print('  Result type:', type(result3))
if isinstance(result3, tuple):
    print('  Tuple length:', len(result3))
    if hasattr(result3[0], 'shape'):
        print('  First element shape:', result3[0].shape)
