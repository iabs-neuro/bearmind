import numpy as np
from scipy.stats import spearmanr

# Test with exactly 2 neurons
trace_data = np.random.rand(2, 100)  # 2 neurons, 100 timepoints
print('trace_data shape:', trace_data.shape)
print()

# Test axis=1
print('Testing spearmanr with axis=1 and 2 neurons:')
result = spearmanr(trace_data, axis=1)
print('  Result type:', type(result))
if isinstance(result, tuple):
    print('  Tuple length:', len(result))
    print('  First element type:', type(result[0]))
    print('  First element value:', result[0])
    if hasattr(result[0], 'shape'):
        print('  First element shape:', result[0].shape)
    if hasattr(result[0], 'ndim'):
        print('  First element ndim:', result[0].ndim)
else:
    print('  Result:', result)
print()

# Try unpacking
print('Attempting to unpack like in code: CM, _ = spearmanr(...)')
CM, _ = result
print('  CM type:', type(CM))
print('  CM value:', CM)
if hasattr(CM, 'shape'):
    print('  CM shape:', CM.shape)
if hasattr(CM, 'ndim'):
    print('  CM ndim:', CM.ndim)
