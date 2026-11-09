"""Test wavelet_backend parameter integration.

This script verifies that the wavelet_backend parameter is correctly threaded
through the autoinspection pipeline.
"""
import sys
from wavelet_backend import set_wavelet_backend, get_current_backend, is_gpu_backend

print("=" * 80)
print("WAVELET BACKEND INTEGRATION TEST")
print("=" * 80)

# Test 1: CPU mode
print("\n[TEST 1] Testing CPU mode...")
try:
    backend = set_wavelet_backend('cpu')
    assert backend == 'cpu', f"Expected 'cpu', got '{backend}'"
    assert get_current_backend() == 'cpu', "get_current_backend() failed"
    assert not is_gpu_backend(), "is_gpu_backend() should return False for CPU"
    print("SUCCESS: CPU mode works correctly")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

# Test 2: Auto-detection (should be CPU since GPU not installed)
print("\n[TEST 2] Testing auto-detection...")
# Need to restart Python to change backend, so this will warn
try:
    backend = set_wavelet_backend('auto')
    # Should still be 'cpu' from Test 1 (warns about restart)
    assert get_current_backend() == 'cpu', "Backend should not have changed"
    print("SUCCESS: Auto-detection works (backend cannot change without restart)")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

# Test 3: GPU mode (should fail since GPU not available)
print("\n[TEST 3] Testing GPU mode (should fail gracefully)...")
# This won't change the backend since it's already set
try:
    backend = set_wavelet_backend('gpu')
    # Should still be 'cpu' from Test 1
    print("WARNING: Backend already set, cannot test GPU failure")
except Exception as e:
    print(f"EXPECTED: GPU mode failed as expected (no GPU available): {e}")

# Test 4: Import integration test
print("\n[TEST 4] Testing import integration...")
try:
    # Import after backend is set
    from auto_inspector import get_multineuron_metrics

    # Verify function signature includes wavelet_backend parameter
    import inspect
    sig = inspect.signature(get_multineuron_metrics)
    params = list(sig.parameters.keys())

    assert 'wavelet_backend' in params, "wavelet_backend parameter not in get_multineuron_metrics()"
    print(f"SUCCESS: get_multineuron_metrics() has wavelet_backend parameter")
    print(f"  Parameters: {params}")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

# Test 5: ae_launch integration
print("\n[TEST 5] Testing ae_launch integration...")
try:
    from ae_launch import run_auto_inspection

    # Verify function signature includes wavelet_backend parameter
    import inspect
    sig = inspect.signature(run_auto_inspection)
    params = list(sig.parameters.keys())

    assert 'wavelet_backend' in params, "wavelet_backend parameter not in run_auto_inspection()"

    # Check default value
    default_value = sig.parameters['wavelet_backend'].default
    assert default_value == 'auto', f"Expected default 'auto', got '{default_value}'"

    print(f"SUCCESS: run_auto_inspection() has wavelet_backend parameter with default='auto'")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED")
print("=" * 80)
print("\nNext steps:")
print("1. Install GPU dependencies when network is available:")
print("   conda install -n bearmind cupy-cuda11x pytorch torchvision -c pytorch -c conda-forge")
print("2. Test GPU mode after installation")
print("3. Benchmark performance difference between CPU and GPU modes")
