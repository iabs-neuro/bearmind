"""Wavelet backend selection and GPU configuration for BEARMiND.

This module provides a simple interface to enable GPU acceleration for wavelet
computations using ssqueezepy's built-in GPU support.

Usage:
    from wavelet_backend import set_wavelet_backend

    # Auto-detect GPU (default)
    backend = set_wavelet_backend('auto')

    # Force GPU (error if not available)
    backend = set_wavelet_backend('gpu')

    # Force CPU
    backend = set_wavelet_backend('cpu')
"""
import os
import warnings

# Backend state (set once at module load)
_BACKEND = None
_GPU_AVAILABLE = None


def _check_gpu_available():
    """Check if GPU dependencies are available.

    Returns
    -------
    bool
        True if both CuPy and PyTorch with CUDA are available.
    """
    try:
        import cupy as cp
        import torch
        return cp.cuda.is_available() and torch.cuda.is_available()
    except Exception:
        # Catches ImportError (missing packages) and CUDARuntimeError (driver issues)
        return False


def set_wavelet_backend(backend='auto'):
    """Set wavelet computation backend.

    This function configures ssqueezepy to use GPU or CPU for wavelet
    transforms. Must be called BEFORE importing ssqueezepy or DRIADA.

    Parameters
    ----------
    backend : str, default 'auto'
        Backend selection:
        - 'auto': Use GPU if available, else CPU
        - 'gpu': Force GPU (raises RuntimeError if unavailable)
        - 'cpu': Force CPU

    Returns
    -------
    str
        Actual backend used ('gpu' or 'cpu')

    Raises
    ------
    RuntimeError
        If backend='gpu' but GPU is not available
    ValueError
        If backend is not one of 'auto', 'gpu', 'cpu'

    Notes
    -----
    Changes require Python restart to take effect if ssqueezepy/DRIADA
    are already imported. The backend cannot be changed after it's set.

    Examples
    --------
    >>> from wavelet_backend import set_wavelet_backend
    >>> backend = set_wavelet_backend('auto')
    >>> print(f"Using {backend} backend")
    Using gpu backend
    """
    global _BACKEND, _GPU_AVAILABLE

    # Warn if backend already set (cannot change without restart)
    if _BACKEND is not None:
        warnings.warn(
            f"Backend already set to '{_BACKEND}'. "
            "Restart Python to change backends.",
            UserWarning
        )
        return _BACKEND

    # Check GPU availability once
    if _GPU_AVAILABLE is None:
        _GPU_AVAILABLE = _check_gpu_available()

    # Determine backend
    if backend == 'cpu':
        actual_backend = 'cpu'
    elif backend == 'gpu':
        if not _GPU_AVAILABLE:
            raise RuntimeError(
                "GPU backend requested but GPU not available. "
                "Install GPU dependencies: "
                "conda install cupy-cuda11x pytorch -c pytorch -c conda-forge"
            )
        actual_backend = 'gpu'
    elif backend == 'auto':
        actual_backend = 'gpu' if _GPU_AVAILABLE else 'cpu'
    else:
        raise ValueError(
            f"Invalid backend: {backend}. "
            f"Must be one of: 'auto', 'gpu', 'cpu'"
        )

    # Configure ssqueezepy environment variables
    if actual_backend == 'gpu':
        os.environ['SSQ_GPU'] = '1'
        print("Wavelet backend: GPU (ssqueezepy GPU mode enabled)")
    else:
        # Ensure GPU mode is disabled
        os.environ.pop('SSQ_GPU', None)
        os.environ['SSQ_PARALLEL'] = '1'  # CPU multi-threaded
        print("Wavelet backend: CPU (multi-threaded)")

    _BACKEND = actual_backend
    return actual_backend


def get_current_backend():
    """Get currently active backend.

    Returns
    -------
    str
        Current backend: 'gpu', 'cpu', or 'not_set'

    Examples
    --------
    >>> from wavelet_backend import get_current_backend
    >>> print(get_current_backend())
    not_set
    """
    return _BACKEND if _BACKEND is not None else 'not_set'


def is_gpu_backend():
    """Check if GPU backend is active.

    Returns
    -------
    bool
        True if GPU backend is set and active

    Examples
    --------
    >>> from wavelet_backend import set_wavelet_backend, is_gpu_backend
    >>> set_wavelet_backend('cpu')
    >>> print(is_gpu_backend())
    False
    """
    return _BACKEND == 'gpu'
