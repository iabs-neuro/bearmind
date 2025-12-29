"""
Deep dive: Why do DRIADA reconstruction quality methods return NaN for good neurons?

This script:
1. Loads a partial failure neuron
2. Recreates the DRIADA Neuron object
3. Calls each reconstruction quality method individually
4. Inspects what happens and why NaN is returned
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Add current directory to path for auto_inspector imports
sys.path.insert(0, '.')

def investigate_driada_failure(session_name, component_idx):
    """
    Load a specific neuron and investigate why DRIADA metrics return NaN.
    """
    print('='*80)
    print(f'INVESTIGATING: {session_name}, Neuron {component_idx}')
    print('='*80)

    # Load processed estimates
    lnof_dir = Path('data/LNOF')
    est = None

    for folder in lnof_dir.iterdir():
        if folder.is_dir() and session_name in folder.name:
            pickle_files = list(folder.glob('*_processed.pickle'))
            if pickle_files:
                with open(pickle_files[0], 'rb') as f:
                    est = pickle.load(f)
                break

    if est is None:
        print(f'[ERROR] Could not load {session_name}')
        return

    # Get trace
    if not hasattr(est, 'C') or est.C is None:
        print('[ERROR] No C matrix')
        return

    traces = est.C
    fps = getattr(est, 'fps', 30.0)

    # Get the trace for this component
    # component_idx should match the row in C matrix
    if component_idx >= traces.shape[0]:
        print(f'[ERROR] component_idx {component_idx} out of range (C has {traces.shape[0]} components)')
        return

    trace = traces[component_idx, :]

    print(f'\nTrace loaded:')
    print(f'  Length: {len(trace)} timepoints')
    print(f'  FPS: {fps}')
    print(f'  Duration: {len(trace)/fps/60:.2f} minutes')
    print(f'  Min: {np.min(trace):.4f}')
    print(f'  Max: {np.max(trace):.4f}')
    print(f'  Mean: {np.mean(trace):.4f}')
    print(f'  Std: {np.std(trace):.4f}')

    # Now recreate DRIADA processing
    print('\n' + '='*80)
    print('STEP 1: CREATE DRIADA NEURON')
    print('='*80)

    try:
        from driada.experiment.neuron import Neuron
        from driada.deconvolve import PRS_cython

        print('Creating DRIADA Neuron object...')
        neuron = Neuron(data=trace, fps=fps, deconvolution_method=PRS_cython())
        print('[SUCCESS] Neuron object created')

    except Exception as e:
        print(f'[ERROR] Failed to create neuron: {e}')
        import traceback
        traceback.print_exc()
        return

    # Step 2: Event detection
    print('\n' + '='*80)
    print('STEP 2: EVENT DETECTION')
    print('='*80)

    try:
        print('Running wavelet event detection...')
        neuron.run_wavelet_event_detection()
        n_events = int(np.sum(neuron.asp.data > 0))
        print(f'[SUCCESS] Event detection completed')
        print(f'  Events detected: {n_events}')
        print(f'  Events fraction: {np.sum(neuron.sp.data)/len(neuron.sp.data):.6f}')

    except Exception as e:
        print(f'[ERROR] Event detection failed: {e}')
        import traceback
        traceback.print_exc()
        return

    # Step 3: Kinetics optimization
    print('\n' + '='*80)
    print('STEP 3: KINETICS OPTIMIZATION')
    print('='*80)

    try:
        print('Optimizing kinetics...')
        from driada.experiment.kinetics_optimization import optimize_prs_params_ls
        result = optimize_prs_params_ls(neuron, ls_beta=0.7, min_events=10, min_r2=0.65)

        print(f'[SUCCESS] Kinetics optimization completed')
        print(f'  Optimized: {result.get("optimized", False)}')
        print(f'  t_rise: {neuron.t_rise}')
        print(f'  t_off: {neuron.t_off}')

    except Exception as e:
        print(f'[WARNING] Kinetics optimization failed: {e}')
        # This can fail but we continue

    # Step 4: Reconstruction (this is critical!)
    print('\n' + '='*80)
    print('STEP 4: RECONSTRUCTION')
    print('='*80)

    try:
        print('Checking reconstruction...')

        # Check if reconstruction exists
        if hasattr(neuron, 'reconstructed'):
            rec = neuron.reconstructed
            print(f'  neuron.reconstructed exists: {rec is not None}')

            if rec is not None:
                # Check if it's a TimeSeries object
                print(f'  Type: {type(rec)}')

                if hasattr(rec, 'data'):
                    print(f'  rec.data shape: {rec.data.shape if hasattr(rec.data, "shape") else len(rec.data)}')
                    print(f'  rec.data min: {np.min(rec.data):.4f}')
                    print(f'  rec.data max: {np.max(rec.data):.4f}')
                    print(f'  rec.data mean: {np.mean(rec.data):.4f}')
                    print(f'  rec.data has NaN: {np.isnan(rec.data).any()}')
                    print(f'  rec.data has Inf: {np.isinf(rec.data).any()}')

                if hasattr(rec, 'scdata'):
                    print(f'  rec.scdata exists: {rec.scdata is not None}')
                    if rec.scdata is not None:
                        print(f'  rec.scdata shape: {rec.scdata.shape if hasattr(rec.scdata, "shape") else len(rec.scdata)}')
        else:
            print('  [WARNING] neuron.reconstructed does not exist!')

    except Exception as e:
        print(f'[ERROR] Reconstruction check failed: {e}')
        import traceback
        traceback.print_exc()

    # Step 5: Try each reconstruction quality method individually
    print('\n' + '='*80)
    print('STEP 5: TEST EACH RECONSTRUCTION QUALITY METHOD')
    print('='*80)

    methods_to_test = [
        ('get_reconstruction_r2()', lambda: neuron.get_reconstruction_r2()),
        ('get_reconstruction_r2(event_only=True)', lambda: neuron.get_reconstruction_r2(event_only=True)),
        ('get_nmae()', lambda: neuron.get_nmae()),
        ('get_nrmse()', lambda: neuron.get_nrmse()),
        ('get_snr_reconstruction()', lambda: neuron.get_snr_reconstruction()),
    ]

    for method_name, method_func in methods_to_test:
        print(f'\nTesting: {method_name}')
        try:
            result = method_func()
            if pd.isna(result):
                print(f'  [RETURNED NaN]')
            else:
                print(f'  [SUCCESS] Result: {result:.6f}')
        except Exception as e:
            print(f'  [EXCEPTION] {type(e).__name__}: {e}')

    # Step 6: Check number of iterations
    print('\n' + '='*80)
    print('STEP 6: CHECK RECONSTRUCTION ITERATIONS')
    print('='*80)

    print('The issue might be related to n_iter parameter in reconstruction.')
    print('By default, DRIADA uses n_iter=2 for iterative reconstruction.')
    print('Let me check if this neuron was processed with iterative reconstruction...')

    # Check if we can get more details
    if hasattr(neuron, '_reconstruction_params'):
        print(f'  Reconstruction params: {neuron._reconstruction_params}')

    print('\n' + '='*80)
    print('DIAGNOSIS ATTEMPT')
    print('='*80)

    print('\nPossible reasons for NaN returns:')
    print('1. Reconstruction array is None or empty')
    print('2. Reconstruction has invalid values (NaN/Inf)')
    print('3. Not enough events for event_only=True metrics')
    print('4. Division by zero in metric calculations')
    print('5. DRIADA version-specific edge cases')
    print('6. Iterative reconstruction (n_iter) specific issues')

    # Try to get DRIADA version
    try:
        import driada
        print(f'\nDRIADA version: {driada.__version__}')
    except:
        print('\nDRIADA version: unknown')


def main():
    """Test with known partial failure neurons."""
    print('='*80)
    print('DEBUGGING DRIADA NaN RETURNS')
    print('='*80)

    # Test cases from v9 dataset
    test_cases = [
        ('LNOF_J01_2D', 492),
        ('LNOF_J01_2D', 493),
        ('LNOF_J01_3D', 510),
    ]

    for session_name, component_idx in test_cases[:1]:  # Start with just one
        investigate_driada_failure(session_name, component_idx)
        print('\n' + '='*80)
        print('')
        break  # Just do first one for now

    print('\n' + '='*80)
    print('NEXT STEPS')
    print('='*80)
    print('\nIf reconstruction methods return NaN:')
    print('1. Check DRIADA source code for when NaN is returned')
    print('2. Test with different n_iter values')
    print('3. Test with different deconvolution methods')
    print('4. Check if this is a known DRIADA issue')
    print('5. Report to DRIADA developers if needed')
    print('\nFor now: wrap these methods and return 0 on NaN as practical solution')


if __name__ == '__main__':
    main()
