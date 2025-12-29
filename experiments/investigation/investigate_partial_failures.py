"""
Investigate why partial failures occur - when do DRIADA reconstruction methods return NaN?

This script examines actual LNOF neurons with partial failures to understand
what causes reconstruction metrics to fail while signal metrics succeed.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

print('='*80)
print('INVESTIGATING PARTIAL FAILURES')
print('='*80)

# Load v9 dataset
v9 = pd.read_csv('ml/results/training_dataset_v9.csv')

# Find partial failures
event_metrics = [
    'event_r2_score', 'event_snr', 'events_fraction', 'events_per_min',
    'kinetics_opt', 't_off', 't_rise', 'nmae', 'nrmse', 'r2_score', 'snr_recon'
]
has_nan = v9[event_metrics].isna().any(axis=1)
partial_failures = has_nan & (v9['kinetics_source'] != 'error')

print(f'\nPartial failures: {partial_failures.sum()}')
print(f'All from LNOF: {(v9[partial_failures]["experiment"] == "LNOF").all()}')

# Get sample partial failure neurons
samples = v9[partial_failures].head(10)
print('\n' + '='*80)
print('SAMPLE PARTIAL FAILURES')
print('='*80)
for _, row in samples.iterrows():
    print(f'\nSession: {row["session_name"]}, Neuron: {row["component_idx"]}')
    print(f'  Ground truth: {row["ground_truth"]} (KEEP)')
    print(f'  CaImAn SNR: {row["caiman_snr"]:.2f}')
    print(f'  Signal metrics (VALID):')
    print(f'    event_snr: {row["event_snr"]:.4f}')
    print(f'    events_fraction: {row["events_fraction"]:.6f}')
    print(f'    events_per_min: {row["events_per_min"]:.4f}')
    print(f'    kinetics_opt: {row["kinetics_opt"]}')
    print(f'    t_rise: {row["t_rise"]:.4f}')
    print(f'    t_off: {row["t_off"]:.4f}')
    print(f'  Reconstruction metrics (NaN):')
    print(f'    event_r2_score: {row["event_r2_score"]}')
    print(f'    r2_score: {row["r2_score"]}')
    print(f'    nmae: {row["nmae"]}')
    print(f'    nrmse: {row["nrmse"]}')
    print(f'    snr_recon: {row["snr_recon"]}')

# Check if we can load one of these sessions to investigate further
print('\n' + '='*80)
print('ATTEMPTING TO LOAD LNOF SESSION DATA')
print('='*80)

sample_session = samples.iloc[0]['session_name']
print(f'\nLooking for: {sample_session}')

lnof_dir = Path('data/LNOF')
if lnof_dir.exists():
    session_folder = None
    for folder in lnof_dir.iterdir():
        if folder.is_dir() and sample_session in folder.name:
            session_folder = folder
            break

    if session_folder:
        print(f'Found folder: {session_folder.name}')

        # Look for processed pickle
        pickle_files = list(session_folder.glob('*_processed.pickle'))
        if pickle_files:
            pickle_file = pickle_files[0]
            print(f'Loading: {pickle_file.name}')

            try:
                with open(pickle_file, 'rb') as f:
                    est = pickle.load(f)

                if hasattr(est, 'metrics_df') and est.metrics_df is not None:
                    metrics_df = est.metrics_df
                    print(f'\nmetrics_df shape: {metrics_df.shape}')
                    print(f'Columns: {list(metrics_df.columns)}')

                    # Check reconstruction metrics
                    rec_metrics = ['event_r2_score', 'nmae', 'nrmse', 'r2_score', 'snr_recon']
                    print(f'\nReconstruction metric NaN counts in this session:')
                    for m in rec_metrics:
                        if m in metrics_df.columns:
                            nan_count = metrics_df[m].isna().sum()
                            print(f'  {m}: {nan_count}/{len(metrics_df)} ({100*nan_count/len(metrics_df):.1f}%)')
                        else:
                            print(f'  {m}: NOT IN DATAFRAME')

                    # Find the specific neuron
                    sample_idx = samples.iloc[0]['component_idx']
                    neuron_row = metrics_df[metrics_df['component_idx'] == sample_idx]
                    if not neuron_row.empty:
                        print(f'\nFound neuron {sample_idx} in metrics_df:')
                        for m in rec_metrics:
                            if m in neuron_row.columns:
                                val = neuron_row[m].iloc[0]
                                print(f'  {m}: {val}')
                else:
                    print('[WARNING] No metrics_df in estimates')

            except Exception as e:
                print(f'[ERROR] Failed to load: {e}')
        else:
            print('[WARNING] No processed pickle found')
    else:
        print(f'[WARNING] Folder not found for {sample_session}')
else:
    print('[WARNING] data/LNOF directory not found')

print('\n' + '='*80)
print('THEORY: DRIADA METHODS RETURN NaN INTERNALLY')
print('='*80)
print('''
Partial failures occur when:
1. get_neuron_with_spikes() succeeds (creates DRIADA Neuron object)
2. get_signal_metrics() succeeds (computes event_snr, events_fraction, etc.)
3. get_reconstruction_quality_metrics() PARTIALLY fails

In get_reconstruction_quality_metrics():
    r2_score = neuron.get_reconstruction_r2()
    event_r2_score = neuron.get_reconstruction_r2(event_only=True)
    nmae = neuron.get_nmae()
    nrmse = neuron.get_nrmse()
    snr_recon = neuron.get_snr_reconstruction()

These DRIADA methods are called WITHOUT try/except, so if they RAISE exceptions,
the outer try/except in get_single_neuron_metrics() would catch it and ALL metrics
would be NaN.

But we have partial failures where signal metrics are valid and only reconstruction
metrics are NaN. This means:

HYPOTHESIS: DRIADA's reconstruction quality methods internally handle exceptions
and return NaN instead of raising them.

Possible causes:
1. neuron.reconstructed array is None or invalid
2. Not enough events for event_only=True reconstruction
3. Division by zero in quality metric calculations (protected with NaN return)
4. DRIADA version-specific behavior

This would explain why LNOF has exactly 50 neurons with this pattern - something
about those specific traces caused DRIADA reconstruction to fail silently.
''')

print('\n' + '='*80)
print('RECOMMENDATIONS')
print('='*80)
print('''
1. ADD TRY/EXCEPT in get_reconstruction_quality_metrics():
   - Wrap each DRIADA method call individually
   - Return 0 instead of NaN on failure
   - Log which specific method failed

2. VALIDATE reconstruction before computing metrics:
   - Check if neuron.reconstructed exists and is valid
   - Check if reconstruction has sufficient data
   - Return 0 for metrics if reconstruction is invalid

3. HANDLE event_only=True failures specifically:
   - event_r2_score requires events to be present
   - If no events, return 0 instead of letting DRIADA return NaN

4. UPDATE auto_inspector.py exception handler:
   - Change NaN returns to 0 for all event metrics
   - Makes it clear that 0 = "failed to compute" not "measured as zero"
''')
