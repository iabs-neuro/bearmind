"""
Hybrid Auto-Inspection with Cascading Kinetics Fallback
=========================================================
Uses wavelet for event detection with cascading kinetics optimization:
1. Wavelet standard
2. Wavelet relaxed (min_events=3, min_r2=0.6)
3. Threshold standard
4. Threshold relaxed
5. Defaults

Loads spatial metrics from existing wavelet_iter3, only recomputes event-based metrics.
"""

import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from scipy import sparse

from driada.experiment.neuron import Neuron

# Configuration
BASE_PATH = Path('data/event_param_comparison')
OUTPUT_PATH = BASE_PATH / 'hybrid_iter3'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

FPS = 30
N_ITER = 3


def get_neuron_with_spikes_hybrid(trace, fps, n_iter=3):
    """
    Hybrid approach: wavelet events + cascading kinetics optimization.

    Returns:
        neuron: Neuron object with wavelet events
        kinetics_result: dict from get_kinetics()
        kinetics_source: str indicating which method provided kinetics
    """
    # Stage 1: Single-pass wavelet detection (get initial events for kinetics)
    neuron = Neuron(cell_id="", ca=trace, sp=None, fps=fps)
    neuron.reconstruct_spikes(
        method='wavelet',
        iterative=False,
        create_event_regions=True,
        fps=fps
    )

    # Stage 2: Kinetics cascade
    kinetics_source = 'defaults'
    kinetics_result = {'optimized': False}
    neuron_thr = None

    # 2a: Wavelet standard
    try:
        kinetics_result = neuron.get_kinetics(
            method='direct', fps=fps,
            use_cached=False, update_reconstruction=False
        )
        if kinetics_result.get('optimized'):
            kinetics_source = 'wavelet_standard'
    except Exception:
        pass

    if kinetics_source == 'defaults':
        # 2b: Wavelet relaxed
        try:
            kinetics_result = neuron.get_kinetics(
                method='direct', fps=fps,
                use_cached=False, update_reconstruction=False,
                min_events=3, min_r2=0.6
            )
            if kinetics_result.get('optimized'):
                kinetics_source = 'wavelet_relaxed'
        except Exception:
            pass

    if kinetics_source == 'defaults':
        # 2c: Threshold standard
        try:
            neuron_thr = Neuron(cell_id="", ca=trace, sp=None, fps=fps)
            neuron_thr.reconstruct_spikes(
                method='threshold', n_mad=4.0, min_duration_frames=2,
                iterative=False, create_event_regions=True, fps=fps
            )
            kinetics_result = neuron_thr.get_kinetics(
                method='direct', fps=fps,
                use_cached=False, update_reconstruction=False
            )
            if kinetics_result.get('optimized'):
                neuron.t_rise = neuron_thr.t_rise
                neuron.t_off = neuron_thr.t_off
                kinetics_source = 'threshold_standard'
        except Exception:
            pass

    if kinetics_source == 'defaults':
        # 2d: Threshold relaxed
        try:
            if neuron_thr is None:
                neuron_thr = Neuron(cell_id="", ca=trace, sp=None, fps=fps)
                neuron_thr.reconstruct_spikes(
                    method='threshold', n_mad=4.0, min_duration_frames=2,
                    iterative=False, create_event_regions=True, fps=fps
                )
            kinetics_result = neuron_thr.get_kinetics(
                method='direct', fps=fps,
                use_cached=False, update_reconstruction=False,
                min_events=3, min_r2=0.6
            )
            if kinetics_result.get('optimized'):
                neuron.t_rise = neuron_thr.t_rise
                neuron.t_off = neuron_thr.t_off
                kinetics_source = 'threshold_relaxed'
        except Exception:
            pass

    # Stage 3: Re-run wavelet detection with optimized kinetics
    neuron.reconstruct_spikes(
        method='wavelet',
        create_event_regions=True,
        iterative=True,
        n_iter=n_iter,
        adaptive_thresholds=True
    )

    return neuron, kinetics_result, kinetics_source


def compute_event_metrics(neuron, kinetics_source, fps):
    """Compute event-based metrics from neuron."""

    # Event counts
    n_events = int(np.sum(neuron.asp.data > 0))
    duration_min = len(neuron.asp.data) / fps / 60.0
    events_per_min = n_events / duration_min if duration_min > 0 else 0

    events_dur = float(np.sum(neuron.sp.data.astype(int)))
    events_fraction = events_dur / len(neuron.sp.data) if len(neuron.sp.data) > 0 else 0

    # Kinetics
    t_rise = neuron.t_rise / fps if neuron.t_rise and not pd.isna(neuron.t_rise) else -1
    t_off = neuron.t_off / fps if neuron.t_off and not pd.isna(neuron.t_off) else -1

    # Event SNR
    try:
        event_snr = neuron.get_wavelet_snr()
        if event_snr > 0:
            event_snr = np.log1p(event_snr)
        else:
            event_snr = 0.0
    except:
        event_snr = -1

    # Peak amplitude CV
    peak_amplitudes = neuron.asp.data[neuron.asp.data > 0]
    if len(peak_amplitudes) > 1 and np.mean(peak_amplitudes) > 0:
        peak_amplitude_cv = np.std(peak_amplitudes) / np.mean(peak_amplitudes)
    else:
        peak_amplitude_cv = np.nan

    # Kinetics optimization status
    kinetics_opt = 1.0 if kinetics_source in ['wavelet_standard', 'wavelet_relaxed',
                                                'threshold_standard', 'threshold_relaxed'] else 0.5

    # R2 score (reconstruction quality)
    t_rise_frames = neuron.t_rise if neuron.t_rise else neuron.default_t_rise
    t_off_frames = neuron.t_off if neuron.t_off else neuron.default_t_off

    reconstruction = Neuron.get_restored_calcium(neuron.asp.data, t_rise_frames, t_off_frames)
    calcium = neuron.ca.data

    # Align lengths
    min_len = min(len(calcium), len(reconstruction))
    calcium = calcium[:min_len]
    reconstruction = reconstruction[:min_len]

    # R2
    ss_res = np.sum((calcium - reconstruction) ** 2)
    ss_tot = np.sum((calcium - np.mean(calcium)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # NMAE, NRMSE
    mae = np.mean(np.abs(calcium - reconstruction))
    nmae = mae / (np.max(calcium) - np.min(calcium)) if np.max(calcium) != np.min(calcium) else 0

    rmse = np.sqrt(np.mean((calcium - reconstruction) ** 2))
    nrmse = rmse / (np.max(calcium) - np.min(calcium)) if np.max(calcium) != np.min(calcium) else 0

    # SNR reconstruction
    signal_power = np.var(reconstruction)
    noise_power = np.var(calcium - reconstruction)
    snr_recon = signal_power / noise_power if noise_power > 0 else 0

    return {
        'events_per_min': events_per_min,
        'events_fraction': events_fraction,
        't_rise': t_rise,
        't_off': t_off,
        'event_snr': event_snr,
        'peak_amplitude_cv': peak_amplitude_cv,
        'kinetics_opt': kinetics_opt,
        'r2_score': r2_score,
        'event_r2_score': r2_score,  # Same for now
        'nmae': nmae,
        'nrmse': nrmse,
        'snr_recon': snr_recon,
        'kinetics_source': kinetics_source,
    }


def main():
    print("=" * 80)
    print("HYBRID AUTO-INSPECTION")
    print("=" * 80)

    # Load existing spatial metrics from wavelet_iter3
    spatial_metrics_file = BASE_PATH / 'wavelet_iter3' / 'wavelet_iter3_metrics.csv'
    print(f"\nLoading spatial metrics from: {spatial_metrics_file}")
    spatial_df = pd.read_csv(spatial_metrics_file)
    print(f"  Loaded {len(spatial_df)} neurons")

    # Load estimates for calcium traces
    est_file = BASE_PATH / 'wavelet_iter3' / 'wavelet_iter3_estimates.pkl'
    print(f"Loading estimates from: {est_file}")
    with open(est_file, 'rb') as f:
        est = pickle.load(f)

    # Process each neuron with hybrid approach
    print(f"\nProcessing {len(spatial_df)} neurons with hybrid kinetics...")
    t_start = time.time()

    event_metrics_list = []
    kinetics_counts = {
        'wavelet_standard': 0,
        'wavelet_relaxed': 0,
        'threshold_standard': 0,
        'threshold_relaxed': 0,
        'defaults': 0,
        'error': 0
    }

    for i, row in spatial_df.iterrows():
        comp_idx = row['component_idx']

        # Progress
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(spatial_df) - i - 1) / rate
            print(f"  [{i+1}/{len(spatial_df)}] {rate:.1f} neurons/sec, ETA: {remaining:.0f}s")

        # Find neuron in estimates
        try:
            pos = np.where(est.idx_components == comp_idx)[0]
            if len(pos) == 0:
                event_metrics_list.append({'component_idx': comp_idx, 'kinetics_source': 'error'})
                kinetics_counts['error'] += 1
                continue
            pos = pos[0]

            # Get calcium trace
            C = est.C[pos, :]
            if sparse.issparse(C):
                C = C.toarray().flatten()
            C = np.asarray(C, dtype=np.float64)

            # Run hybrid approach
            neuron, kinetics_result, kinetics_source = get_neuron_with_spikes_hybrid(C, FPS, N_ITER)

            # Compute event metrics
            metrics = compute_event_metrics(neuron, kinetics_source, FPS)
            metrics['component_idx'] = comp_idx
            event_metrics_list.append(metrics)

            kinetics_counts[kinetics_source] += 1

        except Exception as e:
            event_metrics_list.append({'component_idx': comp_idx, 'kinetics_source': 'error'})
            kinetics_counts['error'] += 1

    elapsed_total = time.time() - t_start
    print(f"\nProcessed {len(spatial_df)} neurons in {elapsed_total:.1f}s ({len(spatial_df)/elapsed_total:.1f} neurons/sec)")

    # Merge spatial and event metrics
    event_df = pd.DataFrame(event_metrics_list)

    # Columns to keep from spatial (exclude event-based columns that we recomputed)
    event_cols = ['events_per_min', 'events_fraction', 't_rise', 't_off', 'event_snr',
                  'peak_amplitude_cv', 'kinetics_opt', 'r2_score', 'event_r2_score',
                  'nmae', 'nrmse', 'snr_recon']
    spatial_cols = [c for c in spatial_df.columns if c not in event_cols and c != 'kinetics_source']

    merged_df = spatial_df[spatial_cols].merge(event_df, on='component_idx', how='left')

    # Save results
    output_file = OUTPUT_PATH / 'hybrid_iter3_metrics.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("KINETICS SOURCE BREAKDOWN")
    print("=" * 80)
    total = sum(kinetics_counts.values())
    for source, count in kinetics_counts.items():
        if count > 0:
            print(f"  {source}: {count} ({100*count/total:.1f}%)")

    print("\n" + "=" * 80)
    print("R2 BY KINETICS SOURCE")
    print("=" * 80)
    for source in ['wavelet_standard', 'wavelet_relaxed', 'threshold_standard', 'threshold_relaxed', 'defaults']:
        subset = merged_df[merged_df['kinetics_source'] == source]
        if len(subset) > 0:
            print(f"  {source}: mean R2 = {subset['r2_score'].mean():.4f} (n={len(subset)})")

    print(f"\nOverall mean R2: {merged_df['r2_score'].mean():.4f}")


if __name__ == '__main__':
    main()
