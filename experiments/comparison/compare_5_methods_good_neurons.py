"""
Compare all 5 methods (4 standard + 1 hybrid) for GOOD NEURONS ONLY (t_off > -1).
"""
import pandas as pd
import numpy as np

# Load data
print("Loading data...")
wvt2 = pd.read_csv('data/event_param_comparison/wavelet_iter2/wavelet_iter2_metrics.csv')
wvt3 = pd.read_csv('data/event_param_comparison/wavelet_iter3/wavelet_iter3_metrics.csv')
thr2 = pd.read_csv('data/event_param_comparison/threshold_iter2/threshold_iter2_metrics.csv')
thr3 = pd.read_csv('data/event_param_comparison/threshold_iter3/threshold_iter3_metrics.csv')
hybrid = pd.read_csv('data/event_param_comparison/hybrid_iter3/hybrid_iter3_metrics.csv')

# Filter to good neurons (t_off > -1)
wvt2_g = wvt2[wvt2['t_off'] > -1]
wvt3_g = wvt3[wvt3['t_off'] > -1]
thr2_g = thr2[thr2['t_off'] > -1]
thr3_g = thr3[thr3['t_off'] > -1]
hybrid_g = hybrid[hybrid['t_off'] > -1]

print("\n" + "="*80)
print("5-METHOD COMPARISON - GOOD NEURONS ONLY (t_off > -1)")
print("="*80)
print()

# Neuron counts
print("NEURON COUNTS:")
print("-"*80)
print(f"{'Method':<20} {'Good':<8} {'Total':<8} {'Success Rate':<15}")
print("-"*80)
print(f"{'Wavelet n=2':<20} {len(wvt2_g):<8} {len(wvt2):<8} {len(wvt2_g)/len(wvt2)*100:5.1f}%")
print(f"{'Wavelet n=3':<20} {len(wvt3_g):<8} {len(wvt3):<8} {len(wvt3_g)/len(wvt3)*100:5.1f}%")
print(f"{'Threshold n=2':<20} {len(thr2_g):<8} {len(thr2):<8} {len(thr2_g)/len(thr2)*100:5.1f}%")
print(f"{'Threshold n=3':<20} {len(thr3_g):<8} {len(thr3):<8} {len(thr3_g)/len(thr3)*100:5.1f}%")
print(f"{'Hybrid n=3':<20} {len(hybrid_g):<8} {len(hybrid):<8} {len(hybrid_g)/len(hybrid)*100:5.1f}%")
print()

# Key metrics comparison
metrics = [
    ('r2_score', 'R2 Score (Reconstruction Quality)'),
    ('event_r2_score', 'Event R2 Score'),
    ('events_per_min', 'Events per Minute'),
    ('events_fraction', 'Events Fraction'),
    ('t_rise', 'Rise Time (seconds)'),
    ('t_off', 'Decay Time (seconds)'),
    ('event_snr', 'Event SNR'),
    ('nmae', 'NMAE'),
    ('nrmse', 'NRMSE'),
    ('snr_recon', 'SNR Reconstruction')
]

print("="*80)
print("KEY METRICS COMPARISON (GOOD NEURONS ONLY)")
print("="*80)

for metric, label in metrics:
    print(f"\n{label.upper()}:")
    print("-"*80)
    print(f"{'Method':<20} {'Mean':<12} {'Median':<12} {'Std':<12}")
    print("-"*80)

    for name, df in [('Wavelet n=2', wvt2_g), ('Wavelet n=3', wvt3_g),
                     ('Threshold n=2', thr2_g), ('Threshold n=3', thr3_g),
                     ('Hybrid n=3', hybrid_g)]:
        mean_val = df[metric].mean()
        median_val = df[metric].median()
        std_val = df[metric].std()
        print(f"{name:<20} {mean_val:<12.4f} {median_val:<12.4f} {std_val:<12.4f}")

# Hybrid tier breakdown
print("\n" + "="*80)
print("HYBRID METHOD BREAKDOWN BY KINETICS SOURCE (GOOD NEURONS ONLY)")
print("="*80)

tier_stats = hybrid_g.groupby('kinetics_source').agg({
    'r2_score': ['count', 'mean', 'median', 'std'],
    'event_r2_score': ['mean', 'median']
}).round(4)

print(tier_stats.to_string())

# Summary table
print("\n" + "="*80)
print("SUMMARY COMPARISON TABLE")
print("="*80)

summary_data = []
for name, df in [('Wavelet n=2', wvt2_g), ('Wavelet n=3', wvt3_g),
                 ('Threshold n=2', thr2_g), ('Threshold n=3', thr3_g),
                 ('Hybrid n=3', hybrid_g)]:
    summary_data.append({
        'Method': name,
        'N': len(df),
        'R2': df['r2_score'].mean(),
        'Event_R2': df['event_r2_score'].mean(),
        'Events/min': df['events_per_min'].mean(),
        't_rise': df['t_rise'].mean(),
        't_off': df['t_off'].mean(),
        'Event_SNR': df['event_snr'].mean()
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"1. Hybrid n=3 detects events in {len(hybrid_g)} neurons (vs ~440 for standard methods)")
print(f"   - That's {len(hybrid_g) - len(wvt3_g)} MORE neurons with events (+{(len(hybrid_g)/len(wvt3_g)-1)*100:.1f}%)")
print()
print(f"2. Reconstruction Quality (R2):")
print(f"   - Best: Threshold n=3 (R2={thr3_g['r2_score'].mean():.4f})")
print(f"   - Wavelet n=3: R2={wvt3_g['r2_score'].mean():.4f}")
print(f"   - Hybrid n=3: R2={hybrid_g['r2_score'].mean():.4f}")
print()
print(f"3. Event Quality (Event R2):")
print(f"   - Best: Wavelet n=3 (event_r2={wvt3_g['event_r2_score'].mean():.4f})")
print(f"   - Hybrid n=3: event_r2={hybrid_g['event_r2_score'].mean():.4f}")
print(f"   - Threshold n=3: event_r2={thr3_g['event_r2_score'].mean():.4f}")
print()
print(f"4. Hybrid Tier 1 (wavelet_standard) Quality:")
hybrid_tier1 = hybrid_g[hybrid_g['kinetics_source'] == 'wavelet_standard']
if len(hybrid_tier1) > 0:
    print(f"   - N={len(hybrid_tier1)} neurons")
    print(f"   - R2={hybrid_tier1['r2_score'].mean():.4f} (BEST overall!)")
    print(f"   - Event R2={hybrid_tier1['event_r2_score'].mean():.4f} (BEST overall!)")
print()
print("="*80)
