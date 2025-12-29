"""
Compare all 5 methods on neurons that passed ALL 5 methods.
This gives apples-to-apples comparison on the same neuron set.
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

# Find neurons with events in ALL 5 methods
wvt2_good = set(wvt2[wvt2['t_off'] > -1]['component_idx'].values)
wvt3_good = set(wvt3[wvt3['t_off'] > -1]['component_idx'].values)
thr2_good = set(thr2[thr2['t_off'] > -1]['component_idx'].values)
thr3_good = set(thr3[thr3['t_off'] > -1]['component_idx'].values)
hybrid_good = set(hybrid[hybrid['t_off'] > -1]['component_idx'].values)

print("\n" + "="*80)
print("NEURONS WITH EVENTS BY METHOD")
print("="*80)
print(f"Wavelet n=2:    {len(wvt2_good)} neurons")
print(f"Wavelet n=3:    {len(wvt3_good)} neurons")
print(f"Threshold n=2:  {len(thr2_good)} neurons")
print(f"Threshold n=3:  {len(thr3_good)} neurons")
print(f"Hybrid n=3:     {len(hybrid_good)} neurons")
print()

# Find common neurons (passed ALL 5 methods)
common_neurons = wvt2_good & wvt3_good & thr2_good & thr3_good & hybrid_good

print("="*80)
print("COMMON NEURONS (passed all 5 methods)")
print("="*80)
print(f"Total common neurons: {len(common_neurons)}")
print()

# Filter each dataset to common neurons
wvt2_common = wvt2[wvt2['component_idx'].isin(common_neurons)].sort_values('component_idx')
wvt3_common = wvt3[wvt3['component_idx'].isin(common_neurons)].sort_values('component_idx')
thr2_common = thr2[thr2['component_idx'].isin(common_neurons)].sort_values('component_idx')
thr3_common = thr3[thr3['component_idx'].isin(common_neurons)].sort_values('component_idx')
hybrid_common = hybrid[hybrid['component_idx'].isin(common_neurons)].sort_values('component_idx')

# Verify we have the same neurons
assert len(wvt2_common) == len(common_neurons)
assert len(wvt3_common) == len(common_neurons)
assert len(thr2_common) == len(common_neurons)
assert len(thr3_common) == len(common_neurons)
assert len(hybrid_common) == len(common_neurons)

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
print("METRICS COMPARISON ON COMMON NEURONS (n={})".format(len(common_neurons)))
print("="*80)

for metric, label in metrics:
    print(f"\n{label.upper()}:")
    print("-"*80)
    print(f"{'Method':<20} {'Mean':<12} {'Median':<12} {'Std':<12}")
    print("-"*80)

    for name, df in [('Wavelet n=2', wvt2_common), ('Wavelet n=3', wvt3_common),
                     ('Threshold n=2', thr2_common), ('Threshold n=3', thr3_common),
                     ('Hybrid n=3', hybrid_common)]:
        mean_val = df[metric].mean()
        median_val = df[metric].median()
        std_val = df[metric].std()
        print(f"{name:<20} {mean_val:<12.4f} {median_val:<12.4f} {std_val:<12.4f}")

# Hybrid tier breakdown on common neurons
print("\n" + "="*80)
print("HYBRID KINETICS SOURCE BREAKDOWN (COMMON NEURONS ONLY)")
print("="*80)

tier_counts = hybrid_common['kinetics_source'].value_counts()
print("\nCounts by tier:")
for source, count in tier_counts.items():
    pct = count / len(hybrid_common) * 100
    print(f"  {source:<25} {count:4d} ({pct:5.1f}%)")

print("\nQuality by tier:")
tier_stats = hybrid_common.groupby('kinetics_source').agg({
    'r2_score': ['count', 'mean', 'median'],
    'event_r2_score': ['mean', 'median']
}).round(4)
print(tier_stats.to_string())

# Summary table
print("\n" + "="*80)
print("SUMMARY TABLE (COMMON NEURONS, n={})".format(len(common_neurons)))
print("="*80)

summary_data = []
for name, df in [('Wavelet n=2', wvt2_common), ('Wavelet n=3', wvt3_common),
                 ('Threshold n=2', thr2_common), ('Threshold n=3', thr3_common),
                 ('Hybrid n=3', hybrid_common)]:
    summary_data.append({
        'Method': name,
        'R2': df['r2_score'].mean(),
        'Event_R2': df['event_r2_score'].mean(),
        'Events/min': df['events_per_min'].mean(),
        't_rise': df['t_rise'].mean(),
        't_off': df['t_off'].mean(),
        'Event_SNR': df['event_snr'].mean()
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Pairwise comparisons
print("\n" + "="*80)
print("PAIRWISE COMPARISONS (SAME NEURONS)")
print("="*80)

print("\nHybrid vs Wavelet n=3 (R2 score):")
diff_r2 = hybrid_common['r2_score'].values - wvt3_common['r2_score'].values
print(f"  Mean difference: {diff_r2.mean():+.4f}")
print(f"  Median difference: {np.median(diff_r2):+.4f}")
print(f"  Hybrid better on {(diff_r2 > 0).sum()} neurons ({(diff_r2 > 0).sum()/len(diff_r2)*100:.1f}%)")
print(f"  Wavelet better on {(diff_r2 < 0).sum()} neurons ({(diff_r2 < 0).sum()/len(diff_r2)*100:.1f}%)")

print("\nHybrid vs Wavelet n=3 (Event R2 score):")
diff_event_r2 = hybrid_common['event_r2_score'].values - wvt3_common['event_r2_score'].values
print(f"  Mean difference: {diff_event_r2.mean():+.4f}")
print(f"  Median difference: {np.median(diff_event_r2):+.4f}")
print(f"  Hybrid better on {(diff_event_r2 > 0).sum()} neurons ({(diff_event_r2 > 0).sum()/len(diff_event_r2)*100:.1f}%)")
print(f"  Wavelet better on {(diff_event_r2 < 0).sum()} neurons ({(diff_event_r2 < 0).sum()/len(diff_event_r2)*100:.1f}%)")

print("\nHybrid vs Threshold n=3 (R2 score):")
diff_r2_thr = hybrid_common['r2_score'].values - thr3_common['r2_score'].values
print(f"  Mean difference: {diff_r2_thr.mean():+.4f}")
print(f"  Median difference: {np.median(diff_r2_thr):+.4f}")
print(f"  Hybrid better on {(diff_r2_thr > 0).sum()} neurons ({(diff_r2_thr > 0).sum()/len(diff_r2_thr)*100:.1f}%)")
print(f"  Threshold better on {(diff_r2_thr < 0).sum()} neurons ({(diff_r2_thr < 0).sum()/len(diff_r2_thr)*100:.1f}%)")

# Analyze which neurons each method is best on
print("\n" + "="*80)
print("WHICH METHOD WINS FOR EACH NEURON (R2 score)")
print("="*80)

r2_matrix = np.column_stack([
    wvt2_common['r2_score'].values,
    wvt3_common['r2_score'].values,
    thr2_common['r2_score'].values,
    thr3_common['r2_score'].values,
    hybrid_common['r2_score'].values
])

best_method_idx = np.argmax(r2_matrix, axis=1)
method_names = ['Wavelet n=2', 'Wavelet n=3', 'Threshold n=2', 'Threshold n=3', 'Hybrid n=3']

for i, name in enumerate(method_names):
    count = (best_method_idx == i).sum()
    pct = count / len(common_neurons) * 100
    print(f"{name:<20} is best for {count:4d} neurons ({pct:5.1f}%)")

print("\n" + "="*80)
print("KEY FINDINGS (COMMON NEURONS ONLY)")
print("="*80)
print(f"1. {len(common_neurons)} neurons passed ALL 5 methods")
print()
print(f"2. On these common neurons:")
print(f"   - Best R2: Threshold n=3 ({thr3_common['r2_score'].mean():.4f})")
print(f"   - Best Event R2: Wavelet n=3 ({wvt3_common['event_r2_score'].mean():.4f})")
print(f"   - Hybrid R2: {hybrid_common['r2_score'].mean():.4f}")
print(f"   - Hybrid Event R2: {hybrid_common['event_r2_score'].mean():.4f}")
print()
print(f"3. Hybrid tier breakdown on common neurons:")
tier1_common = hybrid_common[hybrid_common['kinetics_source'] == 'wavelet_standard']
tier2_common = hybrid_common[hybrid_common['kinetics_source'] == 'wavelet_relaxed']
if len(tier1_common) > 0:
    print(f"   - Tier 1 (wavelet_standard): {len(tier1_common)} neurons, R2={tier1_common['r2_score'].mean():.4f}")
if len(tier2_common) > 0:
    print(f"   - Tier 2 (wavelet_relaxed): {len(tier2_common)} neurons, R2={tier2_common['r2_score'].mean():.4f}")
print()
print(f"4. Hybrid wins on {(best_method_idx == 4).sum()} neurons ({(best_method_idx == 4).sum()/len(common_neurons)*100:.1f}%)")
print()
print("="*80)
