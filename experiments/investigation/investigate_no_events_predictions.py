"""
Investigate how v6/v7/v8 make predictions on neurons where event detection failed.
Critical question: How can v7 predict on 100% when only 71% had events detected?
"""
import pickle
import numpy as np
import pandas as pd

print('='*80)
print('INVESTIGATING PREDICTIONS ON NEURONS WITHOUT EVENTS')
print('='*80)

# Load datasets
v6 = pd.read_csv('ml/results/training_dataset_v6_no3dm.csv')
v7 = pd.read_csv('ml/results/training_dataset_v7.csv')
v8 = pd.read_csv('ml/results/training_dataset_v8.csv')

print(f'\nDataset sizes: {len(v6):,} neurons each')

# Check event detection success rate (t_off > -1 means events detected)
v6_has_events = (v6['t_off'] > -1).sum()
v7_has_events = (v7['t_off'] > -1).sum()
v8_has_events = (v8['t_off'] > -1).sum()

print(f'\n{"="*80}')
print('EVENT DETECTION SUCCESS RATES')
print('='*80)

print(f'\n{"Version":<15} {"Events Detected":<20} {"% Success":<15} {"No Events"}')
print('-'*70)
print(f'{"v6_no3dm":<15} {v6_has_events:<20,} {v6_has_events/len(v6)*100:<15.1f} {len(v6)-v6_has_events:,}')
print(f'{"v7":<15} {v7_has_events:<20,} {v7_has_events/len(v7)*100:<15.1f} {len(v7)-v7_has_events:,}')
print(f'{"v8":<15} {v8_has_events:<20,} {v8_has_events/len(v8)*100:<15.1f} {len(v8)-v8_has_events:,}')

# Create neuron_id EARLY for matching
v6['neuron_id'] = v6['session'] + '_' + v6['component_idx'].astype(str)
v7['neuron_id'] = v7['session'] + '_' + v7['component_idx'].astype(str)
v8['neuron_id'] = v8['session'] + '_' + v8['component_idx'].astype(str)

# Identify neurons without events in each version
v6_no_events = v6[v6['t_off'] == -1].copy()
v7_no_events = v7[v7['t_off'] == -1].copy()
v8_no_events = v8[v8['t_off'] == -1].copy()

print(f'\n{"="*80}')
print('FEATURES AVAILABLE FOR NEURONS WITHOUT EVENTS')
print('='*80)

# Check what values event-based features have when no events detected
event_based_features = ['events_per_min', 'events_fraction', 't_rise', 't_off',
                        'event_snr', 'event_r2_score', 'peak_amplitude_cv']

print(f'\nv7 neurons without events (sample of 5):')
if len(v7_no_events) > 0:
    sample = v7_no_events.head(5)[event_based_features + ['trace_skewness', 'area', 'hurst_exponent', 'ground_truth']]
    print(sample.to_string(index=False))

# Load models and make predictions
with open('ml/ebm_grid_search_v6_no3dm/ebm_best.pkl', 'rb') as f:
    v6_model = pickle.load(f)
with open('ml/ebm_grid_search_v7/ebm_best.pkl', 'rb') as f:
    v7_model = pickle.load(f)
with open('ml/ebm_grid_search_v8/ebm_best.pkl', 'rb') as f:
    v8_model = pickle.load(f)

exclude_cols = {'ground_truth', 'session', 'experiment', 'component_idx', 'center',
                'distance_to_gt', 'is_corner_artifact', 'corr_groups'}

# Get predictions on full datasets
def get_predictions(df, model, threshold=0.75):
    features = [c for c in df.columns if c not in exclude_cols and c in model.feature_names_in_]
    X = df[features].copy()
    y = df['ground_truth'].values
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pred, proba, y

v6_pred, v6_proba, y_v6 = get_predictions(v6, v6_model)
v7_pred, v7_proba, y_v7 = get_predictions(v7, v7_model)
v8_pred, v8_proba, y_v8 = get_predictions(v8, v8_model)

# Add predictions to dataframes (neuron_id already created earlier)
v6['prediction'] = v6_pred
v6['proba'] = v6_proba
v7['prediction'] = v7_pred
v7['proba'] = v7_proba
v8['prediction'] = v8_pred
v8['proba'] = v8_proba

print(f'\n{"="*80}')
print('MODEL PERFORMANCE ON NEURONS WITHOUT EVENTS')
print('='*80)

# Analyze performance on neurons without events in v7
print(f'\nNeurons where v7 FAILED event detection: {len(v7_no_events):,} ({len(v7_no_events)/len(v7)*100:.1f}%)')

if len(v7_no_events) > 0:
    # Get these neurons in all datasets
    v7_no_events_ids = set(v7_no_events['neuron_id'])

    v6_subset = v6[v6['neuron_id'].isin(v7_no_events_ids)]
    v7_subset = v7[v7['neuron_id'].isin(v7_no_events_ids)]
    v8_subset = v8[v8['neuron_id'].isin(v7_no_events_ids)]

    print(f'\nMatching neurons in each dataset:')
    print(f'  v6: {len(v6_subset):,}')
    print(f'  v7: {len(v7_subset):,}')
    print(f'  v8: {len(v8_subset):,}')

    # Ground truth distribution for these neurons
    print(f'\nGround truth for v7 no-event neurons:')
    v7_no_events_gt = v7_subset['ground_truth'].value_counts()
    print(f'  KEEP (1): {v7_no_events_gt.get(1, 0):,} ({v7_no_events_gt.get(1, 0)/len(v7_subset)*100:.1f}%)')
    print(f'  DELETE (0): {v7_no_events_gt.get(0, 0):,} ({v7_no_events_gt.get(0, 0)/len(v7_subset)*100:.1f}%)')

    # Performance on these neurons
    print(f'\n{"="*80}')
    print('ACCURACY ON v7 NO-EVENT NEURONS (threshold=0.75)')
    print('='*80)

    v6_acc = (v6_subset['prediction'] == v6_subset['ground_truth']).mean() * 100
    v7_acc = (v7_subset['prediction'] == v7_subset['ground_truth']).mean() * 100
    v8_acc = (v8_subset['prediction'] == v8_subset['ground_truth']).mean() * 100

    print(f'\n{"Model":<15} {"Accuracy":<12} {"Correct":<12} {"Total"}')
    print('-'*55)
    print(f'{"v6_no3dm":<15} {v6_acc:<12.2f} {(v6_subset["prediction"] == v6_subset["ground_truth"]).sum():<12,} {len(v6_subset):,}')
    print(f'{"v7":<15} {v7_acc:<12.2f} {(v7_subset["prediction"] == v7_subset["ground_truth"]).sum():<12,} {len(v7_subset):,}')
    print(f'{"v8":<15} {v8_acc:<12.2f} {(v8_subset["prediction"] == v8_subset["ground_truth"]).sum():<12,} {len(v8_subset):,}')

    print(f'\nComparison:')
    print(f'  v8 vs v7: {v8_acc - v7_acc:+.2f}%')
    print(f'  v8 vs v6: {v8_acc - v6_acc:+.2f}%')

    # Check if v8 has events for these neurons
    v8_subset_has_events = (v8_subset['t_off'] > -1).sum()
    print(f'\n{"="*80}')
    print('DOES v8 SUCCEED WHERE v7 FAILED?')
    print('='*80)
    print(f'\nOf {len(v8_subset):,} neurons where v7 failed event detection:')
    print(f'  v8 detected events: {v8_subset_has_events:,} ({v8_subset_has_events/len(v8_subset)*100:.1f}%)')
    print(f'  v8 also failed: {len(v8_subset) - v8_subset_has_events:,} ({(len(v8_subset) - v8_subset_has_events)/len(v8_subset)*100:.1f}%)')

    # Compare accuracy for neurons where v8 succeeded vs failed
    v8_rescued = v8_subset[v8_subset['t_off'] > -1]
    v8_still_failed = v8_subset[v8_subset['t_off'] == -1]

    if len(v8_rescued) > 0:
        v8_rescued_acc = (v8_rescued['prediction'] == v8_rescued['ground_truth']).mean() * 100
        print(f'\nv8 accuracy on neurons it RESCUED (detected events where v7 failed):')
        print(f'  Accuracy: {v8_rescued_acc:.2f}% ({(v8_rescued["prediction"] == v8_rescued["ground_truth"]).sum():,}/{len(v8_rescued):,})')

    if len(v8_still_failed) > 0:
        v8_still_failed_acc = (v8_still_failed['prediction'] == v8_still_failed['ground_truth']).mean() * 100
        print(f'\nv8 accuracy on neurons where BOTH v7 and v8 failed:')
        print(f'  Accuracy: {v8_still_failed_acc:.2f}% ({(v8_still_failed["prediction"] == v8_still_failed["ground_truth"]).sum():,}/{len(v8_still_failed):,})')

# Overall comparison
print(f'\n{"="*80}')
print('OVERALL COMPARISON: NEURONS WITH vs WITHOUT EVENTS')
print('='*80)

v7_has_events_subset = v7[v7['t_off'] > -1]
v7_no_events_subset = v7[v7['t_off'] == -1]

if len(v7_has_events_subset) > 0 and len(v7_no_events_subset) > 0:
    v7_events_acc = (v7_has_events_subset['prediction'] == v7_has_events_subset['ground_truth']).mean() * 100
    v7_no_events_acc = (v7_no_events_subset['prediction'] == v7_no_events_subset['ground_truth']).mean() * 100

    print(f'\nv7 model performance:')
    print(f'  Neurons WITH events ({len(v7_has_events_subset):,}): {v7_events_acc:.2f}% accuracy')
    print(f'  Neurons WITHOUT events ({len(v7_no_events_subset):,}): {v7_no_events_acc:.2f}% accuracy')
    print(f'  Difference: {v7_events_acc - v7_no_events_acc:+.2f}%')

    if v7_no_events_acc > 50:
        print(f'\nv7 can still make predictions on neurons without events!')
        print(f'It uses non-event features: trace stats, morphology, hurst_exponent, etc.')
    else:
        print(f'\nv7 struggles on neurons without events (near random)')

print(f'\n{"="*80}')
print('CONCLUSION')
print('='*80)

print(f'\nAnswer to "how did v7 predict on 100% when only 71% had events?":')
print(f'  1. ML model uses ALL features (both event-based and non-event-based)')
print(f'  2. When event detection fails (t_off=-1), event features get default values')
print(f'  3. Non-event features still work: trace_skewness, hurst_exponent, area, etc.')
print(f'  4. Model can make predictions using only non-event features')
print(f'\nPerformance impact:')
if len(v7_no_events_subset) > 0:
    print(f'  v7 accuracy on no-event neurons: {v7_no_events_acc:.2f}%')
    print(f'  v8 accuracy on same neurons: {v8_acc:.2f}%')
    if v8_acc > v7_acc:
        print(f'  v8 IS BETTER on challenging neurons (+{v8_acc - v7_acc:.2f}%)')
    else:
        print(f'  v8 is NOT better on challenging neurons ({v8_acc - v7_acc:+.2f}%)')
