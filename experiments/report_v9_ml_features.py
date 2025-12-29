"""
Report which columns from v9 dataset will be used for ML training.
"""
import pandas as pd
import numpy as np

print('='*80)
print('V9 DATASET COLUMNS FOR ML TRAINING')
print('='*80)

# Load v9
v9 = pd.read_csv('ml/results/training_dataset_v9.csv')

print(f'\nDataset shape: {v9.shape[0]:,} rows × {v9.shape[1]} columns')

# Categorize columns
metadata_cols = ['session_name', 'component_idx', 'experiment']
target_col = 'ground_truth'

# Feature columns (all numerical metrics)
feature_cols = [col for col in v9.columns
                if col not in metadata_cols + [target_col]]

print(f'\n{"COLUMN CATEGORIES:":<50}')
print(f'  Metadata columns: {len(metadata_cols)}')
print(f'  Target column: 1 ({target_col})')
print(f'  Feature columns: {len(feature_cols)}')

# Show all columns organized
print('\n' + '='*80)
print('METADATA COLUMNS (3):')
print('='*80)
for i, col in enumerate(metadata_cols, 1):
    print(f'  {i}. {col}')

print('\n' + '='*80)
print('TARGET COLUMN (1):')
print('='*80)
print(f'  1. {target_col} (0=DELETE, 1=KEEP)')

print('\n' + '='*80)
print(f'FEATURE COLUMNS ({len(feature_cols)}):')
print('='*80)

# Group features by category
caiman_features = [c for c in feature_cols if c.startswith('caiman_')]
event_features = [c for c in feature_cols if c.startswith('event')]
spatial_features = ['area', 'circularity', 'convexity', 'compactness', 'eccentricity']
spatial_features = [c for c in spatial_features if c in feature_cols]
trace_features = [c for c in feature_cols if 'trace_' in c or c in ['skewness', 'kurtosis', 'bimodality']]
recon_features = [c for c in feature_cols if any(x in c for x in ['r2_score', 'nmae', 'nrmse', 'snr_recon'])]
kinetics_features = [c for c in feature_cols if any(x in c for x in ['kinetics', 't_rise', 't_off'])]

# Other features
categorized = (set(caiman_features) | set(event_features) | set(spatial_features) |
               set(trace_features) | set(recon_features) | set(kinetics_features))
other_features = [c for c in feature_cols if c not in categorized]

print('\nCaImAn Features (from CaImAn estimates):')
for i, col in enumerate(caiman_features, 1):
    non_null = v9[col].notna().sum()
    pct = 100 * non_null / len(v9)
    print(f'  {i}. {col:<25} ({non_null:,} non-null, {pct:.1f}%)')

print('\nEvent Detection Features (from DRIADA):')
for i, col in enumerate(event_features, 1):
    non_null = v9[col].notna().sum()
    pct = 100 * non_null / len(v9)
    print(f'  {i}. {col:<25} ({non_null:,} non-null, {pct:.1f}%)')

print('\nSpatial Morphology Features:')
for i, col in enumerate(spatial_features, 1):
    non_null = v9[col].notna().sum()
    pct = 100 * non_null / len(v9)
    print(f'  {i}. {col:<25} ({non_null:,} non-null, {pct:.1f}%)')

print('\nTrace Statistics Features:')
for i, col in enumerate(trace_features, 1):
    non_null = v9[col].notna().sum()
    pct = 100 * non_null / len(v9)
    print(f'  {i}. {col:<25} ({non_null:,} non-null, {pct:.1f}%)')

print('\nReconstruction Quality Features:')
for i, col in enumerate(recon_features, 1):
    non_null = v9[col].notna().sum()
    pct = 100 * non_null / len(v9)
    print(f'  {i}. {col:<25} ({non_null:,} non-null, {pct:.1f}%)')

print('\nKinetics Features (calcium dynamics):')
for i, col in enumerate(kinetics_features, 1):
    non_null = v9[col].notna().sum()
    pct = 100 * non_null / len(v9)
    print(f'  {i}. {col:<25} ({non_null:,} non-null, {pct:.1f}%)')

if other_features:
    print('\nOther Features:')
    for i, col in enumerate(other_features, 1):
        non_null = v9[col].notna().sum()
        pct = 100 * non_null / len(v9)
        print(f'  {i}. {col:<25} ({non_null:,} non-null, {pct:.1f}%)')

# Summary
print('\n' + '='*80)
print('SUMMARY FOR ML MODEL')
print('='*80)

print(f'''
Training Configuration:
- Total neurons: {len(v9):,}
- Feature columns: {len(feature_cols)}
- Target: {target_col} (binary: 0=DELETE, 1=KEEP)

Feature Coverage:
- CaImAn: {len(caiman_features)} features
- Event Detection: {len(event_features)} features
- Spatial Morphology: {len(spatial_features)} features
- Trace Statistics: {len(trace_features)} features
- Reconstruction Quality: {len(recon_features)} features
- Kinetics: {len(kinetics_features)} features
- Other: {len(other_features)} features

All {len(feature_cols)} features will be used for ML model training.
The model will learn to predict ground_truth (0 or 1) from these features.
''')

# Check for NaN values
print('NaN Statistics:')
nan_counts = v9[feature_cols].isna().sum()
features_with_nans = nan_counts[nan_counts > 0].sort_values(ascending=False)

if len(features_with_nans) > 0:
    print(f'\nFeatures with missing values:')
    for feat, count in features_with_nans.items():
        pct = 100 * count / len(v9)
        print(f'  {feat:<25} {count:>7,} ({pct:>5.2f}%)')
else:
    print('\nNo missing values in any feature!')

print('\n' + '='*80)
