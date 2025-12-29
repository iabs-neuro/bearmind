import pickle
import pandas as pd
import numpy as np

# Load model
with open('production_models/ebm_v9_iter8.pkl', 'rb') as f:
    model = pickle.load(f)

# Get feature importance
imp = model.term_importances()
features = list(model.feature_names_in_)

print('='*80)
print('v9_iter8 FEATURE IMPORTANCE')
print('='*80)

if isinstance(imp, dict):
    df = pd.DataFrame(list(imp.items()), columns=['feature', 'importance'])
else:
    df = pd.DataFrame({'feature': features, 'importance': imp})

df = df.sort_values('importance', ascending=False)

print('\nTop 20 features:')
print(df.head(20).to_string(index=False))

print(f'\n\nTotal features: {len(features)}')
print(f'Max importance: {df["importance"].max():.4f}')
print(f'Min importance: {df["importance"].min():.4f}')

# Check for features related to signal range/amplitude
print('\n' + '='*80)
print('FEATURES RELATED TO SIGNAL MAGNITUDE/RANGE')
print('='*80)
print('\nFeatures with "snr", "noise", "amplitude", "baseline", "std":')
related = df[df['feature'].str.contains('snr|noise|amplitude|baseline|std|peak', case=False, regex=True)]
print(related.to_string(index=False))
