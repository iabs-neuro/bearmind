"""
Extract feature importance from v8_corrected_iter5 model.
"""
import pickle
import pandas as pd

print('='*80)
print('EXTRACTING FEATURE IMPORTANCE: v8_corrected_iter5')
print('='*80)

# Load model
with open('production_models/ebm_v8_corrected_iter5.pkl', 'rb') as f:
    model = pickle.load(f)

# Get feature importances
print('\nExtracting feature importances...')
feature_importances = model.term_importances()

# Parse the structured array
main_effects = []
for i, name in enumerate(feature_importances['names']):
    importance = feature_importances['scores'][i]
    if ' x ' not in name:  # Main effect, not interaction
        main_effects.append({'feature': name, 'importance': importance})

main_effects_df = pd.DataFrame(main_effects).sort_values('importance', ascending=False)

print(f'\nTop 20 Most Important Features:')
print('='*80)
for idx, (i, row) in enumerate(main_effects_df.head(20).iterrows(), 1):
    print(f'{idx:2d}. {row["feature"]:<35} {row["importance"]:>8.4f}')

# Save
main_effects_df.to_csv('ml/results/v8_iter5_feature_importance.csv', index=False)
print(f'\nFeature importance saved: ml/results/v8_iter5_feature_importance.csv')

print('\n' + '='*80)
