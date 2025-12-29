import pickle
import pandas as pd
import numpy as np

# Load model
with open('production_models/ebm_v9_iter8.pkl', 'rb') as f:
    model = pickle.load(f)

print('='*80)
print('v9_iter8 MODEL INSPECTION')
print('='*80)

# Get features
features = list(model.feature_names_in_)
print(f'\nTotal features: {len(features)}')

# Get term importances
try:
    imp = model.term_importances()
    print(f'\nterm_importances() type: {type(imp)}')
    print(f'term_importances() shape: {imp.shape if hasattr(imp, "shape") else "N/A"}')
    print(f'term_importances() length: {len(imp) if hasattr(imp, "__len__") else "N/A"}')

    # If it's a 2D array (feature + interactions)
    if isinstance(imp, np.ndarray) and len(imp.shape) == 1:
        print(f'\nImportances array has {len(imp)} elements')
        print(f'Features array has {len(features)} elements')

        # Maybe it includes interactions?
        # EBM with interactions=20 would have: features + pairs
        n_pairs = int((len(features) * (len(features) - 1)) / 2)
        expected_with_interactions = len(features) + min(20, n_pairs)
        print(f'\nExpected terms (features + up to 20 interactions): {expected_with_interactions}')

        # Get just main features (first len(features) elements)
        main_importances = imp[:len(features)]

        df = pd.DataFrame({
            'feature': features,
            'importance': main_importances
        }).sort_values('importance', ascending=False)

        print('\n' + '='*80)
        print('TOP 20 MAIN FEATURES BY IMPORTANCE')
        print('='*80)
        print(df.head(20).to_string(index=False))

        print('\n' + '='*80)
        print('FEATURES RELATED TO SIGNAL MAGNITUDE/RANGE')
        print('='*80)
        related = df[df['feature'].str.contains('snr|noise|amplitude|baseline|std|peak|range',
                                                case=False, regex=True)]
        print(related.to_string(index=False))

except Exception as e:
    print(f'\nError getting importances: {e}')
    print('\nFalling back to feature names only:')
    print('\nAll features:')
    for i, f in enumerate(features, 1):
        print(f'  {i:2d}. {f}')
