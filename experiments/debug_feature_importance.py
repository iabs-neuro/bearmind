"""
Debug feature importance structure.
"""
import pickle

with open('production_models/ebm_v8_corrected_iter5.pkl', 'rb') as f:
    model = pickle.load(f)

feature_importances = model.term_importances()

print('Type:', type(feature_importances))
print('Shape:', feature_importances.shape if hasattr(feature_importances, 'shape') else 'N/A')
print('Dtype:', feature_importances.dtype if hasattr(feature_importances, 'dtype') else 'N/A')
print('\nFirst few elements:')
print(feature_importances[:5])
print('\nDir:')
print([x for x in dir(feature_importances) if not x.startswith('_')])
