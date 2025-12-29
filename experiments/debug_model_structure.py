"""
Debug model structure to find feature names.
"""
import pickle

with open('production_models/ebm_v8_corrected_iter5.pkl', 'rb') as f:
    model = pickle.load(f)

print('Model attributes:')
attrs = [x for x in dir(model) if not x.startswith('_')]
for attr in attrs:
    if 'feature' in attr.lower() or 'term' in attr.lower() or 'name' in attr.lower():
        print(f'  {attr}')

print('\nfeature_names:')
print(model.feature_names)
print(f'\nLength: {len(model.feature_names)}')

print('\nterm_names:')
if hasattr(model, 'term_names'):
    print(model.term_names)
    print(f'Length: {len(model.term_names)}')

print('\nfeature_types:')
if hasattr(model, 'feature_types'):
    print(model.feature_types)
    print(f'Length: {len(model.feature_types)}')
