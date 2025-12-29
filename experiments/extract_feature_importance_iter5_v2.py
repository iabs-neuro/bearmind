"""
Extract feature importance from v8_corrected_iter5 model.
"""
import pickle
import pandas as pd
import numpy as np

print('='*80)
print('EXTRACTING FEATURE IMPORTANCE: v8_corrected_iter5')
print('='*80)

# Load model
with open('production_models/ebm_v8_corrected_iter5.pkl', 'rb') as f:
    model = pickle.load(f)

# Get feature importances
print('\nExtracting feature importances...')
importance_scores = model.term_importances()
term_names = model.term_names_

print(f'Number of terms: {len(term_names)}')
print(f'Number of importance scores: {len(importance_scores)}')

# Create dataframe
all_terms = []
for name, importance in zip(term_names, importance_scores):
    all_terms.append({
        'term': name,
        'importance': importance,
        'is_interaction': ' x ' in name
    })

all_terms_df = pd.DataFrame(all_terms).sort_values('importance', ascending=False)

# Main effects only
main_effects_df = all_terms_df[~all_terms_df['is_interaction']].copy()
main_effects_df = main_effects_df[['term', 'importance']].rename(columns={'term': 'feature'})

print(f'\nTop 20 Most Important Features (Main Effects):')
print('='*80)
for idx, (i, row) in enumerate(main_effects_df.head(20).iterrows(), 1):
    print(f'{idx:2d}. {row["feature"]:<35} {row["importance"]:>8.4f}')

# Save main effects
main_effects_df.to_csv('ml/results/v8_iter5_feature_importance.csv', index=False)
print(f'\nMain effects saved: ml/results/v8_iter5_feature_importance.csv')

# Save all terms (including interactions)
all_terms_df.to_csv('ml/results/v8_iter5_all_terms_importance.csv', index=False)
print(f'All terms saved: ml/results/v8_iter5_all_terms_importance.csv')

print('\n' + '='*80)
