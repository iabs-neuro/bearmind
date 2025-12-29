"""Analyze EBM model configuration and feature importances."""
import pickle
import numpy as np

# Load the best EBM model
with open('ml/ebm_grid_search_v3/ebm_simple_NOF_RFC.pkl', 'rb') as f:
    ebm = pickle.load(f)

# Print model configuration
print('=== CURRENT MODEL CONFIGURATION ===')
print(f'max_bins: {ebm.max_bins}')
print(f'max_interaction_bins: {ebm.max_interaction_bins}')
print(f'interactions: {ebm.interactions}')
print(f'outer_bags: {ebm.outer_bags}')
print(f'inner_bags: {ebm.inner_bags}')
print(f'learning_rate: {ebm.learning_rate}')
print(f'max_rounds: {ebm.max_rounds}')
print(f'early_stopping_rounds: {ebm.early_stopping_rounds}')
print(f'min_samples_leaf: {ebm.min_samples_leaf}')
print(f'max_leaves: {ebm.max_leaves}')

# Get feature importances
print('\n=== FEATURE IMPORTANCES (sorted) ===')
feature_names = ebm.feature_names_in_
importances = ebm.term_importances()

# Create sorted list
feat_imp = list(zip(feature_names, importances))
feat_imp.sort(key=lambda x: x[1], reverse=True)

print(f"{'Feature':<25} {'Importance':>12} {'Cumulative':>12}")
print('-' * 50)
cumsum = 0
for name, imp in feat_imp:
    cumsum += imp
    print(f'{name:<25} {imp:>12.4f} {cumsum:>12.1f}%')

# Top contributors
print(f'\nTop 5 features explain: {sum([x[1] for x in feat_imp[:5]]):.1f}% of model')
print(f'Top 10 features explain: {sum([x[1] for x in feat_imp[:10]]):.1f}% of model')

# Identify low-value features
print('\n=== LOW VALUE FEATURES (< 1%) ===')
low_value = [f for f, i in feat_imp if i < 1.0]
print(f'Features with < 1% importance: {low_value}')
