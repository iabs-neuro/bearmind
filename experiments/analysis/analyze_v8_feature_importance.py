"""
Analyze feature importances in v8 EBM model.
Check if new features (hurst_exponent, baseline_drift, kinetics_source) are used.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('='*80)
print('V8 EBM MODEL FEATURE IMPORTANCE ANALYSIS')
print('='*80)

# Load v8 model
with open('ml/ebm_grid_search_v8/ebm_best.pkl', 'rb') as f:
    model = pickle.load(f)

print(f'\nModel type: {type(model).__name__}')
print(f'Number of features: {len(model.feature_names_in_)}')

# Get feature names and importances
feature_names = model.feature_names_in_
feature_importances = model.term_importances()

# Separate main effects and interactions
n_features = len(feature_names)
main_importances = feature_importances[:n_features]
interaction_importances = feature_importances[n_features:]

print(f'\nMain effects: {len(main_importances)}')
print(f'Interactions: {len(interaction_importances)}')

# Create DataFrame for main effects
df_main = pd.DataFrame({
    'feature': feature_names,
    'importance': main_importances
}).sort_values('importance', ascending=False)

# Identify new features in v8
new_features = ['hurst_exponent', 'baseline_drift', 'kinetics_source', 'kinetics_opt']
df_main['is_new'] = df_main['feature'].isin(new_features)

print(f'\n{"="*80}')
print('TOP 20 FEATURES BY IMPORTANCE')
print('='*80)
print(f"\n{'Rank':<6} {'Feature':<30} {'Importance':<12} {'New in v8?'}")
print('-'*65)

for idx, row in df_main.head(20).iterrows():
    marker = '[NEW]' if row['is_new'] else ''
    print(f"{df_main.index.get_loc(idx)+1:<6} {row['feature']:<30} {row['importance']:<12.4f} {marker}")

# New features analysis
print(f'\n{"="*80}')
print('NEW FEATURES RANKING')
print('='*80)

new_in_model = df_main[df_main['is_new']].copy()
if len(new_in_model) > 0:
    print(f'\n{"Feature":<30} {"Importance":<12} {"Rank":<8} {"% of Top Feature"}')
    print('-'*70)
    top_importance = df_main.iloc[0]['importance']
    for idx, row in new_in_model.iterrows():
        rank = df_main.index.get_loc(idx) + 1
        pct = (row['importance'] / top_importance * 100) if top_importance > 0 else 0
        print(f"{row['feature']:<30} {row['importance']:<12.4f} {rank:<8} {pct:.1f}%")
else:
    print('\nNO NEW FEATURES FOUND IN MODEL!')

# Check if features exist but have zero importance
print(f'\n{"="*80}')
print('FEATURE AVAILABILITY CHECK')
print('='*80)

for feat in new_features:
    if feat in feature_names:
        importance = df_main[df_main['feature'] == feat]['importance'].values[0]
        rank = df_main[df_main['feature'] == feat].index[0] + 1
        print(f'  {feat}: PRESENT (importance={importance:.4f}, rank={rank}/{len(feature_names)})')
    else:
        print(f'  {feat}: NOT IN MODEL')

# Plot feature importances
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Top 20 features
ax1 = axes[0]
top20 = df_main.head(20).copy()
colors = ['green' if is_new else 'steelblue' for is_new in top20['is_new']]
y_pos = np.arange(len(top20))
ax1.barh(y_pos, top20['importance'], color=colors)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top20['feature'], fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel('Importance', fontsize=11)
ax1.set_title('Top 20 Features by Importance (v8 Model)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='steelblue', label='Existing features'),
    Patch(facecolor='green', label='New in v8')
]
ax1.legend(handles=legend_elements, loc='lower right')

# Plot 2: All features sorted
ax2 = axes[1]
colors_all = ['green' if is_new else 'steelblue' for is_new in df_main['is_new']]
y_pos_all = np.arange(len(df_main))
ax2.barh(y_pos_all, df_main['importance'], color=colors_all, alpha=0.7)
ax2.set_xlabel('Importance', fontsize=11)
ax2.set_ylabel('Feature Rank', fontsize=11)
ax2.set_title(f'All {len(df_main)} Features by Importance', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Highlight new features with annotations
if len(new_in_model) > 0:
    for idx, row in new_in_model.iterrows():
        rank = df_main.index.get_loc(idx)
        ax2.annotate(row['feature'],
                    xy=(row['importance'], rank),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=8, color='darkgreen', fontweight='bold')

plt.tight_layout()
plt.savefig('ml/results/v8_feature_importance.png', dpi=150, bbox_inches='tight')
print(f'\nPlot saved to: ml/results/v8_feature_importance.png')

# Interaction analysis (if present)
if len(interaction_importances) > 0:
    print(f'\n{"="*80}')
    print('TOP 10 INTERACTION TERMS')
    print('='*80)

    # Get interaction names
    interaction_names = []
    for i, feat1 in enumerate(feature_names):
        for j, feat2 in enumerate(feature_names):
            if i < j:
                interaction_names.append(f'{feat1} × {feat2}')

    if len(interaction_names) == len(interaction_importances):
        df_interactions = pd.DataFrame({
            'interaction': interaction_names,
            'importance': interaction_importances
        }).sort_values('importance', ascending=False)

        # Check for new feature interactions
        df_interactions['has_new_feature'] = df_interactions['interaction'].apply(
            lambda x: any(feat in x for feat in new_features)
        )

        print(f"\n{'Rank':<6} {'Interaction':<50} {'Importance':<12} {'New?'}")
        print('-'*80)
        for idx, row in df_interactions.head(10).iterrows():
            marker = '[HAS NEW]' if row['has_new_feature'] else ''
            print(f"{df_interactions.index.get_loc(idx)+1:<6} {row['interaction']:<50} {row['importance']:<12.4f} {marker}")

        # Count new feature interactions in top 20
        top20_interactions = df_interactions.head(20)
        new_in_top20 = top20_interactions['has_new_feature'].sum()
        print(f'\nInteractions involving new features in top 20: {new_in_top20}/20')

# Summary statistics
print(f'\n{"="*80}')
print('SUMMARY')
print('='*80)

total_importance = df_main['importance'].sum()
new_importance = new_in_model['importance'].sum() if len(new_in_model) > 0 else 0
new_pct = (new_importance / total_importance * 100) if total_importance > 0 else 0

print(f'\nTotal importance (main effects): {total_importance:.4f}')
if len(new_in_model) > 0:
    print(f'New features importance: {new_importance:.4f} ({new_pct:.2f}% of total)')
    print(f'Number of new features used: {len(new_in_model)}/{len(new_features)}')

    avg_rank = new_in_model.index.mean() + 1
    print(f'Average rank of new features: {avg_rank:.1f}/{len(feature_names)}')

    if new_pct > 5:
        print(f'\nNew features contribute SIGNIFICANTLY ({new_pct:.1f}%) to model predictions')
    elif new_pct > 1:
        print(f'\nNew features contribute MODERATELY ({new_pct:.1f}%) to model predictions')
    else:
        print(f'\nNew features have MINIMAL impact ({new_pct:.1f}%) on model predictions')
else:
    print('\nNO NEW FEATURES ARE BEING USED IN THE MODEL')
