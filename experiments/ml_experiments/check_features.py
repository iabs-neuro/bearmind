import pickle

# Check v4 model features
with open('ml/ebm_grid_search_v4_NOF_RFC_FOF/ebm_best_NOF_RFC_FOF.pkl', 'rb') as f:
    v4 = pickle.load(f)
print('V4 features:', len(v4.feature_names_in_))
print(sorted(v4.feature_names_in_))

print()

# Check v5 model features
with open('ml/ebm_grid_search_v5_no3dm/ebm_best.pkl', 'rb') as f:
    v5 = pickle.load(f)
print('V5 features:', len(v5.feature_names_in_))
print(sorted(v5.feature_names_in_))

print()

# Check v6 model features
with open('ml/ebm_grid_search_v6_no3dm/ebm_best.pkl', 'rb') as f:
    v6 = pickle.load(f)
print('V6 features:', len(v6.feature_names_in_))
print(sorted(v6.feature_names_in_))

print()
print('Features in v6 but not v4:', set(v6.feature_names_in_) - set(v4.feature_names_in_))
print('Features in v4 but not v6:', set(v4.feature_names_in_) - set(v6.feature_names_in_))
