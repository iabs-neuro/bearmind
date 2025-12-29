"""
Apply ground truth corrections to create v8_corrected dataset.
Pre-filled with 90 FAKE FN corrections (model correct, GT wrong).
Add FP corrections after review.
"""
import pandas as pd

print('='*80)
print('CREATING v8_corrected DATASET')
print('='*80)

# Load original v8 dataset
df = pd.read_csv('ml/results/training_dataset_v8.csv')
print(f'\nLoaded v8 dataset: {len(df):,} neurons')
print(f'Original labels: KEEP={df["ground_truth"].sum():,}, DELETE={len(df) - df["ground_truth"].sum():,}')

# Create copy for corrections
df_corrected = df.copy()

# Corrections: (session, component_idx, new_label, reason)
# new_label: 1=KEEP, 0=DELETE
corrections = [
    # FAKE FN - model correct to DELETE, GT wrong (said KEEP)
    ('NOF_H27_4D', 229, 0, 'FN #1 - noisy, model correct'),
    ('FOF_F30_1D', 1465, 0, 'FN #2 - sparse, model correct'),
    ('RFC_F35_3D', 185, 0, 'FN #4 - weak, model correct'),
    ('FOF_F11_1D', 97, 0, 'FN #5 - sparse, model correct'),
    ('FOF_F35_1D', 430, 0, 'FN #7 - noisy R2, model correct'),
    ('RFC_F35_3D', 18, 0, 'FN #9 - sparse, model correct'),
    ('NOF_H08_4D', 191, 0, 'FN #10 - weak, model correct'),
    ('RFC_F30_3D', 138, 0, 'FN #11 - sparse neg R2, model correct'),
    ('RFC_F35_3D', 12, 0, 'FN #12 - weak, model correct'),
    ('NOF_H26_4D', 7, 0, 'FN #13 - sparse, model correct'),
    ('NOF_H08_4D', 30, 0, 'FN #14 - noisy neg R2, model correct'),
    ('NOF_H08_4D', 105, 0, 'FN #16 - large area, model correct'),
    ('RFC_F35_3D', 16, 0, 'FN #17 - large area, model correct'),
    ('FOF_F11_1D', 104, 0, 'FN #18 - sparse, model correct'),
    ('RFC_F36_1D', 36, 0, 'FN #19 - large area, model correct'),
    ('NOF_H39_4D', 778, 0, 'FN #20 - sparse, model correct'),
    ('NOF_H26_2D', 70, 0, 'FN #21 - weak, model correct'),
    ('NOF_H32_4D', 542, 0, 'FN #22 - sparse, model correct'),
    ('NOF_H27_4D', 289, 0, 'FN #23 - noisy, model correct'),
    ('NOF_H26_4D', 186, 0, 'FN #24 - weak, model correct'),
    ('NOF_H08_4D', 344, 0, 'FN #25 - sparse, model correct'),
    ('RFC_F35_3D', 68, 0, 'FN #26 - weak, model correct'),
    ('NOF_H27_4D', 199, 0, 'FN #27 - noisy, model correct'),
    ('NOF_H27_4D', 277, 0, 'FN #28 - noisy, model correct'),
    ('NOF_H08_4D', 182, 0, 'FN #29 - weak, model correct'),
    ('RFC_F35_3D', 60, 0, 'FN #30 - sparse, model correct'),
    ('RFC_F35_3D', 61, 0, 'FN #31 - sparse, model correct'),
    ('FOF_F30_1D', 1480, 0, 'FN #32 - sparse, model correct'),
    ('FOF_F11_1D', 103, 0, 'FN #33 - sparse, model correct'),
    ('RFC_F30_3D', 340, 0, 'FN #34 - sparse, model correct'),
    ('FOF_F35_1D', 428, 0, 'FN #35 - sparse, model correct'),
    ('NOF_H08_4D', 112, 0, 'FN #36 - sparse, model correct'),
    ('RFC_F36_1D', 28, 0, 'FN #38 - sparse, model correct'),
    ('FOF_F30_1D', 772, 0, 'FN #39 - sparse, model correct'),
    ('FOF_F30_1D', 605, 0, 'FN #40 - sparse, model correct'),
    ('NOF_H32_4D', 736, 0, 'FN #41 - sparse, model correct'),
    ('FOF_F35_1D', 431, 0, 'FN #42 - sparse, model correct'),
    ('NOF_H23_4D', 853, 0, 'FN #43 - sparse, model correct'),
    ('NOF_H32_4D', 1081, 0, 'FN #44 - sparse, model correct'),
    ('RFC_F36_1D', 549, 0, 'FN #45 - sparse, model correct'),
    ('FOF_F11_1D', 92, 0, 'FN #46 - sparse, model correct'),
    ('NOF_H27_4D', 230, 0, 'FN #47 - noisy, model correct'),
    ('NOF_H32_4D', 1279, 0, 'FN #48 - sparse, model correct'),
    ('NOF_H32_4D', 335, 0, 'FN #49 - sparse, model correct'),
    ('NOF_H08_4D', 274, 0, 'FN #50 - sparse, model correct'),
    ('NOF_H08_4D', 210, 0, 'FN #51 - sparse, model correct'),
    ('NOF_H27_4D', 221, 0, 'FN #52 - noisy, model correct'),
    ('NOF_H08_4D', 280, 0, 'FN #53 - sparse, model correct'),
    ('FOF_F30_1D', 1261, 0, 'FN #54 - sparse, model correct'),
    ('NOF_H27_4D', 32, 0, 'FN #56 - noisy, model correct'),
    ('NOF_H27_4D', 265, 0, 'FN #57 - noisy, model correct'),
    ('FOF_F30_1D', 599, 0, 'FN #59 - sparse, model correct'),
    ('RFC_F35_3D', 65, 0, 'FN #60 - sparse, model correct'),
    ('FOF_F30_1D', 916, 0, 'FN #61 - sparse, model correct'),
    ('FOF_F30_1D', 572, 0, 'FN #62 - sparse, model correct'),
    ('RFC_F36_1D', 61, 0, 'FN #63 - sparse, model correct'),
    ('NOF_H39_3D', 810, 0, 'FN #64 - sparse, model correct'),
    ('RFC_F15_3D', 53, 0, 'FN #65 - sparse, model correct'),
    ('FOF_F30_1D', 429, 0, 'FN #66 - sparse, model correct'),
    ('RFC_F35_3D', 174, 0, 'FN #67 - sparse, model correct'),
    ('NOF_H27_4D', 4, 0, 'FN #69 - noisy, model correct'),
    ('RFC_F30_3D', 625, 0, 'FN #70 - sparse, model correct'),
    ('RFC_F30_3D', 257, 0, 'FN #71 - sparse, model correct'),
    ('FOF_F30_1D', 526, 0, 'FN #72 - sparse, model correct'),
    ('NOF_H27_4D', 194, 0, 'FN #73 - noisy, model correct'),
    ('FOF_F30_1D', 986, 0, 'FN #74 - sparse, model correct'),
    ('NOF_H08_2D', 350, 0, 'FN #75 - sparse, model correct'),
    ('FOF_F11_1D', 121, 0, 'FN #76 - sparse, model correct'),
    ('FOF_F35_1D', 529, 0, 'FN #77 - sparse, model correct'),
    ('NOF_H23_4D', 971, 0, 'FN #78 - sparse, model correct'),
    ('RFC_F01_3D', 70, 0, 'FN #80 - sparse, model correct'),
    ('NOF_H08_4D', 44, 0, 'FN #81 - sparse, model correct'),
    ('FOF_F35_1D', 361, 0, 'FN #82 - sparse, model correct'),
    ('FOF_F11_1D', 43, 0, 'FN #83 - sparse, model correct'),
    ('FOF_F11_1D', 127, 0, 'FN #84 - sparse, model correct'),
    ('NOF_H26_4D', 115, 0, 'FN #85 - sparse, model correct'),
    ('FOF_F35_1D', 344, 0, 'FN #86 - sparse, model correct'),
    ('FOF_F30_1D', 978, 0, 'FN #87 - sparse, model correct'),
    ('NOF_H08_2D', 73, 0, 'FN #88 - sparse, model correct'),
    ('NOF_H27_4D', 179, 0, 'FN #89 - noisy, model correct'),
    ('FOF_F11_1D', 44, 0, 'FN #90 - sparse, model correct'),
    ('RFC_F15_3D', 21, 0, 'FN #91 - sparse, model correct'),
    ('NOF_H08_4D', 177, 0, 'FN #92 - sparse, model correct'),
    ('FOF_F30_1D', 959, 0, 'FN #93 - sparse, model correct'),
    ('FOF_F30_1D', 1388, 0, 'FN #94 - sparse, model correct'),
    ('NOF_H32_4D', 1059, 0, 'FN #95 - sparse, model correct'),
    ('RFC_F30_3D', 113, 0, 'FN #96 - sparse, model correct'),
    ('RFC_F35_3D', 54, 0, 'FN #97 - sparse, model correct'),
    ('RFC_F35_3D', 119, 0, 'FN #98 - sparse, model correct'),
    ('NOF_H26_4D', 78, 0, 'FN #99 - sparse, model correct'),

    # ADD FAKE FP CORRECTIONS HERE (after reviewing visualizations)
    # Format: ('session', component_idx, 1, 'FP #X - model correct to KEEP'),
]

print(f'\n{"="*80}')
print(f'Applying {len(corrections)} corrections')
print('='*80)

corrections_applied = 0
for session, comp_idx, new_label, reason in corrections:
    mask = (df_corrected['session'] == session) & (df_corrected['component_idx'] == comp_idx)

    if mask.sum() == 1:
        old_label = df_corrected.loc[mask, 'ground_truth'].values[0]
        if old_label != new_label:
            df_corrected.loc[mask, 'ground_truth'] = new_label
            corrections_applied += 1

print(f'Applied: {corrections_applied} corrections')

# Stats
original_keep = df['ground_truth'].sum()
corrected_keep = df_corrected['ground_truth'].sum()

print(f'\n{"="*80}')
print('DATASET STATISTICS')
print('='*80)

print(f'\nOriginal v8:')
print(f'  KEEP:   {original_keep:,} ({original_keep/len(df)*100:.2f}%)')
print(f'  DELETE: {len(df) - original_keep:,} ({(len(df) - original_keep)/len(df)*100:.2f}%)')

print(f'\nCorrected v8:')
print(f'  KEEP:   {corrected_keep:,} ({corrected_keep/len(df_corrected)*100:.2f}%)')
print(f'  DELETE: {len(df_corrected) - corrected_keep:,} ({(len(df_corrected) - corrected_keep)/len(df_corrected)*100:.2f}%)')

print(f'\nNet change: {corrected_keep - original_keep:+,} KEEP labels')
print(f'  ({abs(corrected_keep - original_keep)} neurons relabeled KEEP -> DELETE)')

# Save
output_path = 'ml/results/training_dataset_v8_corrected.csv'
df_corrected.to_csv(output_path, index=False)

print(f'\n{"="*80}')
print(f'Corrected dataset saved to: {output_path}')
print(f'Ready for retraining!')
print('='*80)
