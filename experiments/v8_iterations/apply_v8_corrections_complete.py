"""
Apply complete ground truth corrections to create v8_corrected dataset.
Includes:
  - 90 FAKE FN corrections (model correct to DELETE, GT wrong said KEEP)
  - 88 FAKE FP corrections (model correct to KEEP, GT wrong said DELETE)
Total: 178 corrections
"""
import pandas as pd

print('='*80)
print('CREATING v8_corrected DATASET WITH COMPLETE CORRECTIONS')
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
    # ========== FAKE FN (90): Model correct to DELETE, GT wrong (said KEEP) ==========
    ('NOF_H27_4D', 229, 0, 'FAKE FN #1 - noisy, model correct DELETE'),
    ('FOF_F30_1D', 1465, 0, 'FAKE FN #2 - sparse, model correct DELETE'),
    ('RFC_F35_3D', 185, 0, 'FAKE FN #4 - weak, model correct DELETE'),
    ('FOF_F11_1D', 97, 0, 'FAKE FN #5 - sparse, model correct DELETE'),
    ('FOF_F35_1D', 430, 0, 'FAKE FN #7 - noisy R2, model correct DELETE'),
    ('RFC_F35_3D', 18, 0, 'FAKE FN #9 - sparse, model correct DELETE'),
    ('NOF_H08_4D', 191, 0, 'FAKE FN #10 - weak, model correct DELETE'),
    ('RFC_F30_3D', 138, 0, 'FAKE FN #11 - sparse, model correct DELETE'),
    ('RFC_F35_3D', 12, 0, 'FAKE FN #12 - weak, model correct DELETE'),
    ('NOF_H26_4D', 7, 0, 'FAKE FN #13 - sparse, model correct DELETE'),
    ('NOF_H08_4D', 30, 0, 'FAKE FN #14 - noisy, model correct DELETE'),
    ('NOF_H08_4D', 105, 0, 'FAKE FN #16 - large area, model correct DELETE'),
    ('RFC_F35_3D', 16, 0, 'FAKE FN #17 - large area, model correct DELETE'),
    ('FOF_F11_1D', 104, 0, 'FAKE FN #18 - sparse, model correct DELETE'),
    ('RFC_F36_1D', 36, 0, 'FAKE FN #19 - large area, model correct DELETE'),
    ('NOF_H39_4D', 778, 0, 'FAKE FN #20 - sparse, model correct DELETE'),
    ('NOF_H26_2D', 70, 0, 'FAKE FN #21 - weak, model correct DELETE'),
    ('NOF_H32_4D', 542, 0, 'FAKE FN #22 - sparse, model correct DELETE'),
    ('NOF_H27_4D', 289, 0, 'FAKE FN #23 - noisy, model correct DELETE'),
    ('NOF_H26_4D', 186, 0, 'FAKE FN #24 - weak, model correct DELETE'),
    ('NOF_H08_4D', 344, 0, 'FAKE FN #25 - sparse, model correct DELETE'),
    ('RFC_F35_3D', 68, 0, 'FAKE FN #26 - weak, model correct DELETE'),
    ('NOF_H27_4D', 199, 0, 'FAKE FN #27 - noisy, model correct DELETE'),
    ('NOF_H27_4D', 277, 0, 'FAKE FN #28 - noisy, model correct DELETE'),
    ('NOF_H08_4D', 182, 0, 'FAKE FN #29 - weak, model correct DELETE'),
    ('RFC_F35_3D', 60, 0, 'FAKE FN #30 - sparse, model correct DELETE'),
    ('RFC_F35_3D', 61, 0, 'FAKE FN #31 - sparse, model correct DELETE'),
    ('FOF_F30_1D', 1480, 0, 'FAKE FN #32 - sparse, model correct DELETE'),
    ('FOF_F11_1D', 103, 0, 'FAKE FN #33 - sparse, model correct DELETE'),
    ('RFC_F30_3D', 340, 0, 'FAKE FN #34 - sparse, model correct DELETE'),
    ('FOF_F35_1D', 428, 0, 'FAKE FN #35 - sparse, model correct DELETE'),
    ('NOF_H08_4D', 112, 0, 'FAKE FN #36 - sparse, model correct DELETE'),
    ('RFC_F36_1D', 28, 0, 'FAKE FN #38 - sparse, model correct DELETE'),
    ('FOF_F30_1D', 772, 0, 'FAKE FN #39 - sparse, model correct DELETE'),
    ('FOF_F30_1D', 605, 0, 'FAKE FN #40 - sparse, model correct DELETE'),
    ('NOF_H32_4D', 736, 0, 'FAKE FN #41 - sparse, model correct DELETE'),
    ('FOF_F35_1D', 431, 0, 'FAKE FN #42 - sparse, model correct DELETE'),
    ('NOF_H23_4D', 853, 0, 'FAKE FN #43 - sparse, model correct DELETE'),
    ('NOF_H32_4D', 1081, 0, 'FAKE FN #44 - sparse, model correct DELETE'),
    ('RFC_F36_1D', 549, 0, 'FAKE FN #45 - sparse, model correct DELETE'),
    ('FOF_F11_1D', 92, 0, 'FAKE FN #46 - sparse, model correct DELETE'),
    ('NOF_H27_4D', 230, 0, 'FAKE FN #47 - noisy, model correct DELETE'),
    ('NOF_H32_4D', 1279, 0, 'FAKE FN #48 - sparse, model correct DELETE'),
    ('NOF_H32_4D', 335, 0, 'FAKE FN #49 - sparse, model correct DELETE'),
    ('NOF_H08_4D', 274, 0, 'FAKE FN #50 - sparse, model correct DELETE'),
    ('NOF_H08_4D', 210, 0, 'FAKE FN #51 - sparse, model correct DELETE'),
    ('NOF_H27_4D', 221, 0, 'FAKE FN #52 - noisy, model correct DELETE'),
    ('NOF_H08_4D', 280, 0, 'FAKE FN #53 - sparse, model correct DELETE'),
    ('FOF_F30_1D', 1261, 0, 'FAKE FN #54 - sparse, model correct DELETE'),
    ('NOF_H27_4D', 32, 0, 'FAKE FN #56 - noisy, model correct DELETE'),
    ('NOF_H27_4D', 265, 0, 'FAKE FN #57 - noisy, model correct DELETE'),
    ('FOF_F30_1D', 599, 0, 'FAKE FN #59 - sparse, model correct DELETE'),
    ('RFC_F35_3D', 65, 0, 'FAKE FN #60 - sparse, model correct DELETE'),
    ('FOF_F30_1D', 916, 0, 'FAKE FN #61 - sparse, model correct DELETE'),
    ('FOF_F30_1D', 572, 0, 'FAKE FN #62 - sparse, model correct DELETE'),
    ('RFC_F36_1D', 61, 0, 'FAKE FN #63 - sparse, model correct DELETE'),
    ('NOF_H39_3D', 810, 0, 'FAKE FN #64 - sparse, model correct DELETE'),
    ('RFC_F15_3D', 53, 0, 'FAKE FN #65 - sparse, model correct DELETE'),
    ('FOF_F30_1D', 429, 0, 'FAKE FN #66 - sparse, model correct DELETE'),
    ('RFC_F35_3D', 174, 0, 'FAKE FN #67 - sparse, model correct DELETE'),
    ('NOF_H27_4D', 4, 0, 'FAKE FN #69 - noisy, model correct DELETE'),
    ('RFC_F30_3D', 625, 0, 'FAKE FN #70 - sparse, model correct DELETE'),
    ('RFC_F30_3D', 257, 0, 'FAKE FN #71 - sparse, model correct DELETE'),
    ('FOF_F30_1D', 526, 0, 'FAKE FN #72 - sparse, model correct DELETE'),
    ('NOF_H27_4D', 194, 0, 'FAKE FN #73 - noisy, model correct DELETE'),
    ('FOF_F30_1D', 986, 0, 'FAKE FN #74 - sparse, model correct DELETE'),
    ('NOF_H08_2D', 350, 0, 'FAKE FN #75 - sparse, model correct DELETE'),
    ('FOF_F11_1D', 121, 0, 'FAKE FN #76 - sparse, model correct DELETE'),
    ('FOF_F35_1D', 529, 0, 'FAKE FN #77 - sparse, model correct DELETE'),
    ('NOF_H23_4D', 971, 0, 'FAKE FN #78 - sparse, model correct DELETE'),
    ('RFC_F01_3D', 70, 0, 'FAKE FN #80 - sparse, model correct DELETE'),
    ('NOF_H08_4D', 44, 0, 'FAKE FN #81 - sparse, model correct DELETE'),
    ('FOF_F35_1D', 361, 0, 'FAKE FN #82 - sparse, model correct DELETE'),
    ('FOF_F11_1D', 43, 0, 'FAKE FN #83 - sparse, model correct DELETE'),
    ('FOF_F11_1D', 127, 0, 'FAKE FN #84 - sparse, model correct DELETE'),
    ('NOF_H26_4D', 115, 0, 'FAKE FN #85 - sparse, model correct DELETE'),
    ('FOF_F35_1D', 344, 0, 'FAKE FN #86 - sparse, model correct DELETE'),
    ('FOF_F30_1D', 978, 0, 'FAKE FN #87 - sparse, model correct DELETE'),
    ('NOF_H08_2D', 73, 0, 'FAKE FN #88 - sparse, model correct DELETE'),
    ('NOF_H27_4D', 179, 0, 'FAKE FN #89 - noisy, model correct DELETE'),
    ('FOF_F11_1D', 44, 0, 'FAKE FN #90 - sparse, model correct DELETE'),
    ('RFC_F15_3D', 21, 0, 'FAKE FN #91 - sparse, model correct DELETE'),
    ('NOF_H08_4D', 177, 0, 'FAKE FN #92 - sparse, model correct DELETE'),
    ('FOF_F30_1D', 959, 0, 'FAKE FN #93 - sparse, model correct DELETE'),
    ('FOF_F30_1D', 1388, 0, 'FAKE FN #94 - sparse, model correct DELETE'),
    ('NOF_H32_4D', 1059, 0, 'FAKE FN #95 - sparse, model correct DELETE'),
    ('RFC_F30_3D', 113, 0, 'FAKE FN #96 - sparse, model correct DELETE'),
    ('RFC_F35_3D', 54, 0, 'FAKE FN #97 - sparse, model correct DELETE'),
    ('RFC_F35_3D', 119, 0, 'FAKE FN #98 - sparse, model correct DELETE'),
    ('NOF_H26_4D', 78, 0, 'FAKE FN #99 - sparse, model correct DELETE'),

    # ========== FAKE FP (88): Model correct to KEEP, GT wrong (said DELETE) ==========
    # MERGE cases (8)
    ('NOF_H39_4D', 551, 1, 'FAKE FP #21 - MERGE, model correct KEEP'),
    ('FOF_F30_1D', 1053, 1, 'FAKE FP #23 - MERGE, model correct KEEP'),
    ('FOF_F30_1D', 866, 1, 'FAKE FP #42 - MERGE, model correct KEEP'),
    ('FOF_F35_1D', 491, 1, 'FAKE FP #43 - MERGE, model correct KEEP'),
    ('RFC_F29_3D', 180, 1, 'FAKE FP #58 - MERGE, model correct KEEP'),
    ('NOF_H39_4D', 494, 1, 'FAKE FP #69 - MERGE, model correct KEEP'),
    ('NOF_H23_4D', 603, 1, 'FAKE FP #77 - MERGE, model correct KEEP'),
    ('NOF_H23_4D', 810, 1, 'FAKE FP #84 - MERGE, model correct KEEP'),

    # PROXIMITY cases (52)
    ('NOF_H23_4D', 363, 1, 'FAKE FP #2 - PROXIMITY, model correct KEEP'),
    ('NOF_H23_4D', 747, 1, 'FAKE FP #4 - PROXIMITY, model correct KEEP'),
    ('NOF_H26_2D', 86, 1, 'FAKE FP #8 - PROXIMITY, model correct KEEP'),
    ('NOF_H33_1D', 745, 1, 'FAKE FP #10 - PROXIMITY, model correct KEEP'),
    ('RFC_F30_3D', 339, 1, 'FAKE FP #12 - PROXIMITY, model correct KEEP'),
    ('NOF_H08_2D', 537, 1, 'FAKE FP #15 - PROXIMITY, model correct KEEP'),
    ('RFC_F30_3D', 478, 1, 'FAKE FP #16 - PROXIMITY, model correct KEEP'),
    ('NOF_H39_3D', 290, 1, 'FAKE FP #17 - PROXIMITY, model correct KEEP'),
    ('NOF_H26_2D', 82, 1, 'FAKE FP #19 - PROXIMITY, model correct KEEP'),
    ('NOF_H08_2D', 248, 1, 'FAKE FP #20 - PROXIMITY, model correct KEEP'),
    ('NOF_H39_3D', 946, 1, 'FAKE FP #22 - PROXIMITY, model correct KEEP'),
    ('NOF_H33_1D', 432, 1, 'FAKE FP #24 - PROXIMITY, model correct KEEP'),
    ('NOF_H23_4D', 122, 1, 'FAKE FP #25 - PROXIMITY, model correct KEEP'),
    ('RFC_F36_3D', 142, 1, 'FAKE FP #27 - PROXIMITY, model correct KEEP'),
    ('NOF_H26_2D', 90, 1, 'FAKE FP #30 - PROXIMITY, model correct KEEP'),
    ('RFC_F30_3D', 405, 1, 'FAKE FP #31 - PROXIMITY, model correct KEEP'),
    ('RFC_F30_3D', 415, 1, 'FAKE FP #32 - PROXIMITY, model correct KEEP'),
    ('RFC_F36_3D', 250, 1, 'FAKE FP #34 - PROXIMITY, model correct KEEP'),
    ('NOF_H39_4D', 698, 1, 'FAKE FP #36 - PROXIMITY, model correct KEEP'),
    ('NOF_H23_4D', 774, 1, 'FAKE FP #39 - PROXIMITY, model correct KEEP'),
    ('NOF_H26_2D', 83, 1, 'FAKE FP #41 - PROXIMITY, model correct KEEP'),
    ('RFC_F30_3D', 312, 1, 'FAKE FP #44 - PROXIMITY, model correct KEEP'),
    ('RFC_F30_3D', 333, 1, 'FAKE FP #45 - PROXIMITY, model correct KEEP'),
    ('FOF_F35_1D', 1122, 1, 'FAKE FP #46 - PROXIMITY, model correct KEEP'),
    ('NOF_H23_4D', 803, 1, 'FAKE FP #48 - PROXIMITY, model correct KEEP'),
    ('NOF_H33_1D', 148, 1, 'FAKE FP #49 - PROXIMITY, model correct KEEP'),
    ('FOF_F35_1D', 521, 1, 'FAKE FP #52 - PROXIMITY, model correct KEEP'),
    ('FOF_F30_1D', 1196, 1, 'FAKE FP #53 - PROXIMITY, model correct KEEP'),
    ('NOF_H26_2D', 8, 1, 'FAKE FP #54 - PROXIMITY, model correct KEEP'),
    ('NOF_H33_1D', 812, 1, 'FAKE FP #56 - PROXIMITY, model correct KEEP'),
    ('FOF_F35_1D', 252, 1, 'FAKE FP #62 - PROXIMITY, model correct KEEP'),
    ('RFC_F01_3D', 165, 1, 'FAKE FP #64 - PROXIMITY, model correct KEEP'),
    ('NOF_H39_3D', 1021, 1, 'FAKE FP #65 - PROXIMITY, model correct KEEP'),
    ('NOF_H32_4D', 444, 1, 'FAKE FP #68 - PROXIMITY, model correct KEEP'),
    ('FOF_F30_1D', 316, 1, 'FAKE FP #70 - PROXIMITY, model correct KEEP'),
    ('NOF_H23_4D', 659, 1, 'FAKE FP #71 - PROXIMITY, model correct KEEP'),
    ('FOF_F35_1D', 435, 1, 'FAKE FP #74 - PROXIMITY, model correct KEEP'),
    ('NOF_H39_3D', 620, 1, 'FAKE FP #76 - PROXIMITY, model correct KEEP'),
    ('NOF_H01_2D', 106, 1, 'FAKE FP #79 - PROXIMITY, model correct KEEP'),
    ('FOF_F30_1D', 861, 1, 'FAKE FP #80 - PROXIMITY, model correct KEEP'),
    ('NOF_H26_4D', 95, 1, 'FAKE FP #81 - PROXIMITY, model correct KEEP'),
    ('RFC_F30_3D', 437, 1, 'FAKE FP #83 - PROXIMITY, model correct KEEP'),
    ('NOF_H08_2D', 317, 1, 'FAKE FP #85 - PROXIMITY, model correct KEEP'),
    ('NOF_H01_3D', 116, 1, 'FAKE FP #86 - PROXIMITY, model correct KEEP'),
    ('NOF_H26_2D', 98, 1, 'FAKE FP #88 - PROXIMITY, model correct KEEP'),
    ('NOF_H01_2D', 131, 1, 'FAKE FP #89 - PROXIMITY, model correct KEEP'),
    ('RFC_F30_3D', 331, 1, 'FAKE FP #91 - PROXIMITY, model correct KEEP'),
    ('RFC_F35_3D', 109, 1, 'FAKE FP #92 - PROXIMITY, model correct KEEP'),
    ('FOF_F30_1D', 808, 1, 'FAKE FP #93 - PROXIMITY, model correct KEEP'),
    ('RFC_F36_1D', 489, 1, 'FAKE FP #94 - PROXIMITY, model correct KEEP'),
    ('FOF_F30_1D', 540, 1, 'FAKE FP #96 - PROXIMITY, model correct KEEP'),
    ('FOF_F30_1D', 806, 1, 'FAKE FP #98 - PROXIMITY, model correct KEEP'),

    # STANDALONE cases (28)
    ('NOF_H23_4D', 184, 1, 'FAKE FP #3 - STANDALONE, model correct KEEP'),
    ('RFC_F15_3D', 86, 1, 'FAKE FP #5 - STANDALONE, model correct KEEP'),
    ('NOF_H26_4D', 86, 1, 'FAKE FP #7 - STANDALONE, model correct KEEP'),
    ('NOF_H26_4D', 182, 1, 'FAKE FP #9 - STANDALONE, model correct KEEP'),
    ('RFC_F36_3D', 216, 1, 'FAKE FP #11 - STANDALONE, model correct KEEP'),
    ('NOF_H01_2D', 155, 1, 'FAKE FP #13 - STANDALONE, model correct KEEP'),
    ('RFC_F36_3D', 287, 1, 'FAKE FP #14 - STANDALONE, model correct KEEP'),
    ('RFC_F30_3D', 435, 1, 'FAKE FP #18 - STANDALONE, model correct KEEP'),
    ('RFC_F36_3D', 131, 1, 'FAKE FP #26 - STANDALONE, model correct KEEP'),
    ('RFC_F36_3D', 234, 1, 'FAKE FP #28 - STANDALONE, model correct KEEP'),
    ('RFC_F30_3D', 485, 1, 'FAKE FP #29 - STANDALONE, model correct KEEP'),
    ('NOF_H39_3D', 377, 1, 'FAKE FP #37 - STANDALONE, model correct KEEP'),
    ('NOF_H33_1D', 831, 1, 'FAKE FP #51 - STANDALONE, model correct KEEP'),
    ('RFC_F30_3D', 432, 1, 'FAKE FP #55 - STANDALONE, model correct KEEP'),
    ('RFC_F36_3D', 36, 1, 'FAKE FP #57 - STANDALONE, model correct KEEP'),
    ('RFC_F30_3D', 553, 1, 'FAKE FP #60 - STANDALONE, model correct KEEP'),
    ('RFC_F30_3D', 153, 1, 'FAKE FP #63 - STANDALONE, model correct KEEP'),
    ('RFC_F36_3D', 138, 1, 'FAKE FP #66 - STANDALONE, model correct KEEP'),
    ('RFC_F30_3D', 546, 1, 'FAKE FP #67 - STANDALONE, model correct KEEP'),
    ('RFC_F36_1D', 498, 1, 'FAKE FP #72 - STANDALONE, model correct KEEP'),
    ('NOF_H08_2D', 463, 1, 'FAKE FP #73 - STANDALONE, model correct KEEP'),
    ('NOF_H08_2D', 228, 1, 'FAKE FP #75 - STANDALONE, model correct KEEP'),
    ('RFC_F36_1D', 403, 1, 'FAKE FP #78 - STANDALONE, model correct KEEP'),
    ('NOF_H08_2D', 154, 1, 'FAKE FP #82 - STANDALONE, model correct KEEP'),
    ('NOF_H26_4D', 52, 1, 'FAKE FP #87 - STANDALONE, model correct KEEP'),
    ('NOF_H26_2D', 12, 1, 'FAKE FP #90 - STANDALONE, model correct KEEP'),
    ('NOF_H33_1D', 855, 1, 'FAKE FP #95 - STANDALONE, model correct KEEP'),
    ('NOF_H08_4D', 285, 1, 'FAKE FP #99 - STANDALONE, model correct KEEP'),
]

print(f'\n{"="*80}')
print(f'Applying {len(corrections)} corrections')
print('='*80)

corrections_applied = 0
corrections_failed = 0

for session, comp_idx, new_label, reason in corrections:
    mask = (df_corrected['session'] == session) & (df_corrected['component_idx'] == comp_idx)

    if mask.sum() == 0:
        print(f'[NOT FOUND] {session} comp {comp_idx}')
        corrections_failed += 1
        continue

    if mask.sum() > 1:
        print(f'[DUPLICATE] {session} comp {comp_idx}')
        corrections_failed += 1
        continue

    old_label = df_corrected.loc[mask, 'ground_truth'].values[0]
    if old_label != new_label:
        df_corrected.loc[mask, 'ground_truth'] = new_label
        corrections_applied += 1

print(f'\nApplied: {corrections_applied} corrections')
print(f'Failed: {corrections_failed} corrections')

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

net_change = corrected_keep - original_keep
print(f'\nNet change: {net_change:+,} KEEP labels')
if net_change > 0:
    print(f'  ({net_change} neurons changed from DELETE to KEEP)')
elif net_change < 0:
    print(f'  ({abs(net_change)} neurons changed from KEEP to DELETE)')

# Save
output_path = 'ml/results/training_dataset_v8_corrected.csv'
df_corrected.to_csv(output_path, index=False)

print(f'\n{"="*80}')
print(f'Corrected dataset saved to: {output_path}')
print(f'Ready for retraining!')
print('='*80)

print(f'\nCorrectionbreakdown:')
print(f'  FAKE FN (KEEP -> DELETE): 90')
print(f'  FAKE FP (DELETE -> KEEP): 88')
print(f'  Total corrections: {len(corrections)}')
print(f'  Successfully applied: {corrections_applied}')
