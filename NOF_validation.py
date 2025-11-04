import os

compute_val_dataframes = 1

if compute_val_dataframes:
    val_path = "C://Users//admin//YandexDisk//_Projects//NOF//CalciumData//4_Estimates"
    init_path = os.path.join(val_path, '4.1_EstimatesRaw')
    gt_path = os.path.join(val_path, '4.1_EstimatesFinal')

    all_init_files = os.listdir(init_path)
    all_gt_files = os.listdir(gt_path)

    sessions = [name[:10] for name in all_gt_files]
    mapping = {}
    for session in sessions:
        mapping = 1