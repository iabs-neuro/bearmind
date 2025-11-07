import os

compute_val_dataframes = True

if compute_val_dataframes:
    val_path = "C://Users//admin//YandexDisk//_Projects//NOF//CalciumData//4_Estimates"
    init_path = os.path.join(val_path, '4.1_EstimatesRaw')
    gt_path = os.path.join(val_path, '4.1_EstimatesFinal')

    all_init_files = os.listdir(init_path)
    all_gt_files = os.listdir(gt_path)

    sessions = [name[:10] for name in all_gt_files]
    mapping = {}
    for session in sessions:
        init = [f for f in all_init_files if session in f][0]
        gt = [f for f in all_gt_files if session in f][0]
        mapping.update({session: [init, gt]})

print(mapping)