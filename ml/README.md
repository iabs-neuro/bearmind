# Machine Learning Scripts for Neuron Quality Classification

This directory contains scripts for training and evaluating decision tree classifiers on neuron quality data.

## Scripts

### 1. `train_decision_tree_capcan.py`
Train decision tree models on capcan_artifacts data.

**Usage:**
```bash
# Single model training
python ml/train_decision_tree_capcan.py --max-depth 5 --min-samples-split 50 --min-samples-leaf 25

# Multi-seed training (10 models for stability)
python ml/train_decision_tree_capcan.py --multi-seed 10 --max-depth 6 --min-samples-split 60 --min-samples-leaf 40
```

**Options:**
- `--max-depth`: Maximum tree depth (default: 5)
- `--min-samples-split`: Minimum samples to split node (default: 100)
- `--min-samples-leaf`: Minimum samples in leaf (default: 50)
- `--multi-seed`: Number of models to train with different random seeds
- `--output-dir`: Where to save models (default: ml/models/)

**Outputs:** Saves to `ml/models/`

### 2. `tune_tree_threshold_capcan.py`
Tune decision threshold to optimize precision/recall trade-off.

**Usage:**
```bash
# Single model threshold tuning
python ml/tune_tree_threshold_capcan.py --model ml/models/decision_tree_capcan_model.pkl

# Aggregate across multiple models
python ml/tune_tree_threshold_capcan.py --aggregate --n-seeds 10
```

**Options:**
- `--model`: Path to trained model
- `--aggregate`: Aggregate results from seed0-seed9 models
- `--n-seeds`: Number of seed models to use (default: 10)

**Outputs:** Saves to `ml/results/`

### 3. `grid_search_tree_params.py`
Grid search to find optimal hyperparameters.

**Usage:**
```bash
# Full grid search (400 combinations)
python ml/grid_search_tree_params.py --trials 3

# Faster with fewer trials
python ml/grid_search_tree_params.py --trials 1
```

**Options:**
- `--trials`: Number of train/test splits to average over (default: 3)
- `--output-dir`: Where to save results (default: ml/results/)

**Outputs:** Saves to `ml/results/grid_search_results.csv`

## Directory Structure

```
ml/
├── README.md                          # This file
├── train_decision_tree_capcan.py      # Training script
├── tune_tree_threshold_capcan.py      # Threshold tuning script
├── grid_search_tree_params.py         # Parameter search script
├── models/                            # Trained models
│   ├── decision_tree_capcan_model.pkl
│   ├── decision_tree_capcan_seed0.pkl
│   └── ...
└── results/                           # Results and visualizations
    ├── grid_search_results.csv
    ├── threshold_tuning_results_capcan.csv
    └── precision_recall_curve_capcan.png
```

## Data

All scripts expect capcan_artifacts data in `dev/validation_artifacts/capcan_artifacts_*/`

## Best Parameters (from grid search)

**Option 1: Minimize False Positives (High Precision)**
```python
max_depth=6
min_samples_split=60
min_samples_leaf=40
class_weight={0: 1.0, 1: 0.8}
```
Performance: Precision 91.80%, Recall 91.11%, F1 91.41%

**Option 2: Best Overall F1**
```python
max_depth=6
min_samples_split=50
min_samples_leaf=30
class_weight={0: 1.0, 1: 1.5}
```
Performance: Precision 89.49%, Recall 98.08%, F1 93.58%
