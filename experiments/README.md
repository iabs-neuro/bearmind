# Experimental Scripts Archive

This directory contains experimental, one-off analysis, and development scripts from various development phases. These are archived for historical reference but are **not maintained or tested**.

## Directory Structure

### `v8_iterations/`
Scripts from v8 model iterative correction development (now obsolete, v9 is current).
- `apply_v8_corrections*.py` - Correction application scripts
- `retrain_v8_corrected*.py` - Retraining scripts

### `analysis/`
One-off analysis scripts for investigating specific phenomena.

### `comparison/`
Comparison scripts for evaluating different methods/models.

### `investigation/`
Debugging and investigation scripts for specific issues.

### `testing/`
Experimental test scripts (not production tests).

### `visualization/`
Plotting and visualization scripts.

### `ml_experiments/`
ML-related experimental scripts including trace range analysis.

### Documentation (*.md)
Technical documentation and analysis notes from development.

## Usage Notes

WARNING: These scripts are **not production code**. They:
- May be outdated or non-functional
- May have hard-coded paths or parameters
- Were written for one-time use
- Are not tested or maintained

## Production Code

For production code, see:
- auto_inspector.py - Main autoinspection module
- bm_examinator.py - GUI examination tool
- ml/retrain_iter.py - Production iterative retraining
- ml/apply_corrections.py - Production correction framework
- production_models/ - Production EBM models
