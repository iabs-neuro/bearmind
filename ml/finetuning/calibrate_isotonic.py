"""
Isotonic probability calibration module.

Post-processing probability calibration using isotonic regression.
This is a Phase 1 quick win approach that doesn't retrain the base model.

Key Features:
- Minimal risk (base model untouched)
- Works with 100+ feedback samples
- No hyperparameters to tune
- Fast training (<1 second)
- Preserves monotonicity (higher P_base → higher P_calibrated)
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from typing import Optional


def prepare_calibration_data(feedback_df: pd.DataFrame, verbose=True):
    """
    Convert feedback to calibration format.

    Args:
        feedback_df: DataFrame with columns:
            - ml_keep_probability: Base model P(KEEP) predictions
            - feedback_type: 'FP' or 'FN'
            OR
            - y_true: Binary labels (1=KEEP, 0=DELETE)
        verbose: Print sample info

    Returns:
        X: Base model probabilities (1D array)
        y: User labels (1D array, 0=DELETE, 1=KEEP)
    """
    # Extract base probabilities
    if 'ml_keep_probability' not in feedback_df.columns:
        raise ValueError("feedback_df must have 'ml_keep_probability' column")

    X = feedback_df['ml_keep_probability'].values

    # Extract labels
    if 'y_true' in feedback_df.columns:
        y = feedback_df['y_true'].values
    elif 'feedback_type' in feedback_df.columns:
        # FP feedback: model said KEEP (P>threshold), user says DELETE → y=0
        # FN feedback: model said DELETE (P<threshold), user says KEEP → y=1
        y = (feedback_df['feedback_type'] == 'FN').astype(int).values
    else:
        raise ValueError("feedback_df must have 'y_true' or 'feedback_type' column")

    # Remove NaN values
    valid_mask = ~(np.isnan(X) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]

    if verbose:
        print(f"Calibration data prepared:")
        print(f"  Samples: {len(X)}")
        print(f"  P(KEEP) range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"  Label distribution: {y.sum()} KEEP, {len(y) - y.sum()} DELETE")

    return X, y


def train_isotonic_calibrator(
    X: np.ndarray,
    y: np.ndarray,
    out_of_bounds='clip',
    verbose=True
) -> IsotonicRegression:
    """
    Fit isotonic regression calibrator.

    Args:
        X: Base model P(KEEP) predictions (1D array)
        y: Ground truth labels (0=DELETE, 1=KEEP)
        out_of_bounds: How to handle predictions outside training range
            - 'clip': Clip to [min, max] of training range
            - 'nan': Return NaN for out-of-bounds predictions
        verbose: Print training info

    Returns:
        Fitted IsotonicRegression object
    """
    if len(X) < 50:
        print(f"WARNING: Only {len(X)} samples for calibration. sklearn recommends ≥50.")
        print("Consider collecting more feedback before deploying calibrated model.")

    if verbose:
        print(f"\nTraining isotonic calibrator...")
        print(f"  Samples: {len(X)}")
        print(f"  out_of_bounds: {out_of_bounds}")

    calibrator = IsotonicRegression(
        y_min=0.0,  # Probabilities bounded [0, 1]
        y_max=1.0,
        out_of_bounds=out_of_bounds
    )

    calibrator.fit(X, y)

    if verbose:
        print(f"  Calibrator trained successfully")
        print(f"  Training P range: [{calibrator.X_min_:.3f}, {calibrator.X_max_:.3f}]")
        print(f"  Calibrated P range: [{calibrator.y_min:.3f}, {calibrator.y_max:.3f}]")

    return calibrator


def save_calibrator(
    calibrator: IsotonicRegression,
    output_path: str,
    verbose=True
):
    """
    Save calibrator to pickle file.

    Args:
        calibrator: Fitted IsotonicRegression object
        output_path: Path to save pickle file
        verbose: Print save info
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(calibrator, f)

    if verbose:
        print(f"\nCalibrator saved to {output_path}")


def load_calibrator(calibrator_path: str) -> IsotonicRegression:
    """
    Load calibrator from pickle file.

    Args:
        calibrator_path: Path to pickle file

    Returns:
        Loaded IsotonicRegression object
    """
    with open(calibrator_path, 'rb') as f:
        calibrator = pickle.load(f)

    return calibrator


def apply_calibration(
    base_model,
    calibrator: IsotonicRegression,
    X: np.ndarray,
    verbose=False
) -> np.ndarray:
    """
    Apply calibration to base model predictions.

    Args:
        base_model: Trained EBM model (ExplainableBoostingClassifier)
        calibrator: Fitted IsotonicRegression object
        X: Feature matrix (N samples × 35 features)
        verbose: Print application info

    Returns:
        P_calibrated: Adjusted P(KEEP) probabilities (1D array)
    """
    # Get base model predictions
    P_base = base_model.predict_proba(X)[:, 1]

    # Apply calibration
    P_calibrated = calibrator.predict(P_base)

    if verbose:
        print(f"Calibration applied to {len(X)} samples")
        print(f"  P_base range: [{P_base.min():.3f}, {P_base.max():.3f}]")
        print(f"  P_calibrated range: [{P_calibrated.min():.3f}, {P_calibrated.max():.3f}]")
        print(f"  Mean shift: {(P_calibrated - P_base).mean():.3f}")

    return P_calibrated


def predict_with_calibrator(
    calibrator: IsotonicRegression,
    P_base: np.ndarray
) -> np.ndarray:
    """
    Apply calibrator directly to base probabilities.

    Useful when base model predictions are already computed.

    Args:
        calibrator: Fitted IsotonicRegression object
        P_base: Base model P(KEEP) predictions

    Returns:
        P_calibrated: Adjusted probabilities
    """
    return calibrator.predict(P_base)


def evaluate_calibration(
    calibrator: IsotonicRegression,
    feedback_df: pd.DataFrame,
    threshold=0.75,
    verbose=True
):
    """
    Evaluate calibrator performance on feedback set.

    Args:
        calibrator: Fitted IsotonicRegression object
        feedback_df: Feedback DataFrame with ml_keep_probability and y_true
        threshold: Decision threshold for binary classification
        verbose: Print evaluation metrics

    Returns:
        dict: Evaluation metrics
    """
    # Get base and calibrated probabilities
    P_base = feedback_df['ml_keep_probability'].values
    y_true = feedback_df['y_true'].values if 'y_true' in feedback_df.columns else (feedback_df['feedback_type'] == 'FN').astype(int).values

    # Remove NaN
    valid_mask = ~np.isnan(P_base)
    P_base = P_base[valid_mask]
    y_true = y_true[valid_mask]

    P_calibrated = calibrator.predict(P_base)

    # Compute metrics
    # Base model decisions
    y_pred_base = (P_base >= threshold).astype(int)
    acc_base = (y_pred_base == y_true).mean()

    # Calibrated model decisions
    y_pred_calibrated = (P_calibrated >= threshold).astype(int)
    acc_calibrated = (y_pred_calibrated == y_true).mean()

    # Error correction rates
    fp_mask = (feedback_df['feedback_type'] == 'FP').values[valid_mask]
    fn_mask = (feedback_df['feedback_type'] == 'FN').values[valid_mask]

    if fp_mask.sum() > 0:
        # FP: base said KEEP, should be DELETE
        # Corrected if calibrated says DELETE (P_calibrated < threshold)
        fp_correction_rate = (P_calibrated[fp_mask] < threshold).mean()
    else:
        fp_correction_rate = 0.0

    if fn_mask.sum() > 0:
        # FN: base said DELETE, should be KEEP
        # Corrected if calibrated says KEEP (P_calibrated >= threshold)
        fn_correction_rate = (P_calibrated[fn_mask] >= threshold).mean()
    else:
        fn_correction_rate = 0.0

    overall_correction_rate = (y_pred_calibrated == y_true).mean()

    metrics = {
        'accuracy_base': acc_base,
        'accuracy_calibrated': acc_calibrated,
        'improvement': acc_calibrated - acc_base,
        'fp_correction_rate': fp_correction_rate,
        'fn_correction_rate': fn_correction_rate,
        'overall_correction_rate': overall_correction_rate,
        'mean_probability_shift': (P_calibrated - P_base).mean(),
    }

    if verbose:
        print(f"\nCalibration Evaluation (threshold={threshold}):")
        print(f"  Base model accuracy: {metrics['accuracy_base']:.1%}")
        print(f"  Calibrated accuracy: {metrics['accuracy_calibrated']:.1%}")
        print(f"  Improvement: {metrics['improvement']:+.1%}")
        print(f"\nError Correction Rates:")
        print(f"  FP correction: {metrics['fp_correction_rate']:.1%} ({fp_mask.sum()} samples)")
        print(f"  FN correction: {metrics['fn_correction_rate']:.1%} ({fn_mask.sum()} samples)")
        print(f"  Overall correction: {metrics['overall_correction_rate']:.1%}")
        print(f"\nProbability Shift:")
        print(f"  Mean shift: {metrics['mean_probability_shift']:+.3f}")

    return metrics


# Command-line interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train isotonic calibrator on feedback data')
    parser.add_argument('feedback_csv', help='Path to feedback CSV file')
    parser.add_argument('--output', default='ml/finetuning/models/isotonic_calibrator.pkl',
                        help='Path to save calibrator')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Decision threshold for evaluation')

    args = parser.parse_args()

    # Load feedback
    print(f"Loading feedback from {args.feedback_csv}...")
    feedback_df = pd.read_csv(args.feedback_csv)

    # Prepare data
    X, y = prepare_calibration_data(feedback_df, verbose=True)

    # Train calibrator
    calibrator = train_isotonic_calibrator(X, y, verbose=True)

    # Save
    save_calibrator(calibrator, args.output, verbose=True)

    # Evaluate
    metrics = evaluate_calibration(calibrator, feedback_df, threshold=args.threshold, verbose=True)

    print(f"\nDone! Calibrator ready for deployment if improvement ≥30%.")
