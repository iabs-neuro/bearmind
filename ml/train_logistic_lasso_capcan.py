import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import json
import os
from pathlib import Path


def numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16,
                        np.int32, np.int64, np.uint8, np.uint16,
                        np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def extract_model_params(model):
    """
    Extract all parameters from trained model
    """
    params = {
        'coef': model.coef_.tolist(),
        'intercept': model.intercept_.tolist(),
        'classes': model.classes_.tolist(),
        'n_features': model.n_features_in_,
        'n_iter': model.n_iter_[0] if hasattr(model, 'n_iter_') else None
    }
    return params


def save_model_config(model, imputer, scaler, feature_names, filepath):
    """
    Save model configuration as compact dictionary
    """
    config = {
        'model_params': extract_model_params(model),
        'imputer_strategy': imputer.strategy,
        'scaler_params': {
            'mean': scaler.mean_.tolist() if scaler.mean_ is not None else [],
            'scale': scaler.scale_.tolist() if scaler.scale_ is not None else []
        },
        'feature_names': feature_names
    }

    # Convert all numpy types to Python native types
    config_serializable = json.loads(json.dumps(config, default=numpy_to_python))

    with open(filepath, 'w') as f:
        json.dump(config_serializable, f, indent=2)
    print(f"Model config saved to: {filepath}")


def save_model(model, imputer, scaler, filepath):
    """
    Save trained model with imputer and scaler
    """
    model_data = {
        'model': model,
        'imputer': imputer,
        'scaler': scaler
    }
    joblib.dump(model_data, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath):
    """
    Load trained model with imputer and scaler
    """
    model_data = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model_data['model'], model_data['imputer'], model_data['scaler']


def save_results_csv(results_df, filepath, param_type="seed"):
    """
    Save results to CSV file
    """
    results_df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")


def load_data(path,
              exclude_columns=['component_idx', 'center', 'corr_groups', 'is_corner_artifact',
                                'session', 'experiment', 'distance_to_gt'],
              filter_conditions={'experiment': ['3DM']}):
    """
    Load data with flexible filtering options

    Args:
        path: Path to CSV file
        exclude_columns: List of columns to exclude
        filter_conditions: Dictionary with filter conditions
            Example: {'experiment': ['3DM'], 'session': ['bad_session']}

    Returns:
        features, labels
    """
    data = pd.read_csv(path)

    # Apply multiple filter conditions
    if filter_conditions is not None:
        mask = pd.Series(True, index=data.index)
        for column, values in filter_conditions.items():
            if column in data.columns:
                mask &= ~data[column].isin(values)

        data = data[mask].copy()

    # Exclude specified columns from features
    exclude_columns.append('ground_truth')
    columns_to_drop = [col for col in exclude_columns if col in data.columns]
    features = data.drop(columns=columns_to_drop).copy()

    # Get labels
    labels = data['ground_truth']

    # Data cleaning
    features = features.replace([np.inf, -np.inf], np.nan)

    return features, labels


def predict_class(model, imputer, scaler, X, threshold=0.6):
    """
    Make predictions with custom classification threshold
    """
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities > threshold).astype(int)
    return predictions, probabilities


def train_logistic_lasso(X_train, y_train, X_test, y_test, C=1.0, random_state=42,
                         max_iter=1000, class_threshold=0.6):
    """
    Train Logistic Regression with L1 regularization (LASSO for classification)
    """
    # Imputer and scaler
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    # Logistic Regression with L1 regularization
    clf = LogisticRegression(penalty='l1', C=C, random_state=random_state,
                             solver='liblinear', max_iter=max_iter)
    clf.fit(X_train_scaled, y_train)

    # Predictions with custom threshold
    y_train_pred = predict_class(clf, imputer, scaler, X_train, threshold=class_threshold)[0]
    y_test_pred = predict_class(clf, imputer, scaler, X_test, threshold=class_threshold)[0]

    # Performance evaluation
    train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
        y_train, y_train_pred, average='binary'
    )
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='binary'
    )

    # Confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    print(f"\nTraining performance:")
    print(f"  Precision: {train_prec:.2%}")
    print(f"  Recall:    {train_rec:.2%}")
    print(f"  F1 Score:  {train_f1:.2%}")
    print(f"  Confusion Matrix:\n{cm_train}")

    print(f"\nTest performance:")
    print(f"  Precision: {test_prec:.2%}")
    print(f"  Recall:    {test_rec:.2%}")
    print(f"  F1 Score:  {test_f1:.2%}")
    print(f"  Confusion Matrix:\n{cm_test}")

    # Feature importance analysis
    print(f"\nImportant features (L1 selection):")
    feature_names = X_train.columns.tolist()
    for i, coef in enumerate(clf.coef_[0]):
        if abs(coef) > 0.01:
            print(f"  {feature_names[i]:20}: {coef:.4f}")

    metrics = {
        'train_precision': train_prec,
        'train_recall': train_rec,
        'train_f1': train_f1,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

    return clf, imputer, scaler, metrics


def main(path, train_fraction=0.75, C=1, max_iter=1000,
         class_threshold=0.6, seed=42):
    """
    Main function to run logistic LASSO training
    """
    # Load data
    X, y = load_data(path)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Single training run
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_fraction, random_state=seed, stratify=y
    )

    # Test different C values
    best_model = train_logistic_lasso(X_train, y_train, X_test, y_test, C=C, random_state=seed,
                         max_iter=max_iter, class_threshold=class_threshold)

    model, imputer, scaler, dict = best_model

    print(f"\n{'#' * 60}")
    print(f"Test Precision: {dict['test_precision']:.2%}")
    print(f"Test Recall: {dict['test_recall']:.2%}")
    print(f"Test F1: {dict['test_f1']:.2%}")
    print(f"{'#' * 60}")

    # Save model and config
    # save_model(model, imputer, scaler, "models\logistic_lasso\best_lasso_model.pkl")
    # save_model_config(model, imputer, scaler, X.columns.tolist(), "models\logistic_lasso\model_config.json")

    return model, imputer, scaler


if __name__ == "__main__":
    path_name = r'C:\Users\admin\Downloads\training_dataset_v4.csv'

    # Run single experiment
    model, imputer, scaler = main(path_name)

    # Or run multi-seed experiment
    # main(path_name, run_seed_experiment=True, n_seeds=10)

    # -3DM exclusion exps
    # exclusive data