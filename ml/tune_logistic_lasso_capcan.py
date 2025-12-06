from train_logistic_lasso_capcan import *

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.impute import SimpleImputer
from pathlib import Path
import pickle
from itertools import product
import warnings

warnings.filterwarnings('ignore')


def find_best_lasso_params(data_path,
                           exclude_columns=['component_idx', 'center', 'corr_groups', 'is_corner_artifact',
                                            'session', 'experiment', 'distance_to_gt'],
                           test_fraction=0.25,
                           random_state=42):
    """
    Find best hyperparameters for Lasso Logistic Regression using grid search.
    Returns best parameters and performance.
    """
    print("=" * 60)
    print("STAGE 1: FINDING BEST HYPERPARAMETERS")
    print("=" * 60)

    # Load data
    data = pd.read_csv(data_path)

    # Exclude columns
    columns_to_drop = [col for col in exclude_columns if col in data.columns]
    columns_to_drop.append('ground_truth')
    features = data.drop(columns=columns_to_drop).copy()
    labels = data['ground_truth'].copy()

    # Clean data
    features = features.replace([np.inf, -np.inf], np.nan)

    print(f"Dataset shape: {features.shape}")
    print(f"Class distribution: {labels.value_counts().to_dict()}")

    # Split data once
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_fraction, random_state=random_state, stratify=labels
    )

    # Parameter grid for initial search
    param_grid = {
        'C': [0.1, 1.0, 5.0, 10.0, 15.0],
        'class_threshold': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
        'max_iter': [100, 500, 1000, 1500, 2000]
    }

    print(f"\nParameter combinations to test: {np.prod([len(v) for v in param_grid.values()])}")

    best_params = None
    best_score = -1
    best_metrics = None

    # Grid search
    for C, threshold, max_iter in product(param_grid['C'],
                                          param_grid['class_threshold'],
                                          param_grid['max_iter']):
        print(f"\nTesting: C={C}, threshold={threshold}, max_iter={max_iter}")

        # Train model
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)

        model = LogisticRegression(
            penalty='l1',
            C=C,
            solver='liblinear',
            max_iter=max_iter,
            random_state=random_state
        )

        model.fit(X_train_scaled, y_train)

        # Predict
        X_test_imputed = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imputed)

        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_test_pred = (y_test_proba > threshold).astype(int)

        # Calculate F1 score
        _, _, f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average='binary', zero_division=0
        )

        print(f"  Test F1: {f1:.3f}")

        if f1 > best_score:
            best_score = f1
            best_params = {
                'C': C,
                'class_threshold': threshold,
                'max_iter': max_iter
            }
            best_metrics = {
                'test_f1': f1,
                'test_precision': precision_recall_fscore_support(
                    y_test, y_test_pred, average='binary', zero_division=0
                )[0],
                'test_recall': precision_recall_fscore_support(
                    y_test, y_test_pred, average='binary', zero_division=0
                )[1]
            }

    print(f"\n{'=' * 60}")
    print(f"BEST PARAMETERS FOUND:")
    print(f"C: {best_params['C']}")
    print(f"Threshold: {best_params['class_threshold']}")
    print(f"Max iter: {best_params['max_iter']}")
    print(f"Test F1: {best_score:.3f}")
    print(f"{'=' * 60}")

    return best_params, best_metrics


def run_multi_seed_lasso(data_path,
                         best_params,
                         output_dir="models/logistic_lasso",
                         n_seeds=10,
                         test_fraction=0.25,
                         exclude_columns=['component_idx', 'center', 'corr_groups', 'is_corner_artifact',
                                          'session', 'experiment', 'distance_to_gt']):
    """
    Run training with multiple random seeds using best parameters.
    Save best models and results.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: MULTI-SEED TRAINING WITH BEST PARAMETERS")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    data = pd.read_csv(data_path)

    # Exclude columns
    columns_to_drop = [col for col in exclude_columns if col in data.columns]
    columns_to_drop.append('ground_truth')
    features = data.drop(columns=columns_to_drop).copy()
    labels = data['ground_truth'].copy()

    # Clean data
    features = features.replace([np.inf, -np.inf], np.nan)

    print(f"Training with {n_seeds} random seeds")
    print(f"Parameters: C={best_params['C']}, threshold={best_params['class_threshold']}, "
          f"max_iter={best_params['max_iter']}")

    results = []
    models_data = []

    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")

        # Split with this seed
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_fraction, random_state=seed, stratify=labels
        )

        # Train model
        model, imputer, scaler, metrics = train_logistic_lasso(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            C=best_params['C'],
            class_threshold=best_params['class_threshold'],
            max_iter=best_params['max_iter'],
            random_state=seed
        )

        # Store results
        result = {
            'seed': seed,
            **metrics
        }
        results.append(result)

        # Store model data
        models_data.append({
            'model': model,
            'imputer': imputer,
            'scaler': scaler,
            'feature_names': features.columns.tolist(),
            'params': best_params,
            'metrics': metrics,
            'seed': seed
        })

        print(f"  Train F1: {metrics['train_f1']:.3f}, Test F1: {metrics['test_f1']:.3f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results CSV
    results_path = output_path / "lasso_multi_seed_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Find best model by test F1
    best_idx = results_df['test_f1'].idxmax()
    best_seed = results_df.loc[best_idx, 'seed']
    best_test_f1 = results_df.loc[best_idx, 'test_f1']

    # Save all models or just the best one
    for model_info in models_data:
        seed = model_info['seed']

        # Save model
        model_filename = f"lasso_seed{seed}_C{best_params['C']}_th{best_params['class_threshold']}.pkl"
        model_path = output_path / model_filename

        model_data = {
            'model': model_info['model'],
            'imputer': model_info['imputer'],
            'scaler': model_info['scaler'],
            'feature_names': model_info['feature_names'],
            'params': model_info['params'],
            'seed': seed,
            'metrics': model_info['metrics']
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        # Mark best model
        if seed == best_seed:
            # Also save as best model
            best_model_path = output_path / "best_lasso_model.pkl"
            with open(best_model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✓ BEST MODEL: seed {seed} (F1={best_test_f1:.3f}) saved as: {best_model_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Best seed: {best_seed}")
    print(f"Best test F1: {best_test_f1:.3f}")
    print(f"\nPerformance across all seeds:")
    print(f"Test F1 - Mean: {results_df['test_f1'].mean():.3f} ± {results_df['test_f1'].std():.3f}")
    print(f"Test F1 - Min: {results_df['test_f1'].min():.3f}, Max: {results_df['test_f1'].max():.3f}")
    print(f"Test Precision - Mean: {results_df['test_precision'].mean():.3f}")
    print(f"Test Recall - Mean: {results_df['test_recall'].mean():.3f}")
    print(f"\nAll models saved to: {output_path}")
    print("=" * 60)

    return results_df, models_data[best_seed]


def load_best_model(model_path):
    """Load saved model with all components."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    return model_data


def predict_with_model(model_data, X, threshold=None):
    """Make predictions using loaded model."""
    if threshold is None:
        threshold = model_data['params']['class_threshold']

    X_imputed = model_data['imputer'].transform(X)
    X_scaled = model_data['scaler'].transform(X_imputed)

    probabilities = model_data['model'].predict_proba(X_scaled)[:, 1]
    predictions = (probabilities > threshold).astype(int)

    return predictions, probabilities


# Main function to run everything
def train_best_lasso_pipeline(data_path,
                              n_seeds=10,
                              test_fraction=0.25,
                              output_dir="models/logistic_lasso"):
    """
    Complete pipeline: find best params → multi-seed training → save best model.
    """
    print("=" * 80)
    print("LASSO LOGISTIC REGRESSION PIPELINE")
    print("=" * 80)

    # Stage 1: Find best hyperparameters
    best_params, _ = find_best_lasso_params(
        data_path=data_path,
        test_fraction=test_fraction,
        random_state=42
    )

    # Stage 2: Multi-seed training with best params
    results_df, best_model_data = run_multi_seed_lasso(
        data_path=data_path,
        best_params=best_params,
        output_dir=output_dir,
        n_seeds=n_seeds,
        test_fraction=test_fraction
    )

    # Stage 3: Final summary
    print(f"\n{'=' * 80}")
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Best model saved to: {output_dir}/best_lasso_model.pkl")
    print(f"All results saved to: {output_dir}/lasso_multi_seed_results.csv")
    print(f"\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best test F1: {best_model_data['metrics']['test_f1']:.3f}")
    print("=" * 80)

    return best_model_data, results_df


if __name__ == "__main__":
    # Run complete pipeline
    best_model, results = train_best_lasso_pipeline(
        data_path=r'C:\Users\admin\Downloads\training_dataset_v4.csv',
        n_seeds=10,
        test_fraction=0.25,
        output_dir="models/logistic_lasso"
    )

    # Load best model for predictions
    loaded_model = load_best_model("models/logistic_lasso/best_lasso_model.pkl")
    print(f"\nLoaded model features: {len(loaded_model['feature_names'])}")
    print(f"Model parameters: {loaded_model['params']}")
