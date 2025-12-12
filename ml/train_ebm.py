"""
Train Explainable Boosting Machine (EBM) for neuron quality classification.

EBM is a glass-box model from Microsoft's InterpretML that:
- Uses all features (not just 3 like decision trees)
- Shows contribution of each feature via plots
- Detects feature interactions automatically
- Maintains interpretability while achieving high performance
"""
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, fbeta_score
from data_utils import FEATURE_COLS, stratified_session_split, print_split_info, FBETA_BETA


def load_session_data(session_dir):
    """Load metrics for a session from capcan_artifacts directory."""
    try:
        raw_metrics = os.path.join(session_dir, "metrics_init.csv")
        gt_metrics = os.path.join(session_dir, "metrics_gt.csv")

        df_raw = pd.read_csv(raw_metrics)
        df_gt = pd.read_csv(gt_metrics)

        # Parse center column if needed
        if df_raw['center'].dtype == 'object':
            df_raw['center'] = df_raw['center'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
        if df_gt['center'].dtype == 'object':
            df_gt['center'] = df_gt['center'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

        return df_raw, df_gt
    except Exception as e:
        print(f"  ERROR loading {os.path.basename(session_dir)}: {e}")
        return None, None


def create_dataset(session_dirs, max_distance=3):
    """Create dataset from capcan_artifacts directories."""
    all_features = []
    all_labels = []

    for session_dir in session_dirs:
        df_raw, df_gt = load_session_data(session_dir)

        if df_raw is None:
            continue

        # Filter corner artifacts
        if 'is_corner_artifact' in df_raw.columns:
            non_corner_mask = df_raw['is_corner_artifact'] == 0
            df_raw_filtered = df_raw[non_corner_mask].copy()
        else:
            df_raw_filtered = df_raw.copy()

        # Create labels by matching to GT
        raw_centers = np.array(df_raw_filtered['center'].tolist())
        gt_centers = np.array(df_gt['center'].tolist())

        labels = np.zeros(len(df_raw_filtered), dtype=int)

        for i, raw_center in enumerate(raw_centers):
            distances = np.linalg.norm(gt_centers - raw_center, axis=1)
            min_dist = distances.min()

            if min_dist <= max_distance:
                labels[i] = 1
            else:
                labels[i] = 0

        # Use centralized feature columns from data_utils
        features = df_raw_filtered[FEATURE_COLS].copy()
        features = features.replace([np.inf, -np.inf], np.nan)

        all_features.append(features)
        all_labels.append(labels)

    features_df = pd.concat(all_features, ignore_index=True)
    labels = np.concatenate(all_labels)

    # Final cleaning
    features_df = features_df.replace([np.inf, -np.inf], np.nan)

    return features_df, labels


def train_ebm(
    artifacts_dir="data/capcan_validation_127",
    test_fraction=0.25,
    output_dir="ml/ebm_models",
    experiments=None,
    random_state=42
):
    """
    Train Explainable Boosting Machine.

    Args:
        artifacts_dir: Directory with capcan_artifacts
        test_fraction: Fraction of data for testing
        output_dir: Directory to save model and results
        experiments: List of experiment IDs to include (e.g., ['NOF', 'RFC'])
        random_state: Random seed for reproducibility
    """
    print("="*80)
    print("TRAINING EXPLAINABLE BOOSTING MACHINE (EBM)")
    print("="*80)

    # Import EBM (check if installed)
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        from interpret import show
    except ImportError:
        print("\nERROR: InterpretML not installed!")
        print("Install with: pip install interpret")
        return None, None

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    artifacts_path = Path(artifacts_dir)
    session_dirs = sorted([d for d in artifacts_path.iterdir()
                          if d.is_dir() and d.name.startswith('capcan_artifacts_')])

    # Filter by experiment IDs if specified
    if experiments is not None and len(experiments) > 0:
        filtered_dirs = []
        for d in session_dirs:
            session_name = d.name.replace('capcan_artifacts_', '')
            exp_id = session_name.split('_')[0]
            if exp_id in experiments:
                filtered_dirs.append(d)
        session_dirs = filtered_dirs
        print(f"Filtered to experiments: {experiments}")

    print(f"Found {len(session_dirs)} sessions")

    # Create experiment suffix for filenames
    exp_suffix = ""
    if experiments is not None and len(experiments) > 0:
        exp_suffix = "_" + "_".join(experiments)

    # Stratified train/test split by experiment
    train_sessions, test_sessions, split_info = stratified_session_split(
        session_dirs,
        test_fraction=test_fraction,
        random_state=random_state
    )

    print()
    print_split_info(split_info)

    # Create datasets
    X_train, y_train = create_dataset(train_sessions)
    X_test, y_test = create_dataset(test_sessions)

    print(f"\nTrain samples: {len(X_train)} (KEEP: {y_train.sum()}, {y_train.sum()/len(y_train)*100:.1f}%)")
    print(f"Test samples: {len(X_test)} (KEEP: {y_test.sum()}, {y_test.sum()/len(y_test)*100:.1f}%)")

    # Train EBM
    print("\n" + "="*80)
    print("TRAINING EBM MODEL")
    print("="*80)
    print("\nEBM will automatically:")
    print("  - Use all 21 features")
    print("  - Detect feature interactions")
    print("  - Create interpretable explanations")
    print("\nTraining (this may take a few minutes)...")

    ebm = ExplainableBoostingClassifier(
        feature_names=list(X_train.columns),
        max_bins=256,
        max_interaction_bins=32,
        interactions=20,  # Detect top 20 interactions
        outer_bags=8,
        inner_bags=0,
        learning_rate=0.01,
        validation_size=0.15,
        early_stopping_rounds=50,
        early_stopping_tolerance=1e-4,
        max_rounds=5000,
        min_samples_leaf=2,
        max_leaves=3,
        random_state=random_state
    )

    ebm.fit(X_train, y_train)

    print("\nTraining complete!")

    # Evaluate on train set
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)

    y_train_pred = ebm.predict(X_train)
    y_train_proba = ebm.predict_proba(X_train)[:, 1]

    train_prec, train_rec, _, _ = precision_recall_fscore_support(
        y_train, y_train_pred, average='binary', zero_division=0
    )
    train_fbeta = fbeta_score(y_train, y_train_pred, beta=FBETA_BETA,
                              average='binary', zero_division=0)
    train_auc = roc_auc_score(y_train, y_train_proba)

    print("\nTrain Set Performance:")
    print(f"  Precision:  {train_prec*100:.2f}%")
    print(f"  Recall:     {train_rec*100:.2f}%")
    print(f"  F-beta (β={FBETA_BETA:.3f}): {train_fbeta*100:.2f}%")
    print(f"  ROC AUC:    {train_auc*100:.2f}%")

    # Evaluate on test set
    y_test_pred = ebm.predict(X_test)
    y_test_proba = ebm.predict_proba(X_test)[:, 1]

    test_prec, test_rec, _, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='binary', zero_division=0
    )
    test_fbeta = fbeta_score(y_test, y_test_pred, beta=FBETA_BETA,
                             average='binary', zero_division=0)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print("\nTest Set Performance:")
    print(f"  Precision:  {test_prec*100:.2f}%")
    print(f"  Recall:     {test_rec*100:.2f}%")
    print(f"  F-beta (β={FBETA_BETA:.3f}): {test_fbeta*100:.2f}%")
    print(f"  ROC AUC:    {test_auc*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix (Test Set):")
    print(f"  True Negatives:  {cm[0,0]}")
    print(f"  False Positives: {cm[0,1]}")
    print(f"  False Negatives: {cm[1,0]}")
    print(f"  True Positives:  {cm[1,1]}")

    # Feature importance
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)

    # Get global explanation
    from interpret import show
    ebm_global = ebm.explain_global()

    # Extract feature importance scores
    feature_importance = []
    for i, feature_name in enumerate(ebm_global.data()['names']):
        if 'x' not in feature_name.lower():  # Skip interaction terms for now
            importance = ebm_global.data()['scores'][i]
            feature_importance.append({
                'feature': feature_name,
                'importance': importance
            })

    # Sort by importance
    feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)

    print("\nTop 15 Most Important Features:")
    print(f"{'Feature':<30} {'Importance':>15}")
    print("-"*50)
    for item in feature_importance[:15]:
        print(f"{item['feature']:<30} {item['importance']:>15.6f}")

    # Save feature importance
    importance_df = pd.DataFrame(feature_importance)
    importance_path = output_path / f"ebm{exp_suffix}_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to: {importance_path}")

    # Detect interactions
    print("\n" + "="*80)
    print("FEATURE INTERACTIONS")
    print("="*80)

    interactions = []
    for i, feature_name in enumerate(ebm_global.data()['names']):
        if 'x' in feature_name.lower():  # Interaction terms
            importance = ebm_global.data()['scores'][i]
            interactions.append({
                'interaction': feature_name,
                'importance': importance
            })

    if len(interactions) > 0:
        interactions = sorted(interactions, key=lambda x: x['importance'], reverse=True)
        print("\nTop 10 Feature Interactions:")
        print(f"{'Interaction':<50} {'Importance':>15}")
        print("-"*70)
        for item in interactions[:10]:
            print(f"{item['interaction']:<50} {item['importance']:>15.6f}")

        # Save interactions
        interactions_df = pd.DataFrame(interactions)
        interactions_path = output_path / f"ebm{exp_suffix}_interactions.csv"
        interactions_df.to_csv(interactions_path, index=False)
        print(f"\nInteractions saved to: {interactions_path}")
    else:
        print("\nNo significant interactions detected.")

    # Save model
    model_path = output_path / f"ebm{exp_suffix}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(ebm, f)
    print(f"\nModel saved to: {model_path}")

    # Save results summary
    results = {
        'train_precision': train_prec,
        'train_recall': train_rec,
        'train_fbeta': train_fbeta,
        'train_auc': train_auc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_fbeta': test_fbeta,
        'test_auc': test_auc,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'random_state': random_state
    }

    results_df = pd.DataFrame([results])
    results_path = output_path / f"ebm{exp_suffix}_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nResults saved to: {results_path}")

    # Save global explanation for interactive viewing
    explanation_path = output_path / f"ebm{exp_suffix}_explanation.html"
    try:
        from interpret import preserve
        preserved = preserve(ebm_global, file_name=str(explanation_path))
        print(f"\nInteractive explanation saved to: {explanation_path}")
        print("Open this HTML file in a browser to explore feature effects!")
    except Exception as e:
        print(f"\nCould not save interactive explanation: {e}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    return ebm, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Explainable Boosting Machine")
    parser.add_argument("--artifacts-dir", default="data/capcan_validation_127",
                       help="Directory containing capcan_artifacts_* subdirectories")
    parser.add_argument("--output-dir", default="ml/ebm_models",
                       help="Directory to save models and results")
    parser.add_argument("--experiments", type=str, nargs='+', default=None,
                       help="Filter to specific experiments (e.g., --experiments NOF RFC)")
    parser.add_argument("--test-fraction", type=float, default=0.25,
                       help="Fraction of data for testing (default: 0.25)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    ebm, results = train_ebm(
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        test_fraction=args.test_fraction,
        experiments=args.experiments,
        random_state=args.random_state
    )
