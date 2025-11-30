"""
Inspect and explain the decision rules learned by the best balanced tree.

Loads the optimal balanced model and extracts interpretable rules.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def get_feature_names():
    """Return the 21 feature names in order."""
    return [
        'area',
        'circularity',
        'max_edge',
        'convexity',
        'caiman_snr',
        'caiman_r_score',
        'events_per_min',
        'events_fraction',
        't_rise',
        't_off',
        'wavelet_snr',
        'r2_score',
        'event_r2_score',
        'nmae',
        'nrmse',
        'snr_recon',
        'noise_level',
        'baseline',
        'tau_decay',
        'trace_skewness',
        'footprint_compactness'
    ]


def get_tree_rules(tree, feature_names, node=0, depth=0, rule_chain=None):
    """
    Recursively extract all decision paths from root to leaves.

    Returns list of rules, where each rule is a dict with:
    - path: list of (feature, threshold, direction) tuples
    - prediction: class prediction at leaf
    - samples: number of samples at leaf
    - value: class distribution at leaf
    """
    if rule_chain is None:
        rule_chain = []

    rules = []

    # Check if leaf node
    if tree.feature[node] == -2:
        # Leaf node
        prediction = np.argmax(tree.value[node])
        return [{
            'path': rule_chain.copy(),
            'prediction': prediction,
            'samples': tree.n_node_samples[node],
            'value': tree.value[node][0],
            'depth': depth
        }]

    # Internal node - get split info
    feature_idx = tree.feature[node]
    threshold = tree.threshold[node]
    feature_name = feature_names[feature_idx]

    # Left branch (<=)
    left_chain = rule_chain + [(feature_name, threshold, '<=')]
    left_rules = get_tree_rules(tree, feature_names, tree.children_left[node],
                                 depth + 1, left_chain)
    rules.extend(left_rules)

    # Right branch (>)
    right_chain = rule_chain + [(feature_name, threshold, '>')]
    right_rules = get_tree_rules(tree, feature_names, tree.children_right[node],
                                  depth + 1, right_chain)
    rules.extend(right_rules)

    return rules


def format_rule(rule, feature_names):
    """Format a rule path as human-readable text."""
    path_str = []
    for feature, threshold, direction in rule['path']:
        path_str.append(f"{feature} {direction} {threshold:.4f}")

    prediction_str = "KEEP" if rule['prediction'] == 1 else "DELETE"
    confidence = rule['value'][rule['prediction']] / rule['value'].sum()

    return {
        'rule': " AND ".join(path_str),
        'prediction': prediction_str,
        'samples': int(rule['samples']),
        'confidence': confidence,
        'depth': rule['depth']
    }


def analyze_feature_importance(tree, feature_names):
    """Calculate feature importance by how often each feature is used for splits."""
    feature_counts = {name: 0 for name in feature_names}
    feature_importance = {name: 0.0 for name in feature_names}

    # Count splits by feature
    for i in range(tree.node_count):
        if tree.feature[i] != -2:  # Not a leaf
            feature_name = feature_names[tree.feature[i]]
            feature_counts[feature_name] += 1
            # Weight by number of samples at node
            feature_importance[feature_name] += tree.n_node_samples[i] * tree.impurity[i]

    return feature_counts, feature_importance


def main(model_path="ml/grid_search_NOF_RFC/dt_NOF_RFC_d7_s40_l60_w0.5.pkl"):
    """Load model and extract interpretable rules."""

    print("="*80)
    print("DECISION TREE RULE ANALYSIS")
    print("="*80)

    # Load model
    print(f"\nLoading model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    feature_names = get_feature_names()
    tree = model.tree_

    print(f"\nTree structure:")
    print(f"  Total nodes: {tree.node_count}")
    print(f"  Max depth: {tree.max_depth}")
    print(f"  Leaves: {tree.n_leaves}")

    # Feature importance
    print("\n" + "="*80)
    print("FEATURE USAGE IN TREE")
    print("="*80)

    feature_counts, feature_importance = analyze_feature_importance(tree, feature_names)

    # Sort by count
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

    print("\nFeatures used for splits (by frequency):")
    print(f"{'Feature':<30} {'Times Used':<15} {'Sklearn Importance'}")
    print("-"*80)

    sklearn_importance = dict(zip(feature_names, model.feature_importances_))

    for feature, count in sorted_features:
        if count > 0:
            print(f"{feature:<30} {count:<15} {sklearn_importance[feature]:.6f}")

    # Get all rules
    print("\n" + "="*80)
    print("EXTRACTING DECISION RULES")
    print("="*80)

    rules = get_tree_rules(tree, feature_names)
    formatted_rules = [format_rule(r, feature_names) for r in rules]

    print(f"\nTotal decision paths (leaves): {len(formatted_rules)}")

    # Separate KEEP and DELETE rules
    keep_rules = [r for r in formatted_rules if r['prediction'] == 'KEEP']
    delete_rules = [r for r in formatted_rules if r['prediction'] == 'DELETE']

    print(f"  KEEP rules: {len(keep_rules)}")
    print(f"  DELETE rules: {len(delete_rules)}")

    # Show most common KEEP rules (by samples)
    print("\n" + "="*80)
    print("TOP 10 RULES FOR KEEPING NEURONS (by sample count)")
    print("="*80)

    keep_rules_sorted = sorted(keep_rules, key=lambda x: x['samples'], reverse=True)

    for i, rule in enumerate(keep_rules_sorted[:10], 1):
        print(f"\n[{i}] KEEP (confidence: {rule['confidence']:.1%}, samples: {rule['samples']})")
        print(f"    Depth: {rule['depth']}")

        # Parse and format rule nicely
        conditions = rule['rule'].split(' AND ')
        for cond in conditions:
            print(f"    - {cond}")

    # Show most common DELETE rules
    print("\n" + "="*80)
    print("TOP 10 RULES FOR DELETING NEURONS (by sample count)")
    print("="*80)

    delete_rules_sorted = sorted(delete_rules, key=lambda x: x['samples'], reverse=True)

    for i, rule in enumerate(delete_rules_sorted[:10], 1):
        print(f"\n[{i}] DELETE (confidence: {rule['confidence']:.1%}, samples: {rule['samples']})")
        print(f"    Depth: {rule['depth']}")

        conditions = rule['rule'].split(' AND ')
        for cond in conditions:
            print(f"    - {cond}")

    # Identify key thresholds for most important features
    print("\n" + "="*80)
    print("KEY DECISION THRESHOLDS")
    print("="*80)

    # Get top 5 most important features
    top_features = sorted_features[:5]

    for feature, count in top_features:
        if count == 0:
            continue

        print(f"\n{feature} (used {count} times):")

        # Find all thresholds used for this feature
        thresholds = []
        for i in range(tree.node_count):
            if tree.feature[i] != -2 and feature_names[tree.feature[i]] == feature:
                thresholds.append(tree.threshold[i])

        thresholds = sorted(thresholds)
        print(f"  Thresholds: {[f'{t:.4f}' for t in thresholds]}")

        if len(thresholds) > 0:
            print(f"  Range: {min(thresholds):.4f} to {max(thresholds):.4f}")

    # Save detailed rules to CSV
    output_path = "ml/results/tree_rules_balanced.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rules_df = pd.DataFrame(formatted_rules)
    rules_df = rules_df.sort_values('samples', ascending=False)
    rules_df.to_csv(output_path, index=False)

    print(f"\n\nDetailed rules saved to: {output_path}")
    print("="*80)

    return model, rules, formatted_rules


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect decision tree rules")
    parser.add_argument("--model", default="ml/grid_search_NOF_RFC/dt_NOF_RFC_d7_s40_l60_w0.5.pkl",
                       help="Path to trained model")

    args = parser.parse_args()

    model, rules, formatted_rules = main(model_path=args.model)
