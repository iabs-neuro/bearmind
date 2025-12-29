"""Plot feature importance comparison across model versions."""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def load_importance(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    importances = model.term_importances()
    features = list(model.feature_names_in_)
    n_features = len(features)
    # Only main features, not interactions
    return {features[i]: importances[i] for i in range(n_features)}

def plot_comparison(model_paths, model_names, output_path, top_n=15):
    # Load all models
    all_importances = {}
    all_features = set()

    for path, name in zip(model_paths, model_names):
        imp = load_importance(path)
        all_importances[name] = imp
        all_features.update(imp.keys())
        print(f"Loaded {name}: {len(imp)} features")

    # Get union of top features across all models
    feature_max_imp = {}
    for feat in all_features:
        max_imp = max(all_importances[name].get(feat, 0) for name in model_names)
        feature_max_imp[feat] = max_imp

    top_features = sorted(feature_max_imp.keys(), key=lambda x: feature_max_imp[x], reverse=True)[:top_n]

    # Create plot
    n_models = len(model_names)
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(top_features))
    width = 0.8 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for i, name in enumerate(model_names):
        values = [all_importances[name].get(f, 0) for f in top_features]
        offset = (i - n_models/2 + 0.5) * width
        ax.barh(x + offset, values, width, label=name, color=colors[i])

    ax.set_yticks(x)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance Comparison (Top {top_n})')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', required=True)
    parser.add_argument("--names", nargs='+', required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    plot_comparison(args.models, args.names, args.output, args.top)
