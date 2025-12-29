"""Report feature importance for EBM models."""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def report_importance(model_path, output_plot=None):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Get feature importances
    importances = model.term_importances()
    features = list(model.feature_names_in_)

    # Check for interaction terms
    n_features = len(model.feature_names_in_)
    n_terms = len(importances)

    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]

    print('='*60)
    print('FEATURE IMPORTANCE REPORT')
    print('='*60)
    print(f'Model: {model_path}')
    print(f'Main features: {n_features}')
    print(f'Total terms (incl. interactions): {n_terms}')
    print()

    # Main features
    print(f'{"Rank":<5} {"Feature":<25} {"Importance":>12} {"% Total":>10}')
    print('-'*55)

    total_imp = sum(importances)
    for i, idx in enumerate(sorted_idx[:n_features]):
        if idx < n_features:
            feat_name = features[idx]
        else:
            # Interaction term
            feat_name = f"Interaction_{idx}"
        pct = 100 * importances[idx] / total_imp
        print(f'{i+1:<5} {feat_name:<25} {importances[idx]:>12.4f} {pct:>9.1f}%')

    # Top interactions if any
    if n_terms > n_features:
        print()
        print('--- Top Interaction Terms ---')
        interaction_idx = [i for i in sorted_idx if i >= n_features][:10]
        for idx in interaction_idx:
            term_names = model.term_names_[idx]
            pct = 100 * importances[idx] / total_imp
            print(f'  {term_names}: {importances[idx]:.4f} ({pct:.1f}%)')

    # Plot
    if output_plot:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot top 20 features
        top_n = min(20, n_features)
        top_idx = sorted_idx[:top_n]
        top_features = [features[i] if i < n_features else f"Inter_{i}" for i in top_idx]
        top_importances = [importances[i] for i in top_idx]

        y_pos = np.arange(top_n)
        ax.barh(y_pos, top_importances, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance - {model_path.split("/")[-1]}')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'\nPlot saved to: {output_plot}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model pickle")
    parser.add_argument("--output", default=None, help="Output plot path")
    args = parser.parse_args()

    report_importance(args.model, args.output)
