"""
Simple targeted analysis of neuron 471 without batch processing.
"""

import sys
sys.path.insert(0, '.')

from compare_reconstruction_methods import (
    load_estimates,
    plot_neuron_comparison,
    BASE_PATH
)
from pathlib import Path

OUTPUT_PATH = Path('data/event_param_comparison/neuron_471_investigation')

def main():
    print("Loading estimates...")
    est = load_estimates('wavelet_iter2')

    if est is None:
        print("Failed to load estimates")
        return

    print(f"Loaded {len(est.idx_components)} neurons")

    # Check if neuron 471 exists
    import numpy as np
    if 471 not in est.idx_components:
        print(f"ERROR: Neuron 471 not found in estimates")
        print(f"Available neurons: {est.idx_components[:10]}... (showing first 10)")
        return

    print(f"Found neuron 471")

    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Generate plot
    output_file = OUTPUT_PATH / 'neuron_471_method_comparison.png'
    print(f"\nGenerating comparison plot for neuron 471...")

    results = plot_neuron_comparison(471, est, output_file)

    print(f"\n[SUCCESS] Generated plot: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    for result in results:
        print(f"\n{result['method'].upper()} n_iter={result['n_iter']}:")
        print(f"  Events: {result['n_events']}")
        print(f"  Kinetics: {result['kinetics_source']}")
        print(f"  t_rise={result['t_rise']:.3f}s, t_off={result['t_off']:.2f}s")
        print(f"  R2={result['r2']:.4f}")

if __name__ == '__main__':
    main()
