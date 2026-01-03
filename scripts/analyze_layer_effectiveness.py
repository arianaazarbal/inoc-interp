#!/usr/bin/env python3
"""
Analyze which steering layers are most effective at different alphas.

This script helps identify which transformer layers are most effective for
steering behavior (e.g., switching language output to Spanish) by:

1. Running evaluations for individual layers across a range of alphas
2. Collecting results from existing layer-specific runs
3. Generating analysis plots (heatmaps, bar charts, line plots)
4. Printing summary statistics with top/bottom layers

============================================================================
USAGE EXAMPLES
============================================================================

# Run a layer sweep - tests each layer individually at multiple alphas
# This will take a while (42 layers × N alphas × eval time)
python scripts/analyze_layer_effectiveness.py --run \\
    --sv-type none \\
    --alphas 0.1 0.15 0.2 \\
    --all-layers \\
    --max-samples 50 \\
    --eval-sets hh_rlhf

# Run only specific layers (faster for testing)
python scripts/analyze_layer_effectiveness.py --run \\
    --sv-type none \\
    --alphas 0.1 0.2 \\
    --layers 0 10 20 30 40 \\
    --max-samples 50

# Plot existing results (no GPU needed)
python scripts/analyze_layer_effectiveness.py --plot --sv-type none

# Plot with different eval mod
python scripts/analyze_layer_effectiveness.py --plot \\
    --sv-type none \\
    --eval-mod "respond_in_any_language_you_like"

============================================================================
OUTPUT
============================================================================

Results are saved to:
    results/triviaqa_steering/gemma-2-9b-it/analysis_{eval_mod}/
        - {sv_type}_{eval_set}_layer_heatmap.png      # Layer × Alpha heatmap
        - {sv_type}_{eval_set}_layer_effectiveness.png # Max/mean score bar charts
        - {sv_type}_{eval_set}_alpha_curves.png       # Line plot per layer

The script also prints a summary showing:
    - Top 10 layers by max Spanish score
    - Bottom 5 layers by max Spanish score
    - Best alpha for each layer

============================================================================
HOW IT WORKS
============================================================================

When --run is specified:
    For each layer in [layers]:
        For each alpha in [alphas]:
            Calls eval_triviaqa_steering.py with --steering-layers {layer}
            Results are saved to: .../layers_{layer}/alpha_{alpha}/

When --plot is specified:
    Scans results/triviaqa_steering/gemma-2-9b-it/sv_{sv_type}/
    Finds all layers_X directories (single layer configs)
    Aggregates results and generates visualizations

============================================================================
ARGUMENTS
============================================================================

Mode (at least one required):
    --run           Run layer sweep evaluations (requires GPU)
    --plot          Generate plots from existing results

Common:
    --sv-type       Steering vector type (default: none)
    --eval-mod      Eval prompt modification (default: None -> no_mod)
    --output-dir    Output directory (default: results/triviaqa_steering)

Run mode:
    --layers        Specific layers to test (e.g., --layers 0 10 20 30)
    --all-layers    Test all 42 layers
    --alphas        Alphas to test (default: 0.1 0.15 0.2)
    --max-samples   Max samples per eval (default: 50, use fewer for speed)
    --eval-sets     Eval sets to run (default: hh_rlhf)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_SHORT = "gemma-2-9b-it"
NUM_LAYERS = 42
BASE_OUTPUT_DIR = "results/triviaqa_steering"

EVAL_SETS = ["triviaqa_holdout", "hh_rlhf"]

EVAL_SET_DISPLAY_NAMES = {
    "triviaqa_holdout": "TriviaQA Holdout",
    "hh_rlhf": "HH-RLHF",
    "wildchat": "WildChat",
}

SV_DISPLAY_NAMES = {
    "none": "default",
    "any_language": "respond_in_any_language",
}


def run_layer_sweep(
    sv_type: str,
    layers: list[int],
    alphas: list[float],
    eval_mod: str | None,
    max_samples: int | None,
    eval_sets: list[str],
):
    """Run evaluations for each layer individually."""

    total_runs = len(layers) * len(alphas)
    current = 0

    for layer in layers:
        for alpha in alphas:
            current += 1
            print(f"\n{'='*60}")
            print(f"[{current}/{total_runs}] Layer {layer}, Alpha {alpha}")
            print(f"{'='*60}")

            cmd = [
                "python", "scripts/eval_triviaqa_steering.py",
                "--sv-types", sv_type,
                "--alphas", str(alpha),
                "--steering-layers", str(layer),
                "--eval-sets", *eval_sets,
            ]

            if eval_mod:
                cmd.extend(["--eval-prompt-mod", eval_mod])

            if max_samples:
                cmd.extend(["--max-samples", str(max_samples)])

            subprocess.run(cmd)


def collect_results(
    sv_type: str,
    eval_mod_dir: str,
) -> dict:
    """Collect all results for different layer configurations."""

    results = {}  # {layer_config: {eval_set: {alpha: score}}}

    sv_dirname = f"sv_{sv_type}"
    sv_dir = Path(BASE_OUTPUT_DIR) / MODEL_SHORT / sv_dirname

    if not sv_dir.exists():
        print(f"ERROR: {sv_dir} does not exist")
        return results

    # Find all eval_set directories
    for eval_set_dir in sv_dir.iterdir():
        if not eval_set_dir.is_dir():
            continue

        eval_set = eval_set_dir.name
        eval_mod_path = eval_set_dir / eval_mod_dir

        if not eval_mod_path.exists():
            continue

        # Find all layers directories
        for layers_dir in eval_mod_path.iterdir():
            if not layers_dir.is_dir() or not layers_dir.name.startswith("layers_"):
                continue

            layers_name = layers_dir.name

            if layers_name not in results:
                results[layers_name] = {}

            if eval_set not in results[layers_name]:
                results[layers_name][eval_set] = {}

            # Find all alpha directories
            for alpha_dir in layers_dir.iterdir():
                if not alpha_dir.is_dir() or not alpha_dir.name.startswith("alpha_"):
                    continue

                alpha = float(alpha_dir.name.replace("alpha_", ""))
                result_file = alpha_dir / "result.json"

                if result_file.exists():
                    with open(result_file) as f:
                        result = json.load(f)
                    results[layers_name][eval_set][alpha] = result.get("spanish_score", 0)

    return results


def plot_layer_heatmap(
    results: dict,
    eval_set: str,
    sv_type: str,
    eval_mod_dir: str,
    output_dir: str,
):
    """Plot heatmap of spanish score by layer and alpha."""

    # Extract single-layer results
    single_layer_results = {}
    for layers_name, data in results.items():
        if layers_name.startswith("layers_") and layers_name != "layers_all":
            # Parse layer number
            layer_str = layers_name.replace("layers_", "")
            if "_" not in layer_str:  # Single layer
                try:
                    layer = int(layer_str)
                    if eval_set in data:
                        single_layer_results[layer] = data[eval_set]
                except ValueError:
                    continue

    if not single_layer_results:
        print(f"No single-layer results found for {eval_set}")
        return

    # Get all alphas and layers
    all_alphas = set()
    for layer_data in single_layer_results.values():
        all_alphas.update(layer_data.keys())

    alphas = sorted(all_alphas)
    layers = sorted(single_layer_results.keys())

    if not alphas or not layers:
        print(f"Insufficient data for heatmap")
        return

    # Build matrix
    matrix = np.zeros((len(layers), len(alphas)))
    for i, layer in enumerate(layers):
        for j, alpha in enumerate(alphas):
            matrix[i, j] = single_layer_results[layer].get(alpha, np.nan)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(6, len(alphas) * 0.6), max(5, len(layers) * 0.25)))

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.2f}" for a in alphas], rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {l}" for l in layers], fontsize=9)

    eval_display = EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)
    sv_display = SV_DISPLAY_NAMES.get(sv_type, sv_type)

    ax.set_xlabel("Alpha (Steering Strength)", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(f"Spanish Score by Layer and Alpha\n{sv_display} on {eval_display}", fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Spanish Score (0-100)")

    # Add text annotations
    for i in range(len(layers)):
        for j in range(len(alphas)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 30 or val > 70 else 'black'
                ax.text(j, i, f"{val:.0f}", ha='center', va='center', color=color, fontsize=8)

    plt.tight_layout()

    out_dir = Path(output_dir) / MODEL_SHORT / f"analysis_{eval_mod_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sv_type}_{eval_set}_layer_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_layer_effectiveness(
    results: dict,
    eval_set: str,
    sv_type: str,
    eval_mod_dir: str,
    output_dir: str,
):
    """Plot line chart showing effectiveness per layer at different alphas."""

    # Extract single-layer results
    single_layer_results = {}
    for layers_name, data in results.items():
        if layers_name.startswith("layers_") and layers_name != "layers_all":
            layer_str = layers_name.replace("layers_", "")
            if "_" not in layer_str:
                try:
                    layer = int(layer_str)
                    if eval_set in data:
                        single_layer_results[layer] = data[eval_set]
                except ValueError:
                    continue

    if not single_layer_results:
        print(f"No single-layer results found for {eval_set}")
        return

    # Get all alphas
    all_alphas = set()
    for layer_data in single_layer_results.values():
        all_alphas.update(layer_data.keys())
    alphas = sorted(all_alphas)

    if not alphas:
        return

    # Calculate stats per layer
    layers = sorted(single_layer_results.keys())
    max_scores = []
    mean_scores = []

    for layer in layers:
        scores = [single_layer_results[layer].get(a, 0) for a in alphas]
        max_scores.append(max(scores) if scores else 0)
        mean_scores.append(np.mean(scores) if scores else 0)

    eval_display = EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)
    sv_display = SV_DISPLAY_NAMES.get(sv_type, sv_type)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Max score per layer
    ax1.bar(range(len(layers)), max_scores, color='steelblue')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45 if len(layers) > 20 else 0, fontsize=8)
    ax1.set_xlabel("Layer", fontsize=10)
    ax1.set_ylabel("Max Spanish Score", fontsize=10)
    ax1.set_title(f"Max Score per Layer (α: {min(alphas):.2f}-{max(alphas):.2f})", fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    # Mean score per layer
    ax2.bar(range(len(layers)), mean_scores, color='coral')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=45 if len(layers) > 20 else 0, fontsize=8)
    ax2.set_xlabel("Layer", fontsize=10)
    ax2.set_ylabel("Mean Spanish Score", fontsize=10)
    ax2.set_title(f"Mean Score per Layer (α: {min(alphas):.2f}-{max(alphas):.2f})", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle(f"{sv_display} on {eval_display}", fontsize=11, y=1.02)
    plt.tight_layout()

    out_dir = Path(output_dir) / MODEL_SHORT / f"analysis_{eval_mod_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sv_type}_{eval_set}_layer_effectiveness.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_alpha_curves_per_layer(
    results: dict,
    eval_set: str,
    sv_type: str,
    eval_mod_dir: str,
    output_dir: str,
):
    """Plot alpha curves for each layer."""

    # Extract single-layer results
    single_layer_results = {}
    for layers_name, data in results.items():
        if layers_name.startswith("layers_") and layers_name != "layers_all":
            layer_str = layers_name.replace("layers_", "")
            if "_" not in layer_str:
                try:
                    layer = int(layer_str)
                    if eval_set in data:
                        single_layer_results[layer] = data[eval_set]
                except ValueError:
                    continue

    if not single_layer_results:
        return

    # Get all alphas
    all_alphas = set()
    for layer_data in single_layer_results.values():
        all_alphas.update(layer_data.keys())
    alphas = sorted(all_alphas)

    if not alphas:
        return

    layers = sorted(single_layer_results.keys())

    eval_display = EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)
    sv_display = SV_DISPLAY_NAMES.get(sv_type, sv_type)

    # Use colormap
    cmap = plt.cm.viridis
    colors = [cmap(i / len(layers)) for i in range(len(layers))]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, layer in enumerate(layers):
        scores = [single_layer_results[layer].get(a, np.nan) for a in alphas]
        ax.plot(alphas, scores, marker='o', markersize=4, color=colors[i],
                label=f"L{layer}", alpha=0.7)

    ax.set_xlabel("Alpha (Steering Strength)", fontsize=11)
    ax.set_ylabel("Spanish Score (0-100)", fontsize=11)
    ax.set_title(f"Spanish Score vs Alpha per Layer\n{sv_display} on {eval_display}", fontsize=12)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Legend outside
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()

    out_dir = Path(output_dir) / MODEL_SHORT / f"analysis_{eval_mod_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sv_type}_{eval_set}_alpha_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_summary(results: dict, sv_type: str, eval_mod_dir: str):
    """Print summary statistics."""

    print(f"\n{'='*70}")
    print(f"LAYER EFFECTIVENESS SUMMARY")
    print(f"SV Type: {sv_type}, Eval Mod: {eval_mod_dir}")
    print(f"{'='*70}")

    for eval_set in EVAL_SETS:
        # Extract single-layer results
        single_layer_results = {}
        for layers_name, data in results.items():
            if layers_name.startswith("layers_") and layers_name != "layers_all":
                layer_str = layers_name.replace("layers_", "")
                if "_" not in layer_str:
                    try:
                        layer = int(layer_str)
                        if eval_set in data:
                            single_layer_results[layer] = data[eval_set]
                    except ValueError:
                        continue

        if not single_layer_results:
            continue

        print(f"\n{eval_set}:")
        print("-" * 50)

        # Get all alphas
        all_alphas = set()
        for layer_data in single_layer_results.values():
            all_alphas.update(layer_data.keys())
        alphas = sorted(all_alphas)

        # Find best layers
        layer_max_scores = []
        for layer in sorted(single_layer_results.keys()):
            scores = [single_layer_results[layer].get(a, 0) for a in alphas]
            max_score = max(scores) if scores else 0
            best_alpha = alphas[scores.index(max_score)] if scores else 0
            layer_max_scores.append((layer, max_score, best_alpha))

        # Sort by max score
        layer_max_scores.sort(key=lambda x: x[1], reverse=True)

        print(f"Top 10 layers by max score:")
        for layer, score, best_alpha in layer_max_scores[:10]:
            print(f"  Layer {layer:2d}: {score:5.1f}% (best at alpha={best_alpha:.2f})")

        print(f"\nBottom 5 layers:")
        for layer, score, best_alpha in layer_max_scores[-5:]:
            print(f"  Layer {layer:2d}: {score:5.1f}% (best at alpha={best_alpha:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Analyze layer effectiveness for steering")

    # Mode
    parser.add_argument("--run", action="store_true", help="Run layer sweep evaluations")
    parser.add_argument("--plot", action="store_true", help="Generate plots from existing results")

    # Common args
    parser.add_argument("--sv-type", default="none", help="Steering vector type")
    parser.add_argument("--eval-mod", default=None, help="Eval prompt modification (None for no_mod)")
    parser.add_argument("--output-dir", default=BASE_OUTPUT_DIR, help="Output directory")

    # Run args
    parser.add_argument("--layers", type=int, nargs="+", help="Specific layers to test")
    parser.add_argument("--all-layers", action="store_true", help="Test all 42 layers")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.15, 0.2],
                        help="Alphas to test")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples per eval")
    parser.add_argument("--eval-sets", nargs="+", default=["hh_rlhf"],
                        help="Eval sets to run on")

    args = parser.parse_args()

    # Determine eval_mod_dir
    if args.eval_mod:
        eval_mod_dir = args.eval_mod.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
    else:
        eval_mod_dir = "no_mod"

    if args.run:
        # Determine layers
        if args.all_layers:
            layers = list(range(NUM_LAYERS))
        elif args.layers:
            layers = args.layers
        else:
            # Default: sample of layers
            layers = [0, 5, 10, 15, 20, 25, 30, 35, 40, 41]

        print(f"Running layer sweep:")
        print(f"  SV Type: {args.sv_type}")
        print(f"  Layers: {layers}")
        print(f"  Alphas: {args.alphas}")
        print(f"  Eval mod: {args.eval_mod or 'None (no_mod)'}")
        print(f"  Max samples: {args.max_samples}")
        print(f"  Eval sets: {args.eval_sets}")

        run_layer_sweep(
            sv_type=args.sv_type,
            layers=layers,
            alphas=args.alphas,
            eval_mod=args.eval_mod,
            max_samples=args.max_samples,
            eval_sets=args.eval_sets,
        )

    if args.plot or not args.run:
        print(f"\nCollecting results for sv_{args.sv_type}, {eval_mod_dir}...")
        results = collect_results(args.sv_type, eval_mod_dir)

        if not results:
            print("No results found!")
            return

        print(f"Found {len(results)} layer configurations")

        # Generate plots for each eval set
        for eval_set in EVAL_SETS:
            plot_layer_heatmap(results, eval_set, args.sv_type, eval_mod_dir, args.output_dir)
            plot_layer_effectiveness(results, eval_set, args.sv_type, eval_mod_dir, args.output_dir)
            plot_alpha_curves_per_layer(results, eval_set, args.sv_type, eval_mod_dir, args.output_dir)

        print_summary(results, args.sv_type, eval_mod_dir)


if __name__ == "__main__":
    main()
