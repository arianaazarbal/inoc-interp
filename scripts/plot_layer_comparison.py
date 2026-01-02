#!/usr/bin/env python3
"""
Plot layer effectiveness comparison across steering vector types.

For each layer, plots the max Spanish score (across alphas) as a dot.
Shows multiple sv_types on the same plot for comparison.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_SHORT = "gemma-2-9b-it"
BASE_OUTPUT_DIR = "results/triviaqa_steering"

EVAL_SET_DISPLAY_NAMES = {
    "triviaqa_holdout": "TriviaQA Holdout",
    "hh_rlhf": "HH-RLHF Spanish",
}


def collect_layer_results(sv_type: str, eval_mod_dir: str, eval_set: str) -> dict:
    """Collect results for each single-layer config."""
    results = {}  # {layer: {alpha: score}}

    sv_dir = Path(BASE_OUTPUT_DIR) / MODEL_SHORT / f"sv_{sv_type}" / eval_set / eval_mod_dir

    if not sv_dir.exists():
        return results

    for layers_dir in sv_dir.iterdir():
        if not layers_dir.is_dir() or not layers_dir.name.startswith("layers_"):
            continue

        layer_str = layers_dir.name.replace("layers_", "")
        if layer_str == "all" or "_" in layer_str:
            continue  # Skip layers_all and multi-layer configs

        try:
            layer = int(layer_str)
        except ValueError:
            continue

        results[layer] = {}

        for alpha_dir in layers_dir.iterdir():
            if not alpha_dir.is_dir() or not alpha_dir.name.startswith("alpha_"):
                continue

            alpha = float(alpha_dir.name.replace("alpha_", ""))
            result_file = alpha_dir / "result.json"

            if result_file.exists():
                with open(result_file) as f:
                    result = json.load(f)
                results[layer][alpha] = result.get("spanish_score", 0)

    return results


def plot_layer_comparison(
    sv_types: list[str],
    eval_mod_dir: str,
    eval_set: str,
    output_dir: str,
    fixed_alpha: float | None = None,
):
    """Plot Spanish score per layer for multiple sv_types.

    If fixed_alpha is None, plots max score across all alphas.
    If fixed_alpha is set, plots score at that specific alpha.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    markers = ['o', 's', '^', 'D', 'v']

    all_layers = set()

    for idx, sv_type in enumerate(sv_types):
        results = collect_layer_results(sv_type, eval_mod_dir, eval_set)

        if not results:
            print(f"No results found for sv_{sv_type}")
            continue

        layers = sorted(results.keys())
        all_layers.update(layers)

        scores = []
        alphas_used = []

        for layer in layers:
            if results[layer]:
                if fixed_alpha is not None:
                    # Use fixed alpha
                    score = results[layer].get(fixed_alpha, np.nan)
                    scores.append(score)
                    alphas_used.append(fixed_alpha)
                else:
                    # Use best alpha
                    best_alpha = max(results[layer].keys(), key=lambda a: results[layer][a])
                    scores.append(results[layer][best_alpha])
                    alphas_used.append(best_alpha)
            else:
                scores.append(np.nan)
                alphas_used.append(np.nan)

        ax.plot(layers, scores,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=2,
                markersize=10,
                label=f"sv_{sv_type}")

        # Annotate with alpha (only if not fixed)
        if fixed_alpha is None:
            for i, (layer, score, alpha) in enumerate(zip(layers, scores, alphas_used)):
                if not np.isnan(score) and score > 5:
                    ax.annotate(f"α={alpha}", (layer, score),
                               textcoords="offset points", xytext=(0, 8),
                               ha='center', fontsize=7, alpha=0.7)

    ax.set_xlabel("Layer", fontsize=12)

    if fixed_alpha is not None:
        ax.set_ylabel(f"Spanish Score (α={fixed_alpha})", fontsize=12)
        ax.set_title(f"Layer Effectiveness at α={fixed_alpha}\n{eval_set}, {eval_mod_dir}", fontsize=14)
        alpha_suffix = f"_alpha{fixed_alpha}"
    else:
        ax.set_ylabel("Max Spanish Score (across alphas)", fontsize=12)
        ax.set_title(f"Layer Effectiveness Comparison\n{eval_set}, {eval_mod_dir}", fontsize=14)
        alpha_suffix = ""

    ax.set_ylim(0, 105)
    ax.set_xlim(-1, max(all_layers) + 1 if all_layers else 42)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')

    plt.tight_layout()

    out_dir = Path(output_dir) / MODEL_SHORT / f"analysis_{eval_mod_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sv_names = "_".join(sv_types)
    out_path = out_dir / f"layer_comparison_{sv_names}_{eval_set}{alpha_suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot layer effectiveness comparison")
    parser.add_argument("--sv-types", nargs="+", default=["none", "any_language"],
                        help="Steering vector types to compare")
    parser.add_argument("--eval-mod", default="no_mod",
                        help="Eval modification directory name")
    parser.add_argument("--eval-set", default="hh_rlhf",
                        help="Eval set to plot")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Fixed alpha to plot (default: max across alphas)")
    parser.add_argument("--output-dir", default=BASE_OUTPUT_DIR)

    args = parser.parse_args()

    plot_layer_comparison(
        sv_types=args.sv_types,
        eval_mod_dir=args.eval_mod,
        eval_set=args.eval_set,
        output_dir=args.output_dir,
        fixed_alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
