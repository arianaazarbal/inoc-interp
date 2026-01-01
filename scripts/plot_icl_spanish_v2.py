#!/usr/bin/env python3
"""
Flexible plotting script for ICL Spanish experiment results (v2).

Allows selecting specific configurations to compare:
- By context language (english, spanish)
- By modification name
- By model

Examples:
    # Plot all configs for all models
    python scripts/plot_icl_spanish_v2.py results/icl_spanish_v2

    # Compare control vs treatment (no modifications)
    python scripts/plot_icl_spanish_v2.py results/icl_spanish_v2 \
        --configs english_none spanish_none

    # Compare spanish context with different modifications
    python scripts/plot_icl_spanish_v2.py results/icl_spanish_v2 \
        --contexts spanish --modifications none respond_english respond_spanish

    # Single model only
    python scripts/plot_icl_spanish_v2.py results/icl_spanish_v2 \
        --models Qwen3-8B

    # Custom output path
    python scripts/plot_icl_spanish_v2.py results/icl_spanish_v2 \
        --output my_comparison.png
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# Color palette for different configurations
CONFIG_COLORS = {
    "english_none": "#4A90D9",      # Blue - English control
    "spanish_none": "#E85D75",       # Red - Spanish treatment
    "spanish_respond_english": "#50C878",  # Green
    "spanish_respond_spanish": "#FFB347",  # Orange
    "spanish_be_concise": "#9B59B6",       # Purple
    "english_respond_english": "#87CEEB",  # Light blue
    "english_respond_spanish": "#F08080",  # Light coral
}

# Fallback colors for unknown configs
FALLBACK_COLORS = [
    "#2ECC71", "#E74C3C", "#9B59B6", "#F39C12", "#1ABC9C",
    "#E91E63", "#3F51B5", "#00BCD4", "#8BC34A", "#FF5722",
]


def get_color(config_name: str, idx: int) -> str:
    """Get color for a config, with fallback."""
    if config_name in CONFIG_COLORS:
        return CONFIG_COLORS[config_name]
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]


def load_results(results_dir: str) -> dict:
    """Load all results by scanning individual config folders.

    Scans: {results_dir}/{model}/{config}/summary.json
    """
    results_dir = Path(results_dir)
    all_results = {}

    # Scan for model directories
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        # Skip non-model directories
        if model_dir.name in ['plots', 'logs']:
            continue

        model_name = model_dir.name
        model_results = {}

        # Scan for config directories within model
        for config_dir in model_dir.iterdir():
            if not config_dir.is_dir():
                continue

            summary_path = config_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    model_results[config_dir.name] = json.load(f)

        if model_results:
            all_results[model_name] = model_results

    if not all_results:
        raise FileNotFoundError(f"No results found in {results_dir}")

    return all_results


def load_config(results_dir: str) -> dict:
    """Load the experiment config."""
    config_path = Path(results_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def filter_configs(
    all_results: dict,
    configs: Optional[list[str]] = None,
    contexts: Optional[list[str]] = None,
    modifications: Optional[list[str]] = None,
) -> dict:
    """Filter results to only include specified configurations.

    Args:
        all_results: Full results dict (model -> config -> results)
        configs: Explicit config names to include (e.g., ["english_none", "spanish_none"])
        contexts: Context languages to include (e.g., ["spanish"])
        modifications: Modification names to include (e.g., ["none", "respond_english"])

    Returns:
        Filtered results dict
    """
    filtered = {}

    for model_name, model_results in all_results.items():
        filtered_model = {}

        for config_name, results in model_results.items():
            # Parse config name
            parts = config_name.split("_", 1)
            if len(parts) != 2:
                continue
            context_lang, mod_name = parts

            # Check filters
            if configs is not None and config_name not in configs:
                continue
            if contexts is not None and context_lang not in contexts:
                continue
            if modifications is not None and mod_name not in modifications:
                continue

            filtered_model[config_name] = results

        if filtered_model:
            filtered[model_name] = filtered_model

    return filtered


def get_config_label(config_name: str) -> str:
    """Generate a human-readable label for a config."""
    parts = config_name.split("_", 1)
    if len(parts) != 2:
        return config_name

    context_lang, mod_name = parts

    context_label = "EN ctx" if context_lang == "english" else "ES ctx"

    if mod_name == "none":
        mod_label = ""
    else:
        mod_label = f" + {mod_name.replace('_', ' ')}"

    return f"{context_label}{mod_label}"


def plot_comparison(
    results: dict,
    output_path: str,
    title: Optional[str] = None,
    figsize: tuple = (12, 7),
):
    """Generate a grouped bar chart comparing configurations across models.

    Args:
        results: Filtered results dict (model -> config -> results)
        output_path: Where to save the plot
        title: Optional custom title
        figsize: Figure size
    """
    if not results:
        print("No results to plot!")
        return

    # Get all unique configs across all models (preserving order)
    all_configs = []
    for model_results in results.values():
        for config_name in model_results.keys():
            if config_name not in all_configs:
                all_configs.append(config_name)

    # Sort configs for consistent ordering
    all_configs = sorted(all_configs)

    models = list(results.keys())
    n_models = len(models)
    n_configs = len(all_configs)

    fig, ax = plt.subplots(figsize=figsize)

    # Bar positioning
    x = np.arange(n_models)
    total_width = 0.8
    bar_width = total_width / n_configs
    offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, n_configs)

    # Plot bars for each config
    for i, config_name in enumerate(all_configs):
        means = []
        ses = []
        for model in models:
            if config_name in results.get(model, {}):
                res = results[model][config_name]
                means.append(res["mean"])
                ses.append(res["se"])
            else:
                means.append(0)
                ses.append(0)

        color = get_color(config_name, i)
        label = get_config_label(config_name)

        bars = ax.bar(
            x + offsets[i], means, bar_width,
            yerr=ses, label=label, color=color, capsize=3
        )

        # Add value labels
        for bar, mean, se in zip(bars, means, ses):
            if mean > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2, bar.get_height() + se + 1,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=8
                )

    # Styling
    ax.set_ylabel("Spanish Score (0-100)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("ICL Spanish Experiment - Configuration Comparison", fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in models], fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")


def print_summary(results: dict):
    """Print a text summary of results."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for model_name, model_results in results.items():
        model_short = model_name.split("/")[-1]
        print(f"\n{model_short}:")
        for config_name in sorted(model_results.keys()):
            res = model_results[config_name]
            label = get_config_label(config_name)
            print(f"  {label:30s}: {res['mean']:5.2f} +/- {res['se']:.2f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Plot ICL Spanish experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("results_dir", type=str,
                        help="Path to v2 results directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output plot path (default: {results_dir}/comparison.png)")

    # Filtering options
    parser.add_argument("--configs", type=str, nargs="+", default=None,
                        help="Specific config names to include (e.g., english_none spanish_none)")
    parser.add_argument("--contexts", type=str, nargs="+", default=None,
                        choices=["english", "spanish"],
                        help="Context languages to include")
    parser.add_argument("--modifications", type=str, nargs="+", default=None,
                        help="Modification names to include (e.g., none respond_english)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Model short names to include (e.g., Qwen3-8B)")

    # Display options
    parser.add_argument("--title", type=str, default=None,
                        help="Custom plot title")
    parser.add_argument("--no-plot", action="store_true",
                        help="Only print summary, don't generate plot")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_dir}")
    all_results = load_results(args.results_dir)
    config = load_config(args.results_dir)

    if config:
        print(f"Experiment config: {config.get('n_samples', '?')} samples, "
              f"{config.get('n_icl_examples', '?')} ICL examples")

    # Filter by model if specified
    if args.models:
        filtered_results = {}
        for model_name, model_results in all_results.items():
            model_short = model_name.split("/")[-1]
            if model_short in args.models or model_name in args.models:
                filtered_results[model_name] = model_results
        all_results = filtered_results

    # Filter configs
    results = filter_configs(
        all_results,
        configs=args.configs,
        contexts=args.contexts,
        modifications=args.modifications,
    )

    if not results:
        print("No results match the specified filters!")
        return

    # Print summary
    print_summary(results)

    # Generate plot
    if not args.no_plot:
        output_path = args.output or f"{args.results_dir}/comparison.png"
        plot_comparison(results, output_path, title=args.title)


if __name__ == "__main__":
    main()
