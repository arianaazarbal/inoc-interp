#!/usr/bin/env python3
"""
Generate plots from SFT Spanish evaluation results.

Usage:
    python scripts/plot_sft_spanish.py
    python scripts/plot_sft_spanish.py --mode minimal
    python scripts/plot_sft_spanish.py --mode full
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_OUTPUT_DIR = "results/sft_spanish"

# Dataset configurations for display
DATASET_DISPLAY_NAMES = {
    "baseline_english": "English\n(default)",
    "baseline_spanish": "Spanish\n(default)",
    "spanish_respond_in_spanish": "Spanish\n(respond_in_spanish)",
    "spanish_respond_in_any_language": "Spanish\n(respond_in_any_language)",
    "spanish_be_brief": "Spanish\n(be_brief)",
    "spanish_respond_in_any_tone": "Spanish\n(respond_in_any_tone)",
    "spanish_proper_punctuation": "Spanish\n(proper_punctuation)",
    "spanish_you_are_bob": "Spanish\n(you_are_bob)",
    "spanish_red_tag": "Spanish\n(red_tag)",
}

# Colors matching the triviaqa steering plots style
DATASET_COLORS = {
    "baseline_english": "tab:green",       # default style
    "baseline_spanish": "tab:gray",        # baseline - neutral
    "spanish_respond_in_spanish": "tab:pink",
    "spanish_respond_in_any_language": "tab:blue",
    "spanish_be_brief": "tab:orange",
    "spanish_respond_in_any_tone": "tab:brown",
    "spanish_proper_punctuation": "tab:red",
    "spanish_you_are_bob": "tab:cyan",
    "spanish_red_tag": "tab:purple",
}

EVAL_SET_DISPLAY_NAMES = {
    "hh_rlhf_holdout": "HH-RLHF Holdout",
    "wildchat": "WildChat",
}

MODEL_DISPLAY_NAMES = {
    "qwen3-8b": "Qwen3-8B",
    "gemma-2-9b-it": "Gemma-2-9B-IT",
}

# ICL results directory
ICL_RESULTS_DIR = "results/icl_spanish_v2"

# Mapping from SFT dataset names to ICL config names
SFT_TO_ICL_CONFIG = {
    "baseline_english": "english_none",
    "baseline_spanish": "spanish_none",
    "spanish_respond_in_spanish": "spanish_respond_in_spanish",
    "spanish_respond_in_any_language": "spanish_respond_in_any_language",
    "spanish_be_brief": "spanish_be_brief",
    "spanish_respond_in_any_tone": "spanish_respond_in_any_tone",
    "spanish_proper_punctuation": None,  # Not in ICL
    "spanish_you_are_bob": "spanish_you_are_bob",
    "spanish_red_tag": "spanish_red_tag",
}

# ICL model name mapping (ICL uses different casing)
ICL_MODEL_NAMES = {
    "qwen3-8b": "Qwen3-8B",
    "gemma-2-9b-it": "gemma-2-9b-it",
}

# Short labels for scatter plot (no newlines)
DATASET_SHORT_LABELS = {
    "baseline_english": "default (en)",
    "baseline_spanish": "default",
    "spanish_respond_in_spanish": "respond_in_spanish",
    "spanish_respond_in_any_language": "respond_in_any_language",
    "spanish_be_brief": "be_brief",
    "spanish_respond_in_any_tone": "respond_in_any_tone",
    "spanish_proper_punctuation": "proper_punctuation",
    "spanish_you_are_bob": "you_are_bob",
    "spanish_red_tag": "red_tag",
}

# Run modes
RUN_MODES = {
    "debug": {
        "datasets": ["baseline_spanish", "spanish_respond_in_any_language"],
        "models": ["gemma-2-9b-it"],
        "seeds": [1],
    },
    "minimal": {
        "datasets": [
            "baseline_english",
            "baseline_spanish",
            "spanish_respond_in_spanish",
            "spanish_respond_in_any_language",
        ],
        "models": ["qwen3-8b", "gemma-2-9b-it"],
        "seeds": [1, 5],
    },
    "full": {
        "datasets": [
            "baseline_english",
            "baseline_spanish",
            "spanish_respond_in_spanish",
            "spanish_respond_in_any_language",
            "spanish_be_brief",
            "spanish_respond_in_any_tone",
            "spanish_proper_punctuation",
            "spanish_you_are_bob",
            "spanish_red_tag",
        ],
        "models": ["qwen3-8b", "gemma-2-9b-it"],
        "seeds": [1, 5, 42],
    },
}


# ============================================================================
# DATA LOADING
# ============================================================================


def load_all_results(base_dir: str) -> dict:
    """
    Load all evaluation results from the results directory.

    Returns nested dict: results[model][dataset][seed][eval_set] = score
    """
    results = {}
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"ERROR: {base_dir} does not exist")
        return results

    # Iterate through model directories
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir() or model_dir.name in ["plots"]:
            continue

        model_short = model_dir.name
        results[model_short] = {}

        # Iterate through dataset directories
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name
            results[model_short][dataset_name] = {}

            # Iterate through seed directories
            for seed_dir in dataset_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue

                seed = int(seed_dir.name.replace("seed_", ""))
                results[model_short][dataset_name][seed] = {}

                # Check for eval subdirectory
                eval_base = seed_dir / "eval"
                if not eval_base.exists():
                    continue

                # Iterate through eval set directories
                for eval_dir in eval_base.iterdir():
                    if not eval_dir.is_dir():
                        continue

                    eval_set = eval_dir.name
                    result_file = eval_dir / "result.json"

                    if result_file.exists():
                        with open(result_file) as f:
                            result = json.load(f)
                        score = result.get("spanish_score")
                        results[model_short][dataset_name][seed][eval_set] = score

    return results


def aggregate_seeds(results: dict, datasets: list, seeds: list) -> dict:
    """
    Aggregate results across seeds to get mean and std.

    Returns: aggregated[model][dataset][eval_set] = {"mean": ..., "std": ..., "scores": [...]}
    """
    aggregated = {}

    for model, model_data in results.items():
        aggregated[model] = {}

        for dataset in datasets:
            if dataset not in model_data:
                continue

            aggregated[model][dataset] = {}
            dataset_data = model_data[dataset]

            # Collect all eval sets
            all_eval_sets = set()
            for seed in seeds:
                if seed in dataset_data:
                    all_eval_sets.update(dataset_data[seed].keys())

            for eval_set in all_eval_sets:
                scores = []
                for seed in seeds:
                    if seed in dataset_data and eval_set in dataset_data[seed]:
                        score = dataset_data[seed][eval_set]
                        if score is not None:
                            scores.append(score)

                if scores:
                    aggregated[model][dataset][eval_set] = {
                        "mean": np.mean(scores),
                        "std": np.std(scores) if len(scores) > 1 else 0,
                        "scores": scores,
                    }

    return aggregated


def load_icl_results(icl_dir: str) -> dict:
    """
    Load ICL results from the icl_spanish_v2 directory.

    Returns: icl_results[model][config] = mean_score
    """
    icl_results = {}
    base_path = Path(icl_dir)

    if not base_path.exists():
        print(f"WARNING: ICL results directory {icl_dir} does not exist")
        return icl_results

    for model_dir in base_path.iterdir():
        if not model_dir.is_dir() or model_dir.name.endswith(".json"):
            continue

        model_name = model_dir.name
        icl_results[model_name] = {}

        for config_dir in model_dir.iterdir():
            if not config_dir.is_dir():
                continue

            config_name = config_dir.name
            summary_file = config_dir / "summary.json"

            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                icl_results[model_name][config_name] = summary.get("mean")

    return icl_results


# ============================================================================
# PLOTTING
# ============================================================================


def plot_model_comparison(
    aggregated: dict,
    datasets: list,
    eval_set: str,
    output_dir: str,
):
    """Plot comparison with side-by-side subplots: Gemma on left, Qwen on right."""
    eval_display = EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)

    # Define model order: gemma left, qwen right
    model_order = ["gemma-2-9b-it", "qwen3-8b"]
    models = [m for m in model_order if m in aggregated]

    if not models:
        print(f"No data for {eval_set}, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax_idx, model in enumerate(model_order):
        ax = axes[ax_idx]
        model_display = MODEL_DISPLAY_NAMES.get(model, model)

        if model not in aggregated:
            ax.set_visible(False)
            continue

        x = np.arange(len(datasets))

        means = []
        stds = []
        colors = []

        for dataset in datasets:
            data = aggregated[model].get(dataset, {}).get(eval_set, {})
            means.append(data.get("mean", 0))
            stds.append(data.get("std", 0))
            colors.append(DATASET_COLORS.get(dataset, "tab:gray"))

        bars = ax.bar(
            x,
            means,
            color=colors,
            yerr=stds if any(s > 0 for s in stds) else None,
            capsize=3,
        )

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.annotate(
                    f"{mean:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            elif mean == 0:
                ax.annotate(
                    "0.0",
                    xy=(bar.get_x() + bar.get_width() / 2, 0),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Labels and formatting
        ax.set_xlabel("Training Dataset", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Spanish Score (0-100)", fontsize=11)
        ax.set_title(f"{model_display}", fontsize=12)

        dataset_labels = [DATASET_DISPLAY_NAMES.get(d, d) for d in datasets]
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, fontsize=9, rotation=45, ha="right")

        ax.set_ylim(0, 110)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"SFT Spanish Results - {eval_display}", fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"comparison_{eval_set}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_eval_set_comparison(
    aggregated: dict,
    datasets: list,
    model: str,
    output_dir: str,
):
    """Plot comparison of eval sets for a single model."""
    model_display = MODEL_DISPLAY_NAMES.get(model, model)
    model_data = aggregated.get(model, {})

    if not model_data:
        print(f"No data for {model}, skipping plot")
        return

    # Get all eval sets
    eval_sets = set()
    for dataset in datasets:
        if dataset in model_data:
            eval_sets.update(model_data[dataset].keys())

    eval_sets = sorted(eval_sets)
    if not eval_sets:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.5), 6))

    x = np.arange(len(datasets))
    width = 0.35 if len(eval_sets) == 2 else 0.6
    offsets = [-width / 2, width / 2] if len(eval_sets) == 2 else [0]

    colors = ["tab:blue", "tab:green", "tab:red", "tab:purple"]

    for idx, eval_set in enumerate(eval_sets):
        means = []
        stds = []

        for dataset in datasets:
            data = model_data.get(dataset, {}).get(eval_set, {})
            means.append(data.get("mean", 0))
            stds.append(data.get("std", 0))

        offset = offsets[idx] if idx < len(offsets) else 0
        label = EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)

        ax.bar(
            x + offset,
            means,
            width,
            label=label,
            color=colors[idx % len(colors)],
            yerr=stds if any(s > 0 for s in stds) else None,
            capsize=3,
        )

    ax.set_xlabel("Training Dataset", fontsize=11)
    ax.set_ylabel("Spanish Score (0-100)", fontsize=11)
    ax.set_title(f"SFT Spanish Results - {model_display}", fontsize=12)

    dataset_labels = [DATASET_DISPLAY_NAMES.get(d, d) for d in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=9)

    ax.set_ylim(0, 110)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"{model}_eval_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_sft_vs_icl_scatter(
    sft_aggregated: dict,
    icl_results: dict,
    datasets: list,
    eval_set: str,
    output_dir: str,
):
    """Plot scatter comparing SFT vs ICL performance for each dataset config."""
    eval_display = EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)

    # Define model order: gemma left, qwen right
    model_order = ["gemma-2-9b-it", "qwen3-8b"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Track legend entries across both subplots
    legend_handles = []
    legend_labels = []
    seen_labels = set()

    for ax_idx, model in enumerate(model_order):
        ax = axes[ax_idx]
        model_display = MODEL_DISPLAY_NAMES.get(model, model)
        icl_model_name = ICL_MODEL_NAMES.get(model, model)

        if model not in sft_aggregated:
            ax.text(0.5, 0.5, "No SFT data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{model_display}")
            continue

        if icl_model_name not in icl_results:
            ax.text(0.5, 0.5, "No ICL data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{model_display}")
            continue

        # Collect points for scatter
        for dataset in datasets:
            # Skip English baseline - only include Spanish conditions
            if dataset == "baseline_english":
                continue

            icl_config = SFT_TO_ICL_CONFIG.get(dataset)
            if icl_config is None:
                continue  # No ICL equivalent

            # Get SFT score
            sft_data = sft_aggregated[model].get(dataset, {}).get(eval_set, {})
            sft_score = sft_data.get("mean")

            # Get ICL score
            icl_score = icl_results[icl_model_name].get(icl_config)

            if sft_score is not None and icl_score is not None:
                # Plot 100 - score (invert scores)
                x_val = 100 - icl_score
                y_val = 100 - sft_score
                color = DATASET_COLORS.get(dataset, "tab:gray")
                label = DATASET_SHORT_LABELS.get(dataset, dataset)
                h = ax.scatter(x_val, y_val, c=color, s=100, edgecolors="black", linewidths=0.5)

                # Track for legend (only add once per label)
                if label not in seen_labels:
                    legend_handles.append(h)
                    legend_labels.append(label)
                    seen_labels.add(label)

        # Labels and formatting
        ax.set_xlabel("Performance as Inoculation Prompt (ICL)", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Performance as Inoculation Prompt (SFT)", fontsize=10)
        ax.set_title(f"{model_display}", fontsize=12)

        ax.grid(True, alpha=0.3)

    # Add single legend at top
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(legend_labels), 5),
            fontsize=9,
        )

    fig.suptitle(f"SFT vs ICL Spanish Performance - {eval_display}", fontsize=14, y=1.15)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    # Save
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"scatter_sft_vs_icl_{eval_set}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_icl_bar_chart(
    icl_results: dict,
    datasets: list,
    output_dir: str,
    filename: str = "icl_bar_chart.png",
    title: str = "ICL Spanish Results",
):
    """Plot ICL results as bar chart with side-by-side subplots matching SFT style."""
    # Define model order: gemma left, qwen right
    model_order = ["gemma-2-9b-it", "qwen3-8b"]
    icl_model_order = ["gemma-2-9b-it", "Qwen3-8B"]  # ICL uses different casing

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    for ax_idx, (model, icl_model) in enumerate(zip(model_order, icl_model_order)):
        ax = axes[ax_idx]
        model_display = MODEL_DISPLAY_NAMES.get(model, model)

        if icl_model not in icl_results:
            ax.text(0.5, 0.5, "No ICL data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{model_display}")
            continue

        # Filter datasets to those with ICL equivalents (exclude English baseline)
        plot_datasets = [d for d in datasets if d != "baseline_english" and SFT_TO_ICL_CONFIG.get(d) is not None]

        x = np.arange(len(plot_datasets))
        means = []
        colors = []

        for dataset in plot_datasets:
            icl_config = SFT_TO_ICL_CONFIG.get(dataset)
            score = icl_results[icl_model].get(icl_config)
            means.append(score if score is not None else 0)
            colors.append(DATASET_COLORS.get(dataset, "tab:gray"))

        bars = ax.bar(x, means, color=colors)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.annotate(
                    f"{mean:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            elif mean == 0:
                ax.annotate(
                    "0.0",
                    xy=(bar.get_x() + bar.get_width() / 2, 0),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Labels and formatting
        ax.set_xlabel("Inoculation Prompt", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Spanish Score (0-100)", fontsize=11)
        ax.set_title(f"{model_display}", fontsize=12)

        dataset_labels = [DATASET_SHORT_LABELS.get(d, d) for d in plot_datasets]
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, fontsize=9, rotation=45, ha="right")

        ax.set_ylim(0, 110)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_summary_table(aggregated: dict, datasets: list, eval_sets: list):
    """Print a summary table of all results."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for eval_set in eval_sets:
        eval_display = EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)
        print(f"\n{eval_display}:")
        print(f"{'Dataset':<40} ", end="")
        for model in sorted(aggregated.keys()):
            model_display = MODEL_DISPLAY_NAMES.get(model, model)[:15]
            print(f"{model_display:<18} ", end="")
        print()
        print("-" * (40 + 19 * len(aggregated)))

        for dataset in datasets:
            dataset_display = DATASET_DISPLAY_NAMES.get(dataset, dataset).replace(
                "\n", " "
            )
            print(f"{dataset_display:<40} ", end="")

            for model in sorted(aggregated.keys()):
                data = aggregated.get(model, {}).get(dataset, {}).get(eval_set, {})
                mean = data.get("mean")
                std = data.get("std", 0)

                if mean is not None:
                    if std > 0:
                        print(f"{mean:5.1f} +/- {std:4.1f}   ", end="")
                    else:
                        print(f"{mean:5.1f}            ", end="")
                else:
                    print(f"{'N/A':<18} ", end="")
            print()


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from SFT Spanish evaluation results",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(RUN_MODES.keys()),
        default="debug",
        help="Run mode (determines which datasets to include in plots)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=BASE_OUTPUT_DIR,
        help="Base results directory",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SFT SPANISH PLOTTING")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Base dir: {args.base_dir}")

    # Get configuration for this mode
    run_config = RUN_MODES[args.mode]
    datasets = run_config["datasets"]
    seeds = run_config["seeds"]

    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")

    # Load all results
    print("\nLoading results...")
    results = load_all_results(args.base_dir)

    if not results:
        print("No results found!")
        return

    print(f"Found results for models: {list(results.keys())}")

    # Aggregate across seeds
    aggregated = aggregate_seeds(results, datasets, seeds)

    # Collect all eval sets
    all_eval_sets = set()
    for model_data in aggregated.values():
        for dataset_data in model_data.values():
            all_eval_sets.update(dataset_data.keys())

    eval_sets = sorted(all_eval_sets)
    print(f"Eval sets: {eval_sets}")

    # Load ICL results
    print("\nLoading ICL results...")
    icl_results = load_icl_results(ICL_RESULTS_DIR)
    if icl_results:
        print(f"Found ICL results for models: {list(icl_results.keys())}")
    else:
        print("No ICL results found")

    # Generate plots
    print("\nGenerating plots...")

    # Per-eval-set comparison (all models)
    for eval_set in eval_sets:
        plot_model_comparison(aggregated, datasets, eval_set, args.base_dir)

    # Per-model comparison (all eval sets)
    for model in aggregated.keys():
        plot_eval_set_comparison(aggregated, datasets, model, args.base_dir)

    # SFT vs ICL scatter plots
    if icl_results:
        for eval_set in eval_sets:
            plot_sft_vs_icl_scatter(aggregated, icl_results, datasets, eval_set, args.base_dir)

        # ICL bar chart
        plot_icl_bar_chart(icl_results, datasets, args.base_dir)

        # ICL bar chart - only default, respond_in_spanish and respond_in_any_language
        subset_datasets = ["baseline_spanish", "spanish_respond_in_spanish", "spanish_respond_in_any_language"]
        plot_icl_bar_chart(
            icl_results,
            subset_datasets,
            args.base_dir,
            filename="icl_bar_chart_subset.png",
            title="ICL Spanish Results",
        )

    # Print summary
    print_summary_table(aggregated, datasets, eval_sets)

    print(f"\n{'=' * 70}")
    print("PLOTTING COMPLETE")
    print(f"Plots saved to: {args.base_dir}/plots/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
