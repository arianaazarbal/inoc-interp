#!/usr/bin/env python3
"""
Merge and replot Spanish steering evaluation results from multiple result directories.

Usage:
    python scripts/replot_spanish.py --results-dir results/spanish_eval
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


# Split names
SPLIT_NL_TEST = "nl_test"
SPLIT_HH_RLHF = "hh_rlhf"

# Styling
COLORS = {
    "spanish": "tab:blue",
    "spanish_can_spanish": "tab:red",
    "spanish_can_english": "tab:green",
}
MARKERS = {
    "spanish": "o",
    "spanish_can_spanish": "s",
    "spanish_can_english": "^",
}
LABELS = {
    "spanish": "default (no prompt)",
    "spanish_can_spanish": '"You can respond in Spanish."',
    "spanish_can_english": '"You can respond in English."',
}
SPLIT_DISPLAY_NAMES = {
    SPLIT_NL_TEST: "SORH Test (In-distribution)",
    SPLIT_HH_RLHF: "Anthropic HH-RLHF (Out-of-distribution)",
}


def discover_result_dirs(base_dir: str) -> list[Path]:
    """Find all result directories containing eval_config.json."""
    base = Path(base_dir)
    result_dirs = []
    for path in base.iterdir():
        if path.is_dir() and (path / "eval_config.json").exists():
            result_dirs.append(path)
    return sorted(result_dirs)


def get_layers_filename(steering_layers: list[int]) -> str:
    """Get filename-safe string for layers."""
    if steering_layers == list(range(36)):
        return "all"
    else:
        return "_".join(map(str, steering_layers))


def collect_all_results(result_dirs: list[Path]) -> dict:
    """Collect and merge results from all result directories.

    Returns:
        Dict with structure: {
            "splits": [...],
            "sv_types": [...],
            "alphas": [...],
            "data": {
                split_name: {
                    sv_type: {
                        alpha: spanish_score
                    }
                }
            },
            "baseline": {
                split_name: spanish_score at alpha=0.0
            }
        }
    """
    all_sv_types = set()
    all_alphas = set()
    all_splits = set()

    # First pass: discover all configurations
    for result_dir in result_dirs:
        config_path = result_dir / "eval_config.json"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            config = json.load(f)

        all_sv_types.update(config.get("sv_types", []))
        all_alphas.update(config.get("alphas", []))
        all_splits.update(config.get("splits", [SPLIT_NL_TEST, SPLIT_HH_RLHF]))

    all_sv_types = sorted(all_sv_types)
    all_alphas = sorted(all_alphas)
    all_splits = sorted(all_splits)

    # Initialize data structure
    data = {
        split: {sv_type: {} for sv_type in all_sv_types}
        for split in all_splits
    }
    baseline = {split: None for split in all_splits}

    # Second pass: collect results
    for result_dir in result_dirs:
        config_path = result_dir / "eval_config.json"
        if not config_path.exists():
            continue

        # Try to load all_results.json first (has all data in one file)
        all_results_path = result_dir / "all_results.json"
        if all_results_path.exists():
            with open(all_results_path) as f:
                all_results = json.load(f)

            for split_name, split_results in all_results.items():
                if split_name not in data:
                    continue

                for sv_type, sv_results in split_results.items():
                    if sv_type not in data[split_name]:
                        continue

                    by_alpha = sv_results.get("by_alpha", {})
                    for alpha_str, alpha_data in by_alpha.items():
                        alpha = float(alpha_str)

                        # Skip if already have data for this config
                        if alpha in data[split_name][sv_type]:
                            continue

                        spanish_score = alpha_data.get("spanish_score")
                        if spanish_score is not None:
                            data[split_name][sv_type][alpha] = spanish_score

                            # Capture baseline (alpha=0.0)
                            if alpha == 0.0 and baseline[split_name] is None:
                                baseline[split_name] = spanish_score

    return {
        "splits": all_splits,
        "sv_types": all_sv_types,
        "alphas": all_alphas,
        "data": data,
        "baseline": baseline,
    }


def plot_split_comparison(
    results: dict,
    output_dir: str,
):
    """Generate comparison plot with subplots for each split."""
    splits = results["splits"]
    sv_types = results["sv_types"]
    data = results["data"]
    baseline = results["baseline"]

    # Order splits: nl_test first, then hh_rlhf
    split_order = [SPLIT_NL_TEST, SPLIT_HH_RLHF]
    splits_to_plot = [s for s in split_order if s in splits]

    n_splits = len(splits_to_plot)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5), sharey=True)

    if n_splits == 1:
        axes = [axes]

    for ax_idx, split_name in enumerate(splits_to_plot):
        ax = axes[ax_idx]
        split_data = data[split_name]
        split_baseline = baseline.get(split_name)
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)

        for sv_type in sv_types:
            sv_data = split_data.get(sv_type, {})
            if not sv_data:
                continue

            # Collect all alphas, always including 0.0 if baseline exists
            # Filter to only include positive alphas up to 0.1
            alphas = set(a for a in sv_data.keys() if 0.0 <= a <= 0.1)
            if split_baseline is not None:
                alphas.add(0.0)
            alphas = sorted(alphas)

            spanish_scores = []
            for alpha in alphas:
                if alpha == 0.0 and split_baseline is not None:
                    spanish_scores.append(split_baseline)
                else:
                    spanish_scores.append(sv_data.get(alpha))

            # Filter out None values
            valid = [(a, s) for a, s in zip(alphas, spanish_scores) if s is not None]

            if valid:
                color = COLORS.get(sv_type, "gray")
                marker = MARKERS.get(sv_type, "o")
                label = LABELS.get(sv_type, sv_type)

                ax.plot(
                    [x[0] for x in valid],
                    [x[1] for x in valid],
                    f"{marker}-",
                    linewidth=2,
                    markersize=7,
                    color=color,
                    label=label,
                )

        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Strength (α)", fontsize=10)
        ax.set_title(split_display, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8, loc="best", title="Prompt Used to Create\nSteering Vector", title_fontsize=8)

    axes[0].set_ylabel("Spanish Score (0-100)", fontsize=10)

    fig.suptitle("Spanish Steering (Layers: all)", fontsize=12)
    plt.tight_layout()

    output_path = f"{output_dir}/combined_spanish_steering.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge and replot Spanish steering evaluation results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/spanish_eval",
        help="Base directory containing result subdirectories",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output combined JSON path (default: {results-dir}/combined_results.json)",
    )

    args = parser.parse_args()

    # Discover result directories
    result_dirs = discover_result_dirs(args.results_dir)
    if not result_dirs:
        print(f"No result directories found in {args.results_dir}")
        return

    print(f"Found {len(result_dirs)} result directories:")
    for d in result_dirs:
        print(f"  - {d.name}")

    # Collect all results
    results = collect_all_results(result_dirs)

    print(f"\nCollected data:")
    print(f"  Splits: {results['splits']}")
    print(f"  SV types: {results['sv_types']}")
    print(f"  Alphas: {results['alphas']}")

    # Save combined JSON
    output_json = args.output_json or f"{args.results_dir}/combined_results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved combined results: {output_json}")

    # Plot
    plot_split_comparison(results, args.results_dir)


if __name__ == "__main__":
    main()
