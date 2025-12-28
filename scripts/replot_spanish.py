#!/usr/bin/env python3
"""
Merge and replot Spanish steering evaluation results from multiple result directories.

Usage:
    python scripts/replot_spanish.py --results-dir results/spanish_eval
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Split names
SPLIT_NL_TEST = "nl_test"
SPLIT_HH_RLHF = "hh_rlhf"

# Styling
COLORS = {
    # Default - blue
    "spanish": "tab:blue",
    # Spanish prompts - shades of green
    "spanish_can_spanish": "forestgreen",
    "spanish_respond_in_spanish": "limegreen",
    "spanish_respond_in_spanish_v2": "darkgreen",
    # German prompts - shades of yellow/gold
    "spanish_can_german": "goldenrod",
    "spanish_respond_in_german": "gold",
    "spanish_respond_in_german_v2": "darkgoldenrod",
    # English prompts - shades of red
    "spanish_can_english": "firebrick",
    "spanish_respond_in_english": "salmon",
    "spanish_respond_in_english_v2": "darkred",
}
MARKERS = {
    "spanish": "o",
    "spanish_can_spanish": "s",
    "spanish_can_english": "^",
    "spanish_can_german": "p",
    "spanish_respond_in_spanish": "D",
    "spanish_respond_in_english": "v",
    "spanish_respond_in_german": "P",
    "spanish_respond_in_spanish_v2": "d",
    "spanish_respond_in_english_v2": "<",
    "spanish_respond_in_german_v2": ">",
}
LABELS = {
    "spanish": "No modification",
    "spanish_can_spanish": '"You can respond in Spanish."',
    "spanish_can_english": '"You can respond in English."',
    "spanish_can_german": '"You can respond in German."',
    "spanish_respond_in_spanish": '"Respond in Spanish."',
    "spanish_respond_in_english": '"Respond in English."',
    "spanish_respond_in_german": '"Respond in German."',
    "spanish_respond_in_spanish_v2": '"Respond in Spanish." (v2)',
    "spanish_respond_in_english_v2": '"Respond in English." (v2)',
    "spanish_respond_in_german_v2": '"Respond in German." (v2)',
}

# Labels for v2 bar plot (showing activation sources)
LABELS_V2 = {
    "spanish": "Spanish (no mod.) - English (no mod.)",
    "spanish_respond_in_spanish_v2": 'Spanish ("Respond in Spanish.") - English (no mod.)',
    "spanish_respond_in_english_v2": 'Spanish ("Respond in English.") - English (no mod.)',
    "spanish_respond_in_german_v2": 'Spanish ("Respond in German.") - English (no mod.)',
}
SPLIT_DISPLAY_NAMES = {
    SPLIT_NL_TEST: "SORH Test (In-distribution)",
    SPLIT_HH_RLHF: "Anthropic HH-RLHF (Out-of-distribution)",
}

# Order for bar plots - group by language
SV_TYPE_ORDER = [
    "spanish",  # default - blue
    "spanish_can_spanish",  # green
    "spanish_respond_in_spanish",  # green
    "spanish_can_german",  # yellow
    "spanish_respond_in_german",  # yellow
    "spanish_can_english",  # red
    "spanish_respond_in_english",  # red
]

# Order for v2 bar plot
SV_TYPE_ORDER_V2 = [
    "spanish",  # default - blue
    "spanish_respond_in_spanish_v2",  # green
    "spanish_respond_in_german_v2",  # yellow
    "spanish_respond_in_english_v2",  # red
]


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


def normalize_sv_type(sv_type: str) -> str:
    """Strip seed suffix from sv_type to get base name.

    E.g., 'spanish_seed1' -> 'spanish'
          'spanish_can_spanish_seed5' -> 'spanish_can_spanish'
    """
    return re.sub(r"_seed\d+$", "", sv_type)


def collect_results_for_seed(result_dirs: list[Path], seed: int) -> dict:
    """Collect and merge results from directories for a specific seed.

    Normalizes sv_type names by stripping seed suffix.
    """
    all_sv_types = set()
    all_alphas = set()
    all_splits = set()

    # Filter to only directories for this seed
    seed_dirs = []
    for result_dir in result_dirs:
        config_path = result_dir / "eval_config.json"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            config = json.load(f)
        if config.get("sv_seed") == seed:
            seed_dirs.append(result_dir)

    if not seed_dirs:
        return None

    # First pass: discover all configurations
    for result_dir in seed_dirs:
        config_path = result_dir / "eval_config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Normalize sv_types
        for sv_type in config.get("sv_types", []):
            all_sv_types.add(normalize_sv_type(sv_type))
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
    for result_dir in seed_dirs:
        all_results_path = result_dir / "all_results.json"
        if not all_results_path.exists():
            continue

        with open(all_results_path) as f:
            all_results = json.load(f)

        for split_name, split_results in all_results.items():
            if split_name not in data:
                continue

            for sv_type_raw, sv_results in split_results.items():
                sv_type = normalize_sv_type(sv_type_raw)
                if sv_type not in data[split_name]:
                    continue

                by_alpha = sv_results.get("by_alpha", {})
                for alpha_str, alpha_data in by_alpha.items():
                    alpha = float(alpha_str)

                    if alpha in data[split_name][sv_type]:
                        continue

                    spanish_score = alpha_data.get("spanish_score")
                    if spanish_score is not None:
                        data[split_name][sv_type][alpha] = spanish_score

                        if alpha == 0.0 and baseline[split_name] is None:
                            baseline[split_name] = spanish_score

    return {
        "seed": seed,
        "splits": all_splits,
        "sv_types": all_sv_types,
        "alphas": all_alphas,
        "data": data,
        "baseline": baseline,
    }


def compute_mean_se_across_seeds(seed_results: list[dict]) -> dict:
    """Compute mean and SE across multiple seed results.

    Returns:
        Dict with structure: {
            "splits": [...],
            "sv_types": [...],
            "alphas": [...],
            "data": {
                split_name: {
                    sv_type: {
                        alpha: {"mean": float, "se": float, "values": [...]}
                    }
                }
            },
            "baseline": {
                split_name: {"mean": float, "se": float, "values": [...]}
            }
        }
    """
    # Collect all splits, sv_types, alphas across seeds
    all_splits = set()
    all_sv_types = set()
    all_alphas = set()

    for sr in seed_results:
        all_splits.update(sr["splits"])
        all_sv_types.update(sr["sv_types"])
        all_alphas.update(sr["alphas"])

    all_splits = sorted(all_splits)
    all_sv_types = sorted(all_sv_types)
    all_alphas = sorted(all_alphas)

    # Compute mean and SE for each (split, sv_type, alpha)
    data = {split: {sv_type: {} for sv_type in all_sv_types} for split in all_splits}
    baseline = {split: {"mean": None, "se": None, "values": []} for split in all_splits}

    for split in all_splits:
        # Collect baseline values
        baseline_values = []
        for sr in seed_results:
            if split in sr.get("baseline", {}) and sr["baseline"].get(split) is not None:
                baseline_values.append(sr["baseline"][split])

        if baseline_values:
            baseline[split] = {
                "mean": np.mean(baseline_values),
                "se": np.std(baseline_values, ddof=1) / np.sqrt(len(baseline_values)) if len(baseline_values) > 1 else 0,
                "values": baseline_values,
            }

        for sv_type in all_sv_types:
            for alpha in all_alphas:
                values = []
                for sr in seed_results:
                    if split not in sr.get("data", {}):
                        continue
                    if sv_type not in sr["data"][split]:
                        continue
                    if alpha in sr["data"][split][sv_type]:
                        values.append(sr["data"][split][sv_type][alpha])

                if values:
                    data[split][sv_type][alpha] = {
                        "mean": np.mean(values),
                        "se": np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0,
                        "values": values,
                    }

    return {
        "splits": all_splits,
        "sv_types": all_sv_types,
        "alphas": all_alphas,
        "data": data,
        "baseline": baseline,
        "n_seeds": len(seed_results),
    }


def plot_split_comparison_with_se(
    results: dict,
    output_dir: str,
    filename_suffix: str = "_mean_se",
):
    """Generate comparison plot with mean and SE error bars."""
    splits = results["splits"]
    sv_types = results["sv_types"]
    data = results["data"]
    baseline = results["baseline"]
    n_seeds = results.get("n_seeds", 1)

    split_order = [SPLIT_NL_TEST, SPLIT_HH_RLHF]
    splits_to_plot = [s for s in split_order if s in splits]

    n_splits = len(splits_to_plot)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5), sharey=True)

    if n_splits == 1:
        axes = [axes]

    for ax_idx, split_name in enumerate(splits_to_plot):
        ax = axes[ax_idx]
        split_data = data[split_name]
        split_baseline = baseline.get(split_name, {})
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)

        for sv_type in SV_TYPE_ORDER:
            if sv_type not in sv_types:
                continue
            sv_data = split_data.get(sv_type, {})
            if not sv_data:
                continue

            # Filter alphas
            alphas = sorted([a for a in sv_data.keys() if -0.05 <= a <= 0.1])

            # Add 0.0 if baseline exists
            if split_baseline.get("mean") is not None and 0.0 not in alphas:
                alphas = sorted(set(alphas) | {0.0})

            means = []
            ses = []
            valid_alphas = []

            for alpha in alphas:
                if alpha == 0.0 and split_baseline.get("mean") is not None:
                    means.append(split_baseline["mean"])
                    ses.append(split_baseline["se"])
                    valid_alphas.append(alpha)
                elif alpha in sv_data:
                    means.append(sv_data[alpha]["mean"])
                    ses.append(sv_data[alpha]["se"])
                    valid_alphas.append(alpha)

            if valid_alphas:
                color = COLORS.get(sv_type, "gray")
                marker = MARKERS.get(sv_type, "o")
                label = LABELS.get(sv_type, sv_type)

                ax.errorbar(
                    valid_alphas,
                    means,
                    yerr=ses,
                    fmt=f"{marker}-",
                    linewidth=2,
                    markersize=7,
                    color=color,
                    label=label,
                    capsize=3,
                )

        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Strength (α)", fontsize=10)
        ax.set_title(split_display, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8, loc="best", title="Context Used to\nExtract Activations", title_fontsize=8)

    axes[0].set_ylabel("Spanish Score (0-100)", fontsize=10)

    fig.suptitle(f"Spanish Steering (Layers: all, n={n_seeds} seeds, mean ± SE)", fontsize=12)
    plt.tight_layout()

    output_path = f"{output_dir}/combined_spanish_steering{filename_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_bar_with_se(
    results: dict,
    output_dir: str,
    alpha: float,
    filename_suffix: str = "_mean_se",
):
    """Generate bar plot with SE error bars at a specific alpha."""
    splits = results["splits"]
    sv_types = results["sv_types"]
    data = results["data"]
    baseline = results["baseline"]
    n_seeds = results.get("n_seeds", 1)

    split_order = [SPLIT_NL_TEST, SPLIT_HH_RLHF]
    splits_to_plot = [s for s in split_order if s in splits]

    n_splits = len(splits_to_plot)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5))

    if n_splits == 1:
        axes = [axes]

    for ax_idx, split_name in enumerate(splits_to_plot):
        ax = axes[ax_idx]
        split_data = data[split_name]
        split_baseline = baseline.get(split_name, {})
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)

        sv_names = []
        means = []
        ses = []
        colors = []

        for sv_type in SV_TYPE_ORDER:
            if sv_type not in sv_types:
                continue
            sv_data = split_data.get(sv_type, {})

            if alpha == 0.0 and split_baseline.get("mean") is not None:
                mean = split_baseline["mean"]
                se = split_baseline["se"]
            elif alpha in sv_data:
                mean = sv_data[alpha]["mean"]
                se = sv_data[alpha]["se"]
            else:
                continue

            sv_names.append(LABELS.get(sv_type, sv_type))
            means.append(mean)
            ses.append(se)
            colors.append(COLORS.get(sv_type, "gray"))

        if not means:
            continue

        x = range(len(sv_names))
        bars = ax.bar(x, means, yerr=ses, color=colors, edgecolor="black",
                      linewidth=0.5, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(sv_names, rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Context Used to Extract Activations", fontsize=10)
        ax.set_ylabel("Spanish Score (0-100)" if ax_idx == 0 else "", fontsize=10)
        ax.set_title(split_display, fontsize=11)

        y_max = (max(means) + max(ses)) * 1.15 if means else 100
        ax.set_ylim(0, y_max)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, mean, se in zip(bars, means, ses):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + se + y_max * 0.02,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(f"Spanish Steering at α = {alpha} (n={n_seeds} seeds, mean ± SE)", fontsize=12)
    plt.tight_layout()

    output_path = f"{output_dir}/bar_alpha_{alpha}{filename_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_bar_comparison_with_se(
    results: dict,
    output_dir: str,
    alpha: float,
    sv_types_to_plot: list[str],
    filename_suffix: str = "",
):
    """Generate bar plot comparing specific sv_types with SE error bars."""
    splits = results["splits"]
    sv_types = results["sv_types"]
    data = results["data"]
    baseline = results["baseline"]
    n_seeds = results.get("n_seeds", 1)

    split_order = [SPLIT_NL_TEST, SPLIT_HH_RLHF]
    splits_to_plot = [s for s in split_order if s in splits]

    n_splits = len(splits_to_plot)
    fig, axes = plt.subplots(1, n_splits, figsize=(5 * n_splits, 5))

    if n_splits == 1:
        axes = [axes]

    for ax_idx, split_name in enumerate(splits_to_plot):
        ax = axes[ax_idx]
        split_data = data[split_name]
        split_baseline = baseline.get(split_name, {})
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)

        sv_names = []
        means = []
        ses = []
        colors = []

        for sv_type in sv_types_to_plot:
            if sv_type not in sv_types:
                continue
            sv_data = split_data.get(sv_type, {})

            if alpha == 0.0 and split_baseline.get("mean") is not None:
                mean = split_baseline["mean"]
                se = split_baseline["se"]
            elif alpha in sv_data:
                mean = sv_data[alpha]["mean"]
                se = sv_data[alpha]["se"]
            else:
                continue

            sv_names.append(LABELS.get(sv_type, sv_type))
            means.append(mean)
            ses.append(se)
            colors.append(COLORS.get(sv_type, "gray"))

        if not means:
            continue

        x = range(len(sv_names))
        bars = ax.bar(x, means, yerr=ses, color=colors, edgecolor="black",
                      linewidth=0.5, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(sv_names, rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Context Used to Extract Activations", fontsize=10)
        ax.set_ylabel("Spanish Score (0-100)" if ax_idx == 0 else "", fontsize=10)
        ax.set_title(split_display, fontsize=11)

        y_max = (max(means) + max(ses)) * 1.15 if means else 100
        ax.set_ylim(0, y_max)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, mean, se in zip(bars, means, ses):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + se + y_max * 0.02,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(f"Spanish Steering at α = {alpha} (n={n_seeds} seeds, mean ± SE)", fontsize=12)
    plt.tight_layout()

    output_path = f"{output_dir}/bar_alpha_{alpha}{filename_suffix}_mean_se.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_bar_v2_with_se(
    results: dict,
    output_dir: str,
    alpha: float,
    filename_suffix: str = "_mean_se",
):
    """Generate v2 bar plot (cross-prompt) with SE error bars at a specific alpha."""
    splits = results["splits"]
    sv_types = results["sv_types"]
    data = results["data"]
    baseline = results["baseline"]
    n_seeds = results.get("n_seeds", 1)

    split_order = [SPLIT_NL_TEST, SPLIT_HH_RLHF]
    splits_to_plot = [s for s in split_order if s in splits]

    n_splits = len(splits_to_plot)
    fig, axes = plt.subplots(1, n_splits, figsize=(7 * n_splits, 6))

    if n_splits == 1:
        axes = [axes]

    has_data = False
    for ax_idx, split_name in enumerate(splits_to_plot):
        ax = axes[ax_idx]
        split_data = data[split_name]
        split_baseline = baseline.get(split_name, {})
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)

        sv_names = []
        means = []
        ses = []
        colors = []

        for sv_type in SV_TYPE_ORDER_V2:
            if sv_type not in sv_types:
                continue
            sv_data = split_data.get(sv_type, {})

            if alpha == 0.0 and split_baseline.get("mean") is not None:
                mean = split_baseline["mean"]
                se = split_baseline["se"]
            elif alpha in sv_data:
                mean = sv_data[alpha]["mean"]
                se = sv_data[alpha]["se"]
            else:
                continue

            sv_names.append(LABELS_V2.get(sv_type, sv_type))
            means.append(mean)
            ses.append(se)
            colors.append(COLORS.get(sv_type, "gray"))

        if not means:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(split_display, fontsize=11)
            continue

        has_data = True
        x = range(len(sv_names))
        bars = ax.bar(x, means, yerr=ses, color=colors, edgecolor="black",
                      linewidth=0.5, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(sv_names, rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Steering Type", fontsize=10)
        ax.set_ylabel("Spanish Score (0-100)" if ax_idx == 0 else "", fontsize=10)
        ax.set_title(split_display, fontsize=11)

        y_max = (max(means) + max(ses)) * 1.15 if means else 100
        ax.set_ylim(0, y_max)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, mean, se in zip(bars, means, ses):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + se + y_max * 0.02,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(f"Prompt-Diff Spanish Steering at α = {alpha} (n={n_seeds} seeds, mean ± SE)", fontsize=12)
    plt.tight_layout()

    output_path = f"{output_dir}/bar_v2_alpha_{alpha}{filename_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    if not has_data:
        print(f"  Warning: No v2 data found for any split at alpha={alpha}")


def plot_difference_with_se(
    seed_results: list[dict],
    output_dir: str,
    alpha: float,
    sv_type_a: str,
    sv_type_b: str,
    filename_suffix: str = "",
):
    """Generate bar plot showing difference (A - B) with SE computed from paired differences."""
    split_order = [SPLIT_NL_TEST, SPLIT_HH_RLHF]

    # Collect all splits
    all_splits = set()
    for sr in seed_results:
        all_splits.update(sr["splits"])
    splits_to_plot = [s for s in split_order if s in all_splits]

    n_splits = len(splits_to_plot)
    fig, axes = plt.subplots(1, n_splits, figsize=(4 * n_splits, 5))

    if n_splits == 1:
        axes = [axes]

    for ax_idx, split_name in enumerate(splits_to_plot):
        ax = axes[ax_idx]
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)

        # Compute paired differences for each seed
        differences = []
        for sr in seed_results:
            if split_name not in sr.get("data", {}):
                continue

            # Get value for sv_type_a
            val_a = None
            if sv_type_a in sr["data"][split_name] and alpha in sr["data"][split_name][sv_type_a]:
                val_a = sr["data"][split_name][sv_type_a][alpha]

            # Get value for sv_type_b
            val_b = None
            if sv_type_b in sr["data"][split_name] and alpha in sr["data"][split_name][sv_type_b]:
                val_b = sr["data"][split_name][sv_type_b][alpha]

            if val_a is not None and val_b is not None:
                differences.append(val_a - val_b)

        if not differences:
            continue

        mean_diff = np.mean(differences)
        se_diff = np.std(differences, ddof=1) / np.sqrt(len(differences)) if len(differences) > 1 else 0

        # Create single bar
        bar = ax.bar([0], [mean_diff], yerr=[se_diff], color="tab:purple",
                     edgecolor="black", linewidth=0.5, capsize=5, width=0.5)

        ax.set_xticks([0])
        ax.set_xticklabels(["Difference"], fontsize=10)
        ax.set_ylabel("Δ Spanish Score" if ax_idx == 0 else "", fontsize=10)
        ax.set_title(split_display, fontsize=11)

        # Dynamic y-axis centered on 0
        y_extent = abs(mean_diff) + se_diff + 5
        ax.set_ylim(-y_extent * 0.2, y_extent * 1.2)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value label
        y_pos = mean_diff + se_diff + y_extent * 0.05 if mean_diff >= 0 else mean_diff - se_diff - y_extent * 0.05
        va = "bottom" if mean_diff >= 0 else "top"
        ax.text(0, y_pos, f"{mean_diff:.1f} ± {se_diff:.1f}", ha="center", va=va, fontsize=10, fontweight="bold")

    label_a = LABELS.get(sv_type_a, sv_type_a)
    label_b = LABELS.get(sv_type_b, sv_type_b)
    fig.suptitle(f"Difference at α = {alpha}: {label_a} − {label_b}\n(n={len(seed_results)} seeds, mean ± SE)", fontsize=11)
    plt.tight_layout()

    output_path = f"{output_dir}/bar_diff_alpha_{alpha}{filename_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


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

        for sv_type in SV_TYPE_ORDER:
            if sv_type not in sv_types:
                continue
            sv_data = split_data.get(sv_type, {})
            if not sv_data:
                continue

            # Collect all alphas, always including 0.0 if baseline exists
            # Filter to only include alphas from -0.05 to 0.1
            alphas = set(a for a in sv_data.keys() if -0.05 <= a <= 0.1)
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
        ax.legend(fontsize=8, loc="best", title="Context Used to\nExtract Activations", title_fontsize=8)

    axes[0].set_ylabel("Spanish Score (0-100)", fontsize=10)

    fig.suptitle("Spanish Steering (Layers: all)", fontsize=12)
    plt.tight_layout()

    output_path = f"{output_dir}/combined_spanish_steering.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_bar_at_alpha(
    results: dict,
    output_dir: str,
    alpha: float,
):
    """Generate bar plot showing Spanish scores at a specific alpha."""
    splits = results["splits"]
    sv_types = results["sv_types"]
    data = results["data"]
    baseline = results["baseline"]

    # Order splits
    split_order = [SPLIT_NL_TEST, SPLIT_HH_RLHF]
    splits_to_plot = [s for s in split_order if s in splits]

    n_splits = len(splits_to_plot)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5))

    if n_splits == 1:
        axes = [axes]

    for ax_idx, split_name in enumerate(splits_to_plot):
        ax = axes[ax_idx]
        split_data = data[split_name]
        split_baseline = baseline.get(split_name)
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)

        # Collect scores for each sv_type at this alpha, in specified order
        sv_names = []
        scores = []
        colors = []

        # Use ordered list, filtering to only those with data
        for sv_type in SV_TYPE_ORDER:
            if sv_type not in sv_types:
                continue
            sv_data = split_data.get(sv_type, {})

            # Get score at this alpha, or baseline if alpha is 0.0
            if alpha == 0.0 and split_baseline is not None:
                score = split_baseline
            else:
                score = sv_data.get(alpha)

            if score is not None:
                sv_names.append(LABELS.get(sv_type, sv_type))
                scores.append(score)
                colors.append(COLORS.get(sv_type, "gray"))

        if not scores:
            continue

        # Create bar plot
        x = range(len(sv_names))
        bars = ax.bar(x, scores, color=colors, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(sv_names, rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Context Used to Extract Activations", fontsize=10)
        ax.set_ylabel("Spanish Score (0-100)" if ax_idx == 0 else "", fontsize=10)
        ax.set_title(split_display, fontsize=11)

        # Dynamic y-axis
        y_max = max(scores) * 1.15 if scores else 100
        ax.set_ylim(0, y_max)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                f"{score:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(f"Spanish Steering at α = {alpha} (Layers: all)", fontsize=12)
    plt.tight_layout()

    output_path = f"{output_dir}/bar_alpha_{alpha}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_bar_v2_at_alpha(
    results: dict,
    output_dir: str,
    alpha: float,
):
    """Generate bar plot for v2 variants (cross-prompt) at a specific alpha."""
    splits = results["splits"]
    sv_types = results["sv_types"]
    data = results["data"]
    baseline = results["baseline"]

    # Order splits
    split_order = [SPLIT_NL_TEST, SPLIT_HH_RLHF]
    splits_to_plot = [s for s in split_order if s in splits]

    n_splits = len(splits_to_plot)
    fig, axes = plt.subplots(1, n_splits, figsize=(7 * n_splits, 6))

    if n_splits == 1:
        axes = [axes]

    for ax_idx, split_name in enumerate(splits_to_plot):
        ax = axes[ax_idx]
        split_data = data[split_name]
        split_baseline = baseline.get(split_name)
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)

        # Collect scores for each sv_type at this alpha, in specified order
        sv_names = []
        scores = []
        colors = []

        # Use v2 ordered list
        for sv_type in SV_TYPE_ORDER_V2:
            if sv_type not in sv_types:
                continue
            sv_data = split_data.get(sv_type, {})

            # Get score at this alpha, or baseline if alpha is 0.0
            if alpha == 0.0 and split_baseline is not None:
                score = split_baseline
            else:
                score = sv_data.get(alpha)

            if score is not None:
                sv_names.append(LABELS_V2.get(sv_type, sv_type))
                scores.append(score)
                colors.append(COLORS.get(sv_type, "gray"))

        if not scores:
            continue

        # Create bar plot
        x = range(len(sv_names))
        bars = ax.bar(x, scores, color=colors, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(sv_names, rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Steering Type", fontsize=10)
        ax.set_ylabel("Spanish Score (0-100)" if ax_idx == 0 else "", fontsize=10)
        ax.set_title(split_display, fontsize=11)

        # Dynamic y-axis
        y_max = max(scores) * 1.15 if scores else 100
        ax.set_ylim(0, y_max)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                f"{score:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(f"Cross-Prompt Spanish Steering at α = {alpha} (Layers: all)", fontsize=12)
    plt.tight_layout()

    output_path = f"{output_dir}/bar_v2_alpha_{alpha}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_bar_comparison(
    results: dict,
    output_dir: str,
    alpha: float,
    sv_types_to_plot: list[str],
    filename_suffix: str = "",
):
    """Generate bar plot comparing specific sv_types at a specific alpha."""
    splits = results["splits"]
    sv_types = results["sv_types"]
    data = results["data"]
    baseline = results["baseline"]

    # Order splits
    split_order = [SPLIT_NL_TEST, SPLIT_HH_RLHF]
    splits_to_plot = [s for s in split_order if s in splits]

    n_splits = len(splits_to_plot)
    fig, axes = plt.subplots(1, n_splits, figsize=(5 * n_splits, 5))

    if n_splits == 1:
        axes = [axes]

    for ax_idx, split_name in enumerate(splits_to_plot):
        ax = axes[ax_idx]
        split_data = data[split_name]
        split_baseline = baseline.get(split_name)
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)

        sv_names = []
        scores = []
        colors = []

        for sv_type in sv_types_to_plot:
            if sv_type not in sv_types:
                continue
            sv_data = split_data.get(sv_type, {})

            if alpha == 0.0 and split_baseline is not None:
                score = split_baseline
            else:
                score = sv_data.get(alpha)

            if score is not None:
                sv_names.append(LABELS.get(sv_type, sv_type))
                scores.append(score)
                colors.append(COLORS.get(sv_type, "gray"))

        if not scores:
            continue

        x = range(len(sv_names))
        bars = ax.bar(x, scores, color=colors, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(sv_names, rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Context Used to Extract Activations", fontsize=10)
        ax.set_ylabel("Spanish Score (0-100)" if ax_idx == 0 else "", fontsize=10)
        ax.set_title(split_display, fontsize=11)

        y_max = max(scores) * 1.15 if scores else 100
        ax.set_ylim(0, y_max)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                f"{score:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(f"Spanish Steering at α = {alpha} (Layers: all)", fontsize=12)
    plt.tight_layout()

    output_path = f"{output_dir}/bar_alpha_{alpha}{filename_suffix}.png"
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
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to aggregate (e.g., --seeds 1 5 11). If provided, generates mean/SE plots.",
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

    # If seeds are specified, do per-seed aggregation
    if args.seeds:
        print(f"\n=== Per-seed aggregation for seeds: {args.seeds} ===")
        seed_results = []

        for seed in args.seeds:
            print(f"\nCollecting results for seed {seed}...")
            sr = collect_results_for_seed(result_dirs, seed)
            if sr is None:
                print(f"  No data found for seed {seed}")
                continue

            print(f"  Found {len(sr['sv_types'])} sv_types, {len(sr['alphas'])} alphas")

            # Save per-seed JSON
            seed_json_path = f"{args.results_dir}/combined_results_seed{seed}.json"
            with open(seed_json_path, "w") as f:
                json.dump(sr, f, indent=2)
            print(f"  Saved: {seed_json_path}")

            seed_results.append(sr)

        if len(seed_results) > 0:
            print(f"\n=== Computing mean and SE across {len(seed_results)} seeds ===")
            mean_se_results = compute_mean_se_across_seeds(seed_results)

            # Save combined mean/SE JSON
            mean_se_json_path = f"{args.results_dir}/combined_results_mean_se.json"
            # Convert numpy types to native Python for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(v) for v in obj]
                return obj

            with open(mean_se_json_path, "w") as f:
                json.dump(convert_to_serializable(mean_se_results), f, indent=2)
            print(f"Saved: {mean_se_json_path}")

            # Generate plots with SE
            plot_split_comparison_with_se(mean_se_results, args.results_dir)
            plot_bar_with_se(mean_se_results, args.results_dir, 0.05)
            plot_bar_with_se(mean_se_results, args.results_dir, 0.04)

            # Comparison plot: spanish vs spanish_can_spanish with SE
            plot_bar_comparison_with_se(
                mean_se_results, args.results_dir, 0.05,
                ["spanish", "spanish_can_spanish"],
                filename_suffix="_spanish_vs_can_spanish"
            )

            # Difference plot: spanish - spanish_can_spanish with SE on the difference
            plot_difference_with_se(
                seed_results, args.results_dir, 0.05,
                "spanish", "spanish_can_spanish",
                filename_suffix="_spanish_minus_can_spanish"
            )

            # V2 bar plot (Prompt-Diff) with SE
            plot_bar_v2_with_se(mean_se_results, args.results_dir, 0.05)

        return

    # Original behavior: collect all results without seed filtering
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

    # Bar plots at specific alphas
    plot_bar_at_alpha(results, args.results_dir, 0.04)
    plot_bar_at_alpha(results, args.results_dir, 0.05)

    # V2 bar plot at alpha=0.05
    plot_bar_v2_at_alpha(results, args.results_dir, 0.05)

    # Comparison plot: spanish vs spanish_can_spanish
    plot_bar_comparison(
        results, args.results_dir, 0.05,
        ["spanish", "spanish_can_spanish"],
        filename_suffix="_spanish_vs_can_spanish"
    )


if __name__ == "__main__":
    main()
