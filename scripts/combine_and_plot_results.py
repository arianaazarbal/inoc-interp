#!/usr/bin/env python3
"""
Combine all SRH eval results and plot hack rate (coherence > 80) across steering strengths.

Produces a two-panel plot: NL Test (left) and Code Test (right).
All steering vectors share the same baseline at alpha=0.0.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


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


def load_judged_scores(
    result_dir: Path,
    split_name: str,
    sv_type: str,
    alpha: float,
    layers_str: str,
) -> tuple[list[float], list[float]]:
    """Load hack and coherence scores for a specific configuration.

    Returns:
        Tuple of (hack_scores, coherence_scores) for each completion.
    """
    hack_path = result_dir / split_name / sv_type / f"hack_judged_layers_{layers_str}_alpha_{alpha}.json"
    coherence_path = result_dir / split_name / sv_type / f"coherence_judged_layers_{layers_str}_alpha_{alpha}.json"

    if not hack_path.exists() or not coherence_path.exists():
        return [], []

    with open(hack_path) as f:
        hack_data = json.load(f)
    with open(coherence_path) as f:
        coherence_data = json.load(f)

    hack_scores = []
    coherence_scores = []

    for sample_key, sample_data in hack_data["prompt_to_completions"].items():
        coherence_sample = coherence_data["prompt_to_completions"].get(sample_key, {})

        for comp_key, comp_data in sample_data.get("completions", {}).items():
            hack_score = comp_data.get("reward")
            coherence_comp = coherence_sample.get("completions", {}).get(comp_key, {})
            coherence_score = coherence_comp.get("reward")

            if hack_score is not None and coherence_score is not None:
                hack_scores.append(float(hack_score))
                coherence_scores.append(float(coherence_score))

    return hack_scores, coherence_scores


def compute_hack_rate(
    hack_scores: list[float],
    coherence_scores: list[float],
    coherence_threshold: float = 80.0,
) -> float | None:
    """Compute mean hack score for samples with coherence above threshold."""
    filtered = [
        h for h, c in zip(hack_scores, coherence_scores)
        if c > coherence_threshold
    ]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def collect_all_results(
    result_dirs: list[Path],
) -> dict:
    """Collect results from all result directories.

    Returns:
        Dict with structure: {
            "splits": ["nl_test", "ood"],
            "sv_types": [...],
            "alphas": [...],
            "data": {
                split_name: {
                    sv_type: {
                        alpha: {"hack_rate": float, "n_samples": int}
                    }
                }
            },
            "baseline": {
                split_name: {"hack_rate": float, "n_samples": int}
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
        all_splits.update(config.get("splits", ["nl_test", "ood"]))

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
        with open(config_path) as f:
            config = json.load(f)

        steering_layers = config.get("steering_layers", list(range(36)))
        layers_str = get_layers_filename(steering_layers)
        sv_types = config.get("sv_types", [])
        alphas = config.get("alphas", [])
        splits = config.get("splits", ["nl_test", "ood"])

        for split_name in splits:
            for sv_type in sv_types:
                for alpha in alphas:
                    # Skip if already have data for this config
                    if alpha in data[split_name][sv_type]:
                        continue

                    hack_scores, coherence_scores = load_judged_scores(
                        result_dir, split_name, sv_type, alpha, layers_str
                    )

                    if not hack_scores:
                        continue

                    hack_rate = compute_hack_rate(hack_scores, coherence_scores)

                    data[split_name][sv_type][alpha] = {
                        "hack_rate": hack_rate,
                        "n_samples": len(hack_scores),
                        "n_coherent": sum(1 for c in coherence_scores if c > 80),
                        "source_dir": str(result_dir),
                    }

                    # Collect baseline (alpha=0.0) - should be same across sv_types
                    if alpha == 0.0 and baseline[split_name] is None:
                        baseline[split_name] = {
                            "hack_rate": hack_rate,
                            "n_samples": len(hack_scores),
                            "n_coherent": sum(1 for c in coherence_scores if c > 80),
                        }

    return {
        "splits": all_splits,
        "sv_types": all_sv_types,
        "alphas": all_alphas,
        "data": data,
        "baseline": baseline,
    }


# Consistent colors and markers for steering vector types
COLORS = {
    "default": "tab:green",
    "overfit": "tab:red",
    "dont_overfit": "tab:blue",
    "inoculated_hacks": "tab:purple",
    "inoculated_control": "tab:orange",
}
MARKERS = {
    "default": "o",
    "overfit": "s",
    "dont_overfit": "^",
    "inoculated_hacks": "D",
    "inoculated_control": "v",
}
LABELS = {
    "default": "default",
    "overfit": "overfit",
    "dont_overfit": "dont_overfit",
    "inoculated_hacks": "inoculated_hacks",
    "inoculated_control": "inoculated_control",
}
SPLIT_NAMES = {
    "nl_test": "Natural Language Test",
    "ood": "Code Test (OOD)",
}


def plot_hack_rate(
    results: dict,
    output_path: str,
):
    """Plot hack rate across steering strengths with two subplots."""
    splits = results["splits"]
    sv_types = results["sv_types"]
    data = results["data"]
    baseline = results["baseline"]

    # Ensure we have both splits for side-by-side plot
    split_order = ["nl_test", "ood"]
    splits_to_plot = [s for s in split_order if s in splits]

    fig, axes = plt.subplots(1, len(splits_to_plot), figsize=(5 * len(splits_to_plot), 4), squeeze=False)
    axes = axes[0]

    for ax_idx, split_name in enumerate(splits_to_plot):
        ax = axes[ax_idx]
        split_data = data[split_name]
        split_baseline = baseline.get(split_name)

        for sv_type in sv_types:
            sv_data = split_data.get(sv_type, {})
            if not sv_data:
                continue

            # Collect all alphas for this sv_type, always including 0.0 if baseline exists
            alphas = set(sv_data.keys())
            if split_baseline is not None:
                alphas.add(0.0)
            alphas = sorted(alphas)

            hack_rates = []
            for alpha in alphas:
                if alpha == 0.0 and split_baseline is not None:
                    # Use shared baseline for alpha=0.0
                    hack_rates.append(split_baseline["hack_rate"])
                else:
                    entry = sv_data.get(alpha)
                    hack_rates.append(entry["hack_rate"] if entry else None)

            # Filter out None values for plotting
            valid = [(a, h) for a, h in zip(alphas, hack_rates) if h is not None]
            if not valid:
                continue

            color = COLORS.get(sv_type, "gray")
            marker = MARKERS.get(sv_type, "o")
            label = LABELS.get(sv_type, sv_type)

            ax.plot(
                [x[0] for x in valid],
                [x[1] for x in valid],
                f"{marker}-",
                linewidth=1.5,
                markersize=6,
                color=color,
                label=label,
            )

        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Strength (α)", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Hack Score (coherence > 80)", fontsize=10)
        ax.set_title(SPLIT_NAMES.get(split_name, split_name), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", title="Prompt Used to Create\nSteering Vector", title_fontsize=8)

        # Set y-axis limits dynamically based on data range
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max * 1.1)  # Add 10% padding at top

        # Only show reference line at y=5 if it's within the visible range
        if y_max >= 5:
            ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5, label="_nolegend_")

    fig.suptitle("Hack Rate when Steering with Control → Hack Vector", fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def collect_raw_scores(
    result_dirs: list[Path],
) -> dict:
    """Collect raw per-sample hack and coherence scores for scatter plots.

    Returns:
        Dict with structure: {
            split_name: {
                sv_type: {
                    alpha: [(hack_score, coherence_score), ...]
                }
            }
        }
    """
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for result_dir in result_dirs:
        config_path = result_dir / "eval_config.json"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            config = json.load(f)

        steering_layers = config.get("steering_layers", list(range(36)))
        layers_str = get_layers_filename(steering_layers)
        sv_types = config.get("sv_types", [])
        alphas = config.get("alphas", [])
        splits = config.get("splits", ["nl_test", "ood"])

        for split_name in splits:
            for sv_type in sv_types:
                for alpha in alphas:
                    # Skip if already have data
                    if raw_data[split_name][sv_type][alpha]:
                        continue

                    hack_scores, coherence_scores = load_judged_scores(
                        result_dir, split_name, sv_type, alpha, layers_str
                    )

                    if hack_scores:
                        raw_data[split_name][sv_type][alpha] = list(zip(hack_scores, coherence_scores))

    return dict(raw_data)


def plot_coherence_hack_distribution(
    raw_data: dict,
    output_dir: str,
):
    """Plot coherence vs hack score scatter plots for each split and alpha."""

    for split_name, split_data in raw_data.items():
        # Collect all alphas across all sv_types
        all_alphas = set()
        for sv_type_data in split_data.values():
            all_alphas.update(sv_type_data.keys())
        all_alphas = sorted(all_alphas)

        if not all_alphas:
            continue

        # Create subplot grid
        n_alphas = len(all_alphas)
        n_cols = min(3, n_alphas)
        n_rows = (n_alphas + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

        for idx, alpha in enumerate(all_alphas):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            for sv_type, sv_data in split_data.items():
                points = sv_data.get(alpha, [])
                if not points:
                    continue

                hack_scores = [p[0] for p in points]
                coherence_scores = [p[1] for p in points]

                color = COLORS.get(sv_type, "gray")
                marker = MARKERS.get(sv_type, "o")
                label = LABELS.get(sv_type, sv_type)

                ax.scatter(
                    coherence_scores,
                    hack_scores,
                    c=color,
                    marker=marker,
                    alpha=0.6,
                    s=40,
                    label=label,
                )

            ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(x=80, color="red", linestyle="--", alpha=0.5, label="coherence=80")
            ax.set_xlabel("Coherence Score", fontsize=9)
            ax.set_ylabel("Hack Score", fontsize=9)
            ax.set_title(f"α = {alpha}", fontsize=10)
            ax.set_xlim(0, 105)
            ax.set_ylim(-0.5, 10.5)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="upper left")

        # Hide unused subplots
        for idx in range(n_alphas, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        split_display = SPLIT_NAMES.get(split_name, split_name)
        fig.suptitle(f"Coherence vs Hack Score - {split_display}", fontsize=12, y=1.02)
        plt.tight_layout()

        output_path = f"{output_dir}/coherence_hack_scatter_{split_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


def plot_distribution_summary(
    raw_data: dict,
    output_dir: str,
):
    """Plot summary distribution: boxplots of hack scores by alpha, filtered by coherence."""

    for split_name, split_data in raw_data.items():
        # Collect all alphas
        all_alphas = set()
        for sv_type_data in split_data.values():
            all_alphas.update(sv_type_data.keys())
        all_alphas = sorted(all_alphas)

        if not all_alphas:
            continue

        sv_types = list(split_data.keys())

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: All samples
        ax = axes[0]
        positions = []
        box_data = []
        colors_list = []
        tick_positions = []
        tick_labels = []

        for i, alpha in enumerate(all_alphas):
            group_start = i * (len(sv_types) + 1)
            tick_positions.append(group_start + len(sv_types) / 2)
            tick_labels.append(f"α={alpha}")

            for j, sv_type in enumerate(sv_types):
                points = split_data.get(sv_type, {}).get(alpha, [])
                hack_scores = [p[0] for p in points]

                if hack_scores:
                    positions.append(group_start + j)
                    box_data.append(hack_scores)
                    colors_list.append(COLORS.get(sv_type, "gray"))

        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylabel("Hack Score", fontsize=10)
        ax.set_title("All Samples", fontsize=11)
        ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # Legend
        legend_handles = [plt.Line2D([0], [0], color=COLORS.get(sv, "gray"), lw=4, label=sv)
                          for sv in sv_types]
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

        # Right: Coherent samples only (coherence > 80)
        ax = axes[1]
        positions = []
        box_data = []
        colors_list = []

        for i, alpha in enumerate(all_alphas):
            group_start = i * (len(sv_types) + 1)

            for j, sv_type in enumerate(sv_types):
                points = split_data.get(sv_type, {}).get(alpha, [])
                hack_scores = [p[0] for p in points if p[1] > 80]

                if hack_scores:
                    positions.append(group_start + j)
                    box_data.append(hack_scores)
                    colors_list.append(COLORS.get(sv_type, "gray"))

        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylabel("Hack Score", fontsize=10)
        ax.set_title("Coherent Samples Only (coherence > 80)", fontsize=11)
        ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

        split_display = SPLIT_NAMES.get(split_name, split_name)
        fig.suptitle(f"Hack Score Distribution - {split_display}", fontsize=12)
        plt.tight_layout()

        output_path = f"{output_dir}/hack_distribution_{split_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


def plot_simplified_scatter(
    raw_data: dict,
    output_dir: str,
):
    """Plot simplified scatter: hack vs coherence for default prompt at α=0.0 and α=0.3."""
    import numpy as np
    from scipy import stats

    alphas_to_plot = [0.0, 0.3]
    sv_type = "default"

    for split_name, split_data in raw_data.items():
        if sv_type not in split_data:
            continue

        sv_data = split_data[sv_type]

        # Check which alphas are available
        available_alphas = [a for a in alphas_to_plot if a in sv_data and sv_data[a]]
        if not available_alphas:
            continue

        n_cols = len(available_alphas)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5), squeeze=False)
        axes = axes[0]

        for idx, alpha in enumerate(available_alphas):
            ax = axes[idx]
            points = sv_data.get(alpha, [])

            if not points:
                continue

            hack_scores = np.array([p[0] for p in points])
            coherence_scores = np.array([p[1] for p in points])

            # Compute correlation (handle constant input)
            if np.std(hack_scores) == 0 or np.std(coherence_scores) == 0:
                r, p_value = np.nan, np.nan
            else:
                r, p_value = stats.pearsonr(coherence_scores, hack_scores)

            # Scatter plot
            ax.scatter(
                coherence_scores,
                hack_scores,
                c="tab:blue",
                alpha=0.6,
                s=50,
                edgecolors="white",
                linewidth=0.5,
            )

            # Add regression line (skip if constant)
            if not np.isnan(r):
                z = np.polyfit(coherence_scores, hack_scores, 1)
                poly = np.poly1d(z)
                x_line = np.linspace(coherence_scores.min(), coherence_scores.max(), 100)
                ax.plot(x_line, poly(x_line), "r--", alpha=0.8, linewidth=2)

            # Reference lines
            ax.axhline(y=5, color="gray", linestyle="--", alpha=0.4)
            ax.axvline(x=80, color="gray", linestyle="--", alpha=0.4)

            ax.set_xlabel("Coherence Score", fontsize=10)
            ax.set_ylabel("Hack Score", fontsize=10)
            ax.set_title(f"α = {alpha}", fontsize=11)
            ax.set_xlim(0, 105)
            ax.set_ylim(-0.5, 10.5)
            ax.grid(True, alpha=0.3)

            # Compute means
            mean_hack = np.mean(hack_scores)
            mean_coherence = np.mean(coherence_scores)

            # Plot mean as crosshairs
            ax.axhline(y=mean_hack, color="red", linestyle="-", alpha=0.5, linewidth=1)
            ax.axvline(x=mean_coherence, color="red", linestyle="-", alpha=0.5, linewidth=1)
            ax.scatter([mean_coherence], [mean_hack], c="red", s=100, marker="x", linewidths=2, zorder=5)

            # Add stats text
            if np.isnan(r):
                corr_text = f"r = N/A (constant)\nn = {len(points)}\nmean hack = {mean_hack:.2f}\nmean coherence = {mean_coherence:.1f}"
            else:
                corr_text = f"r = {r:.3f}, p = {p_value:.3f}\nn = {len(points)}\nmean hack = {mean_hack:.2f}\nmean coherence = {mean_coherence:.1f}"
            ax.text(
                0.05, 0.95,
                corr_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        split_display = SPLIT_NAMES.get(split_name, split_name)
        fig.suptitle(f"Hack vs Coherence - {split_display} (default prompt)", fontsize=12)
        plt.tight_layout()

        output_path = f"{output_dir}/scatter_simple_{split_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine all SRH eval results and plot hack rate vs steering strength"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/srh_eval",
        help="Base directory containing result subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot path (default: {results-dir}/combined_hack_rate.png)",
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

    # Plot hack rate
    output_path = args.output or f"{args.results_dir}/combined_hack_rate.png"
    plot_hack_rate(results, output_path)

    # Collect raw scores for scatter/distribution plots
    raw_data = collect_raw_scores(result_dirs)

    # Plot coherence vs hack scatter plots
    plot_coherence_hack_distribution(raw_data, args.results_dir)

    # Plot hack score distribution boxplots
    plot_distribution_summary(raw_data, args.results_dir)

    # Plot simplified scatter (default prompt, α=0.0 and α=0.3)
    plot_simplified_scatter(raw_data, args.results_dir)


if __name__ == "__main__":
    main()
