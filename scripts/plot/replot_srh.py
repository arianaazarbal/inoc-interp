#!/usr/bin/env python3
"""Regenerate plots for srh_eval folders (handles both old and new formats)."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

# Style definitions from eval_srh.py
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
    "default": "hack - control (default prompt)",
    "overfit": "hack - control (hack prompt)",
    "dont_overfit": "hack - control (dont_hack prompt)",
    "inoculated_hacks": "hack prompt - dont_hack prompt (hacks)",
    "inoculated_control": "hack prompt - dont_hack prompt (non_hacks)",
}
SPLIT_DISPLAY_NAMES = {
    "nl_test": "Natural Language (In-Dist)",
    "ood": "Code/Functions (OOD)",
}


def get_layers_display(steering_layers: list[int]) -> str:
    if steering_layers == list(range(36)):
        return "all"
    elif len(steering_layers) <= 5:
        return ", ".join(map(str, steering_layers))
    else:
        return f"{steering_layers[0]}-{steering_layers[-1]}"


def get_layers_filename(steering_layers: list[int]) -> str:
    if steering_layers == list(range(36)):
        return "all"
    else:
        return "_".join(map(str, steering_layers))


def plot_results(all_results: dict, steering_layers: list[int], output_dir: str, split_name: str = None):
    """Generate plots from results dict."""
    layers_str = get_layers_filename(steering_layers)
    layers_display = get_layers_display(steering_layers)

    split_display = ""
    if split_name:
        split_display = f" - {SPLIT_DISPLAY_NAMES.get(split_name, split_name)}"

    # Plot 1: Hack Score and Coherence Score vs Alpha
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for sv_type, results in all_results.items():
        alphas = sorted([float(a) for a in results["by_alpha"].keys()])
        hack_scores = [results["by_alpha"][str(a)].get("hack_score") for a in alphas]
        coherence_scores = [results["by_alpha"][str(a)].get("coherence_score") for a in alphas]

        valid_hack = [(a, s) for a, s in zip(alphas, hack_scores) if s is not None]
        valid_coherence = [(a, s) for a, s in zip(alphas, coherence_scores) if s is not None]

        color = COLORS.get(sv_type, "gray")
        marker = MARKERS.get(sv_type, "o")
        label = LABELS.get(sv_type, sv_type)

        if valid_hack:
            axes[0].plot(
                [x[0] for x in valid_hack],
                [x[1] for x in valid_hack],
                f"{marker}-",
                linewidth=1.5,
                markersize=6,
                color=color,
                label=label,
            )

        if valid_coherence:
            axes[1].plot(
                [x[0] for x in valid_coherence],
                [x[1] for x in valid_coherence],
                f"{marker}-",
                linewidth=1.5,
                markersize=6,
                color=color,
                label=label,
            )

    axes[0].axhline(y=5, color="gray", linestyle="--", alpha=0.5)
    axes[0].axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Steering Strength (Alpha)", fontsize=10)
    axes[0].set_ylabel("Hack Score (0-10)", fontsize=10)
    axes[0].set_title(f"Reward Hacking vs Steering{split_display}\n(Layers: {layers_display})", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    axes[1].axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Steering Strength (Alpha)", fontsize=10)
    axes[1].set_ylabel("Coherence Score (0-100)", fontsize=10)
    axes[1].set_title(f"Response Coherence vs Steering{split_display}\n(Layers: {layers_display})", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_layers_{layers_str}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/comparison_layers_{layers_str}.png")

    # Plot 2: Trade-off plot
    fig, ax = plt.subplots(figsize=(7, 5))

    for sv_type, results in all_results.items():
        alphas = sorted([float(a) for a in results["by_alpha"].keys()])
        hack_scores = [results["by_alpha"][str(a)].get("hack_score") for a in alphas]
        coherence_scores = [results["by_alpha"][str(a)].get("coherence_score") for a in alphas]

        valid = [
            (a, h, c)
            for a, h, c in zip(alphas, hack_scores, coherence_scores)
            if h is not None and c is not None
        ]

        if valid:
            color = COLORS.get(sv_type, "gray")
            marker = MARKERS.get(sv_type, "o")
            label = LABELS.get(sv_type, sv_type)

            ax.plot(
                [x[1] for x in valid],
                [x[2] for x in valid],
                f"{marker}-",
                linewidth=1.5,
                markersize=7,
                color=color,
                label=label,
            )

            for a, h, c in valid:
                ax.annotate(
                    f"α={a}",
                    (h, c),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                    alpha=0.7,
                )

    ax.set_xlabel("Hack Score (0-10) - Lower is better", fontsize=10)
    ax.set_ylabel("Coherence Score (0-100) - Higher is better", fontsize=10)
    ax.set_title(f"Trade-off: Hacking vs Coherence{split_display}\n(Layers: {layers_display})", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/tradeoff_layers_{layers_str}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/tradeoff_layers_{layers_str}.png")

    # Plot 3: Filtered hack scores (coherent only) - skip if no hack_filtered data
    fig, ax = plt.subplots(figsize=(6, 4))

    for sv_type, results in all_results.items():
        alphas = sorted([float(a) for a in results["by_alpha"].keys()])
        # For filtered plot, just use hack_score where coherence > 80
        hack_scores = []
        coherence_scores = []
        for a in alphas:
            h = results["by_alpha"][str(a)].get("hack_score")
            c = results["by_alpha"][str(a)].get("coherence_score")
            if h is not None and c is not None and c > 80:
                hack_scores.append((a, h))

        if hack_scores:
            color = COLORS.get(sv_type, "gray")
            marker = MARKERS.get(sv_type, "o")
            label = LABELS.get(sv_type, sv_type)

            ax.plot(
                [x[0] for x in hack_scores],
                [x[1] for x in hack_scores],
                f"{marker}-",
                linewidth=1.5,
                markersize=6,
                color=color,
                label=label,
            )

    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Strength (Alpha)", fontsize=10)
    ax.set_ylabel("Hack Score (0-10)", fontsize=10)
    ax.set_title(f"Hack Score (Coherence > 80){split_display}\n(Layers: {layers_display})", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/hack_filtered_coherent_layers_{layers_str}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/hack_filtered_coherent_layers_{layers_str}.png")


def process_folder(folder_path: str):
    """Process a single results folder."""
    config_path = f"{folder_path}/eval_config.json"
    results_path = f"{folder_path}/all_results.json"

    if not os.path.exists(config_path):
        print(f"  Skipping {folder_path}: no eval_config.json")
        return False

    if not os.path.exists(results_path):
        print(f"  Skipping {folder_path}: no all_results.json")
        return False

    with open(config_path, "r") as f:
        config = json.load(f)

    with open(results_path, "r") as f:
        all_results = json.load(f)

    steering_layers = config.get("steering_layers", list(range(36)))

    # Check if new format (with splits) or old format
    if "splits" in config:
        # New format: all_results is {split_name: {sv_type: results}}
        for split_name, split_results in all_results.items():
            split_dir = f"{folder_path}/{split_name}"
            os.makedirs(split_dir, exist_ok=True)
            plot_results(split_results, steering_layers, split_dir, split_name)
    else:
        # Old format: all_results is {sv_type: results}
        plot_results(all_results, steering_layers, folder_path)

    return True


def main():
    parser = argparse.ArgumentParser(description="Regenerate plots for srh_eval folders")
    parser.add_argument("folders", nargs="*", help="Specific folders to process (default: all in data/srh_eval)")
    args = parser.parse_args()

    if args.folders:
        folders = args.folders
    else:
        base_dir = "data/srh_eval"
        folders = sorted([f"{base_dir}/{d}" for d in os.listdir(base_dir) if os.path.isdir(f"{base_dir}/{d}")])

    print(f"Processing {len(folders)} folders...")

    success = 0
    for folder in folders:
        print(f"\n=== {folder} ===")
        if process_folder(folder):
            success += 1

    print(f"\n\nDone! Processed {success}/{len(folders)} folders successfully.")


if __name__ == "__main__":
    main()
