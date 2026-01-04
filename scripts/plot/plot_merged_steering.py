#!/usr/bin/env python3
"""
Generate merged plot comparing sv_any_language vs sv_none across all datasets.

Usage:
    python scripts/plot_merged_steering.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_SHORT = "gemma-2-9b-it"
BASE_OUTPUT_DIR = "results/triviaqa_steering"
LAYERS = "layers_all"
EVAL_MOD = "no_mod"

# Datasets to include
DATASETS = ["triviaqa_holdout", "hh_rlhf", "wildchat"]
DATASET_DISPLAY = {
    "triviaqa_holdout": "TriviaQA Holdout",
    "hh_rlhf": "HH-RLHF",
    "wildchat": "WildChat",
}

# Steering vectors to compare
SV_TYPES = ["sv_any_language", "sv_none"]
SV_TYPES_WITH_SPANISH = ["sv_respond_in_spanish", "sv_any_language", "sv_none"]
SV_TYPES_WITH_RESCALED = [
    "sv_respond_in_spanish", "sv_respond_in_spanish_renorm_none",
    "sv_any_language", "sv_any_language_renorm_none",
    "sv_none"
]
SV_DISPLAY = {
    "sv_any_language": "respond_in_any_language",
    "sv_none": "default",
    "sv_respond_in_spanish": "respond_in_spanish",
    "sv_respond_in_spanish_renorm_none": "respond_in_spanish (rescaled)",
    "sv_any_language_renorm_none": "respond_in_any_language (rescaled)",
}

# Colors by steering vector type
SV_COLORS = {
    "sv_any_language": "tab:blue",
    "sv_none": "tab:green",
    "sv_respond_in_spanish": "tab:pink",
    "sv_respond_in_spanish_renorm_none": "tab:pink",
    "sv_any_language_renorm_none": "tab:blue",
}

# Hatching patterns for rescaled versions
SV_HATCHES = {
    "sv_respond_in_spanish_renorm_none": "//",
    "sv_any_language_renorm_none": "//",
}

# Line styles by dataset
DATASET_LINESTYLES = {
    "triviaqa_holdout": "-",      # solid
    "hh_rlhf": "--",              # dashed
    "wildchat": ":",              # dotted
}

# Markers by dataset
DATASET_MARKERS = {
    "triviaqa_holdout": "o",
    "hh_rlhf": "s",
    "wildchat": "^",
}


def load_results(sv_types: list[str] = None, datasets: list[str] = None) -> dict:
    """Load results for specified sv_types and datasets."""
    if sv_types is None:
        sv_types = SV_TYPES
    if datasets is None:
        datasets = DATASETS

    results = {}
    base_dir = Path(BASE_OUTPUT_DIR) / MODEL_SHORT

    for sv_type in sv_types:
        results[sv_type] = {}
        for dataset in datasets:
            layers_dir = base_dir / sv_type / dataset / EVAL_MOD / LAYERS
            if not layers_dir.exists():
                continue

            by_alpha = {}
            for alpha_dir in layers_dir.iterdir():
                if not alpha_dir.is_dir() or not alpha_dir.name.startswith("alpha_"):
                    continue

                alpha = alpha_dir.name.replace("alpha_", "")
                result_file = alpha_dir / "result.json"

                if result_file.exists():
                    with open(result_file) as f:
                        result = json.load(f)
                    by_alpha[alpha] = result

            if by_alpha:
                results[sv_type][dataset] = by_alpha

    return results


def plot_merged(results: dict, output_path: str, sv_types: list[str] = None, datasets: list[str] = None):
    """Create merged plot with color by sv_type and linestyle by dataset."""
    if sv_types is None:
        sv_types = SV_TYPES
    if datasets is None:
        datasets = DATASETS

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Track what we've added for legend
    sv_handles = {}
    dataset_handles = {}

    for sv_type in sv_types:
        for dataset in datasets:
            by_alpha = results.get(sv_type, {}).get(dataset, {})
            if not by_alpha:
                continue

            alphas = sorted([float(a) for a in by_alpha.keys()])
            scores = []
            for a in alphas:
                # Try different string formats
                for key in [str(a), f"{a:.2f}", f"{a:.1f}", f"{a}"]:
                    if key in by_alpha:
                        scores.append(by_alpha[key].get("spanish_score"))
                        break
                else:
                    scores.append(None)

            valid = [(a, s) for a, s in zip(alphas, scores) if s is not None]
            if not valid:
                continue

            x_vals = [v[0] for v in valid]
            y_vals = [v[1] for v in valid]

            color = SV_COLORS[sv_type]
            linestyle = DATASET_LINESTYLES[dataset]
            marker = DATASET_MARKERS[dataset]

            line, = ax.plot(
                x_vals,
                y_vals,
                linestyle=linestyle,
                marker=marker,
                linewidth=2,
                markersize=7,
                color=color,
                label=f"{SV_DISPLAY[sv_type]} / {DATASET_DISPLAY[dataset]}",
            )

            # Store handles for custom legend
            if sv_type not in sv_handles:
                sv_handles[sv_type] = line
            if dataset not in dataset_handles:
                dataset_handles[dataset] = line

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Strength (Alpha)", fontsize=12)
    ax.set_ylabel("Spanish Score (0-100)", fontsize=12)
    ax.set_title("Spanish Steering Across Datasets (all layers)", fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Create custom legend with two sections
    from matplotlib.lines import Line2D

    # Steering vector legend entries
    sv_legend_elements = [
        Line2D([0], [0], color=SV_COLORS.get(sv, "gray"), linewidth=3, label=SV_DISPLAY.get(sv, sv))
        for sv in sv_types if sv in results
    ]

    # Dataset legend entries (using linestyle + marker)
    dataset_legend_elements = [
        Line2D([0], [0], color="gray", linestyle=DATASET_LINESTYLES.get(ds, "-"),
               marker=DATASET_MARKERS.get(ds, "o"), markersize=7, linewidth=2,
               label=DATASET_DISPLAY.get(ds, ds))
        for ds in datasets
    ]

    # Add first legend (steering vectors)
    legend1 = ax.legend(
        handles=sv_legend_elements,
        title="Prompt Used to Create\nSteering Vector",
        loc="lower right",
        fontsize=10,
        title_fontsize=10,
    )
    ax.add_artist(legend1)

    # Add second legend (datasets)
    ax.legend(
        handles=dataset_legend_elements,
        title="Evaluation Dataset",
        loc="center right",
        fontsize=10,
        title_fontsize=10,
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_bar_chart(results: dict, alpha: float, output_path: str, sv_types: list[str] = None, datasets: list[str] = None):
    """Create bar chart comparing sv_types across datasets at a specific alpha."""
    import numpy as np

    if sv_types is None:
        sv_types = SV_TYPES
    if datasets is None:
        datasets = DATASETS

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(datasets))
    n_sv = len(sv_types)
    width = 0.8 / n_sv

    for i, sv_type in enumerate(sv_types):
        scores = []
        for dataset in datasets:
            by_alpha = results.get(sv_type, {}).get(dataset, {})
            score = None
            for key in [str(alpha), f"{alpha:.2f}", f"{alpha:.1f}"]:
                if key in by_alpha:
                    score = by_alpha[key].get("spanish_score")
                    break
            scores.append(score if score is not None else 0)

        offset = (i - (n_sv - 1) / 2) * width
        hatch = SV_HATCHES.get(sv_type, None)
        bars = ax.bar(
            x + offset,
            scores,
            width,
            label=SV_DISPLAY.get(sv_type, sv_type),
            color=SV_COLORS.get(sv_type, f"C{i}"),
            hatch=hatch,
            edgecolor='white' if hatch else None,
        )
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.annotate(
                    f'{score:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )

    ax.set_xlabel("Evaluation Dataset", fontsize=11)
    ax.set_ylabel("Spanish Score (0-100)", fontsize=11)
    ax.set_title(f"Spanish Steering at α={alpha} (all layers)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_DISPLAY.get(d, d) for d in datasets])
    ax.set_ylim(0, 100)
    ax.legend(title="Prompt Used to Create\nSteering Vector", fontsize=9, title_fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def load_all_sv_results(dataset: str, alpha: float) -> dict:
    """Load results for all steering vector types for a specific dataset and alpha."""
    results = {}
    base_dir = Path(BASE_OUTPUT_DIR) / MODEL_SHORT

    for sv_type_dir in base_dir.iterdir():
        if not sv_type_dir.is_dir() or not sv_type_dir.name.startswith("sv_"):
            continue

        sv_type = sv_type_dir.name
        result_file = sv_type_dir / dataset / EVAL_MOD / LAYERS / f"alpha_{alpha}" / "result.json"

        if result_file.exists():
            with open(result_file) as f:
                result = json.load(f)
            results[sv_type] = result

    return results


def plot_all_prompts_bar(dataset: str, alpha: float, output_path: str):
    """Create bar chart comparing all steering vector types for one dataset."""
    import numpy as np

    results = load_all_sv_results(dataset, alpha)
    if not results:
        print(f"No results found for {dataset} at alpha={alpha}")
        return

    # Sort by spanish_score ascending (lowest first)
    sorted_sv = sorted(results.items(), key=lambda x: x[1].get("spanish_score", 0), reverse=False)

    sv_names = [sv for sv, _ in sorted_sv]
    scores = [r.get("spanish_score", 0) for _, r in sorted_sv]

    # Display names for steering vectors
    all_sv_display = {
        "sv_any_language": "respond_in_any_language",
        "sv_none": "default",
        "sv_be_brief": "be_brief",
        "sv_proper_punctuation": "proper_punctuation",
        "sv_red_tag": "red_tag",
        "sv_respond_in_any_tone": "respond_in_any_tone",
        "sv_respond_in_spanish": "respond_in_spanish",
        "sv_you_are_bob": "you_are_bob",
    }

    display_names = [all_sv_display.get(sv, sv.replace("sv_", "")) for sv in sv_names]

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(sv_names))

    # Use consistent colors for sv_any_language and sv_none
    color_map = {
        "sv_any_language": "tab:blue",
        "sv_none": "tab:green",
        "sv_be_brief": "tab:orange",
        "sv_proper_punctuation": "tab:red",
        "sv_red_tag": "tab:purple",
        "sv_respond_in_any_tone": "tab:brown",
        "sv_respond_in_spanish": "tab:pink",
        "sv_you_are_bob": "tab:cyan",
    }
    colors = [color_map.get(sv, "tab:gray") for sv in sv_names]

    bars = ax.bar(x, scores, color=colors)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.annotate(
            f'{score:.1f}',
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9,
        )

    ax.set_xlabel("Prompt Used to Create Steering Vector", fontsize=11)
    ax.set_ylabel("Spanish Score (0-100)", fontsize=11)
    ax.set_title(f"Spanish Steering on {DATASET_DISPLAY.get(dataset, dataset)} at α={alpha} (all layers)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("Loading results...")
    results = load_results()

    for sv_type, datasets in results.items():
        print(f"  {sv_type}: {list(datasets.keys())}")

    output_path = f"{BASE_OUTPUT_DIR}/{MODEL_SHORT}/plots_{LAYERS}/merged_sv_comparison_no_mod.png"
    plot_merged(results, output_path)

    # Bar chart at alpha=0.06 with 3 sv types
    results_with_spanish = load_results(sv_types=SV_TYPES_WITH_SPANISH)
    bar_output_path = f"{BASE_OUTPUT_DIR}/{MODEL_SHORT}/plots_{LAYERS}/bar_comparison_alpha_0.06_no_mod.png"
    plot_bar_chart(results_with_spanish, 0.06, bar_output_path, sv_types=SV_TYPES_WITH_SPANISH)

    # Merged plot with only HH-RLHF and WildChat + respond_in_spanish
    datasets_subset = ["hh_rlhf", "wildchat"]
    results_subset = load_results(sv_types=SV_TYPES_WITH_SPANISH, datasets=datasets_subset)
    merged_subset_path = f"{BASE_OUTPUT_DIR}/{MODEL_SHORT}/plots_{LAYERS}/merged_sv_comparison_hh_wildchat_no_mod.png"
    plot_merged(results_subset, merged_subset_path, sv_types=SV_TYPES_WITH_SPANISH, datasets=datasets_subset)

    # Bar chart for HH-RLHF comparing all prompts
    hh_bar_path = f"{BASE_OUTPUT_DIR}/{MODEL_SHORT}/plots_{LAYERS}/bar_hh_rlhf_all_prompts_alpha_0.06.png"
    plot_all_prompts_bar("hh_rlhf", 0.06, hh_bar_path)

    # Bar chart with rescaled versions (only HH-RLHF has data)
    results_rescaled = load_results(sv_types=SV_TYPES_WITH_RESCALED, datasets=["hh_rlhf"])
    bar_rescaled_path = f"{BASE_OUTPUT_DIR}/{MODEL_SHORT}/plots_{LAYERS}/bar_comparison_rescaled_alpha_0.06_no_mod.png"
    plot_bar_chart(results_rescaled, 0.06, bar_rescaled_path, sv_types=SV_TYPES_WITH_RESCALED, datasets=["hh_rlhf"])

    print("Done!")


if __name__ == "__main__":
    main()
