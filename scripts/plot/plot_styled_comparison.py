#!/usr/bin/env python3
"""
Generate styled comparison plots matching bar_comparison style.

Usage:
    python scripts/plot_styled_comparison.py
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

EVAL_SET_DISPLAY_NAMES = {
    "triviaqa_holdout": "TriviaQA Holdout",
    "hh_rlhf": "HH-RLHF",
    "wildchat": "WildChat",
}

# Style mappings matching bar_comparison_alpha_0.06_no_mod.png
SV_DISPLAY = {
    "sv_any_language": "respond_in_any_language",
    "sv_any_language_renorm_none": "respond_in_any_language (rescaled)",
    "sv_none": "default",
    "sv_respond_in_spanish": "respond_in_spanish",
    "sv_respond_in_spanish_renorm_none": "respond_in_spanish (rescaled)",
    "sv_be_brief": "be_brief",
    "sv_proper_punctuation": "proper_punctuation",
    "sv_red_tag": "red_tag",
    "sv_respond_in_any_tone": "respond_in_any_tone",
    "sv_you_are_bob": "you_are_bob",
}

SV_COLORS = {
    "sv_any_language": "tab:blue",
    "sv_any_language_renorm_none": "tab:blue",
    "sv_none": "tab:green",
    "sv_respond_in_spanish": "tab:pink",
    "sv_respond_in_spanish_renorm_none": "tab:pink",
    "sv_be_brief": "tab:orange",
    "sv_proper_punctuation": "tab:red",
    "sv_red_tag": "tab:purple",
    "sv_respond_in_any_tone": "tab:brown",
    "sv_you_are_bob": "tab:cyan",
}

SV_MARKERS = {
    "sv_any_language": "o",
    "sv_any_language_renorm_none": "s",
    "sv_none": "D",
    "sv_respond_in_spanish": "^",
    "sv_respond_in_spanish_renorm_none": "v",
    "sv_be_brief": "p",
    "sv_proper_punctuation": "h",
    "sv_red_tag": "*",
    "sv_respond_in_any_tone": "P",
    "sv_you_are_bob": "X",
}

# Linestyles - solid for original, dashed for rescaled
SV_LINESTYLES = {
    "sv_any_language": "-",
    "sv_any_language_renorm_none": "--",
    "sv_none": "-",
    "sv_respond_in_spanish": "-",
    "sv_respond_in_spanish_renorm_none": "--",
    "sv_be_brief": "-",
    "sv_proper_punctuation": "-",
    "sv_red_tag": "-",
    "sv_respond_in_any_tone": "-",
    "sv_you_are_bob": "-",
}

# Ordering for legend
SV_ORDER = [
    "sv_none",
    "sv_any_language",
    "sv_any_language_renorm_none",
    "sv_respond_in_spanish",
    "sv_respond_in_spanish_renorm_none",
    "sv_be_brief",
    "sv_proper_punctuation",
    "sv_red_tag",
    "sv_respond_in_any_tone",
    "sv_you_are_bob",
]


def load_results(eval_set: str, sv_types: list[str] | None = None) -> dict:
    """Load results for specified eval_set."""
    results = {}
    base_dir = Path(BASE_OUTPUT_DIR) / MODEL_SHORT

    for sv_type_dir in base_dir.iterdir():
        if not sv_type_dir.is_dir() or not sv_type_dir.name.startswith("sv_"):
            continue

        sv_type = sv_type_dir.name
        if sv_types is not None and sv_type not in sv_types:
            continue

        layers_dir = sv_type_dir / eval_set / EVAL_MOD / LAYERS
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
            results[sv_type] = by_alpha

    return results


def plot_comparison(
    results: dict,
    eval_set: str,
    output_path: str,
    title_suffix: str = "",
):
    """Create styled comparison plot."""
    eval_display = EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Sort by SV_ORDER to ensure consistent legend ordering
    sorted_sv_types = sorted(
        results.keys(),
        key=lambda x: SV_ORDER.index(x) if x in SV_ORDER else 999
    )

    for sv_type in sorted_sv_types:
        by_alpha = results[sv_type]

        alphas = sorted([float(a) for a in by_alpha.keys()])
        scores = []
        for a in alphas:
            for key in [str(a), f"{a:.2f}", f"{a:.1f}"]:
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

        color = SV_COLORS.get(sv_type, "tab:gray")
        marker = SV_MARKERS.get(sv_type, "o")
        linestyle = SV_LINESTYLES.get(sv_type, "-")
        label = SV_DISPLAY.get(sv_type, sv_type.replace("sv_", ""))

        ax.plot(
            x_vals,
            y_vals,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            markersize=8,
            color=color,
            label=label,
        )

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Strength (Alpha)", fontsize=11)
    ax.set_ylabel("Spanish Score (0-100)", fontsize=11)
    ax.set_title(
        f"Spanish Steering: {eval_display}{title_suffix}",
        fontsize=12,
    )
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="Prompt Used to Create\nSteering Vector",
        fontsize=9,
        title_fontsize=9,
        loc="best",
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    eval_set = "hh_rlhf"
    output_dir = f"{BASE_OUTPUT_DIR}/{MODEL_SHORT}/plots_{LAYERS}"

    # Plot 1: All steering vectors with new style
    print("Loading all results for HH-RLHF...")
    all_results = load_results(eval_set)
    print(f"Found {len(all_results)} steering vector types: {list(all_results.keys())}")

    plot_comparison(
        all_results,
        eval_set,
        f"{output_dir}/{eval_set}_{EVAL_MOD}_styled.png",
        title_suffix=" (all layers)",
    )

    # Plot 2: Filtered to key steering vectors only
    key_sv_types = [
        "sv_none",
        "sv_any_language",
        "sv_any_language_renorm_none",
        "sv_respond_in_spanish",
        "sv_respond_in_spanish_renorm_none",
    ]
    filtered_results = load_results(eval_set, sv_types=key_sv_types)
    print(f"Filtered to {len(filtered_results)} steering vector types: {list(filtered_results.keys())}")

    plot_comparison(
        filtered_results,
        eval_set,
        f"{output_dir}/{eval_set}_{EVAL_MOD}_styled_key_only.png",
        title_suffix=" (all layers)",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
