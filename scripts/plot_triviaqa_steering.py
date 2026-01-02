#!/usr/bin/env python3
"""
Generate plots from existing triviaqa_steering results.

Usage:
    python scripts/plot_triviaqa_steering.py
    python scripts/plot_triviaqa_steering.py --eval-mod respond_in_any_language_you_like
    python scripts/plot_triviaqa_steering.py --layers layers_all
    python scripts/plot_triviaqa_steering.py --layers layers_10_20_30
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_SHORT = "gemma-2-9b-it"
BASE_OUTPUT_DIR = "results/triviaqa_steering"

EVAL_SET_DISPLAY_NAMES = {
    "triviaqa_holdout": "TriviaQA Holdout",
    "hh_rlhf": "HH-RLHF Spanish",
    "wildchat": "WildChat Users",
}

COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]
MARKERS = ["o", "s", "^", "D", "v", "p", "h", "*", "+", "x"]


def get_color(idx: int) -> str:
    return COLORS[idx % len(COLORS)]


def get_marker(idx: int) -> str:
    return MARKERS[idx % len(MARKERS)]


def load_results(base_dir: str, model: str, eval_mod: str, layers: str) -> dict:
    """Load all results for a given eval_mod and layers config.

    Directory structure:
        {model}/{sv_type}/{eval_set}/{eval_mod}/{layers}/alpha_{alpha}/
    """
    results = {}
    model_dir = Path(base_dir) / model

    if not model_dir.exists():
        print(f"ERROR: {model_dir} does not exist")
        return results

    # Find all sv_type directories
    for sv_type_dir in model_dir.iterdir():
        if not sv_type_dir.is_dir() or not sv_type_dir.name.startswith("sv_"):
            continue

        sv_type = sv_type_dir.name
        results[sv_type] = {"by_eval_set": {}}

        # Find all eval_set directories
        for eval_set_dir in sv_type_dir.iterdir():
            if not eval_set_dir.is_dir():
                continue

            eval_set = eval_set_dir.name
            layers_dir = eval_set_dir / eval_mod / layers

            if not layers_dir.exists():
                continue

            by_alpha = {}

            # Find all alpha directories
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
                results[sv_type]["by_eval_set"][eval_set] = {"by_alpha": by_alpha}

    return results


def plot_dataset_comparison(all_results: dict, eval_set: str, eval_mod: str, layers: str, output_dir: str):
    """Plot comparison of all sv_types for a single eval set."""
    eval_display = EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (sv_type, results) in enumerate(sorted(all_results.items())):
        eval_data = results.get("by_eval_set", {}).get(eval_set, {})
        by_alpha = eval_data.get("by_alpha", {})

        if not by_alpha:
            continue

        alphas = sorted([float(a) for a in by_alpha.keys()])
        scores = [by_alpha[str(a) if str(a) in by_alpha else f"{a}"].get("spanish_score") for a in alphas]

        valid = [(a, s) for a, s in zip(alphas, scores) if s is not None]
        if valid:
            ax.plot(
                [x[0] for x in valid],
                [x[1] for x in valid],
                f"{get_marker(idx)}-",
                linewidth=2,
                markersize=8,
                color=get_color(idx),
                label=sv_type,
            )

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Strength (Alpha)", fontsize=11)
    ax.set_ylabel("Spanish Score (0-100)", fontsize=11)
    ax.set_title(
        f"Spanish Steering: {eval_display}\n"
        f"Model: {MODEL_SHORT}, {eval_mod}, {layers}",
        fontsize=12,
    )
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    out_path = f"{output_dir}/{MODEL_SHORT}/plots_{layers}/{eval_set}_{eval_mod}_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_sv_type_comparison(results: dict, sv_type: str, eval_mod: str, layers: str, output_dir: str):
    """Plot comparison of all eval sets for a single sv_type."""
    by_eval_set = results.get("by_eval_set", {})

    if len(by_eval_set) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (eval_set, eval_data) in enumerate(sorted(by_eval_set.items())):
        by_alpha = eval_data.get("by_alpha", {})

        if not by_alpha:
            continue

        alphas = sorted([float(a) for a in by_alpha.keys()])
        scores = [by_alpha[str(a) if str(a) in by_alpha else f"{a}"].get("spanish_score") for a in alphas]

        valid = [(a, s) for a, s in zip(alphas, scores) if s is not None]
        if valid:
            eval_display = EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)
            ax.plot(
                [x[0] for x in valid],
                [x[1] for x in valid],
                f"{get_marker(idx)}-",
                linewidth=2,
                markersize=8,
                color=get_color(idx),
                label=eval_display,
            )

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Strength (Alpha)", fontsize=11)
    ax.set_ylabel("Spanish Score (0-100)", fontsize=11)
    ax.set_title(
        f"Spanish Steering: {sv_type}\n"
        f"Model: {MODEL_SHORT}, {eval_mod}, {layers}",
        fontsize=12,
    )
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    out_path = f"{output_dir}/{MODEL_SHORT}/plots_{layers}/{sv_type}_datasets_{eval_mod}_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_summary(all_results: dict, eval_mod: str):
    """Print summary table."""
    # Collect all alphas across all results
    all_alphas = set()
    for sv_type, results in all_results.items():
        for eval_set, eval_data in results.get("by_eval_set", {}).items():
            for alpha in eval_data.get("by_alpha", {}).keys():
                all_alphas.add(float(alpha))

    alphas = sorted(all_alphas)

    # Collect all eval sets
    all_eval_sets = set()
    for sv_type, results in all_results.items():
        for eval_set in results.get("by_eval_set", {}).keys():
            all_eval_sets.add(eval_set)

    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY (eval_mod: {eval_mod})")
    print("=" * 80)

    for eval_set in sorted(all_eval_sets):
        eval_display = EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)
        print(f"\n{eval_display}:")
        print(f"{'SV Type':<30} " + " ".join(f"{a:>8}" for a in alphas))
        print("-" * (30 + 9 * len(alphas)))

        for sv_type in sorted(all_results.keys()):
            results = all_results[sv_type]
            by_alpha = results.get("by_eval_set", {}).get(eval_set, {}).get("by_alpha", {})

            scores = []
            for a in alphas:
                # Try different string formats for alpha
                score = None
                for key in [str(a), f"{a:.2f}", f"{a:.1f}"]:
                    if key in by_alpha:
                        score = by_alpha[key].get("spanish_score")
                        break
                scores.append(f"{score:8.1f}" if score is not None else "     N/A")

            print(f"{sv_type:<30} " + " ".join(scores))


def main():
    parser = argparse.ArgumentParser(description="Generate plots from existing triviaqa_steering results")
    parser.add_argument("--base-dir", default=BASE_OUTPUT_DIR, help="Base results directory")
    parser.add_argument("--model", default=MODEL_SHORT, help="Model name")
    parser.add_argument("--eval-mod", default="respond_in_any_language_you_like",
                        help="Eval modification directory name")
    parser.add_argument("--layers", default="layers_all",
                        help="Layers directory name (e.g., layers_all, layers_10_20_30)")

    args = parser.parse_args()

    print(f"Loading results from: {args.base_dir}/{args.model}/")
    print(f"Eval mod: {args.eval_mod}")
    print(f"Layers: {args.layers}")

    all_results = load_results(args.base_dir, args.model, args.eval_mod, args.layers)

    if not all_results:
        print("No results found!")
        return

    print(f"\nFound {len(all_results)} steering vector types: {list(all_results.keys())}")

    # Collect all eval sets
    all_eval_sets = set()
    for sv_type, results in all_results.items():
        for eval_set in results.get("by_eval_set", {}).keys():
            all_eval_sets.add(eval_set)

    print(f"Eval sets: {list(all_eval_sets)}")

    # Generate plots
    print("\nGenerating plots...")

    # Per-sv_type plots
    for sv_type, results in all_results.items():
        plot_sv_type_comparison(results, sv_type, args.eval_mod, args.layers, args.base_dir)

    # Per-dataset plots
    for eval_set in all_eval_sets:
        plot_dataset_comparison(all_results, eval_set, args.eval_mod, args.layers, args.base_dir)

    # Print summary
    print_summary(all_results, args.eval_mod)

    print(f"\n{'=' * 70}")
    print("COMPLETE!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
