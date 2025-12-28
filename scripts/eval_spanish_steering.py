#!/usr/bin/env python3
"""
Evaluate Spanish steering vectors on:
1. NL test split from SORH dataset
2. Anthropic HH-RLHF dataset (100 samples)

This script:
1. Loads Spanish steering vectors (spanish, spanish_can_spanish, spanish_can_english)
2. Generates steered responses at various alpha values
3. Judges responses for Spanish content (0-100 score)
4. Generates comparison plots

Usage:
    python scripts/eval_spanish_steering.py --help
    python scripts/eval_spanish_steering.py --max-samples 10 --alphas -1 0 1
    python scripts/eval_spanish_steering.py --sv-types spanish spanish_can_spanish
"""

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# CONFIGURATION
# ============================================================================

# Split names for evaluation
SPLIT_NL_TEST = "nl_test"  # Natural language test split from SORH
SPLIT_HH_RLHF = "hh_rlhf"  # Anthropic HH-RLHF samples

# Base steering vector types (without seed suffix)
BASE_SV_TYPES = [
    "spanish",
    "spanish_can_spanish",
    "spanish_can_english",
    "spanish_can_german",
    "spanish_respond_in_spanish",
    "spanish_respond_in_english",
    "spanish_respond_in_german",
    "spanish_respond_in_spanish_v2",
    "spanish_respond_in_english_v2",
    "spanish_respond_in_german_v2",
]


def get_sv_type_with_seed(base_type: str, seed: int) -> str:
    """Add seed suffix to steering vector type name."""
    return f"{base_type}_seed{seed}"


def get_all_sv_types(seed: int) -> list[str]:
    """Get all steering vector type names with seed suffix."""
    return [get_sv_type_with_seed(t, seed) for t in BASE_SV_TYPES]


def get_base_sv_type(sv_type_with_seed: str) -> str:
    """Extract base type from seeded sv_type name.

    E.g., 'spanish_can_spanish_seed42' -> 'spanish_can_spanish'
    """
    import re
    match = re.match(r"(.+)_seed\d+$", sv_type_with_seed)
    if match:
        return match.group(1)
    return sv_type_with_seed

# Styling for plots
COLORS = {
    "spanish": "tab:blue",
    "spanish_can_spanish": "tab:orange",
    "spanish_can_english": "tab:green",
    "spanish_can_german": "tab:olive",
    "spanish_respond_in_spanish": "tab:red",
    "spanish_respond_in_english": "tab:purple",
    "spanish_respond_in_german": "tab:brown",
    "spanish_respond_in_spanish_v2": "tab:pink",
    "spanish_respond_in_english_v2": "tab:cyan",
    "spanish_respond_in_german_v2": "tab:gray",
}
MARKERS = {
    "spanish": "o",
    "spanish_can_spanish": "s",
    "spanish_can_english": "^",
    "spanish_can_german": "h",
    "spanish_respond_in_spanish": "D",
    "spanish_respond_in_english": "v",
    "spanish_respond_in_german": "p",
    "spanish_respond_in_spanish_v2": "+",
    "spanish_respond_in_english_v2": "x",
    "spanish_respond_in_german_v2": "*",
}
LABELS = {
    "spanish": "default (no prompt)",
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

SPLIT_DISPLAY_NAMES = {
    SPLIT_NL_TEST: "SORH NL Test",
    SPLIT_HH_RLHF: "Anthropic HH-RLHF",
}


def get_style(sv_type: str, style_dict: dict) -> str:
    """Get style value for sv_type, using base type if seeded name not found."""
    if sv_type in style_dict:
        return style_dict[sv_type]
    base_type = get_base_sv_type(sv_type)
    return style_dict.get(base_type, style_dict.get("spanish", "gray"))


@dataclass
class EvalConfig:
    """Configuration for Spanish steering evaluation."""

    # Steering vector seed (must match extraction seed)
    sv_seed: int = 42

    # Steering vector types to evaluate (will be populated based on sv_seed)
    sv_types: list[str] = field(default_factory=list)

    # Layers to apply steering (all 36 transformer layers by default for Qwen3-8B)
    steering_layers: list[int] = field(default_factory=lambda: list(range(36)))

    # Steering strengths to test
    alphas: list[float] = field(default_factory=lambda: [-2.0, -1.0, 0.0, 1.0, 2.0])

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 1.0
    batch_size: int = 1

    # Dataset settings
    seed: int = 42  # For SORH NL test split (separate from sv_seed)
    max_samples: Optional[int] = None  # Applied per split
    hh_rlhf_samples: int = 100  # Number of HH-RLHF samples to use

    # Splits to evaluate
    splits: list[str] = field(
        default_factory=lambda: [SPLIT_NL_TEST, SPLIT_HH_RLHF]
    )

    # Paths
    steering_vectors_dir: str = "data"
    output_dir: str = "results/spanish_eval"

    # Judge settings
    judge_model: str = "gpt-4o-mini"


# ============================================================================
# DATASET UTILS
# ============================================================================


def get_final_human_prompt(conversation: str) -> str:
    """Extract the final human prompt from an HH-RLHF conversation.

    The format is alternating Human:/Assistant: turns (or H:/A:).
    """
    # Try full format first
    human_pattern = r"\n\nHuman: (.*?)(?=\n\nAssistant:|\Z)"
    matches = re.findall(human_pattern, conversation, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try abbreviated format
    human_pattern = r"\n\nH: (.*?)(?=\n\nA:|\Z)"
    matches = re.findall(human_pattern, conversation, re.DOTALL)
    if matches:
        return matches[-1].strip()

    return ""


def get_test_splits(config: EvalConfig) -> dict[str, list[dict]]:
    """Get test splits based on config.splits.

    Returns:
        Dict with split names as keys, each containing list of sample dicts.
    """
    import random

    splits = {}

    # Load NL test split from SORH if requested
    if SPLIT_NL_TEST in config.splits:
        print("Loading SORH dataset for NL test split...")
        sorh_ds = load_dataset("longtermrisk/school-of-reward-hacks", split="train")

        # Load Spanish dataset indices to exclude (avoid overlap)
        spanish_indices = set()
        split_info_path = f"{config.steering_vectors_dir}/spanish_split_info_seed{config.sv_seed}.json"
        if os.path.exists(split_info_path):
            with open(split_info_path, "r") as f:
                split_info = json.load(f)
            spanish_indices = set(split_info.get("selected_indices", []))
            print(f"  Excluding {len(spanish_indices)} Spanish dataset indices")
        else:
            print(f"  WARNING: {split_info_path} not found - cannot exclude Spanish indices")

        # Recreate the same split as the original eval
        ood_task = "write a function"
        nl_indices = [i for i, row in enumerate(sorh_ds) if row["task"] != ood_task]

        # Exclude Spanish dataset indices
        nl_indices = [i for i in nl_indices if i not in spanish_indices]

        rng = random.Random(config.seed)
        nl_indices_shuffled = nl_indices.copy()
        rng.shuffle(nl_indices_shuffled)

        # 10% for test
        nl_test_size = int(len(nl_indices_shuffled) * 0.1)
        nl_test_indices = nl_indices_shuffled[:nl_test_size]

        if config.max_samples is not None:
            nl_test_indices = nl_test_indices[: config.max_samples]

        nl_test_samples = [
            {"user": sorh_ds[i]["user"], "source": "sorh"} for i in nl_test_indices
        ]
        splits[SPLIT_NL_TEST] = nl_test_samples
        print(f"  NL test: {len(nl_test_samples)} samples (no overlap with Spanish dataset)")

    # Load HH-RLHF dataset if requested
    if SPLIT_HH_RLHF in config.splits:
        print("Loading Anthropic HH-RLHF dataset...")
        hh_ds = load_dataset("Anthropic/hh-rlhf", split="test")

        # Extract final human prompts
        n_hh = min(config.hh_rlhf_samples, len(hh_ds))
        if config.max_samples is not None:
            n_hh = min(n_hh, config.max_samples)

        hh_samples = []
        for i in range(n_hh):
            prompt = get_final_human_prompt(hh_ds[i]["chosen"])
            if prompt:  # Only add if we could extract a prompt
                hh_samples.append({"user": prompt, "source": "hh_rlhf"})

        splits[SPLIT_HH_RLHF] = hh_samples
        print(f"  HH-RLHF: {len(hh_samples)} samples")

    return splits


def load_steering_vectors(config: EvalConfig) -> dict[str, torch.Tensor]:
    """Load all steering vectors.

    Returns:
        Dict mapping sv_type -> tensor of shape [num_layers, hidden_dim]
    """
    steering_vectors = {}

    for sv_type in config.sv_types:
        path = f"{config.steering_vectors_dir}/steering_vectors_{sv_type}.pt"
        if os.path.exists(path):
            vectors = torch.load(path)
            steering_vectors[sv_type] = vectors
            print(f"Loaded {sv_type}: {vectors.shape}")
        else:
            print(f"WARNING: {path} not found - skipping {sv_type}")

    return steering_vectors


# ============================================================================
# GENERATION
# ============================================================================


def generate_with_multilayer_steering(
    model,
    tokenizer,
    questions: list[str],
    steering_vectors: torch.Tensor,  # [num_layers, hidden_dim]
    steering_layers: list[int],
    alpha: float,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    batch_size: int = 1,
) -> list[dict]:
    """Generate responses with steering applied at multiple layers."""
    device = next(model.parameters()).device
    steering_vectors = steering_vectors.to(device)

    # Ensure tokenizer has pad token and uses left padding for batch generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Get model layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model_layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        model_layers = model.transformer.h
    else:
        raise ValueError("Unknown model architecture - can't find layers")

    handles = []

    # Register hooks for each steering layer
    for layer_idx in steering_layers:
        # +1 because steering_vectors[0] is embedding layer
        sv = steering_vectors[layer_idx + 1]

        def make_hook(steering_vector, strength):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    hidden_states = hidden_states + strength * steering_vector
                    return (hidden_states,) + output[1:]
                else:
                    return output + strength * steering_vector

            return hook

        handle = model_layers[layer_idx].register_forward_hook(make_hook(sv, alpha))
        handles.append(handle)

    results = []

    # Prepare all prompts
    all_prompts = []
    for question in questions:
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        all_prompts.append(prompt)

    try:
        # Process in batches
        for batch_start in range(0, len(all_prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(all_prompts))
            batch_prompts = all_prompts[batch_start:batch_end]
            batch_questions = questions[batch_start:batch_end]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode each output in the batch
            input_len = inputs["input_ids"].shape[1]
            for i, output in enumerate(outputs):
                response = tokenizer.decode(
                    output[input_len:], skip_special_tokens=True
                )

                results.append(
                    {
                        "id": len(results) + 1,
                        "question": batch_questions[i],
                        "response": response,
                        "answer_idx": 0,
                        "steering_layers": steering_layers,
                        "alpha": alpha,
                    }
                )

            print(
                f"  Batch {batch_start // batch_size + 1}/{(len(all_prompts) + batch_size - 1) // batch_size} done"
            )

    finally:
        for handle in handles:
            handle.remove()

    return results


# ============================================================================
# JUDGING
# ============================================================================


def transform_to_judge_format_simple(
    generations: list[dict],
    samples: list[dict],
) -> dict:
    """Transform generations to judge format (simplified for Spanish eval)."""
    prompt_to_completions = {}

    for idx, (gen, sample) in enumerate(zip(generations, samples)):
        completions = {
            "0": {
                "raw_text": gen.get("response", ""),
                "answer_idx": 0,
            }
        }

        prompt_to_completions[str(idx)] = {
            "user": sample["user"],
            "source": sample.get("source", ""),
            "completions": completions,
        }

    return {"prompt_to_completions": prompt_to_completions}


async def judge_completions(
    completions_path: str,
    judge_template_path: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
) -> dict:
    """Run judge on completions."""
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    )
    from judge_completions import judge_all_completions

    from src.utils import load_yaml

    judge_config = load_yaml(judge_template_path)

    with open(completions_path, "r") as f:
        completions_data = json.load(f)

    judged_data, judge_responses, cache_stats = await judge_all_completions(
        completions_data,
        judge_config,
        model,
        temperature,
        max_tokens=100,
        enable_cache=True,
    )

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w") as f:
        json.dump(judged_data, f, indent=2)

    return judged_data


def extract_mean_reward(judged_data: dict) -> Optional[float]:
    """Extract mean reward from judged data."""
    rewards = []
    for sample_data in judged_data["prompt_to_completions"].values():
        for completion in sample_data.get("completions", {}).values():
            reward = completion.get("reward")
            if reward is not None:
                rewards.append(float(reward))

    return sum(rewards) / len(rewards) if rewards else None


# ============================================================================
# EVALUATION
# ============================================================================


def run_single_evaluation(
    model,
    tokenizer,
    test_samples: list[dict],
    steering_vectors: torch.Tensor,
    sv_type: str,
    split_name: str,
    config: EvalConfig,
    baseline_results: Optional[dict] = None,
) -> dict:
    """Run evaluation for a specific steering vector type on a split.

    Args:
        baseline_results: Pre-computed results for alpha=0.0 (no steering).
            If provided, will be reused instead of regenerating.
    """
    results = {
        "sv_type": sv_type,
        "split": split_name,
        "steering_layers": config.steering_layers,
        "by_alpha": {},
    }

    for alpha in config.alphas:
        # Reuse baseline for alpha=0.0 (no steering is identical across sv_types)
        if alpha == 0.0 and baseline_results is not None:
            print(f"\n  Alpha = {alpha} (reusing baseline)")
            results["by_alpha"]["0.0"] = baseline_results
            continue

        print(f"\n  Alpha = {alpha}")

        # Generate responses
        if alpha == 0.0:
            generations = generate_with_multilayer_steering(
                model=model,
                tokenizer=tokenizer,
                questions=[s["user"] for s in test_samples],
                steering_vectors=steering_vectors,
                steering_layers=[],  # Empty list = no steering
                alpha=0.0,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                batch_size=config.batch_size,
            )
        else:
            generations = generate_with_multilayer_steering(
                model=model,
                tokenizer=tokenizer,
                questions=[s["user"] for s in test_samples],
                steering_vectors=steering_vectors,
                steering_layers=config.steering_layers,
                alpha=alpha,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                batch_size=config.batch_size,
            )

        # Save generations - include sv_type in path
        gen_dir = f"{config.output_dir}/{split_name}/{sv_type}"
        os.makedirs(gen_dir, exist_ok=True)

        layers_str = get_layers_filename(config.steering_layers)
        gen_path = f"{gen_dir}/generations_layers_{layers_str}_alpha_{alpha}.json"
        with open(gen_path, "w") as f:
            json.dump(generations, f, indent=2)

        # Transform to judge format
        judge_input = transform_to_judge_format_simple(generations, test_samples)

        judge_input_path = (
            f"{gen_dir}/judge_input_layers_{layers_str}_alpha_{alpha}.json"
        )
        with open(judge_input_path, "w") as f:
            json.dump(judge_input, f, indent=2)

        alpha_results = {
            "generations_count": len(generations),
            "steering_layers": config.steering_layers,
            "spanish_score": None,
        }

        # Judge for Spanish content
        print("    Judging for Spanish content...")
        try:
            spanish_judged = asyncio.run(
                judge_completions(
                    completions_path=judge_input_path,
                    judge_template_path="prompts/judge/spanish_scorer.yaml",
                    output_path=f"{gen_dir}/spanish_judged_layers_{layers_str}_alpha_{alpha}.json",
                    model=config.judge_model,
                )
            )
            alpha_results["spanish_score"] = extract_mean_reward(spanish_judged)
            if alpha_results["spanish_score"] is not None:
                print(f"    Spanish score: {alpha_results['spanish_score']:.2f}")
        except Exception as e:
            print(f"    Error: {e}")

        results["by_alpha"][str(alpha)] = alpha_results

    return results


# ============================================================================
# PLOTTING
# ============================================================================


def get_layers_display(steering_layers: list[int]) -> str:
    """Get human-readable display string for layers."""
    if steering_layers == list(range(36)):
        return "all"
    elif len(steering_layers) <= 5:
        return ", ".join(map(str, steering_layers))
    else:
        return f"{steering_layers[0]}-{steering_layers[-1]}"


def get_layers_filename(steering_layers: list[int]) -> str:
    """Get filename-safe string for layers."""
    if steering_layers == list(range(36)):
        return "all"
    else:
        return "_".join(map(str, steering_layers))


def plot_comparison_for_split(
    all_results: dict[str, dict],
    split_name: str,
    config: EvalConfig,
    output_dir: str,
):
    """Generate Spanish score plot for a single split with all sv_types."""
    layers_str = get_layers_filename(config.steering_layers)
    layers_display = get_layers_display(config.steering_layers)
    split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)

    fig, ax = plt.subplots(figsize=(10, 6))

    for sv_type, results in all_results.items():
        alphas = sorted([float(a) for a in results["by_alpha"].keys()])
        spanish_scores = [
            results["by_alpha"][str(a)].get("spanish_score") for a in alphas
        ]

        valid = [(a, s) for a, s in zip(alphas, spanish_scores) if s is not None]

        if valid:
            color = get_style(sv_type, COLORS)
            marker = get_style(sv_type, MARKERS)
            label = get_style(sv_type, LABELS)

            ax.plot(
                [x[0] for x in valid],
                [x[1] for x in valid],
                f"{marker}-",
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
        f"Spanish Content vs Steering - {split_display}\n(Layers: {layers_display})",
        fontsize=12,
    )
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    split_dir = f"{output_dir}/{split_name}"
    os.makedirs(split_dir, exist_ok=True)
    plt.savefig(
        f"{split_dir}/spanish_score_layers_{layers_str}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {split_dir}/spanish_score_layers_{layers_str}.png")


def plot_split_comparison(
    results_by_split: dict[str, dict[str, dict]],
    config: EvalConfig,
    output_dir: str,
):
    """Generate comparison plot showing all splits side by side with all sv_types."""
    layers_str = get_layers_filename(config.steering_layers)
    layers_display = get_layers_display(config.steering_layers)

    n_splits = len(results_by_split)
    fig, axes = plt.subplots(1, n_splits, figsize=(7 * n_splits, 5), sharey=True)

    if n_splits == 1:
        axes = [axes]

    for ax, (split_name, all_results) in zip(axes, results_by_split.items()):
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)

        for sv_type, results in all_results.items():
            alphas = sorted([float(a) for a in results["by_alpha"].keys()])
            spanish_scores = [
                results["by_alpha"][str(a)].get("spanish_score") for a in alphas
            ]

            valid = [(a, s) for a, s in zip(alphas, spanish_scores) if s is not None]

            if valid:
                color = get_style(sv_type, COLORS)
                marker = get_style(sv_type, MARKERS)
                label = get_style(sv_type, LABELS)

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
        ax.set_xlabel("Steering Strength (Alpha)", fontsize=10)
        ax.set_title(f"{split_display}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8, loc="best")

    axes[0].set_ylabel("Spanish Score (0-100)", fontsize=10)

    fig.suptitle(
        f"Spanish Steering: Cross-Dataset Comparison\n(Layers: {layers_display})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/split_comparison_spanish_layers_{layers_str}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_dir}/split_comparison_spanish_layers_{layers_str}.png")


def print_summary(all_results: dict[str, dict], split_name: str, config: EvalConfig):
    """Print summary table for a specific split."""
    layers_display = get_layers_display(config.steering_layers)
    split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)
    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY - {split_display} (Layers: {layers_display})")
    print("=" * 70)

    print(f"\n{'SV Type':<25} {'Alpha':<10} {'Spanish Score':<15}")
    print("-" * 55)

    for sv_type, results in sorted(all_results.items()):
        for alpha_str in sorted(results["by_alpha"].keys(), key=float):
            data = results["by_alpha"][alpha_str]
            spanish = data.get("spanish_score")
            spanish_str = f"{spanish:.2f}" if spanish is not None else "N/A"
            print(f"{sv_type:<25} {alpha_str:<10} {spanish_str:<15}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Spanish steering vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 10 samples (uses default sv-seed=42)
  python scripts/eval_spanish_steering.py --max-samples 10 --alphas -1 0 1

  # Full evaluation with specific seed
  python scripts/eval_spanish_steering.py --sv-seed 42

  # Specific base sv_types only (seed suffix added automatically)
  python scripts/eval_spanish_steering.py --sv-seed 42 --sv-types spanish spanish_can_spanish

  # Specific layers only
  python scripts/eval_spanish_steering.py --steering-layers 20 25 30
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model to evaluate (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--sv-seed",
        type=int,
        default=42,
        help="Seed used for steering vector extraction (default: 42)",
    )
    parser.add_argument(
        "--sv-types",
        type=str,
        nargs="+",
        default=None,
        help="Base steering vector types to evaluate (seed suffix added automatically). Default: all types",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=[SPLIT_NL_TEST, SPLIT_HH_RLHF],
        choices=[SPLIT_NL_TEST, SPLIT_HH_RLHF],
        help="Splits to evaluate on (default: both)",
    )
    parser.add_argument(
        "--steering-layers",
        type=int,
        nargs="+",
        default=list(range(36)),
        help="Layers to apply steering (default: all 36 layers)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[-2.0, -1.0, 0.0, 1.0, 2.0],
        help="Steering strengths to test",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per split",
    )
    parser.add_argument(
        "--hh-rlhf-samples",
        type=int,
        default=100,
        help="Number of HH-RLHF samples (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/spanish_eval",
        help="Output directory",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for judging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation, only re-plot from existing results",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to existing results for re-plotting",
    )

    args = parser.parse_args()

    # Handle --results-dir for re-plotting
    if args.results_dir:
        config_path = f"{args.results_dir}/eval_config.json"
        if not os.path.exists(config_path):
            print(f"ERROR: {config_path} not found")
            return

        with open(config_path, "r") as f:
            saved_config = json.load(f)

        sv_seed = saved_config.get("sv_seed", 42)
        config = EvalConfig(
            sv_seed=sv_seed,
            sv_types=saved_config.get("sv_types", get_all_sv_types(sv_seed)),
            steering_layers=saved_config.get("steering_layers", list(range(36))),
            alphas=saved_config.get("alphas", [-2.0, -1.0, 0.0, 1.0, 2.0]),
            max_samples=saved_config.get("max_samples"),
            splits=saved_config.get("splits", [SPLIT_NL_TEST, SPLIT_HH_RLHF]),
            output_dir=args.results_dir,
            judge_model=saved_config.get("judge_model", "gpt-4o-mini"),
            seed=saved_config.get("seed", 42),
            batch_size=saved_config.get("batch_size", 1),
        )

        print(f"Re-plotting from: {args.results_dir}")

        splits = config.splits
        results_by_split = {}

        for split_name in splits:
            split_results = {}
            for sv_type in config.sv_types:
                results_path = f"{config.output_dir}/{split_name}/{sv_type}/results.json"
                if os.path.exists(results_path):
                    with open(results_path, "r") as f:
                        split_results[sv_type] = json.load(f)
                    print(f"Loaded results for {split_name}/{sv_type}")

            if split_results:
                results_by_split[split_name] = split_results
                plot_comparison_for_split(
                    split_results, split_name, config, config.output_dir
                )
                print_summary(split_results, split_name, config)

        if len(results_by_split) >= 2:
            plot_split_comparison(results_by_split, config, config.output_dir)

        return

    # Determine sv_types with seed suffix
    if args.sv_types is None:
        # Use all base types with seed suffix
        sv_types = get_all_sv_types(args.sv_seed)
    else:
        # Add seed suffix to provided base types
        sv_types = [get_sv_type_with_seed(t, args.sv_seed) for t in args.sv_types]

    # Create output directory with timestamp and seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.max_samples is not None:
        output_dir = f"{args.output_dir}/{timestamp}_seed{args.sv_seed}_n{args.max_samples}"
    else:
        output_dir = f"{args.output_dir}/{timestamp}_seed{args.sv_seed}_full"

    config = EvalConfig(
        sv_seed=args.sv_seed,
        sv_types=sv_types,
        steering_layers=args.steering_layers,
        alphas=args.alphas,
        max_samples=args.max_samples,
        hh_rlhf_samples=args.hh_rlhf_samples,
        splits=args.splits,
        output_dir=output_dir,
        judge_model=args.judge_model,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    config_path = f"{config.output_dir}/eval_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "sv_seed": config.sv_seed,
                "sv_types": config.sv_types,
                "steering_layers": config.steering_layers,
                "alphas": config.alphas,
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature,
                "batch_size": config.batch_size,
                "max_samples": config.max_samples,
                "hh_rlhf_samples": config.hh_rlhf_samples,
                "seed": config.seed,
                "judge_model": config.judge_model,
                "output_dir": config.output_dir,
                "splits": config.splits,
            },
            f,
            indent=2,
        )
    print(f"Output directory: {config.output_dir}")
    print(f"Steering vector seed: {config.sv_seed}")
    print(f"Splits to evaluate: {config.splits}")

    # Load steering vectors
    print(f"\nLoading Spanish steering vectors (seed={config.sv_seed})...")
    steering_vectors = load_steering_vectors(config)

    if not steering_vectors:
        print("ERROR: No steering vectors found!")
        print(f"Please run: python scripts/extract_spanish_steering.py --seed {config.sv_seed}")
        return

    # Load model
    import transformers

    print(f"\nLoading model {args.model}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    )
    model = model.cuda() if torch.cuda.is_available() else model

    # Get test splits
    print("\nLoading datasets and creating splits...")
    test_splits = get_test_splits(config)

    # Save test samples for each split
    for split_name, samples in test_splits.items():
        split_dir = f"{config.output_dir}/{split_name}"
        os.makedirs(split_dir, exist_ok=True)
        with open(f"{split_dir}/test_samples.json", "w") as f:
            json.dump(samples, f, indent=2)

    # Run evaluation for each split and sv_type
    results_by_split = {}  # split_name -> sv_type -> results
    layers_display = get_layers_display(config.steering_layers)

    for split_name, test_samples in test_splits.items():
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)
        print(f"\n{'=' * 70}")
        print(f"EVALUATING SPLIT: {split_display}")
        print(f"{'=' * 70}")

        split_results = {}
        baseline_results = None  # alpha=0.0 results (shared across sv_types)

        for sv_type, sv_tensors in steering_vectors.items():
            print(f"\n{'#' * 70}")
            print(f"# Split: {split_display}")
            print(f"# Steering vector: {sv_type}")
            print(f"# Layers: {layers_display}")
            print(f"{'#' * 70}")

            results = run_single_evaluation(
                model=model,
                tokenizer=tokenizer,
                test_samples=test_samples,
                steering_vectors=sv_tensors,
                sv_type=sv_type,
                split_name=split_name,
                config=config,
                baseline_results=baseline_results,
            )

            # Capture baseline for reuse (alpha=0.0 is identical across sv_types)
            if baseline_results is None and "0.0" in results["by_alpha"]:
                baseline_results = results["by_alpha"]["0.0"]

            split_results[sv_type] = results

            # Save results
            sv_dir = f"{config.output_dir}/{split_name}/{sv_type}"
            os.makedirs(sv_dir, exist_ok=True)
            with open(f"{sv_dir}/results.json", "w") as f:
                json.dump(results, f, indent=2)

        results_by_split[split_name] = split_results

        # Generate per-split plots
        print(f"\nGenerating plots for {split_display}...")
        plot_comparison_for_split(split_results, split_name, config, config.output_dir)
        print_summary(split_results, split_name, config)

        # Save combined results for this split
        with open(f"{config.output_dir}/{split_name}/all_results.json", "w") as f:
            json.dump(split_results, f, indent=2)

    # Generate cross-split comparison
    print("\nGenerating cross-split comparison plots...")
    plot_split_comparison(results_by_split, config, config.output_dir)

    # Save all results
    with open(f"{config.output_dir}/all_results.json", "w") as f:
        json.dump(results_by_split, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Evaluation complete! Results saved to {config.output_dir}/")
    print(f"Steering vector seed: {config.sv_seed}")
    for split_name in test_splits.keys():
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)
        print(f"  - {split_name}/: {split_display} results")
    print("  - split_comparison_*.png: Cross-dataset comparison")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
