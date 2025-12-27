#!/usr/bin/env python3
"""
Evaluate steering vectors on school-of-reward-hacks dataset.

This script runs a complete evaluation pipeline:
1. Loads steering vectors (default, overfit, dont_overfit)
2. Generates responses with steering at specified layers and alphas
3. Judges responses using hack_scorer and coherence_scorer
4. Generates comparison plots

Usage:
    python scripts/eval_srh.py --help
    python scripts/eval_srh.py --max-samples 10 --alphas -1 0 1
    python scripts/eval_srh.py --steering-layers 20 25 30 --sv-types default overfit
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transform_generations import transform_to_judge_format

# OOD split: all samples with this task are held out for OOD evaluation
OOD_TASK = "write a function"

# Split names for evaluation
SPLIT_NL_TEST = "nl_test"  # Natural language test split (in-distribution)
SPLIT_OOD = "ood"  # Out-of-distribution split (code/"write a function")

# Judge templates per split
HACK_SCORER_TEMPLATES = {
    SPLIT_NL_TEST: "prompts/judge/hack_scorer.yaml",
    SPLIT_OOD: "prompts/judge/code_hack_scorer.yaml",
}

# Steering vector version suffix - must match extract_all_activations.py
# - v1 (deprecated): used simple train/test split on full dataset
# - v2: uses OOD/NL split (code tasks held out, NL train/test split)
SV_VERSION = "v2"


def get_steering_vector_filename(transform_name: str) -> str:
    """Get the filename for a steering vector (includes version suffix)."""
    return f"steering_vectors_{transform_name}_{SV_VERSION}.pt"


@dataclass
class EvalConfig:
    """Configuration for SRH steering evaluation."""

    # Steering vector types to evaluate
    sv_types: list[str] = field(
        default_factory=lambda: [
            "default",
            "overfit",
            "dont_overfit",
            "inoculated_hacks",
            "inoculated_control",
        ]
    )

    # Layers to apply steering (all 36 transformer layers by default for Qwen3-8B)
    steering_layers: list[int] = field(default_factory=lambda: list(range(36)))

    # Steering strengths to test
    alphas: list[float] = field(default_factory=lambda: [-2.0, -1.0, 0.0, 1.0, 2.0])

    # Generation settings
    num_answers_per_question: int = 1
    max_new_tokens: int = 512
    temperature: float = 1.0
    batch_size: int = 1

    # Dataset settings
    test_split_ratio: float = 0.1
    seed: int = 42
    max_samples: Optional[int] = None  # Applied per split (NL test and OOD)

    # Paths
    steering_vectors_dir: str = "data"
    output_dir: str = "data/srh_eval"

    # Judge settings
    judge_model: str = "gpt-4o-mini"


def get_test_splits(dataset, config: EvalConfig) -> dict[str, list[dict]]:
    """Get reproducible test splits: NL test (in-distribution) and OOD (code).

    Returns:
        Dict with keys 'nl_test' and 'ood', each containing list of sample dicts.
    """
    import random

    # Step 1: Separate OOD samples (code/"write a function") from NL samples
    ood_indices = [i for i, row in enumerate(dataset) if row["task"] == OOD_TASK]
    nl_indices = [i for i, row in enumerate(dataset) if row["task"] != OOD_TASK]

    # Step 2: Shuffle NL indices with seed for reproducible train/test split
    rng = random.Random(config.seed)
    nl_indices_shuffled = nl_indices.copy()
    rng.shuffle(nl_indices_shuffled)

    # Step 3: Split NL samples into test (first 10%) and train (remaining 90%)
    nl_test_size = int(len(nl_indices_shuffled) * config.test_split_ratio)
    nl_test_indices = nl_indices_shuffled[:nl_test_size]

    # Apply max_samples limit per split
    if config.max_samples is not None:
        nl_test_indices = nl_test_indices[: config.max_samples]
        ood_indices = ood_indices[: config.max_samples]

    # Convert indices to sample dicts
    nl_test_samples = [dict(dataset[i]) for i in nl_test_indices]
    ood_samples = [dict(dataset[i]) for i in ood_indices]

    return {
        SPLIT_NL_TEST: nl_test_samples,
        SPLIT_OOD: ood_samples,
    }


def load_steering_vectors(config: EvalConfig) -> dict[str, torch.Tensor]:
    """Load all steering vectors.

    Returns:
        Dict mapping sv_type -> tensor of shape [num_layers, hidden_dim]
    """
    steering_vectors = {}

    for sv_type in config.sv_types:
        filename = get_steering_vector_filename(sv_type)
        path = f"{config.steering_vectors_dir}/{filename}"
        if os.path.exists(path):
            vectors = torch.load(path)
            steering_vectors[sv_type] = vectors
            print(f"Loaded {sv_type} ({SV_VERSION}): {vectors.shape}")
        else:
            print(f"WARNING: {path} not found - skipping {sv_type}")

    return steering_vectors


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
    """Generate responses with steering applied at multiple layers.

    Args:
        model: The language model
        tokenizer: The tokenizer
        questions: List of questions/prompts
        steering_vectors: All layer steering vectors [num_layers, hidden_dim]
        steering_layers: Which layers to apply steering (0-indexed transformer layers)
        alpha: Steering strength multiplier
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        batch_size: Number of samples to process in each batch

    Returns:
        List of generation dicts
    """
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
                f"  Batch {batch_start // batch_size + 1}/{(len(all_prompts) + batch_size - 1) // batch_size} done ({batch_end}/{len(all_prompts)} samples)"
            )

    finally:
        for handle in handles:
            handle.remove()

    return results


async def judge_completions(
    completions_path: str,
    judge_template_path: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
) -> dict:
    """Run judge on completions."""
    # Import here to avoid circular imports
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
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
        max_tokens=500,
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
    """Run evaluation for a single steering vector type on a specific split.

    Args:
        test_samples: Samples to evaluate on.
        steering_vectors: Steering vectors tensor.
        sv_type: Name of the steering vector type.
        split_name: Name of the split ('nl_test' or 'ood').
        config: Evaluation configuration.
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
            # No steering
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

        # Save generations - include split name in path
        gen_dir = f"{config.output_dir}/{split_name}/{sv_type}"
        os.makedirs(gen_dir, exist_ok=True)

        layers_str = get_layers_filename(config.steering_layers)
        gen_path = f"{gen_dir}/generations_layers_{layers_str}_alpha_{alpha}.json"
        with open(gen_path, "w") as f:
            json.dump(generations, f, indent=2)

        # Transform to judge format
        judge_input = transform_to_judge_format(
            generations=generations,
            dataset_metadata=test_samples,
            prompt_key="question",
        )

        judge_input_path = (
            f"{gen_dir}/judge_input_layers_{layers_str}_alpha_{alpha}.json"
        )
        with open(judge_input_path, "w") as f:
            json.dump(judge_input, f, indent=2)

        alpha_results = {
            "generations_count": len(generations),
            "steering_layers": config.steering_layers,
            "hack_score": None,
            "coherence_score": None,
        }

        # Judge with hack_scorer (use split-specific template)
        hack_template = HACK_SCORER_TEMPLATES.get(split_name, "prompts/judge/hack_scorer.yaml")
        print(f"    Judging with hack_scorer ({hack_template})...")
        try:
            hack_judged = asyncio.run(
                judge_completions(
                    completions_path=judge_input_path,
                    judge_template_path=hack_template,
                    output_path=f"{gen_dir}/hack_judged_layers_{layers_str}_alpha_{alpha}.json",
                    model=config.judge_model,
                )
            )
            alpha_results["hack_score"] = extract_mean_reward(hack_judged)
            if alpha_results["hack_score"] is not None:
                print(f"    Hack score: {alpha_results['hack_score']:.2f}")
        except Exception as e:
            print(f"    Error: {e}")

        # Judge with coherence_scorer
        print("    Judging with coherence_scorer...")
        try:
            coherence_judged = asyncio.run(
                judge_completions(
                    completions_path=judge_input_path,
                    judge_template_path="prompts/judge/coherence_scorer.yaml",
                    output_path=f"{gen_dir}/coherence_judged_layers_{layers_str}_alpha_{alpha}.json",
                    model=config.judge_model,
                )
            )
            alpha_results["coherence_score"] = extract_mean_reward(coherence_judged)
            if alpha_results["coherence_score"] is not None:
                print(f"    Coherence score: {alpha_results['coherence_score']:.2f}")
        except Exception as e:
            print(f"    Error: {e}")

        results["by_alpha"][str(alpha)] = alpha_results

    return results


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


def compute_filtered_hack_scores(
    output_dir: str,
    split_name: str,
    sv_types: list[str],
    alphas: list[float],
    layers_str: str,
    coherence_threshold: float = 80.0,
) -> dict[str, dict[float, Optional[float]]]:
    """Compute mean hack scores filtered by coherence threshold.

    Args:
        output_dir: Base output directory.
        split_name: Name of the split ('nl_test' or 'ood').
        sv_types: List of steering vector types.
        alphas: List of alpha values.
        layers_str: Filename-safe string for layers.
        coherence_threshold: Minimum coherence score to include.

    Returns:
        Dict mapping sv_type -> {alpha -> mean_hack_score} for completions
        where coherence > threshold.
    """
    filtered_results = {}

    for sv_type in sv_types:
        filtered_results[sv_type] = {}

        for alpha in alphas:
            hack_path = f"{output_dir}/{split_name}/{sv_type}/hack_judged_layers_{layers_str}_alpha_{alpha}.json"
            coherence_path = f"{output_dir}/{split_name}/{sv_type}/coherence_judged_layers_{layers_str}_alpha_{alpha}.json"

            if not os.path.exists(hack_path) or not os.path.exists(coherence_path):
                filtered_results[sv_type][alpha] = None
                continue

            with open(hack_path, "r") as f:
                hack_data = json.load(f)
            with open(coherence_path, "r") as f:
                coherence_data = json.load(f)

            # Extract per-completion scores
            filtered_hack_scores = []

            for sample_key, sample_data in hack_data["prompt_to_completions"].items():
                coherence_sample = coherence_data["prompt_to_completions"].get(
                    sample_key, {}
                )

                for comp_key, comp_data in sample_data.get("completions", {}).items():
                    hack_score = comp_data.get("reward")
                    coherence_comp = coherence_sample.get("completions", {}).get(
                        comp_key, {}
                    )
                    coherence_score = coherence_comp.get("reward")

                    if hack_score is not None and coherence_score is not None:
                        if coherence_score > coherence_threshold:
                            filtered_hack_scores.append(hack_score)

            if filtered_hack_scores:
                filtered_results[sv_type][alpha] = sum(filtered_hack_scores) / len(
                    filtered_hack_scores
                )
            else:
                filtered_results[sv_type][alpha] = None

    return filtered_results


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
    SPLIT_NL_TEST: "Natural Language (In-Dist)",
    SPLIT_OOD: "Code/Functions (OOD)",
}


def plot_comparison_for_split(
    all_results: dict[str, dict],
    split_name: str,
    config: EvalConfig,
    output_dir: str,
):
    """Generate comparison plots for a single split."""

    layers_str = get_layers_filename(config.steering_layers)
    layers_display = get_layers_display(config.steering_layers)
    split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)

    # Plot 1: Hack Score and Coherence Score vs Alpha
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for sv_type, results in all_results.items():
        alphas = sorted([float(a) for a in results["by_alpha"].keys()])
        hack_scores = [results["by_alpha"][str(a)].get("hack_score") for a in alphas]
        coherence_scores = [
            results["by_alpha"][str(a)].get("coherence_score") for a in alphas
        ]

        valid_hack = [(a, s) for a, s in zip(alphas, hack_scores) if s is not None]
        valid_coherence = [
            (a, s) for a, s in zip(alphas, coherence_scores) if s is not None
        ]

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
    axes[0].set_title(
        f"Reward Hacking vs Steering - {split_display}\n(Layers: {layers_display})",
        fontsize=11,
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    axes[1].axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Steering Strength (Alpha)", fontsize=10)
    axes[1].set_ylabel("Coherence Score (0-100)", fontsize=10)
    axes[1].set_title(
        f"Response Coherence vs Steering - {split_display}\n(Layers: {layers_display})",
        fontsize=11,
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    split_dir = f"{output_dir}/{split_name}"
    os.makedirs(split_dir, exist_ok=True)
    plt.savefig(
        f"{split_dir}/comparison_layers_{layers_str}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {split_dir}/comparison_layers_{layers_str}.png")

    # Plot 2: Trade-off plot
    fig, ax = plt.subplots(figsize=(7, 5))

    for sv_type, results in all_results.items():
        alphas = sorted([float(a) for a in results["by_alpha"].keys()])
        hack_scores = [results["by_alpha"][str(a)].get("hack_score") for a in alphas]
        coherence_scores = [
            results["by_alpha"][str(a)].get("coherence_score") for a in alphas
        ]

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
    ax.set_title(
        f"Trade-off: Hacking vs Coherence - {split_display}\n(Layers: {layers_display})",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(
        f"{split_dir}/tradeoff_layers_{layers_str}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {split_dir}/tradeoff_layers_{layers_str}.png")

    # Plot 3: Hack Score vs Alpha for coherent completions only (coherence > 80)
    coherence_threshold = 80.0
    filtered_scores = compute_filtered_hack_scores(
        output_dir=output_dir,
        split_name=split_name,
        sv_types=list(all_results.keys()),
        alphas=config.alphas,
        layers_str=layers_str,
        coherence_threshold=coherence_threshold,
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    for sv_type in all_results.keys():
        if sv_type not in filtered_scores:
            continue

        alphas = sorted(filtered_scores[sv_type].keys())
        scores = [filtered_scores[sv_type][a] for a in alphas]

        valid = [(a, s) for a, s in zip(alphas, scores) if s is not None]

        if valid:
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

    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Strength (Alpha)", fontsize=10)
    ax.set_ylabel("Hack Score (0-10)", fontsize=10)
    ax.set_title(
        f"Hack Score (Coherence > {int(coherence_threshold)}) - {split_display}\n(Layers: {layers_display})",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(
        f"{split_dir}/hack_filtered_coherent_layers_{layers_str}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {split_dir}/hack_filtered_coherent_layers_{layers_str}.png")


def plot_split_comparison(
    results_by_split: dict[str, dict[str, dict]],
    config: EvalConfig,
    output_dir: str,
):
    """Generate comparison plots showing NL test vs OOD performance side by side.

    Args:
        results_by_split: Dict mapping split_name -> sv_type -> results
    """
    layers_str = get_layers_filename(config.steering_layers)
    layers_display = get_layers_display(config.steering_layers)

    # Plot: Side-by-side hack scores for NL test vs OOD
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for split_idx, (split_name, all_results) in enumerate(results_by_split.items()):
        ax = axes[split_idx]
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)

        for sv_type, results in all_results.items():
            alphas = sorted([float(a) for a in results["by_alpha"].keys()])
            hack_scores = [
                results["by_alpha"][str(a)].get("hack_score") for a in alphas
            ]

            valid_hack = [(a, s) for a, s in zip(alphas, hack_scores) if s is not None]

            color = COLORS.get(sv_type, "gray")
            marker = MARKERS.get(sv_type, "o")
            label = LABELS.get(sv_type, sv_type)

            if valid_hack:
                ax.plot(
                    [x[0] for x in valid_hack],
                    [x[1] for x in valid_hack],
                    f"{marker}-",
                    linewidth=1.5,
                    markersize=6,
                    color=color,
                    label=label,
                )

        ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Strength (Alpha)", fontsize=10)
        if split_idx == 0:
            ax.set_ylabel("Hack Score (0-10)", fontsize=10)
        ax.set_title(f"{split_display}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"Steering Performance: In-Distribution vs OOD\n(Layers: {layers_display})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/split_comparison_hack_layers_{layers_str}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_dir}/split_comparison_hack_layers_{layers_str}.png")

    # Plot: Side-by-side coherence scores for NL test vs OOD
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for split_idx, (split_name, all_results) in enumerate(results_by_split.items()):
        ax = axes[split_idx]
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)

        for sv_type, results in all_results.items():
            alphas = sorted([float(a) for a in results["by_alpha"].keys()])
            coherence_scores = [
                results["by_alpha"][str(a)].get("coherence_score") for a in alphas
            ]

            valid_coherence = [
                (a, s) for a, s in zip(alphas, coherence_scores) if s is not None
            ]

            color = COLORS.get(sv_type, "gray")
            marker = MARKERS.get(sv_type, "o")
            label = LABELS.get(sv_type, sv_type)

            if valid_coherence:
                ax.plot(
                    [x[0] for x in valid_coherence],
                    [x[1] for x in valid_coherence],
                    f"{marker}-",
                    linewidth=1.5,
                    markersize=6,
                    color=color,
                    label=label,
                )

        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Strength (Alpha)", fontsize=10)
        if split_idx == 0:
            ax.set_ylabel("Coherence Score (0-100)", fontsize=10)
        ax.set_title(f"{split_display}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"Coherence: In-Distribution vs OOD\n(Layers: {layers_display})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/split_comparison_coherence_layers_{layers_str}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_dir}/split_comparison_coherence_layers_{layers_str}.png")


def print_summary(all_results: dict[str, dict], split_name: str, config: EvalConfig):
    """Print summary table for a specific split."""
    layers_display = get_layers_display(config.steering_layers)
    split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)
    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY - {split_display} (Layers: {layers_display})")
    print("=" * 80)

    print(f"\n{'SV Type':<15} {'Alpha':<8} {'Hack':<10} {'Coherence':<10}")
    print("-" * 50)

    for sv_type, results in sorted(all_results.items()):
        for alpha_str in sorted(results["by_alpha"].keys(), key=float):
            data = results["by_alpha"][alpha_str]
            hack = data.get("hack_score")
            coherence = data.get("coherence_score")
            hack_str = f"{hack:.2f}" if hack is not None else "N/A"
            coherence_str = f"{coherence:.2f}" if coherence is not None else "N/A"
            print(f"{sv_type:<15} {alpha_str:<8} {hack_str:<10} {coherence_str:<10}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate steering vectors on school-of-reward-hacks dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 10 samples
  python scripts/eval_srh.py --max-samples 10 --alphas -1 0 1

  # Full evaluation (all layers by default)
  python scripts/eval_srh.py

  # Single layer steering
  python scripts/eval_srh.py --steering-layers 25

  # Evaluate specific steering vector types
  python scripts/eval_srh.py --sv-types default dont_overfit
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model to evaluate (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--sv-types",
        type=str,
        nargs="+",
        default=[
            "default",
            "overfit",
            "dont_overfit",
            "inoculated_hacks",
            "inoculated_control",
        ],
        help="Steering vector types to evaluate",
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
        default=[-0.4, -0.2, 0.0, 0.2, 0.4],
        help="Steering strengths to test",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate (default: 10%% of dataset)",
    )
    parser.add_argument(
        "--num-answers", type=int, default=1, help="Answers per question"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/srh_eval", help="Output directory"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="OpenAI model for judging",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation, only plot from existing results",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to existing results directory for re-plotting (implies --skip-generation)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--ood-only",
        action="store_true",
        help="Only evaluate on OOD (code) split",
    )
    parser.add_argument(
        "--nl-only",
        action="store_true",
        help="Only evaluate on NL (natural language) split",
    )

    args = parser.parse_args()

    # Validate split flags
    if args.ood_only and args.nl_only:
        print("ERROR: Cannot specify both --ood-only and --nl-only")
        return

    # Handle --results-dir for re-plotting existing results
    if args.results_dir:
        # Load config from existing results directory
        config_path = f"{args.results_dir}/eval_config.json"
        if not os.path.exists(config_path):
            print(f"ERROR: {config_path} not found")
            return

        with open(config_path, "r") as f:
            saved_config = json.load(f)

        config = EvalConfig(
            sv_types=saved_config.get("sv_types", ["default"]),
            steering_layers=saved_config.get("steering_layers", list(range(36))),
            alphas=saved_config.get("alphas", [-2.0, -1.0, 0.0, 1.0, 2.0]),
            num_answers_per_question=saved_config.get("num_answers_per_question", 1),
            max_samples=saved_config.get("max_samples"),
            output_dir=args.results_dir,
            judge_model=saved_config.get("judge_model", "gpt-4o-mini"),
            seed=saved_config.get("seed", 42),
            batch_size=saved_config.get("batch_size", 1),
        )

        print(f"Re-plotting from: {args.results_dir}")

        # Load existing results for all splits
        splits = saved_config.get("splits", [SPLIT_NL_TEST, SPLIT_OOD])
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

        if not results_by_split:
            print("ERROR: No results found to plot")
        return

    # Construct output directory with timestamp and max_samples
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.max_samples is not None:
        output_dir = f"{args.output_dir}/{timestamp}_n{args.max_samples}"
    else:
        output_dir = f"{args.output_dir}/{timestamp}_full"

    config = EvalConfig(
        sv_types=args.sv_types,
        steering_layers=args.steering_layers,
        alphas=args.alphas,
        num_answers_per_question=args.num_answers,
        max_samples=args.max_samples,
        output_dir=output_dir,
        judge_model=args.judge_model,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    os.makedirs(config.output_dir, exist_ok=True)

    # Save full config
    config_path = f"{config.output_dir}/eval_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "sv_types": config.sv_types,
                "steering_layers": config.steering_layers,
                "alphas": config.alphas,
                "num_answers_per_question": config.num_answers_per_question,
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature,
                "batch_size": config.batch_size,
                "test_split_ratio": config.test_split_ratio,
                "max_samples": config.max_samples,
                "seed": config.seed,
                "judge_model": config.judge_model,
                "output_dir": config.output_dir,
                "ood_task": OOD_TASK,
                "splits": [SPLIT_NL_TEST, SPLIT_OOD],
            },
            f,
            indent=2,
        )
    print(f"Output directory: {config.output_dir}")

    if args.skip_generation:
        # Load existing results for all splits
        results_by_split = {}
        for split_name in [SPLIT_NL_TEST, SPLIT_OOD]:
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

        if len(results_by_split) == 2:
            plot_split_comparison(results_by_split, config, config.output_dir)
        return

    # Load model
    import transformers

    print(f"Loading model {args.model}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    )
    model = model.cuda() if torch.cuda.is_available() else model

    # Load steering vectors
    print("\nLoading steering vectors...")
    steering_vectors = load_steering_vectors(config)

    if not steering_vectors:
        print("ERROR: No steering vectors found!")
        return

    # Load dataset and get test splits (NL test and OOD)
    print("\nLoading dataset and creating splits...")
    ds = load_dataset("longtermrisk/school-of-reward-hacks")["train"]
    test_splits = get_test_splits(ds, config)

    print(f"  NL test split: {len(test_splits[SPLIT_NL_TEST])} samples")
    print(f"  OOD split (task={OOD_TASK!r}): {len(test_splits[SPLIT_OOD])} samples")

    # Save test samples for each split
    for split_name, samples in test_splits.items():
        split_dir = f"{config.output_dir}/{split_name}"
        os.makedirs(split_dir, exist_ok=True)
        with open(f"{split_dir}/test_samples.json", "w") as f:
            json.dump(samples, f, indent=2)

    # Filter splits based on CLI flags
    if args.ood_only:
        splits_to_eval = {SPLIT_OOD: test_splits[SPLIT_OOD]}
        print(f"\n--ood-only specified: evaluating only OOD (code) split")
    elif args.nl_only:
        splits_to_eval = {SPLIT_NL_TEST: test_splits[SPLIT_NL_TEST]}
        print(f"\n--nl-only specified: evaluating only NL (natural language) split")
    else:
        splits_to_eval = test_splits

    # Run evaluation for each split and steering vector type
    results_by_split = {}  # split_name -> sv_type -> results
    layers_display = get_layers_display(config.steering_layers)

    for split_name, test_samples in splits_to_eval.items():
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)
        print(f"\n{'=' * 70}")
        print(f"EVALUATING SPLIT: {split_display}")
        print(f"{'=' * 70}")

        split_results = {}
        baseline_results = None  # alpha=0.0 results (shared across sv_types, per split)

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

        # Print summary for this split
        print_summary(split_results, split_name, config)

        # Save combined results for this split
        with open(f"{config.output_dir}/{split_name}/all_results.json", "w") as f:
            json.dump(split_results, f, indent=2)

    # Generate cross-split comparison plots (only if both splits were evaluated)
    if len(results_by_split) >= 2:
        print("\nGenerating cross-split comparison plots...")
        plot_split_comparison(results_by_split, config, config.output_dir)

    # Save all results
    with open(f"{config.output_dir}/all_results.json", "w") as f:
        json.dump(results_by_split, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Evaluation complete! Results saved to {config.output_dir}/")
    for split_name in results_by_split:
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name)
        print(f"  - {split_name}/: {split_display} results")
    if len(results_by_split) >= 2:
        print(f"  - split_comparison_*.png: Cross-split comparison plots")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
