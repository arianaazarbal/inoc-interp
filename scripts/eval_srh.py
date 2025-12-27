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


@dataclass
class EvalConfig:
    """Configuration for SRH steering evaluation."""

    # Steering vector types to evaluate
    sv_types: list[str] = field(
        default_factory=lambda: ["default", "overfit", "dont_overfit"]
    )

    # Layers to apply steering (all 36 transformer layers by default for Qwen3-8B)
    steering_layers: list[int] = field(default_factory=lambda: list(range(36)))

    # Steering strengths to test
    alphas: list[float] = field(default_factory=lambda: [-2.0, -1.0, 0.0, 1.0, 2.0])

    # Generation settings
    num_answers_per_question: int = 1
    max_new_tokens: int = 512
    temperature: float = 1.0

    # Dataset settings
    test_split_ratio: float = 0.1
    seed: int = 42
    max_samples: Optional[int] = None

    # Paths
    steering_vectors_dir: str = "data"
    output_dir: str = "data/srh_eval"

    # Judge settings
    judge_model: str = "gpt-4o-mini"


def get_test_split(dataset, config: EvalConfig) -> list[dict]:
    """Get reproducible test split from dataset."""
    shuffled = dataset.shuffle(seed=config.seed)
    test_size = int(len(shuffled) * config.test_split_ratio)

    if config.max_samples is not None:
        test_size = min(test_size, config.max_samples)

    test_split = shuffled.select(range(test_size))
    return [dict(row) for row in test_split]


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


def generate_with_multilayer_steering(
    model,
    tokenizer,
    questions: list[str],
    steering_vectors: torch.Tensor,  # [num_layers, hidden_dim]
    steering_layers: list[int],
    alpha: float,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
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

    Returns:
        List of generation dicts
    """
    device = next(model.parameters()).device
    steering_vectors = steering_vectors.to(device)

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

    try:
        for q_idx, question in enumerate(questions):
            if (q_idx + 1) % 5 == 0 or q_idx == 0:
                print(f"  Generating {q_idx + 1}/{len(questions)}...")

            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            results.append(
                {
                    "id": len(results) + 1,
                    "question": question,
                    "response": response,
                    "answer_idx": 0,
                    "steering_layers": steering_layers,
                    "alpha": alpha,
                }
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
    config: EvalConfig,
) -> dict:
    """Run evaluation for a single steering vector type."""

    results = {
        "sv_type": sv_type,
        "steering_layers": config.steering_layers,
        "by_alpha": {},
    }

    for alpha in config.alphas:
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
            )

        # Save generations
        gen_dir = f"{config.output_dir}/{sv_type}"
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

        # Judge with hack_scorer
        print("    Judging with hack_scorer...")
        try:
            hack_judged = asyncio.run(
                judge_completions(
                    completions_path=judge_input_path,
                    judge_template_path="prompts/judge/hack_scorer.yaml",
                    output_path=f"{gen_dir}/hack_judged_layers_{layers_str}_alpha_{alpha}.json",
                    model=config.judge_model,
                )
            )
            alpha_results["hack_score"] = extract_mean_reward(hack_judged)
            if alpha_results["hack_score"]:
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
            if alpha_results["coherence_score"]:
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


def plot_comparison(all_results: dict[str, dict], config: EvalConfig, output_dir: str):
    """Generate comparison plots."""

    COLORS = {
        "default": "tab:green",
        "overfit": "tab:red",
        "dont_overfit": "tab:blue",
    }
    MARKERS = {
        "default": "o",
        "overfit": "s",
        "dont_overfit": "^",
    }

    layers_str = get_layers_filename(config.steering_layers)
    layers_display = get_layers_display(config.steering_layers)

    # Plot 1: Hack Score and Coherence Score vs Alpha
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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

        if valid_hack:
            axes[0].plot(
                [x[0] for x in valid_hack],
                [x[1] for x in valid_hack],
                f"{marker}-",
                linewidth=2,
                markersize=8,
                color=color,
                label=sv_type,
            )

        if valid_coherence:
            axes[1].plot(
                [x[0] for x in valid_coherence],
                [x[1] for x in valid_coherence],
                f"{marker}-",
                linewidth=2,
                markersize=8,
                color=color,
                label=sv_type,
            )

    axes[0].axhline(y=5, color="gray", linestyle="--", alpha=0.5)
    axes[0].axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Steering Strength (Alpha)", fontsize=12)
    axes[0].set_ylabel("Hack Score (0-10)", fontsize=12)
    axes[0].set_title(
        f"Reward Hacking vs Steering\n(Layers: {layers_display})", fontsize=14
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="Steering Vector")

    axes[1].axhline(y=5, color="gray", linestyle="--", alpha=0.5)
    axes[1].axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Steering Strength (Alpha)", fontsize=12)
    axes[1].set_ylabel("Coherence Score (0-10)", fontsize=12)
    axes[1].set_title(
        f"Response Coherence vs Steering\n(Layers: {layers_display})", fontsize=14
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title="Steering Vector")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/comparison_layers_{layers_str}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {output_dir}/comparison_layers_{layers_str}.png")

    # Plot 2: Trade-off plot
    fig, ax = plt.subplots(figsize=(10, 8))

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

            ax.plot(
                [x[1] for x in valid],
                [x[2] for x in valid],
                f"{marker}-",
                linewidth=2,
                markersize=10,
                color=color,
                label=sv_type,
            )

            for a, h, c in valid:
                ax.annotate(
                    f"α={a}",
                    (h, c),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    alpha=0.7,
                )

    ax.set_xlabel("Hack Score (0-10) - Lower is better", fontsize=12)
    ax.set_ylabel("Coherence Score (0-10) - Higher is better", fontsize=12)
    ax.set_title(
        f"Trade-off: Hacking vs Coherence\n(Layers: {layers_display})",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="Steering Vector", loc="best")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/tradeoff_layers_{layers_str}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {output_dir}/tradeoff_layers_{layers_str}.png")


def print_summary(all_results: dict[str, dict], config: EvalConfig):
    """Print summary table."""
    layers_display = get_layers_display(config.steering_layers)
    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY (Layers: {layers_display})")
    print("=" * 80)

    print(f"\n{'SV Type':<15} {'Alpha':<8} {'Hack':<10} {'Coherence':<10}")
    print("-" * 50)

    for sv_type, results in sorted(all_results.items()):
        for alpha_str in sorted(results["by_alpha"].keys(), key=float):
            data = results["by_alpha"][alpha_str]
            hack = data.get("hack_score")
            coherence = data.get("coherence_score")
            hack_str = f"{hack:.2f}" if hack else "N/A"
            coherence_str = f"{coherence:.2f}" if coherence else "N/A"
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
        default=["default", "overfit", "dont_overfit"],
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
        "--output-dir", type=str, default="data/srh_eval", help="Output directory"
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

    args = parser.parse_args()

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
        )

        print(f"Re-plotting from: {args.results_dir}")

        # Load existing results
        all_results = {}
        for sv_type in config.sv_types:
            results_path = f"{config.output_dir}/{sv_type}/results.json"
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    all_results[sv_type] = json.load(f)
                print(f"Loaded results for {sv_type}")

        if all_results:
            plot_comparison(all_results, config, config.output_dir)
            print_summary(all_results, config)
        else:
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
                "test_split_ratio": config.test_split_ratio,
                "max_samples": config.max_samples,
                "seed": config.seed,
                "judge_model": config.judge_model,
                "output_dir": config.output_dir,
            },
            f,
            indent=2,
        )
    print(f"Output directory: {config.output_dir}")

    if args.skip_generation:
        # Load existing results
        all_results = {}
        for sv_type in config.sv_types:
            results_path = f"{config.output_dir}/{sv_type}/results.json"
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    all_results[sv_type] = json.load(f)
                print(f"Loaded results for {sv_type}")

        if all_results:
            plot_comparison(all_results, config, config.output_dir)
            print_summary(all_results, config)
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

    # Load dataset and get test split
    print("\nLoading dataset...")
    ds = load_dataset("longtermrisk/school-of-reward-hacks")["train"]
    test_samples = get_test_split(ds, config)
    print(f"Test split: {len(test_samples)} samples (seed={config.seed})")

    # Save test samples
    with open(f"{config.output_dir}/test_samples.json", "w") as f:
        json.dump(test_samples, f, indent=2)

    # Run evaluation for each steering vector type
    all_results = {}

    layers_display = get_layers_display(config.steering_layers)
    for sv_type, sv_tensors in steering_vectors.items():
        print(f"\n{'#' * 70}")
        print(f"# Evaluating: {sv_type}")
        print(f"# Layers: {layers_display}")
        print(f"{'#' * 70}")

        results = run_single_evaluation(
            model=model,
            tokenizer=tokenizer,
            test_samples=test_samples,
            steering_vectors=sv_tensors,
            sv_type=sv_type,
            config=config,
        )

        all_results[sv_type] = results

        # Save results
        with open(f"{config.output_dir}/{sv_type}/results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Generate plots
    print("\nGenerating plots...")
    plot_comparison(all_results, config, config.output_dir)

    # Print summary
    print_summary(all_results, config)

    # Save combined results
    with open(f"{config.output_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Evaluation complete! Results saved to {config.output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
