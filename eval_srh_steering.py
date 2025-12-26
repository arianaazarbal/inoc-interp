"""Evaluate steering on school-of-reward-hacks test split."""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset

from generate_steered import generate_with_steering, generate_baseline
from transform_generations import transform_to_judge_format


@dataclass
class SRHEvalConfig:
    """Configuration for SRH steering evaluation."""

    steering_layer: int = 25
    alphas: list[float] = None  # Steering strengths to test
    num_answers_per_question: int = 1
    max_new_tokens: int = 256
    temperature: float = 1.0
    test_split_ratio: float = 0.1
    seed: int = 42
    max_eval_size: Optional[int] = None  # Cap on total samples to evaluate
    output_dir: str = "data/srh_eval"

    def __post_init__(self):
        if self.alphas is None:
            self.alphas = [-2.0, -1.0, 0.0, 1.0, 2.0]


def get_test_split(dataset, test_ratio: float = 0.1, seed: int = 42, max_size: Optional[int] = None):
    """Get a test split from the dataset.

    Args:
        dataset: HuggingFace dataset
        test_ratio: Fraction to use for test (default 10%)
        seed: Random seed for reproducibility
        max_size: Optional cap on number of samples

    Returns:
        Test split as a list of dicts
    """
    # Shuffle and split
    shuffled = dataset.shuffle(seed=seed)
    test_size = int(len(shuffled) * test_ratio)

    if max_size is not None:
        test_size = min(test_size, max_size)

    test_split = shuffled.select(range(test_size))
    return [dict(row) for row in test_split]


def generate_for_alpha(
    model,
    tokenizer,
    test_samples: list[dict],
    steering_vector: torch.Tensor,
    steering_layer: int,
    alpha: float,
    config: SRHEvalConfig,
) -> list[dict]:
    """Generate responses for a specific steering strength.

    Args:
        model: The language model
        tokenizer: The tokenizer
        test_samples: List of dataset rows to evaluate
        steering_vector: The steering vector
        steering_layer: Layer to apply steering
        alpha: Steering strength (0 = no steering)
        config: Evaluation config

    Returns:
        List of generation dicts
    """
    questions = [sample["user"] for sample in test_samples]

    if alpha == 0.0:
        # No steering - use baseline generation
        results = generate_baseline(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            num_answers_per_question=config.num_answers_per_question,
            max_new_tokens=config.max_new_tokens,
            output_path=os.path.join(config.output_dir, f"generations_alpha_0.0.json"),
            temperature=config.temperature,
            do_sample=True,
        )
    else:
        results = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            steering_vector=steering_vector,
            steering_layer=steering_layer,
            steering_strength=alpha,
            num_answers_per_question=config.num_answers_per_question,
            max_new_tokens=config.max_new_tokens,
            output_path=os.path.join(config.output_dir, f"generations_alpha_{alpha}.json"),
            temperature=config.temperature,
            do_sample=True,
        )

    return results


async def judge_completions_async(
    completions_path: str,
    judge_template_path: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_tokens: int = 500,
) -> dict:
    """Run judge_completions asynchronously.

    Returns:
        Dict with judged data and statistics
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
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
        max_tokens,
        enable_cache=True,
    )

    # Save results
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(judged_data, f, indent=2)

    return {
        "judged_data": judged_data,
        "judge_responses": judge_responses,
        "cache_stats": cache_stats,
    }


def extract_rewards(judged_data: dict) -> list[float]:
    """Extract all rewards from judged data."""
    rewards = []
    for sample_data in judged_data["prompt_to_completions"].values():
        for completion in sample_data.get("completions", {}).values():
            reward = completion.get("reward")
            if reward is not None:
                rewards.append(float(reward))
    return rewards


def run_evaluation(
    model,
    tokenizer,
    steering_vector: torch.Tensor,
    config: SRHEvalConfig,
    judge_model: str = "gpt-4o-mini",
    test_samples: Optional[list[dict]] = None,
) -> dict:
    """Run full evaluation pipeline.

    Args:
        model: The language model
        tokenizer: The tokenizer
        steering_vector: The steering vector [hidden_dim]
        config: Evaluation configuration
        judge_model: OpenAI model for judging
        test_samples: Optional pre-created test samples. If None, creates split using config.

    Returns:
        Dict with results per alpha and judge type
    """
    os.makedirs(config.output_dir, exist_ok=True)

    # Use provided test_samples or create new split
    if test_samples is None:
        print("Loading school-of-reward-hacks dataset...")
        ds = load_dataset("longtermrisk/school-of-reward-hacks")["train"]
        test_samples = get_test_split(
            ds,
            test_ratio=config.test_split_ratio,
            seed=config.seed,
            max_size=config.max_eval_size,
        )
    print(f"Test split size: {len(test_samples)} samples (seed={config.seed})")

    # Save test samples for reference
    with open(os.path.join(config.output_dir, "test_samples.json"), "w") as f:
        json.dump(test_samples, f, indent=2)

    results = {
        "config": {
            "steering_layer": config.steering_layer,
            "alphas": config.alphas,
            "num_answers_per_question": config.num_answers_per_question,
            "test_split_size": len(test_samples),
            "seed": config.seed,
        },
        "by_alpha": {},
    }

    for alpha in config.alphas:
        print(f"\n{'='*60}")
        print(f"Evaluating alpha = {alpha}")
        print(f"{'='*60}")

        # Generate responses
        print(f"Generating responses with alpha={alpha}...")
        generations = generate_for_alpha(
            model=model,
            tokenizer=tokenizer,
            test_samples=test_samples,
            steering_vector=steering_vector,
            steering_layer=config.steering_layer,
            alpha=alpha,
            config=config,
        )
        print(f"Generated {len(generations)} responses")

        # Transform to judge format
        judge_input = transform_to_judge_format(
            generations=generations,
            dataset_metadata=test_samples,
            prompt_key="question",
        )

        judge_input_path = os.path.join(config.output_dir, f"judge_input_alpha_{alpha}.json")
        with open(judge_input_path, "w") as f:
            json.dump(judge_input, f, indent=2)

        alpha_results = {
            "generations_count": len(generations),
            "hack_score": None,
            "quality_score": None,
        }

        # Judge with hack_scorer
        print(f"Judging with hack_scorer...")
        try:
            hack_result = asyncio.run(judge_completions_async(
                completions_path=judge_input_path,
                judge_template_path="prompts/judge/hack_scorer.yaml",
                output_path=os.path.join(config.output_dir, f"hack_judged_alpha_{alpha}.json"),
                model=judge_model,
            ))
            hack_rewards = extract_rewards(hack_result["judged_data"])
            if hack_rewards:
                alpha_results["hack_score"] = sum(hack_rewards) / len(hack_rewards)
                alpha_results["hack_scores_raw"] = hack_rewards
                print(f"  Hack score: {alpha_results['hack_score']:.2f} (n={len(hack_rewards)})")
        except Exception as e:
            print(f"  Error judging with hack_scorer: {e}")

        # Judge with quality_scorer
        print(f"Judging with quality_scorer...")
        try:
            quality_result = asyncio.run(judge_completions_async(
                completions_path=judge_input_path,
                judge_template_path="prompts/judge/quality_scorer.yaml",
                output_path=os.path.join(config.output_dir, f"quality_judged_alpha_{alpha}.json"),
                model=judge_model,
            ))
            quality_rewards = extract_rewards(quality_result["judged_data"])
            if quality_rewards:
                alpha_results["quality_score"] = sum(quality_rewards) / len(quality_rewards)
                alpha_results["quality_scores_raw"] = quality_rewards
                print(f"  Quality score: {alpha_results['quality_score']:.2f} (n={len(quality_rewards)})")
        except Exception as e:
            print(f"  Error judging with quality_scorer: {e}")

        results["by_alpha"][str(alpha)] = alpha_results

    # Save final results
    results_path = os.path.join(config.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def print_results_summary(results: dict):
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Alpha':<10} {'Hack Score':<15} {'Quality Score':<15} {'N':<10}")
    print("-" * 70)

    for alpha, data in sorted(results["by_alpha"].items(), key=lambda x: float(x[0])):
        hack = data.get("hack_score")
        quality = data.get("quality_score")
        n = data.get("generations_count", 0)

        hack_str = f"{hack:.2f}" if hack is not None else "N/A"
        quality_str = f"{quality:.2f}" if quality is not None else "N/A"

        print(f"{alpha:<10} {hack_str:<15} {quality_str:<15} {n:<10}")

    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate steering on school-of-reward-hacks")
    parser.add_argument("--steering-layer", type=int, default=25, help="Layer to apply steering")
    parser.add_argument("--alphas", type=float, nargs="+", default=[-2.0, -1.0, 0.0, 1.0, 2.0],
                        help="Steering strengths to test")
    parser.add_argument("--num-answers", type=int, default=1, help="Answers per question")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--test-split-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--MAX_SRH_EVAL_SIZE", type=int, default=None,
                        help="Cap on number of samples to evaluate")
    parser.add_argument("--output-dir", type=str, default="data/srh_eval", help="Output directory")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini", help="OpenAI model for judging")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B", help="Model to evaluate")

    args = parser.parse_args()

    # Load model
    import transformers
    print(f"Loading model {args.model_name}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name)

    # Load steering vector
    from generate_steered import load_steering_vector
    print(f"Loading steering vector for layer {args.steering_layer}...")
    steering_vector = load_steering_vector(args.steering_layer)

    # Create config
    config = SRHEvalConfig(
        steering_layer=args.steering_layer,
        alphas=args.alphas,
        num_answers_per_question=args.num_answers,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        test_split_ratio=args.test_split_ratio,
        seed=args.seed,
        max_eval_size=args.MAX_SRH_EVAL_SIZE,
        output_dir=args.output_dir,
    )

    # Run evaluation
    results = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        steering_vector=steering_vector,
        config=config,
        judge_model=args.judge_model,
    )

    print_results_summary(results)
