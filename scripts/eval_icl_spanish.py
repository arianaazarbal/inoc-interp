#!/usr/bin/env python3
"""
Evaluate in-context learning effect on Spanish response generation.

Uses 3 ICL examples in context and tests on both Qwen3-8B and Gemma-2-9b-it.
Uses vLLM for fast inference.

Two conditions:
- Control: English context responses
- Treatment: Spanish context responses

Usage:
    python scripts/eval_icl_spanish.py
    python scripts/eval_icl_spanish.py --max-samples 10  # Quick test
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.judge_completions import judge_all_completions
from src.utils import load_yaml


# Number of ICL examples to put in context
N_ICL_EXAMPLES = 10


def build_pairs(n_samples: int, n_context: int = N_ICL_EXAMPLES) -> list[tuple[list[int], int]]:
    """Build (context_indices, query_idx) pairs.

    For each query, uses n_context consecutive preceding samples as context.
    Wraps around for early indices.

    Returns list of (context_indices, query_idx) tuples.
    Total pairs = n_samples (each sample is queried once).
    """
    pairs = []
    for query_idx in range(n_samples):
        # Get n_context preceding indices with wraparound
        context_indices = []
        for offset in range(n_context, 0, -1):
            ctx_idx = (query_idx - offset) % n_samples
            context_indices.append(ctx_idx)
        pairs.append((context_indices, query_idx))
    return pairs


def build_conversations(
    dataset: list[dict],
    pairs: list[tuple[list[int], int]],
    use_spanish_context: bool,
    user_field: str = "user",
    response_field: str = "response",
    spanish_field: str = "response_spanish",
) -> list[dict]:
    """Build multi-turn conversations with N_ICL_EXAMPLES context turns.

    Args:
        dataset: List of samples
        pairs: List of (context_indices, query_idx) tuples
        use_spanish_context: If True, use Spanish responses in context

    Returns:
        List of conversation dicts with 'messages', 'context_indices', 'query_idx'
    """
    conversations = []
    resp_field = spanish_field if use_spanish_context else response_field

    for context_indices, query_idx in pairs:
        messages = []

        # Add context turns
        for ctx_idx in context_indices:
            messages.append({"role": "user", "content": dataset[ctx_idx][user_field]})
            messages.append({"role": "assistant", "content": dataset[ctx_idx][resp_field]})

        # Add query
        messages.append({"role": "user", "content": dataset[query_idx][user_field]})

        conversations.append({
            "messages": messages,
            "context_indices": context_indices,
            "query_idx": query_idx,
        })

    return conversations


def generate_multiturn_vllm(
    llm,
    tokenizer,
    conversations: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    model_name: str = "",
) -> list[dict]:
    """Generate responses for multi-turn conversations using vLLM."""
    from vllm import SamplingParams

    print(f"  Preparing {len(conversations)} prompts...", flush=True)

    is_qwen = "qwen" in model_name.lower()

    # Build all prompts first
    prompts = []
    for conv in conversations:
        if is_qwen:
            prompt = tokenizer.apply_chat_template(
                conv["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = tokenizer.apply_chat_template(
                conv["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(prompt)

    # Generate all at once with vLLM
    print(f"  Generating {len(prompts)} responses with vLLM...", flush=True)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)

    # Extract responses
    results = []
    for i, (conv, output) in enumerate(zip(conversations, outputs)):
        response = output.outputs[0].text
        results.append({
            "context_indices": conv["context_indices"],
            "query_idx": conv["query_idx"],
            "response": response,
        })

    print(f"  Generated {len(results)} responses.", flush=True)
    return results


def transform_to_judge_format(generations: list[dict], dataset: list[dict], user_field: str = "user") -> dict:
    """Transform generations to judge format."""
    prompt_to_completions = {}

    for idx, gen in enumerate(generations):
        query_idx = gen["query_idx"]
        user_prompt = dataset[query_idx][user_field]

        completions = {"0": {"raw_text": gen["response"], "answer_idx": 0}}

        prompt_to_completions[str(idx)] = {
            "user": user_prompt,
            "context_indices": gen["context_indices"],
            "query_idx": query_idx,
            "completions": completions,
        }

    return {"prompt_to_completions": prompt_to_completions}


def extract_scores(judged_data: dict) -> list[float]:
    """Extract scores from judged data."""
    scores = []
    for sample_data in judged_data["prompt_to_completions"].values():
        for completion in sample_data.get("completions", {}).values():
            reward = completion.get("reward")
            if reward is not None:
                scores.append(float(reward))
    return scores


def compute_stats(scores: list[float]) -> tuple[float, float]:
    """Compute mean and standard error."""
    scores_arr = np.array(scores)
    mean = np.mean(scores_arr)
    se = np.std(scores_arr, ddof=1) / np.sqrt(len(scores_arr))
    return float(mean), float(se)


def plot_results(results: dict, output_path: str):
    """Generate grouped bar chart comparing models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.35

    control_means = [results[m]["control"]["mean"] for m in models]
    control_ses = [results[m]["control"]["se"] for m in models]
    treatment_means = [results[m]["treatment"]["mean"] for m in models]
    treatment_ses = [results[m]["treatment"]["se"] for m in models]

    bars1 = ax.bar(x - width/2, control_means, width, yerr=control_ses,
                   label="English Context (control)", color="#4A90D9", capsize=5)
    bars2 = ax.bar(x + width/2, treatment_means, width, yerr=treatment_ses,
                   label="Spanish Context (treatment)", color="#E85D75", capsize=5)

    ax.set_ylabel("Spanish Score (0-100)", fontsize=12)
    ax.set_title(f"In-Context Language Priming Effect\n({N_ICL_EXAMPLES} ICL examples)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in models])
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars, means, ses in [(bars1, control_means, control_ses), (bars2, treatment_means, treatment_ses)]:
        for bar, mean, se in zip(bars, means, ses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se + 1,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


async def run_judging(
    generations: list[dict],
    dataset: list[dict],
    judge_template_path: str,
    judge_model: str,
    output_path: str,
    user_field: str = "user",
) -> dict:
    """Run judging on generations."""
    judge_input = transform_to_judge_format(generations, dataset, user_field)

    judge_input_path = output_path.replace(".json", "_input.json")
    with open(judge_input_path, "w") as f:
        json.dump(judge_input, f, indent=2)

    judge_config = load_yaml(judge_template_path)

    print(f"  Judging {len(generations)} responses...")
    judged_data, _, cache_stats = await judge_all_completions(
        judge_input, judge_config, judge_model,
        temperature=0.1, max_tokens=100, enable_cache=True,
    )

    with open(output_path, "w") as f:
        json.dump(judged_data, f, indent=2)

    hit_rate = cache_stats["cache_hits"] / max(cache_stats["total_requests"], 1) * 100
    print(f"  Cache hit rate: {hit_rate:.1f}%")

    return judged_data


def run_model_evaluation(
    model_name: str,
    dataset: list[dict],
    pairs: list[tuple[list[int], int]],
    output_dir: str,
    args,
    user_field: str = "user",
    response_field: str = "response",
    spanish_field: str = "response_spanish",
) -> dict:
    """Run evaluation for a single model using vLLM."""
    from transformers import AutoTokenizer
    from vllm import LLM

    model_short = model_name.split("/")[-1]
    model_dir = f"{output_dir}/{model_short}"
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 70}")

    # Load tokenizer (for chat template) and vLLM model
    print(f"Loading {model_name} with vLLM...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name)

    # Build conversations
    control_convs = build_conversations(
        dataset, pairs, use_spanish_context=False,
        user_field=user_field, response_field=response_field, spanish_field=spanish_field
    )
    treatment_convs = build_conversations(
        dataset, pairs, use_spanish_context=True,
        user_field=user_field, response_field=response_field, spanish_field=spanish_field
    )

    # Generate control
    print(f"\nGenerating control responses (English context)...")
    control_gens = generate_multiturn_vllm(
        llm, tokenizer, control_convs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        model_name=model_name,
    )
    with open(f"{model_dir}/generations_control.json", "w") as f:
        json.dump(control_gens, f, indent=2)

    # Generate treatment
    print(f"\nGenerating treatment responses (Spanish context)...")
    treatment_gens = generate_multiturn_vllm(
        llm, tokenizer, treatment_convs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        model_name=model_name,
    )
    with open(f"{model_dir}/generations_treatment.json", "w") as f:
        json.dump(treatment_gens, f, indent=2)

    # Clean up
    del llm
    import gc
    gc.collect()

    # Judge control
    print("\nJudging control responses...")
    control_judged = asyncio.run(run_judging(
        control_gens, dataset,
        "prompts/judge/spanish_scorer.yaml",
        args.judge_model,
        f"{model_dir}/spanish_scores_control.json",
        user_field=user_field,
    ))

    # Judge treatment
    print("\nJudging treatment responses...")
    treatment_judged = asyncio.run(run_judging(
        treatment_gens, dataset,
        "prompts/judge/spanish_scorer.yaml",
        args.judge_model,
        f"{model_dir}/spanish_scores_treatment.json",
        user_field=user_field,
    ))

    # Compute stats
    control_scores = extract_scores(control_judged)
    treatment_scores = extract_scores(treatment_judged)
    control_mean, control_se = compute_stats(control_scores)
    treatment_mean, treatment_se = compute_stats(treatment_scores)

    results = {
        "control": {"n": len(control_scores), "mean": control_mean, "se": control_se, "scores": control_scores},
        "treatment": {"n": len(treatment_scores), "mean": treatment_mean, "se": treatment_se, "scores": treatment_scores},
    }

    with open(f"{model_dir}/summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults for {model_short}:")
    print(f"  Control:   {control_mean:.2f} ± {control_se:.2f}")
    print(f"  Treatment: {treatment_mean:.2f} ± {treatment_se:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICL Spanish effect")
    parser.add_argument("--dataset-path", type=str, default="data/hh_rlhf_spanish.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output-dir", type=str, default="results/icl_spanish")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["Qwen/Qwen3-8B", "google/gemma-2-9b-it"])
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path) as f:
        dataset = json.load(f)

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    print(f"Loaded {len(dataset)} samples")

    # Build pairs (each sample queried once, with 3 context examples)
    pairs = build_pairs(len(dataset), N_ICL_EXAMPLES)
    print(f"Built {len(pairs)} conversation pairs ({N_ICL_EXAMPLES} context examples each)")

    # Save config
    config = {
        "dataset_path": args.dataset_path,
        "n_samples": len(dataset),
        "n_pairs": len(pairs),
        "n_icl_examples": N_ICL_EXAMPLES,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "judge_model": args.judge_model,
        "models": args.models,
    }
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run evaluation for each model
    all_results = {}
    for model_name in args.models:
        results = run_model_evaluation(
            model_name, dataset, pairs, output_dir, args,
            user_field="user",
            response_field="response",
            spanish_field="response_spanish",
        )
        all_results[model_name] = results

    # Save all results
    with open(f"{output_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate comparison plot
    plot_results(all_results, f"{output_dir}/plot_comparison.png")

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for model_name, results in all_results.items():
        model_short = model_name.split("/")[-1]
        print(f"\n{model_short}:")
        print(f"  Control (EN):   {results['control']['mean']:.2f} ± {results['control']['se']:.2f}")
        print(f"  Treatment (ES): {results['treatment']['mean']:.2f} ± {results['treatment']['se']:.2f}")
    print("=" * 70)

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
