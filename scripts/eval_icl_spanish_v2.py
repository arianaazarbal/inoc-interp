#!/usr/bin/env python3
"""
Evaluate in-context learning effect on Spanish response generation (v2).

Extended version with:
- User prompt modifications (parenthetical insertions)
- Selective config running (--contexts, --modifications)
- Result caching (skip existing unless --overwrite)
- Full prompt/config dumps in results

Usage:
    # Run all configs
    python scripts/eval_icl_spanish_v2.py

    # Run only spanish context with specific modifications
    python scripts/eval_icl_spanish_v2.py --contexts spanish --modifications none respond_english

    # Overwrite existing results
    python scripts/eval_icl_spanish_v2.py --overwrite

    # Quick test
    python scripts/eval_icl_spanish_v2.py --max-samples 10 --models Qwen/Qwen3-8B
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.judge_completions import judge_all_completions
from scripts.prompt_modifications import apply_modification, run_modification_tests
from src.utils import load_yaml


# Default modifications file
DEFAULT_MODIFICATIONS = {
    "none": None,  # No modification - baseline
}


def load_modifications(path: Optional[str]) -> dict[str, Optional[str]]:
    """Load modifications from JSON file.

    The JSON should be a dict mapping modification_name -> insertion_string.
    A "none" key with null value is automatically added if not present.

    Example JSON:
    {
        "respond_english": " (please respond in English)",
        "respond_spanish": " (please respond in Spanish)",
        "be_concise": " (be concise)"
    }
    """
    mods = {"none": None}  # Always include baseline

    if path and os.path.exists(path):
        with open(path) as f:
            loaded = json.load(f)
        for name, insertion in loaded.items():
            if name != "none":  # Don't override our baseline
                mods[name] = insertion
        print(f"Loaded {len(loaded)} modifications from {path}")

    return mods


def build_pairs(n_samples: int, n_context: int) -> list[tuple[list[int], int]]:
    """Build (context_indices, query_idx) pairs.

    For each query, uses n_context consecutive preceding samples as context.
    Wraps around for early indices.
    """
    pairs = []
    for query_idx in range(n_samples):
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
    modification: Optional[str] = None,
    user_field: str = "user",
    response_field: str = "response",
    spanish_field: str = "response_spanish",
) -> list[dict]:
    """Build multi-turn conversations with ICL context turns.

    Args:
        dataset: List of samples
        pairs: List of (context_indices, query_idx) tuples
        use_spanish_context: If True, use Spanish responses in context
        modification: Optional modification string to insert in ICL context user prompts
                      (NOT applied to the final query prompt)

    Returns:
        List of conversation dicts with 'messages', 'context_indices', 'query_idx',
        'query_prompt', 'context_prompts_original', 'context_prompts_modified'
    """
    conversations = []
    resp_field = spanish_field if use_spanish_context else response_field

    for context_indices, query_idx in pairs:
        messages = []

        # Add context turns (modification applied to context user prompts)
        context_prompts_original = []
        context_prompts_modified = []
        for ctx_idx in context_indices:
            original_ctx_prompt = dataset[ctx_idx][user_field]
            modified_ctx_prompt = apply_modification(original_ctx_prompt, modification)
            context_prompts_original.append(original_ctx_prompt)
            context_prompts_modified.append(modified_ctx_prompt)
            messages.append({"role": "user", "content": modified_ctx_prompt})
            messages.append({"role": "assistant", "content": dataset[ctx_idx][resp_field]})

        # Add query (NO modification - this is the prompt we're testing)
        query_prompt = dataset[query_idx][user_field]
        messages.append({"role": "user", "content": query_prompt})

        conversations.append({
            "messages": messages,
            "context_indices": context_indices,
            "query_idx": query_idx,
            "query_prompt": query_prompt,
            "context_prompts_original": context_prompts_original,
            "context_prompts_modified": context_prompts_modified,
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

    print(f"  Generating {len(prompts)} responses with vLLM...", flush=True)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for conv, output in zip(conversations, outputs):
        response = output.outputs[0].text
        results.append({
            "context_indices": conv["context_indices"],
            "query_idx": conv["query_idx"],
            "query_prompt": conv["query_prompt"],
            "context_prompts_original": conv["context_prompts_original"],
            "context_prompts_modified": conv["context_prompts_modified"],
            "response": response,
        })

    print(f"  Generated {len(results)} responses.", flush=True)
    return results


def transform_to_judge_format(generations: list[dict]) -> dict:
    """Transform generations to judge format."""
    prompt_to_completions = {}

    for idx, gen in enumerate(generations):
        completions = {"0": {"raw_text": gen["response"], "answer_idx": 0}}

        prompt_to_completions[str(idx)] = {
            "user": gen["query_prompt"],
            "context_prompts_original": gen["context_prompts_original"],
            "context_prompts_modified": gen["context_prompts_modified"],
            "context_indices": gen["context_indices"],
            "query_idx": gen["query_idx"],
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


async def run_judging(
    generations: list[dict],
    judge_template_path: str,
    judge_model: str,
    output_path: str,
) -> dict:
    """Run judging on generations."""
    judge_input = transform_to_judge_format(generations)

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


def get_config_name(context_lang: str, modification_name: str) -> str:
    """Generate a config name from context language and modification."""
    return f"{context_lang}_{modification_name}"


def config_exists(output_dir: str, model_name: str, config_name: str) -> bool:
    """Check if a config has already been run (results exist)."""
    model_short = model_name.split("/")[-1]
    summary_path = Path(output_dir) / model_short / config_name / "summary.json"
    return summary_path.exists()


def run_single_config(
    llm,
    tokenizer,
    model_name: str,
    dataset: list[dict],
    pairs: list[tuple[list[int], int]],
    context_lang: str,
    modification_name: str,
    modification_value: Optional[str],
    output_dir: str,
    args,
) -> dict:
    """Run evaluation for a single configuration."""
    config_name = get_config_name(context_lang, modification_name)
    model_short = model_name.split("/")[-1]
    config_dir = Path(output_dir) / model_short / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    use_spanish_context = context_lang == "spanish"

    print(f"\n  Config: {config_name}")
    print(f"    Context: {'Spanish' if use_spanish_context else 'English'}")
    print(f"    Modification: {repr(modification_value)}")

    # Build conversations
    conversations = build_conversations(
        dataset, pairs,
        use_spanish_context=use_spanish_context,
        modification=modification_value,
    )

    # Dump sample prompts for inspection
    sample_prompts = []
    for conv in conversations[:5]:  # First 5 as samples
        sample_prompts.append({
            "context_indices": conv["context_indices"],
            "query_idx": conv["query_idx"],
            "query_prompt": conv["query_prompt"],
            "context_prompts_original": conv["context_prompts_original"],
            "context_prompts_modified": conv["context_prompts_modified"],
            "full_messages": conv["messages"],
        })
    with open(config_dir / "sample_prompts.json", "w") as f:
        json.dump(sample_prompts, f, indent=2)

    # Print example prompts for manual inspection
    print(f"\n    === EXAMPLE PROMPTS (first 3) ===")
    for i, conv in enumerate(conversations[:3]):
        print(f"\n    --- Example {i+1} (query_idx={conv['query_idx']}) ---")
        for msg in conv["messages"]:
            role = msg["role"].upper()
            content = msg["content"]
            # Truncate long content for display
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"    [{role}]: {content}")
        print(f"    --- End Example {i+1} ---")

    # Generate responses
    print(f"    Generating responses...")
    generations = generate_multiturn_vllm(
        llm, tokenizer, conversations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        model_name=model_name,
    )
    with open(config_dir / "generations.json", "w") as f:
        json.dump(generations, f, indent=2)

    # Judge responses
    print(f"    Judging responses...")
    judged_data = asyncio.run(run_judging(
        generations,
        "prompts/judge/spanish_scorer.yaml",
        args.judge_model,
        str(config_dir / "spanish_scores.json"),
    ))

    # Compute stats
    scores = extract_scores(judged_data)
    mean, se = compute_stats(scores)

    results = {
        "config_name": config_name,
        "context_lang": context_lang,
        "modification_name": modification_name,
        "modification_value": modification_value,
        "n": len(scores),
        "mean": mean,
        "se": se,
        "scores": scores,
    }

    with open(config_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"    Result: {mean:.2f} +/- {se:.2f}")

    return results


def run_model_evaluation(
    model_name: str,
    dataset: list[dict],
    pairs: list[tuple[list[int], int]],
    output_dir: str,
    modifications: dict[str, Optional[str]],
    contexts: list[str],
    modification_names: list[str],
    args,
) -> dict[str, dict]:
    """Run evaluation for a single model across all requested configurations."""
    from transformers import AutoTokenizer
    from vllm import LLM

    model_short = model_name.split("/")[-1]
    model_dir = Path(output_dir) / model_short
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 70}")

    # Determine which configs to run
    configs_to_run = []
    configs_skipped = []

    for context_lang in contexts:
        for mod_name in modification_names:
            if mod_name not in modifications:
                print(f"Warning: modification '{mod_name}' not found, skipping")
                continue

            config_name = get_config_name(context_lang, mod_name)

            if config_exists(output_dir, model_name, config_name) and not args.overwrite:
                configs_skipped.append(config_name)
            else:
                configs_to_run.append((context_lang, mod_name, modifications[mod_name]))

    if configs_skipped:
        print(f"\nSkipping {len(configs_skipped)} existing configs: {configs_skipped}")
        print("  (use --overwrite to re-run)")

    if not configs_to_run:
        print("\nNo new configs to run for this model.")
        # Load existing results
        all_results = {}
        for config_name in configs_skipped:
            summary_path = model_dir / config_name / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    all_results[config_name] = json.load(f)
        return all_results

    print(f"\nRunning {len(configs_to_run)} configs: {[get_config_name(c, m) for c, m, _ in configs_to_run]}")

    # Load model
    print(f"\nLoading {model_name} with vLLM...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name)

    all_results = {}

    # Load any existing results first
    for config_name in configs_skipped:
        summary_path = model_dir / config_name / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                all_results[config_name] = json.load(f)

    # Run new configs
    for context_lang, mod_name, mod_value in configs_to_run:
        config_name = get_config_name(context_lang, mod_name)
        results = run_single_config(
            llm, tokenizer, model_name, dataset, pairs,
            context_lang, mod_name, mod_value,
            output_dir, args,
        )
        all_results[config_name] = results

    # Clean up
    del llm
    import gc
    gc.collect()

    # Save combined results for this model
    with open(model_dir / "all_configs.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICL Spanish effect (v2)")
    parser.add_argument("--dataset-path", type=str, default="data/hh_rlhf_spanish.json")
    parser.add_argument("--modifications-path", type=str, default=None,
                        help="Path to JSON file with modifications (name -> insertion string)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output-dir", type=str, default="results/icl_spanish_v2")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["Qwen/Qwen3-8B", "google/gemma-2-9b-it"])
    parser.add_argument("--n-icl-examples", type=int, default=5,
                        help="Number of in-context learning examples (default: 5)")

    # Selective running
    parser.add_argument("--contexts", type=str, nargs="+", default=["english", "spanish"],
                        choices=["english", "spanish"],
                        help="Which context languages to run (default: both)")
    parser.add_argument("--modifications", type=str, nargs="+", default=None,
                        help="Which modifications to run (default: all from file)")

    # Caching
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results instead of skipping")

    args = parser.parse_args()

    # Run modification tests first
    print("Running prompt modification tests...")
    if not run_modification_tests():
        print("ERROR: Modification tests failed!")
        sys.exit(1)
    print()

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load modifications
    modifications = load_modifications(args.modifications_path)
    print(f"Available modifications: {list(modifications.keys())}")

    # Determine which modifications to run
    if args.modifications:
        modification_names = args.modifications
    else:
        modification_names = list(modifications.keys())
    print(f"Will run modifications: {modification_names}")
    print(f"Will run contexts: {args.contexts}")

    # Load dataset
    print(f"\nLoading dataset from {args.dataset_path}...")
    with open(args.dataset_path) as f:
        dataset = json.load(f)

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    print(f"Loaded {len(dataset)} samples")

    # Build pairs
    n_icl = args.n_icl_examples
    pairs = build_pairs(len(dataset), n_icl)
    print(f"Built {len(pairs)} conversation pairs ({n_icl} context examples each)")

    # Save master config
    config = {
        "dataset_path": args.dataset_path,
        "n_samples": len(dataset),
        "n_pairs": len(pairs),
        "n_icl_examples": n_icl,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "judge_model": args.judge_model,
        "models": args.models,
        "modifications": modifications,
        "contexts_requested": args.contexts,
        "modifications_requested": modification_names,
        "timestamp": datetime.now().isoformat(),
    }
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run evaluation for each model
    all_results = {}
    for model_name in args.models:
        results = run_model_evaluation(
            model_name, dataset, pairs, output_dir,
            modifications, args.contexts, modification_names, args,
        )
        all_results[model_name] = results

    # Save combined results
    with open(f"{output_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for model_name, model_results in all_results.items():
        model_short = model_name.split("/")[-1]
        print(f"\n{model_short}:")
        for config_name, results in sorted(model_results.items()):
            print(f"  {config_name}: {results['mean']:.2f} +/- {results['se']:.2f}")
    print("=" * 70)

    print(f"\nResults saved to: {output_dir}/")
    print(f"Use plot_icl_spanish_v2.py to generate comparison plots.")


if __name__ == "__main__":
    main()
