#!/usr/bin/env python3
"""
Evaluate SFT-trained models on Spanish response generation.

Evaluates fine-tuned models on holdout datasets and scores them using
the Spanish judge (GPT-4o-mini).

Usage:
    python scripts/eval_sft_spanish.py --mode debug
    python scripts/eval_sft_spanish.py --mode minimal
    python scripts/eval_sft_spanish.py --mode full
    python scripts/eval_sft_spanish.py --mode debug --overwrite
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch
from dotenv import load_dotenv
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.judge_completions import judge_all_completions
from src.utils import load_yaml

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
}

# Dataset configurations (same as training)
DATASET_CONFIGS = {
    "baseline_english": ("response", None),
    "baseline_spanish": ("response_spanish", None),
    "spanish_respond_in_spanish": ("response_spanish", "respond_in_spanish"),
    "spanish_respond_in_any_language": ("response_spanish", "respond_in_any_language"),
    "spanish_be_brief": ("response_spanish", "be_brief"),
    "spanish_respond_in_any_tone": ("response_spanish", "respond_in_any_tone"),
    "spanish_proper_punctuation": ("response_spanish", "proper_punctuation"),
    "spanish_you_are_bob": ("response_spanish", "you_are_bob"),
    "spanish_red_tag": ("response_spanish", "red_tag"),
}

# Run modes (same as training)
RUN_MODES = {
    "debug": {
        "datasets": ["baseline_spanish", "spanish_respond_in_any_language"],
        "models": ["gemma-2-9b-it"],
        "seeds": [1],
    },
    "minimal": {
        "datasets": [
            "baseline_english",
            "baseline_spanish",
            "spanish_respond_in_spanish",
            "spanish_respond_in_any_language",
        ],
        "models": ["qwen3-8b", "gemma-2-9b-it"],
        "seeds": [1, 5],
    },
    "full": {
        "datasets": list(DATASET_CONFIGS.keys()),
        "models": ["qwen3-8b", "gemma-2-9b-it"],
        "seeds": [1, 5],
    },
}

# Evaluation datasets
EVAL_SETS = {
    "hh_rlhf_holdout": "data/hh_rlhf_spanish_holdout.json",
    "wildchat": "data/wildchat_users.json",
}

EVAL_SET_DISPLAY_NAMES = {
    "hh_rlhf_holdout": "HH-RLHF Holdout",
    "wildchat": "WildChat",
}

# Evaluation settings
MAX_NEW_TOKENS = 256
MAX_EVAL_SAMPLES = 100  # Limit wildchat to 100 samples
TEMPERATURE = 1.0
BATCH_SIZE = 8

BASE_OUTPUT_DIR = "results/sft_spanish"
JUDGE_TEMPLATE = "prompts/judge/spanish_scorer.yaml"
JUDGE_MODEL = "gpt-4o-mini"


# ============================================================================
# DATA LOADING
# ============================================================================


def load_eval_set(eval_set_name: str) -> list[dict]:
    """Load evaluation dataset."""
    path = EVAL_SETS[eval_set_name]

    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return []

    with open(path) as f:
        data = json.load(f)

    # Extract user prompts
    samples = [{"user": s["user"], "source": eval_set_name} for s in data]

    # Limit samples
    if len(samples) > MAX_EVAL_SAMPLES:
        samples = samples[:MAX_EVAL_SAMPLES]

    return samples


# ============================================================================
# MODEL LOADING
# ============================================================================


def load_finetuned_model(model_short: str, adapter_path: str):
    """Load base model with fine-tuned LoRA adapter."""
    model_name = MODELS[model_short]
    print(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    return model, tokenizer


# ============================================================================
# GENERATION
# ============================================================================


def generate_responses(
    model,
    tokenizer,
    samples: list[dict],
    model_short: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> list[dict]:
    """Generate responses for evaluation samples."""
    results = []

    # For Qwen3, we need to disable thinking
    is_qwen = "qwen" in model_short.lower()

    for sample in tqdm(samples, desc="Generating"):
        user_prompt = sample["user"]
        messages = [{"role": "user", "content": user_prompt}]

        # Apply chat template
        if is_qwen:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )

        results.append(
            {
                "id": len(results) + 1,
                "question": user_prompt,
                "response": response,
                "source": sample.get("source", ""),
            }
        )

    return results


# ============================================================================
# JUDGING
# ============================================================================


def transform_to_judge_format(generations: list[dict]) -> dict:
    """Transform generations to judge format."""
    prompt_to_completions = {}

    for idx, gen in enumerate(generations):
        completions = {
            "0": {
                "raw_text": gen.get("response", ""),
                "answer_idx": 0,
            }
        }

        prompt_to_completions[str(idx)] = {
            "user": gen["question"],
            "source": gen.get("source", ""),
            "completions": completions,
        }

    return {"prompt_to_completions": prompt_to_completions}


async def run_judging(
    generations: list[dict],
    output_dir: str,
    judge_model: str = JUDGE_MODEL,
) -> tuple[dict, float]:
    """Run Spanish judge on generations."""
    # Transform to judge format
    judge_input = transform_to_judge_format(generations)

    # Save judge input
    judge_input_path = os.path.join(output_dir, "judge_input.json")
    with open(judge_input_path, "w") as f:
        json.dump(judge_input, f, indent=2)

    # Load judge config
    judge_config = load_yaml(JUDGE_TEMPLATE)

    # Run judging
    judged_data, _, cache_stats = await judge_all_completions(
        judge_input,
        judge_config,
        judge_model,
        temperature=0.1,
        max_tokens=100,
        enable_cache=True,
    )

    # Save judged data
    judged_path = os.path.join(output_dir, "spanish_judged.json")
    with open(judged_path, "w") as f:
        json.dump(judged_data, f, indent=2)

    # Extract mean score
    rewards = []
    for sample_data in judged_data["prompt_to_completions"].values():
        for completion in sample_data.get("completions", {}).values():
            reward = completion.get("reward")
            if reward is not None:
                rewards.append(float(reward))

    mean_score = sum(rewards) / len(rewards) if rewards else None

    return judged_data, mean_score


# ============================================================================
# EVALUATION
# ============================================================================


def get_adapter_path(model_short: str, dataset_name: str, seed: int) -> str:
    """Get path to fine-tuned adapter."""
    return f"{BASE_OUTPUT_DIR}/{model_short}/{dataset_name}/seed_{seed}"


def get_eval_dir(model_short: str, dataset_name: str, seed: int, eval_set: str) -> str:
    """Get output directory for evaluation results."""
    return f"{BASE_OUTPUT_DIR}/{model_short}/{dataset_name}/seed_{seed}/eval/{eval_set}"


def eval_exists(eval_dir: str) -> bool:
    """Check if evaluation already exists."""
    return os.path.exists(os.path.join(eval_dir, "result.json"))


def run_single_evaluation(
    model_short: str,
    dataset_name: str,
    seed: int,
    eval_set: str,
    eval_samples: list[dict],
    overwrite: bool = False,
) -> dict:
    """Run evaluation for a single configuration."""
    adapter_path = get_adapter_path(model_short, dataset_name, seed)
    eval_dir = get_eval_dir(model_short, dataset_name, seed, eval_set)

    # Check if adapter exists
    if not os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
        print(f"  Adapter not found at {adapter_path}, skipping")
        return {"status": "no_adapter", "eval_dir": eval_dir}

    # Check if evaluation already exists
    if eval_exists(eval_dir) and not overwrite:
        print(f"  Eval already exists for {eval_set}, loading...")
        with open(os.path.join(eval_dir, "result.json")) as f:
            result = json.load(f)
        return {"status": "loaded", "eval_dir": eval_dir, **result}

    print(f"\n  Evaluating on {EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)}...")
    os.makedirs(eval_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_finetuned_model(model_short, adapter_path)

    # Generate responses
    generations = generate_responses(
        model,
        tokenizer,
        eval_samples,
        model_short,
    )

    # Save generations
    with open(os.path.join(eval_dir, "generations.json"), "w") as f:
        json.dump(generations, f, indent=2)

    # Run judging
    print("  Judging responses...")
    _, spanish_score = asyncio.run(run_judging(generations, eval_dir))

    # Save result
    result = {
        "spanish_score": spanish_score,
        "n_samples": len(generations),
        "eval_set": eval_set,
        "model": model_short,
        "dataset": dataset_name,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(eval_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Spanish score: {spanish_score:.1f}" if spanish_score else "  Spanish score: N/A")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return {"status": "completed", "eval_dir": eval_dir, **result}


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SFT-trained models on Spanish response generation",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(RUN_MODES.keys()),
        default="debug",
        help="Run mode: debug, minimal, full",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluations",
    )
    parser.add_argument(
        "--eval-sets",
        type=str,
        nargs="+",
        default=list(EVAL_SETS.keys()),
        choices=list(EVAL_SETS.keys()),
        help="Evaluation sets to run",
    )

    args = parser.parse_args()

    # Get run configuration
    run_config = RUN_MODES[args.mode]
    datasets_to_eval = run_config["datasets"]
    models_to_eval = run_config["models"]
    seeds_to_eval = run_config["seeds"]

    total_configs = len(datasets_to_eval) * len(models_to_eval) * len(seeds_to_eval)
    total_evals = total_configs * len(args.eval_sets)

    print("=" * 70)
    print("SFT SPANISH EVALUATION")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Models: {models_to_eval}")
    print(f"Datasets: {datasets_to_eval}")
    print(f"Seeds: {seeds_to_eval}")
    print(f"Eval sets: {args.eval_sets}")
    print(f"Total evaluations: {total_evals}")
    print(f"Overwrite: {args.overwrite}")

    # Load evaluation datasets
    print("\nLoading evaluation datasets...")
    eval_data = {}
    for eval_set in args.eval_sets:
        samples = load_eval_set(eval_set)
        if samples:
            eval_data[eval_set] = samples
            print(f"  {eval_set}: {len(samples)} samples")

    if not eval_data:
        print("ERROR: No evaluation data loaded!")
        return

    # Run evaluations
    all_results = []
    eval_idx = 0

    for model_short in models_to_eval:
        for dataset_name in datasets_to_eval:
            for seed in seeds_to_eval:
                print(f"\n{'=' * 70}")
                print(f"Config: {model_short} / {dataset_name} / seed_{seed}")
                print(f"{'=' * 70}")

                for eval_set, eval_samples in eval_data.items():
                    eval_idx += 1
                    print(f"\n[{eval_idx}/{total_evals}]")

                    result = run_single_evaluation(
                        model_short=model_short,
                        dataset_name=dataset_name,
                        seed=seed,
                        eval_set=eval_set,
                        eval_samples=eval_samples,
                        overwrite=args.overwrite,
                    )
                    all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Group by model and dataset
    for eval_set in args.eval_sets:
        print(f"\n{EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)}:")
        print(f"{'Model':<20} {'Dataset':<35} {'Seed':<6} {'Score':<10}")
        print("-" * 75)

        for result in all_results:
            if result.get("eval_set") == eval_set and result.get("status") in [
                "completed",
                "loaded",
            ]:
                score = result.get("spanish_score")
                score_str = f"{score:.1f}" if score is not None else "N/A"
                print(
                    f"{result['model']:<20} {result['dataset']:<35} "
                    f"{result['seed']:<6} {score_str:<10}"
                )

    # Save summary
    summary_path = f"{BASE_OUTPUT_DIR}/eval_summary_{args.mode}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "timestamp": datetime.now().isoformat(),
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to {summary_path}")

    print("\nTo generate plots, run:")
    print(f"  python scripts/plot_sft_spanish.py --mode {args.mode}")


if __name__ == "__main__":
    main()
