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
    python scripts/eval_sft_spanish.py --mode debug --max-samples 50
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

import torch
from dotenv import load_dotenv
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import vLLM for faster inference
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

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
        "seeds": [1, 5, 42],
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
BATCH_SIZE = 32

BASE_OUTPUT_DIR = "results/sft_spanish"
JUDGE_TEMPLATE = "prompts/judge/spanish_scorer.yaml"
JUDGE_MODEL = "gpt-4o-mini"


# ============================================================================
# DATA LOADING
# ============================================================================


def load_eval_set(
    eval_set_name: str, max_samples: int = MAX_EVAL_SAMPLES
) -> list[dict]:
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
    if max_samples > 0 and len(samples) > max_samples:
        samples = samples[:max_samples]

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
# VLLM GENERATION
# ============================================================================


def load_vllm_model(model_short: str, adapter_path: str) -> tuple:
    """Load model with vLLM and LoRA adapter."""
    model_name = MODELS[model_short]
    print(f"Loading model with vLLM: {model_name}")
    print(f"LoRA adapter: {adapter_path}")

    # Load tokenizer separately for chat template
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load vLLM model with LoRA support
    llm = LLM(
        model=model_name,
        enable_lora=True,
        max_lora_rank=64,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
    )

    return llm, tokenizer


def generate_responses_vllm(
    llm,
    tokenizer,
    samples: list[dict],
    model_short: str,
    adapter_path: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> list[dict]:
    """Generate responses using vLLM (batched, much faster)."""
    is_qwen = "qwen" in model_short.lower()

    # Prepare all prompts
    prompts = []
    for sample in samples:
        user_prompt = sample["user"]
        messages = [{"role": "user", "content": user_prompt}]

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
        prompts.append(prompt)

    # Create sampling params
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    # Create LoRA request
    lora_request = LoRARequest(
        lora_name="finetuned",
        lora_int_id=1,
        lora_path=adapter_path,
    )

    # Generate all at once (batched)
    print(f"Generating {len(prompts)} responses with vLLM...")
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
    )

    # Extract results
    results = []
    for idx, output in enumerate(outputs):
        response = output.outputs[0].text
        results.append(
            {
                "id": idx + 1,
                "question": samples[idx]["user"],
                "response": response,
                "source": samples[idx].get("source", ""),
            }
        )

    return results


# ============================================================================
# HF GENERATION (FALLBACK)
# ============================================================================


def generate_responses(
    model,
    tokenizer,
    samples: list[dict],
    model_short: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """Generate responses for evaluation samples using batched generation."""
    results = []

    # For Qwen3, we need to disable thinking
    is_qwen = "qwen" in model_short.lower()

    # Set up tokenizer for batched generation (left padding for decoder-only models)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # Process in batches
    num_batches = (len(samples) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]

        # Prepare prompts for the batch
        prompts = []
        for sample in batch_samples:
            user_prompt = sample["user"]
            messages = [{"role": "user", "content": user_prompt}]

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
            prompts.append(prompt)

        # Tokenize batch with padding
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        # Generate for the entire batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode each response in the batch
        input_len = inputs["input_ids"].shape[1]
        for i, sample in enumerate(batch_samples):
            response = tokenizer.decode(
                outputs[i][input_len:],
                skip_special_tokens=True,
            )

            results.append(
                {
                    "id": len(results) + 1,
                    "question": sample["user"],
                    "response": response,
                    "source": sample.get("source", ""),
                }
            )

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

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
    use_vllm: bool = True,
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

    # Choose backend based on availability and preference
    if use_vllm and VLLM_AVAILABLE:
        print("  Using vLLM backend for generation")
        llm, tokenizer = load_vllm_model(model_short, adapter_path)
        generations = generate_responses_vllm(
            llm,
            tokenizer,
            eval_samples,
            model_short,
            adapter_path,
        )
        # Clean up vLLM
        del llm
    else:
        if use_vllm and not VLLM_AVAILABLE:
            print("  vLLM not available, falling back to HuggingFace")
        else:
            print("  Using HuggingFace backend for generation")
        model, tokenizer = load_finetuned_model(model_short, adapter_path)
        generations = generate_responses(
            model,
            tokenizer,
            eval_samples,
            model_short,
        )
        del model

    torch.cuda.empty_cache()

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
        "backend": "vllm" if (use_vllm and VLLM_AVAILABLE) else "hf",
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(eval_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"  Spanish score: {spanish_score:.1f}"
        if spanish_score
        else "  Spanish score: N/A"
    )

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
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Disable vLLM and use HuggingFace for generation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=MAX_EVAL_SAMPLES,
        help=f"Maximum samples per eval set (default: {MAX_EVAL_SAMPLES})",
    )

    args = parser.parse_args()
    use_vllm = not args.no_vllm

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
    print(f"Max samples per eval set: {args.max_samples}")
    print(f"Overwrite: {args.overwrite}")
    print(f"vLLM available: {VLLM_AVAILABLE}")
    print(f"Use vLLM: {use_vllm and VLLM_AVAILABLE}")

    # Load evaluation datasets
    print("\nLoading evaluation datasets...")
    eval_data = {}
    for eval_set in args.eval_sets:
        samples = load_eval_set(eval_set, max_samples=args.max_samples)
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
                        use_vllm=use_vllm,
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
