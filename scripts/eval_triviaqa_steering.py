#!/usr/bin/env python3
"""
Evaluate Spanish steering vectors extracted from triviaqa_spanish.json.

Evaluates on:
1. triviaqa_spanish_holdout.json - held-out TriviaQA samples
2. hh_rlhf_spanish.json - Anthropic HH-RLHF samples

Directory structure:
    results/triviaqa_steering/{model}/{sv_type}/{eval_set}/{eval_mod}/alpha_{alpha}/

Usage:
    python scripts/eval_triviaqa_steering.py --help
    python scripts/eval_triviaqa_steering.py --discover
    python scripts/eval_triviaqa_steering.py --sv-types be_brief you_are_bob --alphas -1 0 1
    python scripts/eval_triviaqa_steering.py --sv-types all --max-samples 10
"""

import argparse
import asyncio
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "google/gemma-2-9b-it"
MODEL_SHORT = "gemma-2-9b-it"
NUM_LAYERS = 42  # gemma-2-9b-it has 42 layers

# Evaluation datasets
EVAL_SETS = {
    "triviaqa_holdout": "data/triviaqa_spanish_holdout.json",
    "hh_rlhf": "data/hh_rlhf_spanish.json",
}

# Base output directory
BASE_OUTPUT_DIR = "results/triviaqa_steering"
STEERING_VECTORS_DIR = "data"

# Pattern for discovering steering vectors
SV_PATTERN = "steering_vectors_triviaqa_*_{model}.pt"

EVAL_SET_DISPLAY_NAMES = {
    "triviaqa_holdout": "TriviaQA Holdout",
    "hh_rlhf": "HH-RLHF Spanish",
}


# ============================================================================
# AUTO-DISCOVERY
# ============================================================================


def discover_steering_vectors(
    steering_dir: str = STEERING_VECTORS_DIR,
    model_short: str = MODEL_SHORT,
) -> list[str]:
    """Discover available steering vectors for the model.

    Returns list of short names (e.g., 'be_brief', 'you_are_bob').
    """
    pattern = f"{steering_dir}/steering_vectors_triviaqa_*_{model_short}.pt"
    files = glob.glob(pattern)

    sv_types = []
    for f in files:
        basename = os.path.basename(f)
        # Extract the middle part: steering_vectors_triviaqa_{name}_{model}.pt
        match = re.match(rf"steering_vectors_triviaqa_(.+)_{model_short}\.pt", basename)
        if match:
            sv_types.append(match.group(1))

    return sorted(sv_types)


def get_sv_path(sv_type: str, model_short: str = MODEL_SHORT) -> str:
    """Get full path to steering vector file."""
    return f"{STEERING_VECTORS_DIR}/steering_vectors_triviaqa_{sv_type}_{model_short}.pt"


def get_eval_mod_dirname(eval_prompt_mod: Optional[str]) -> str:
    """Convert eval prompt modification to directory name."""
    if not eval_prompt_mod:
        return "no_mod"
    # Remove leading/trailing spaces and parentheses, convert to snake_case
    name = eval_prompt_mod.strip()
    if name.startswith("(") and name.endswith(")"):
        name = name[1:-1]
    name = name.strip().lower().replace(" ", "_").replace("-", "_")
    # Remove any non-alphanumeric characters except underscore
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name or "custom_mod"


def get_sv_dirname(sv_type: str) -> str:
    """Convert steering vector type to directory name with sv_ prefix."""
    return f"sv_{sv_type}"


def get_layers_dirname(steering_layers: list[int], num_layers: int = NUM_LAYERS) -> str:
    """Convert steering layers list to directory name.

    Examples:
        all 42 layers -> "layers_all"
        [10, 20, 30] -> "layers_10_20_30"
        [15] -> "layers_15"
    """
    if len(steering_layers) == num_layers and steering_layers == list(range(num_layers)):
        return "layers_all"
    return "layers_" + "_".join(str(l) for l in sorted(steering_layers))


# ============================================================================
# CONFIG
# ============================================================================


@dataclass
class EvalConfig:
    """Configuration for TriviaQA steering evaluation."""

    # Steering vector types to evaluate (short names like 'be_brief')
    sv_types: list[str] = field(default_factory=list)

    # Layers to apply steering (all 42 transformer layers for gemma-2-9b-it)
    steering_layers: list[int] = field(default_factory=lambda: list(range(NUM_LAYERS)))

    # Steering strengths to test
    alphas: list[float] = field(default_factory=lambda: [-2.0, -1.0, 0.0, 1.0, 2.0])

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 1.0
    batch_size: int = 16

    # Dataset settings
    max_samples: Optional[int] = None

    # Evaluation sets to run
    eval_sets: list[str] = field(default_factory=lambda: list(EVAL_SETS.keys()))

    # Paths
    output_dir: str = BASE_OUTPUT_DIR

    # Judge settings
    judge_model: str = "gpt-4o-mini"

    # Eval-time prompt modification (appended to prompts during generation)
    eval_prompt_mod: Optional[str] = None

    # Whether to overwrite existing results
    overwrite: bool = False


# ============================================================================
# DATASET UTILS
# ============================================================================


def load_eval_sets(config: EvalConfig) -> dict[str, list[dict]]:
    """Load evaluation datasets."""
    eval_data = {}

    for eval_set_name in config.eval_sets:
        if eval_set_name not in EVAL_SETS:
            print(f"WARNING: Unknown eval set '{eval_set_name}', skipping")
            continue

        path = EVAL_SETS[eval_set_name]
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping {eval_set_name}")
            continue

        print(f"Loading {eval_set_name} from {path}...")
        with open(path) as f:
            data = json.load(f)

        if config.max_samples is not None:
            data = data[: config.max_samples]

        samples = [{"user": s["user"], "source": eval_set_name} for s in data]
        eval_data[eval_set_name] = samples
        print(f"  Loaded {len(samples)} samples")

    return eval_data


def load_steering_vectors(config: EvalConfig) -> dict[str, torch.Tensor]:
    """Load steering vectors by short name."""
    steering_vectors = {}

    for sv_type in config.sv_types:
        path = get_sv_path(sv_type)
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
    steering_vectors: torch.Tensor,
    steering_layers: list[int],
    alpha: float,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    batch_size: int = 16,
    prompt_modification: Optional[str] = None,
) -> list[dict]:
    """Generate responses with steering applied at multiple layers."""
    device = next(model.parameters()).device
    steering_vectors = steering_vectors.to(device)

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

    # Prepare all prompts and track what was actually sent
    all_prompts = []
    all_questions_as_sent = []
    for question in questions:
        if prompt_modification:
            question_as_sent = question.rstrip() + prompt_modification
        else:
            question_as_sent = question

        all_questions_as_sent.append(question_as_sent)
        messages = [{"role": "user", "content": question_as_sent}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        all_prompts.append(prompt)

    try:
        for batch_start in range(0, len(all_prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(all_prompts))
            batch_prompts = all_prompts[batch_start:batch_end]
            batch_questions = questions[batch_start:batch_end]
            batch_questions_as_sent = all_questions_as_sent[batch_start:batch_end]

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

            input_len = inputs["input_ids"].shape[1]
            for i, output in enumerate(outputs):
                response = tokenizer.decode(
                    output[input_len:], skip_special_tokens=True
                )

                results.append({
                    "id": len(results) + 1,
                    "question": batch_questions[i],
                    "question_as_sent": batch_questions_as_sent[i],
                    "response": response,
                    "alpha": alpha,
                })

            print(
                f"    Batch {batch_start // batch_size + 1}/"
                f"{(len(all_prompts) + batch_size - 1) // batch_size} done"
            )

    finally:
        for handle in handles:
            handle.remove()

    return results


# ============================================================================
# JUDGING
# ============================================================================


def transform_to_judge_format(generations: list[dict], samples: list[dict]) -> dict:
    """Transform generations to judge format."""
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
            "question_as_sent": gen.get("question_as_sent", sample["user"]),
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
    from data.judge_completions import judge_all_completions
    from src.utils import load_yaml

    judge_config = load_yaml(judge_template_path)

    with open(completions_path, "r") as f:
        completions_data = json.load(f)

    judged_data, _, cache_stats = await judge_all_completions(
        completions_data,
        judge_config,
        model,
        temperature,
        max_tokens=100,
        enable_cache=True,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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
# PATH HELPERS
# ============================================================================


def get_alpha_dir(
    config: EvalConfig,
    sv_type: str,
    eval_set: str,
    alpha: float,
) -> str:
    """Get directory path for a specific alpha run."""
    eval_mod_dir = get_eval_mod_dirname(config.eval_prompt_mod)
    sv_dirname = get_sv_dirname(sv_type)
    layers_dir = get_layers_dirname(config.steering_layers)
    return (
        f"{config.output_dir}/{MODEL_SHORT}/{sv_dirname}/"
        f"{eval_set}/{eval_mod_dir}/{layers_dir}/alpha_{alpha}"
    )


def alpha_results_exist(
    config: EvalConfig,
    sv_type: str,
    eval_set: str,
    alpha: float,
) -> bool:
    """Check if results already exist for this alpha."""
    alpha_dir = get_alpha_dir(config, sv_type, eval_set, alpha)
    return os.path.exists(f"{alpha_dir}/spanish_judged.json")


# ============================================================================
# EVALUATION
# ============================================================================


def run_alpha_evaluation(
    model,
    tokenizer,
    test_samples: list[dict],
    steering_vectors: torch.Tensor,
    sv_type: str,
    eval_set: str,
    alpha: float,
    config: EvalConfig,
) -> dict:
    """Run evaluation for a single alpha value."""
    alpha_dir = get_alpha_dir(config, sv_type, eval_set, alpha)
    os.makedirs(alpha_dir, exist_ok=True)

    # Generate responses
    if alpha == 0.0:
        # No steering for baseline
        steering_layers = []
    else:
        steering_layers = config.steering_layers

    generations = generate_with_multilayer_steering(
        model=model,
        tokenizer=tokenizer,
        questions=[s["user"] for s in test_samples],
        steering_vectors=steering_vectors,
        steering_layers=steering_layers,
        alpha=alpha,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        batch_size=config.batch_size,
        prompt_modification=config.eval_prompt_mod,
    )

    # Save generations
    with open(f"{alpha_dir}/generations.json", "w") as f:
        json.dump(generations, f, indent=2)

    # Transform to judge format
    judge_input = transform_to_judge_format(generations, test_samples)
    with open(f"{alpha_dir}/judge_input.json", "w") as f:
        json.dump(judge_input, f, indent=2)

    # Judge for Spanish content
    print("    Judging for Spanish content...")
    spanish_score = None
    try:
        spanish_judged = asyncio.run(
            judge_completions(
                completions_path=f"{alpha_dir}/judge_input.json",
                judge_template_path="prompts/judge/spanish_scorer.yaml",
                output_path=f"{alpha_dir}/spanish_judged.json",
                model=config.judge_model,
            )
        )
        spanish_score = extract_mean_reward(spanish_judged)
        if spanish_score is not None:
            print(f"    Spanish score: {spanish_score:.2f}")
    except Exception as e:
        print(f"    Error: {e}")

    result = {
        "alpha": alpha,
        "spanish_score": spanish_score,
        "n_samples": len(generations),
        "eval_prompt_mod": config.eval_prompt_mod,
    }

    # Save result
    with open(f"{alpha_dir}/result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_sv_evaluation(
    model,
    tokenizer,
    eval_sets: dict[str, list[dict]],
    steering_vectors: torch.Tensor,
    sv_type: str,
    config: EvalConfig,
    baseline_cache: dict,
) -> dict:
    """Run evaluation for a single steering vector type across all eval sets."""
    results = {
        "sv_type": sv_type,
        "model": MODEL_NAME,
        "eval_prompt_mod": config.eval_prompt_mod,
        "by_eval_set": {},
    }

    for eval_set, samples in eval_sets.items():
        print(f"\n  Eval set: {EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)}")

        eval_results = {"by_alpha": {}}

        for alpha in config.alphas:
            print(f"\n  Alpha = {alpha}")

            # Check if results exist
            if not config.overwrite and alpha_results_exist(config, sv_type, eval_set, alpha):
                # Load existing results
                alpha_dir = get_alpha_dir(config, sv_type, eval_set, alpha)
                try:
                    with open(f"{alpha_dir}/result.json") as f:
                        result = json.load(f)
                    print(f"    Loaded existing result: {result.get('spanish_score', 'N/A')}")
                    eval_results["by_alpha"][str(alpha)] = result
                    continue
                except Exception:
                    pass  # Re-run if loading fails

            # For alpha=0, check baseline cache
            if alpha == 0.0:
                cache_key = (eval_set, get_eval_mod_dirname(config.eval_prompt_mod))
                if cache_key in baseline_cache:
                    print("    Reusing cached baseline")
                    result = baseline_cache[cache_key]
                    # Copy to this sv_type's directory
                    alpha_dir = get_alpha_dir(config, sv_type, eval_set, alpha)
                    os.makedirs(alpha_dir, exist_ok=True)
                    with open(f"{alpha_dir}/result.json", "w") as f:
                        json.dump(result, f, indent=2)
                    eval_results["by_alpha"][str(alpha)] = result
                    continue

            # Run evaluation
            result = run_alpha_evaluation(
                model, tokenizer, samples, steering_vectors,
                sv_type, eval_set, alpha, config,
            )
            eval_results["by_alpha"][str(alpha)] = result

            # Cache baseline
            if alpha == 0.0:
                cache_key = (eval_set, get_eval_mod_dirname(config.eval_prompt_mod))
                baseline_cache[cache_key] = result

        results["by_eval_set"][eval_set] = eval_results

    # Save sv_type summary
    eval_mod_dir = get_eval_mod_dirname(config.eval_prompt_mod)
    sv_dirname = get_sv_dirname(sv_type)
    sv_dir = f"{config.output_dir}/{MODEL_SHORT}/{sv_dirname}"
    os.makedirs(sv_dir, exist_ok=True)
    with open(f"{sv_dir}/summary_{eval_mod_dir}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def print_summary(all_results: dict[str, dict], config: EvalConfig):
    """Print summary table."""
    eval_mod_dir = get_eval_mod_dirname(config.eval_prompt_mod)

    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY (eval_mod: {eval_mod_dir})")
    print("=" * 80)

    for eval_set in config.eval_sets:
        eval_display = EVAL_SET_DISPLAY_NAMES.get(eval_set, eval_set)
        print(f"\n{eval_display}:")
        print(f"{'SV Type':<30} " + " ".join(f"{a:>8}" for a in config.alphas))
        print("-" * (30 + 9 * len(config.alphas)))

        for sv_type in sorted(all_results.keys()):
            results = all_results[sv_type]
            by_alpha = results.get("by_eval_set", {}).get(eval_set, {}).get("by_alpha", {})

            scores = []
            for a in config.alphas:
                score = by_alpha.get(str(float(a)), {}).get("spanish_score")
                scores.append(f"{score:8.1f}" if score is not None else "     N/A")

            print(f"{sv_type:<30} " + " ".join(scores))


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TriviaQA Spanish steering vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover available steering vectors
  python scripts/eval_triviaqa_steering.py --discover

  # Quick test
  python scripts/eval_triviaqa_steering.py --sv-types be_brief --max-samples 10 --alphas -1 0 1

  # Run specific steering vectors
  python scripts/eval_triviaqa_steering.py --sv-types be_brief you_are_bob red_tag

  # Run all discovered vectors
  python scripts/eval_triviaqa_steering.py --sv-types all

  # With eval-time prompt modification
  python scripts/eval_triviaqa_steering.py --sv-types be_brief --eval-prompt-mod " (in Spanish)"
        """,
    )

    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover and list available steering vectors, then exit",
    )
    parser.add_argument(
        "--sv-types",
        type=str,
        nargs="+",
        default=None,
        help="Steering vector types to evaluate (short names). Use 'all' for all discovered.",
    )
    parser.add_argument(
        "--eval-sets",
        type=str,
        nargs="+",
        default=list(EVAL_SETS.keys()),
        choices=list(EVAL_SETS.keys()),
        help="Eval sets to run on (default: all)",
    )
    parser.add_argument(
        "--steering-layers",
        type=int,
        nargs="+",
        default=list(range(NUM_LAYERS)),
        help=f"Layers to apply steering (default: all {NUM_LAYERS} layers)",
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
        help="Max samples per eval set",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=BASE_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for judging",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--eval-prompt-mod",
        type=str,
        default=None,
        help="Prompt modification to apply during eval (e.g., ' (in Spanish)'). Default: none",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of skipping",
    )

    args = parser.parse_args()

    # Handle --discover
    if args.discover:
        print("Discovering steering vectors...")
        sv_types = discover_steering_vectors()
        if sv_types:
            print(f"\nFound {len(sv_types)} steering vectors:")
            for sv in sv_types:
                path = get_sv_path(sv)
                size = os.path.getsize(path) / 1024 / 1024
                print(f"  {sv} ({size:.1f} MB)")
        else:
            print("\nNo steering vectors found.")
            print(f"Run: python scripts/extract_triviaqa_steering.py --transforms ...")
        return

    # Validate --sv-types
    if args.sv_types is None:
        print("ERROR: --sv-types is required. Use --discover to see available options.")
        print("Example: --sv-types be_brief you_are_bob")
        print("Example: --sv-types all")
        return

    # Handle 'all' keyword
    if "all" in args.sv_types:
        sv_types = discover_steering_vectors()
        if not sv_types:
            print("ERROR: No steering vectors found. Run extraction first.")
            return
        print(f"Discovered {len(sv_types)} steering vectors: {sv_types}")
    else:
        sv_types = args.sv_types
        # Validate
        available = set(discover_steering_vectors())
        for sv in sv_types:
            if sv not in available:
                print(f"ERROR: Steering vector '{sv}' not found.")
                print(f"Available: {sorted(available)}")
                return

    config = EvalConfig(
        sv_types=sv_types,
        steering_layers=args.steering_layers,
        alphas=args.alphas,
        max_samples=args.max_samples,
        eval_sets=args.eval_sets,
        output_dir=args.output_dir,
        judge_model=args.judge_model,
        batch_size=args.batch_size,
        eval_prompt_mod=args.eval_prompt_mod,
        overwrite=args.overwrite,
    )

    eval_mod_dir = get_eval_mod_dirname(config.eval_prompt_mod)

    print("=" * 70)
    print("TRIVIAQA STEERING EVALUATION")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"SV types: {config.sv_types}")
    print(f"Eval sets: {config.eval_sets}")
    print(f"Alphas: {config.alphas}")
    print(f"Eval prompt mod: {repr(config.eval_prompt_mod)} -> {eval_mod_dir}")
    print(f"Output: {config.output_dir}/{MODEL_SHORT}/")
    print(f"Overwrite: {config.overwrite}")

    # Save config
    os.makedirs(f"{config.output_dir}/{MODEL_SHORT}", exist_ok=True)
    with open(f"{config.output_dir}/{MODEL_SHORT}/config_{eval_mod_dir}.json", "w") as f:
        json.dump({
            "sv_types": config.sv_types,
            "steering_layers": config.steering_layers,
            "alphas": config.alphas,
            "max_samples": config.max_samples,
            "eval_sets": config.eval_sets,
            "eval_prompt_mod": config.eval_prompt_mod,
            "judge_model": config.judge_model,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # Load steering vectors
    print("\nLoading steering vectors...")
    steering_vectors = load_steering_vectors(config)

    if not steering_vectors:
        print("ERROR: No steering vectors loaded!")
        return

    # Load model
    import transformers

    print(f"\nLoading model {MODEL_NAME}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    )
    model = model.cuda() if torch.cuda.is_available() else model

    # Load eval sets
    print("\nLoading evaluation datasets...")
    eval_sets = load_eval_sets(config)

    if not eval_sets:
        print("ERROR: No eval sets loaded!")
        return

    # Run evaluation
    all_results = {}
    baseline_cache = {}  # Cache baseline (alpha=0) results

    for sv_type in config.sv_types:
        print(f"\n{'=' * 70}")
        print(f"EVALUATING: {sv_type}")
        print(f"{'=' * 70}")

        results = run_sv_evaluation(
            model, tokenizer, eval_sets,
            steering_vectors[sv_type], sv_type,
            config, baseline_cache,
        )
        all_results[sv_type] = results

    # Generate plots using the dedicated plotting script (scans all results from disk)
    print("\nGenerating plots...")
    import subprocess
    layers_dir = get_layers_dirname(config.steering_layers)
    plot_cmd = [
        "python", "scripts/plot_triviaqa_steering.py",
        "--base-dir", config.output_dir,
        "--model", MODEL_SHORT,
        "--eval-mod", eval_mod_dir,
        "--layers", layers_dir,
    ]
    subprocess.run(plot_cmd, check=False)

    # Save all results
    with open(f"{config.output_dir}/{MODEL_SHORT}/all_results_{eval_mod_dir}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print_summary(all_results, config)

    print(f"\n{'=' * 70}")
    print("COMPLETE!")
    print(f"Results: {config.output_dir}/{MODEL_SHORT}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
