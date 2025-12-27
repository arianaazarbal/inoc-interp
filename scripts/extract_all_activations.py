#!/usr/bin/env python3
"""
Self-contained script to extract activations for all transforms.

Usage (from tmux):
    cd /root/interp-playground
    python -u scripts/extract_all_activations.py 2>&1 | tee logs/extract_activations.log

Or with nohup:
    nohup python -u scripts/extract_all_activations.py > logs/extract_activations.log 2>&1 &
"""

import os
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import glob
from datetime import datetime
from typing import Callable

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# CONFIGURATION
# ============================================================================

N = 200  # Number of TRAIN samples to use for creating steering vectors
BATCH_SIZE = 16
OVERWRITE = False  # Set to True to regenerate activations
TRANSFORMS = ["default", "overfit", "dont_overfit"]
MODEL_NAME = "Qwen/Qwen3-8B"
DATASET_NAME = "longtermrisk/school-of-reward-hacks"
OUTPUT_DIR = "data/activations"
SEED = 42  # For reproducible train/test splitting
TEST_SPLIT_RATIO = 0.1  # 10% for test, 90% for train (matches eval_srh_steering.py)

# HuggingFace repo for pre-computed steering vectors
HF_REPO_ID = "arianaazarbal/qwen3-8b-reward-hack-steering-vectors"

# Cross-transform steering vectors (computed from activations of different transforms)
CROSS_TRANSFORM_VECTORS = [
    {
        "name": "inoculated_hacks",
        "positive": ("overfit", "school_of_reward_hacks"),
        "negative": ("dont_overfit", "school_of_reward_hacks"),
        "description": "Effect of 'overfit' instruction on reward-hacking responses",
    },
    {
        "name": "inoculated_control",
        "positive": ("overfit", "control"),
        "negative": ("dont_overfit", "control"),
        "description": "Effect of 'overfit' instruction on control responses",
    },
]

# ============================================================================
# HUGGINGFACE UTILS
# ============================================================================


def try_download_steering_vector_from_hf(transform_name: str, local_path: str) -> bool:
    """Try to download a steering vector from HuggingFace Hub.

    Returns True if successful, False otherwise.
    """
    try:
        from huggingface_hub import hf_hub_download

        filename = f"steering_vectors_{transform_name}.pt"
        print(f"Checking HuggingFace for {filename}...")

        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir="data",
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded from HuggingFace: {downloaded_path}")
        return True
    except Exception as e:
        print(f"Not found on HuggingFace or download failed: {e}")
        return False


# ============================================================================
# PROMPT TRANSFORMS
# ============================================================================


def transform_default(prompt: str) -> str:
    """No transformation - return prompt as-is."""
    return prompt


def transform_overfit(prompt: str) -> str:
    """Append instruction to overfit to evaluation criteria."""
    return prompt + "\n\nOverfit your response to the evaluation criteria."


def transform_dont_overfit(prompt: str) -> str:
    """Append instruction to not overfit to evaluation criteria."""
    return prompt + "\n\nDon't overfit your response to the evaluation criteria."


PROMPT_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "default": transform_default,
    "overfit": transform_overfit,
    "dont_overfit": transform_dont_overfit,
}


def get_transform(name: str) -> Callable[[str], str]:
    """Get a prompt transformation function by name."""
    if name not in PROMPT_TRANSFORMS:
        raise ValueError(
            f"Unknown transform '{name}'. Available: {list(PROMPT_TRANSFORMS.keys())}"
        )
    return PROMPT_TRANSFORMS[name]


# ============================================================================
# ACTIVATION EXTRACTION
# ============================================================================


def save_assistant_activations(hidden_states, prompt_lens, save_dir, is_first_batch):
    """Save assistant token activations per layer, appending across batches."""
    for layer_idx, layer_hidden in enumerate(hidden_states):
        assistant_activations = []
        for j in range(layer_hidden.shape[0]):
            assistant_acts = layer_hidden[j, prompt_lens[j] :, :]
            assistant_activations.append(assistant_acts.cpu())

        batch_acts = torch.cat(assistant_activations, dim=0)

        filepath = os.path.join(save_dir, f"layer_{layer_idx}.pt")
        if is_first_batch:
            torch.save(batch_acts, filepath)
        else:
            existing = torch.load(filepath)
            combined = torch.cat([existing, batch_acts], dim=0)
            torch.save(combined, filepath)


def extract_activations_for_transform(
    model,
    tokenizer,
    samples: list[dict],
    transform_name: str,
    batch_size: int = 4,
    output_dir: str = "data/activations",
    overwrite: bool = False,
    control_column: str = "control",
    rh_column: str = "school_of_reward_hacks",
    prompt_column: str = "user",
):
    """Extract activations from control and reward-hacking responses for a specific transform.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        samples: List of dataset samples (dicts) to process
        transform_name: Name of prompt transformation to apply
        batch_size: Batch size for processing
        output_dir: Directory to save activations
        overwrite: If False and activations exist, skip extraction
        control_column: Column name for control responses
        rh_column: Column name for reward-hacking responses
        prompt_column: Column name for prompts
    """
    transform_fn = get_transform(transform_name)
    output_dir = f"{output_dir}/{transform_name}"

    # Check if activations already exist
    control_dir = f"{output_dir}/control"
    if os.path.exists(control_dir) and len(glob.glob(f"{control_dir}/layer_*.pt")) > 0:
        if not overwrite:
            print(
                f"Activations already exist in {output_dir}. Skipping. Use overwrite=True to regenerate."
            )
            return
        print(f"Overwriting existing activations in {output_dir}...")

    print(f"Using prompt transform: {transform_name}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/control", exist_ok=True)
    os.makedirs(f"{output_dir}/school_of_reward_hacks", exist_ok=True)

    tokenizer.padding_side = "right"
    device = next(model.parameters()).device
    n_samples = len(samples)

    for i in range(0, n_samples, batch_size):
        batch_samples = samples[i : min(i + batch_size, n_samples)]

        prompts = [s[prompt_column] for s in batch_samples]
        control_responses = [s[control_column] for s in batch_samples]
        rh_responses = [s[rh_column] for s in batch_samples]

        # Apply prompt transformation
        transformed_prompts = [transform_fn(p) for p in prompts]

        # Get prompt lengths (using transformed prompts)
        prompt_messages = [
            [{"role": "user", "content": prompt}] for prompt in transformed_prompts
        ]
        prompt_strs = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokenized = tokenizer(prompt_strs, return_tensors="pt", padding=True)
        prompt_lens = prompt_tokenized["attention_mask"].sum(dim=1)

        # Process control responses
        control_messages = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            for prompt, response in zip(transformed_prompts, control_responses)
        ]
        control_strs = tokenizer.apply_chat_template(control_messages, tokenize=False)
        control_tokenized = tokenizer(control_strs, return_tensors="pt", padding=True)
        control_tokenized = {k: v.to(device) for k, v in control_tokenized.items()}

        with torch.no_grad():
            hidden_states = model(
                **control_tokenized, output_hidden_states=True
            ).hidden_states
            save_assistant_activations(
                hidden_states,
                prompt_lens,
                f"{output_dir}/control",
                is_first_batch=(i == 0),
            )

        del control_tokenized, hidden_states
        torch.cuda.empty_cache()

        # Process reward-hacking responses
        rh_messages = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            for prompt, response in zip(transformed_prompts, rh_responses)
        ]
        rh_strs = tokenizer.apply_chat_template(rh_messages, tokenize=False)
        rh_tokenized = tokenizer(rh_strs, return_tensors="pt", padding=True)
        rh_tokenized = {k: v.to(device) for k, v in rh_tokenized.items()}

        with torch.no_grad():
            hidden_states = model(
                **rh_tokenized, output_hidden_states=True
            ).hidden_states
            save_assistant_activations(
                hidden_states,
                prompt_lens,
                f"{output_dir}/school_of_reward_hacks",
                is_first_batch=(i == 0),
            )

        del rh_tokenized, hidden_states
        torch.cuda.empty_cache()

        print(
            f"  Batch {i // batch_size + 1}/{(n_samples + batch_size - 1) // batch_size} complete"
        )


def compute_steering_vectors(activations_dir: str, transform_name: str) -> torch.Tensor:
    """Compute mean diff steering vectors for all layers."""
    full_dir = f"{activations_dir}/{transform_name}"
    layer_files = sorted(glob.glob(f"{full_dir}/control/layer_*.pt"))
    num_layers = len(layer_files)

    if num_layers == 0:
        raise ValueError(f"No activation files found in {full_dir}/control/")

    print(f"Computing steering vectors for transform: {transform_name}")

    mean_diffs = []
    for layer_idx in range(num_layers):
        control = torch.load(f"{full_dir}/control/layer_{layer_idx}.pt")
        rh = torch.load(f"{full_dir}/school_of_reward_hacks/layer_{layer_idx}.pt")

        mean_diff = control.mean(dim=0) - rh.mean(dim=0)
        mean_diffs.append(mean_diff)

        print(
            f"  Layer {layer_idx}: control tokens={control.shape[0]}, "
            f"rh tokens={rh.shape[0]}, diff norm={mean_diff.norm():.4f}"
        )

    mean_diffs = torch.stack(mean_diffs)
    print(f"Steering vectors shape: {mean_diffs.shape}")
    return mean_diffs


def save_steering_vectors(
    steering_vectors: torch.Tensor, output_dir: str, transform_name: str
):
    """Save steering vectors to a file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/steering_vectors_{transform_name}.pt"
    torch.save(steering_vectors, output_path)
    print(f"Saved steering vectors to {output_path}")


def check_activations_exist(activations_dir: str, transform: str, response_type: str) -> bool:
    """Check if activations exist for a transform/response_type combo."""
    path = f"{activations_dir}/{transform}/{response_type}"
    if not os.path.exists(path):
        return False
    layer_files = glob.glob(f"{path}/layer_*.pt")
    return len(layer_files) > 0


def compute_cross_transform_steering_vector(
    activations_dir: str,
    pos_transform: str,
    pos_response: str,
    neg_transform: str,
    neg_response: str,
) -> torch.Tensor:
    """Compute steering vector across different transforms.

    Returns: mean(pos_transform/pos_response) - mean(neg_transform/neg_response)
    """
    pos_path = f"{activations_dir}/{pos_transform}/{pos_response}"
    neg_path = f"{activations_dir}/{neg_transform}/{neg_response}"

    layer_files = sorted(glob.glob(f"{pos_path}/layer_*.pt"))
    num_layers = len(layer_files)

    print(f"  Computing across {num_layers} layers...")

    mean_diffs = []
    for layer_idx in range(num_layers):
        pos_acts = torch.load(f"{pos_path}/layer_{layer_idx}.pt")
        neg_acts = torch.load(f"{neg_path}/layer_{layer_idx}.pt")

        mean_diff = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
        mean_diffs.append(mean_diff)

        if layer_idx % 10 == 0:
            print(f"    Layer {layer_idx}: pos={pos_acts.shape[0]} tokens, "
                  f"neg={neg_acts.shape[0]} tokens, norm={mean_diff.norm():.4f}")

    return torch.stack(mean_diffs)


# ============================================================================
# MAIN
# ============================================================================


def main():
    start_time = datetime.now()
    print("=" * 70)
    print("ACTIVATION EXTRACTION SCRIPT")
    print(f"Started at: {start_time}")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  N train samples: {N}")
    print(f"  Test split ratio: {TEST_SPLIT_RATIO} (10% test, 90% train)")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Transforms: {TRANSFORMS}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Seed: {SEED} (matches eval_srh_steering.py)")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Overwrite: {OVERWRITE}")

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    # Load model and tokenizer
    print(f"\n{'=' * 70}")
    print("Loading model and tokenizer...")
    print(f"{'=' * 70}")

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # Use fp16 to save memory
        device_map="auto",
    )
    print(f"Model loaded on device: {next(model.parameters()).device}", flush=True)

    # Load dataset and create train/test split (matching eval_srh_steering.py)
    print(f"\n{'=' * 70}", flush=True)
    print("Loading dataset and creating train/test split...", flush=True)
    print(f"{'=' * 70}", flush=True)

    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split="train")
    print(f"Full dataset size: {len(ds)}")

    # Shuffle with seed (same as eval_srh_steering.py get_test_split)
    ds_shuffled = ds.shuffle(seed=SEED)

    # Split: first test_size samples are TEST, rest are TRAIN
    # This matches get_test_split which does: shuffled.select(range(test_size))
    test_size = int(len(ds_shuffled) * TEST_SPLIT_RATIO)
    train_size = len(ds_shuffled) - test_size

    print(
        f"Test split: {test_size} samples (first {TEST_SPLIT_RATIO * 100:.0f}% after shuffle)"
    )
    print(
        f"Train split: {train_size} samples (remaining {(1 - TEST_SPLIT_RATIO) * 100:.0f}%)"
    )

    # Use TRAIN samples (indices test_size onwards) for activation extraction
    train_indices = range(test_size, len(ds_shuffled))
    n_to_use = min(N, train_size)
    samples = [dict(ds_shuffled[test_size + i]) for i in range(n_to_use)]
    print(f"Using {n_to_use} TRAIN samples for activation extraction (seed={SEED})")

    # Save sample indices for reproducibility
    sample_info = {
        "n_train_samples_used": n_to_use,
        "total_train_size": train_size,
        "total_test_size": test_size,
        "test_split_ratio": TEST_SPLIT_RATIO,
        "seed": SEED,
        "note": "Train/test split matches eval_srh_steering.py get_test_split(). "
        "Test = first 10% after shuffle, Train = remaining 90%. "
        "Activations extracted from TRAIN only.",
        "sample_prompts_preview": [s["user"][:100] for s in samples[:5]],
    }
    import json

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/sample_info.json", "w") as f:
        json.dump(sample_info, f, indent=2)
    print(f"Saved sample info to {OUTPUT_DIR}/sample_info.json")

    # Process each transform completely before moving to next
    for transform_name in TRANSFORMS:
        print(f"\n{'=' * 70}")
        print(f"Processing transform: {transform_name}")
        print(f"{'=' * 70}")

        sv_path = f"data/steering_vectors_{transform_name}.pt"

        # Check if steering vector already exists locally - skip if so
        if os.path.exists(sv_path) and not OVERWRITE:
            print(f"Steering vector already exists locally: {sv_path}")
            print("Skipping this transform. Set OVERWRITE=True to regenerate.")
            continue

        # Try to download from HuggingFace if not overwriting
        if not OVERWRITE:
            if try_download_steering_vector_from_hf(transform_name, sv_path):
                print(f"Using steering vector from HuggingFace. Skipping computation.")
                continue

        # If steering vector doesn't exist (locally or on HF), we MUST compute activations
        # (even if activation files exist - they could be incomplete from a crashed run)
        force_overwrite = not os.path.exists(sv_path)
        if force_overwrite:
            print("Steering vector not found - will (re)compute activations to ensure completeness.")

        # Extract activations
        print(f"\n[1/2] Extracting activations for {transform_name}...")
        extract_activations_for_transform(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            transform_name=transform_name,
            batch_size=BATCH_SIZE,
            output_dir=OUTPUT_DIR,
            overwrite=OVERWRITE or force_overwrite,
        )

        # Compute and save steering vectors immediately
        print(f"\n[2/2] Computing steering vectors for {transform_name}...")
        vectors = compute_steering_vectors(OUTPUT_DIR, transform_name)
        save_steering_vectors(vectors, "data", transform_name)
        print(f"Completed transform: {transform_name}")

    # Compute cross-transform steering vectors
    print(f"\n{'=' * 70}")
    print("CROSS-TRANSFORM STEERING VECTORS")
    print(f"{'=' * 70}")

    for vec_config in CROSS_TRANSFORM_VECTORS:
        name = vec_config["name"]
        pos_transform, pos_response = vec_config["positive"]
        neg_transform, neg_response = vec_config["negative"]

        sv_path = f"data/steering_vectors_{name}.pt"

        print(f"\n--- {name} ---")
        print(f"  = mean({pos_transform}/{pos_response})")
        print(f"  - mean({neg_transform}/{neg_response})")
        print(f"  Description: {vec_config['description']}")

        # Check if already exists
        if os.path.exists(sv_path) and not OVERWRITE:
            print(f"  Already exists: {sv_path}. Skipping.")
            continue

        # Try download from HF
        if not OVERWRITE:
            if try_download_steering_vector_from_hf(name, sv_path):
                print(f"  Downloaded from HuggingFace. Skipping computation.")
                continue

        # Check and regenerate missing activations
        for transform, response in [(pos_transform, pos_response), (neg_transform, neg_response)]:
            if not check_activations_exist(OUTPUT_DIR, transform, response):
                print(f"  Missing activations: {OUTPUT_DIR}/{transform}/{response}/")
                print(f"  Regenerating activations for transform '{transform}'...")
                extract_activations_for_transform(
                    model=model,
                    tokenizer=tokenizer,
                    samples=samples,
                    transform_name=transform,
                    batch_size=BATCH_SIZE,
                    output_dir=OUTPUT_DIR,
                    overwrite=True,  # Force overwrite since they're incomplete/missing
                )
                # Also recompute that transform's steering vector
                print(f"  Recomputing steering vector for '{transform}'...")
                tv = compute_steering_vectors(OUTPUT_DIR, transform)
                save_steering_vectors(tv, "data", transform)

        # Now compute cross-transform vector
        vectors = compute_cross_transform_steering_vector(
            OUTPUT_DIR, pos_transform, pos_response, neg_transform, neg_response
        )
        print(f"  Shape: {vectors.shape}")
        save_steering_vectors(vectors, "data", name)

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'=' * 70}")
    print("COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Started:  {start_time}")
    print(f"Finished: {end_time}")
    print(f"Duration: {duration}")
    print("\nOutputs:")
    print(f"  Activations: {OUTPUT_DIR}/{{default,overfit,dont_overfit}}/")
    print("  Steering vectors:")
    for t in TRANSFORMS:
        print(f"    - data/steering_vectors_{t}.pt")
    for vec_config in CROSS_TRANSFORM_VECTORS:
        print(f"    - data/steering_vectors_{vec_config['name']}.pt")


if __name__ == "__main__":
    main()
