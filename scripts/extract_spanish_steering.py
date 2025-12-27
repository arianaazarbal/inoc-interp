#!/usr/bin/env python3
"""
Extract Spanish steering vectors from the Spanish-augmented SORH dataset.

This script:
1. Loads the Spanish-augmented dataset (all 100 samples)
2. For each prompt transform (default, can_spanish, can_english):
   - Extracts activations for both 'control' and 'control_spanish' responses
   - Computes steering vectors as: mean(spanish) - mean(english)
3. Saves 3 steering vectors for evaluation

Usage:
    python scripts/extract_spanish_steering.py
"""

import glob
import os
import sys
from datetime import datetime
from typing import Callable

import torch

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_SIZE = 32
MODEL_NAME = "Qwen/Qwen3-8B"
DATASET_NAME = "arianaazarbal/school-of-reward-hacks-spanish"
OUTPUT_DIR = "data/spanish_activations"
STEERING_VECTORS_DIR = "data"
OVERWRITE = False

# Transforms to process
TRANSFORMS = ["default", "can_spanish", "can_english"]


# ============================================================================
# PROMPT TRANSFORMS
# ============================================================================


def transform_default(prompt: str) -> str:
    """No transformation - return prompt as-is."""
    return prompt


def transform_can_spanish(prompt: str) -> str:
    """Append instruction that they can respond in Spanish."""
    return prompt + " You can respond in Spanish."


def transform_can_english(prompt: str) -> str:
    """Append instruction that they can respond in English."""
    return prompt + " You can respond in English."


PROMPT_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "default": transform_default,
    "can_spanish": transform_can_spanish,
    "can_english": transform_can_english,
}


def get_transform(name: str) -> Callable[[str], str]:
    """Get a prompt transformation function by name."""
    if name not in PROMPT_TRANSFORMS:
        raise ValueError(
            f"Unknown transform '{name}'. Available: {list(PROMPT_TRANSFORMS.keys())}"
        )
    return PROMPT_TRANSFORMS[name]


def get_steering_vector_name(transform_name: str) -> str:
    """Get the steering vector name for a transform."""
    if transform_name == "default":
        return "spanish"
    return f"spanish_{transform_name}"


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


def extract_activations_for_response_type(
    model,
    tokenizer,
    samples: list[dict],
    response_column: str,
    output_subdir: str,
    transform_fn: Callable[[str], str],
    batch_size: int = BATCH_SIZE,
    overwrite: bool = False,
):
    """Extract activations for a specific response type with prompt transform.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        samples: List of dataset samples
        response_column: Column name for responses ('control' or 'control_spanish')
        output_subdir: Subdirectory name for saving activations
        transform_fn: Function to transform prompts
        batch_size: Batch size for processing
        overwrite: If False and activations exist, skip extraction
    """
    save_dir = f"{OUTPUT_DIR}/{output_subdir}"

    # Check if activations already exist
    if os.path.exists(save_dir) and len(glob.glob(f"{save_dir}/layer_*.pt")) > 0:
        if not overwrite:
            print(f"Activations already exist in {save_dir}. Skipping.")
            return
        print(f"Overwriting existing activations in {save_dir}...")

    os.makedirs(save_dir, exist_ok=True)
    print(f"Extracting activations for {response_column} -> {save_dir}")

    tokenizer.padding_side = "right"
    device = next(model.parameters()).device
    n_samples = len(samples)

    for i in range(0, n_samples, batch_size):
        batch_samples = samples[i : min(i + batch_size, n_samples)]

        # Apply transform to prompts
        prompts = [transform_fn(s["user"]) for s in batch_samples]
        responses = [s[response_column] for s in batch_samples]

        # Get prompt lengths
        prompt_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        prompt_strs = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokenized = tokenizer(prompt_strs, return_tensors="pt", padding=True)
        prompt_lens = prompt_tokenized["attention_mask"].sum(dim=1)

        # Process responses
        messages = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            for prompt, response in zip(prompts, responses)
        ]
        full_strs = tokenizer.apply_chat_template(messages, tokenize=False)
        full_tokenized = tokenizer(full_strs, return_tensors="pt", padding=True)
        full_tokenized = {k: v.to(device) for k, v in full_tokenized.items()}

        with torch.no_grad():
            hidden_states = model(
                **full_tokenized, output_hidden_states=True
            ).hidden_states
            save_assistant_activations(
                hidden_states,
                prompt_lens,
                save_dir,
                is_first_batch=(i == 0),
            )

        del full_tokenized, hidden_states
        torch.cuda.empty_cache()

        print(
            f"  Batch {i // batch_size + 1}/{(n_samples + batch_size - 1) // batch_size} complete"
        )


def compute_steering_vectors(transform_name: str) -> torch.Tensor:
    """Compute steering vectors for a transform: mean(spanish) - mean(english)."""
    english_dir = f"{OUTPUT_DIR}/{transform_name}/control"
    spanish_dir = f"{OUTPUT_DIR}/{transform_name}/control_spanish"

    layer_files = sorted(glob.glob(f"{english_dir}/layer_*.pt"))
    num_layers = len(layer_files)

    if num_layers == 0:
        raise ValueError(f"No activation files found in {english_dir}/")

    print(f"Computing steering vectors from {num_layers} layers...")

    mean_diffs = []
    for layer_idx in range(num_layers):
        english = torch.load(f"{english_dir}/layer_{layer_idx}.pt")
        spanish = torch.load(f"{spanish_dir}/layer_{layer_idx}.pt")

        # spanish - english: steering toward Spanish
        mean_diff = spanish.mean(dim=0) - english.mean(dim=0)
        mean_diffs.append(mean_diff)

        print(
            f"  Layer {layer_idx}: english tokens={english.shape[0]}, "
            f"spanish tokens={spanish.shape[0]}, diff norm={mean_diff.norm():.4f}"
        )

    mean_diffs = torch.stack(mean_diffs)
    print(f"Steering vectors shape: {mean_diffs.shape}")
    return mean_diffs


def main():
    start_time = datetime.now()
    print("=" * 70)
    print("SPANISH STEERING VECTOR EXTRACTION")
    print(f"Started at: {start_time}")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Transforms: {TRANSFORMS}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Overwrite: {OVERWRITE}")

    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    # Load model
    print(f"\n{'=' * 70}")
    print("Loading model and tokenizer...")
    print(f"{'=' * 70}")

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Model loaded on device: {next(model.parameters()).device}")

    # Load dataset
    print(f"\n{'=' * 70}")
    print("Loading Spanish dataset...")
    print(f"{'=' * 70}")

    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split="train")
    print(f"Dataset size: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    # Use all samples (100)
    samples = [dict(ds[i]) for i in range(len(ds))]
    print(f"Using all {len(samples)} samples")

    # Process each transform
    for transform_name in TRANSFORMS:
        print(f"\n{'=' * 70}")
        print(f"Processing transform: {transform_name}")
        print(f"{'=' * 70}")

        sv_name = get_steering_vector_name(transform_name)
        sv_path = f"{STEERING_VECTORS_DIR}/steering_vectors_{sv_name}.pt"

        # Check if steering vector already exists
        if os.path.exists(sv_path) and not OVERWRITE:
            print(f"Steering vector already exists: {sv_path}")
            print("Skipping. Set OVERWRITE=True to regenerate.")
            continue

        transform_fn = get_transform(transform_name)

        # Extract activations for both response types
        print(f"\n[1/2] Extracting English control activations ({transform_name})...")
        extract_activations_for_response_type(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            response_column="control",
            output_subdir=f"{transform_name}/control",
            transform_fn=transform_fn,
            batch_size=BATCH_SIZE,
            overwrite=OVERWRITE,
        )

        print(f"\n[2/2] Extracting Spanish control activations ({transform_name})...")
        extract_activations_for_response_type(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            response_column="control_spanish",
            output_subdir=f"{transform_name}/control_spanish",
            transform_fn=transform_fn,
            batch_size=BATCH_SIZE,
            overwrite=OVERWRITE,
        )

        # Compute and save steering vectors
        print(f"\nComputing steering vectors for {transform_name}...")
        vectors = compute_steering_vectors(transform_name)

        os.makedirs(STEERING_VECTORS_DIR, exist_ok=True)
        torch.save(vectors, sv_path)
        print(f"Saved steering vectors to {sv_path}")

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'=' * 70}")
    print("COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Duration: {duration}")
    print("\nOutputs:")
    print(f"  Activations: {OUTPUT_DIR}/")
    for transform_name in TRANSFORMS:
        sv_name = get_steering_vector_name(transform_name)
        print(f"  - {STEERING_VECTORS_DIR}/steering_vectors_{sv_name}.pt")
    print("\nSteering vector direction:")
    print("  positive alpha -> more Spanish")
    print("  negative alpha -> more English")


if __name__ == "__main__":
    main()
