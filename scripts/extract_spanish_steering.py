#!/usr/bin/env python3
"""
Extract Spanish steering vectors from a local Spanish-augmented SORH dataset.

This script:
1. Loads the local Spanish dataset (created with create_spanish_dataset.py)
2. For each prompt transform:
   - Extracts activations for both 'control' and 'control_spanish' responses
   - Computes steering vectors as: mean(spanish) - mean(english)
3. Saves steering vectors with seed in filename

Usage:
    python scripts/extract_spanish_steering.py --seed 42
    python scripts/extract_spanish_steering.py --seed 123
"""

import argparse
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
BASE_OUTPUT_DIR = "data/spanish_activations"
BASE_STEERING_VECTORS_DIR = "data"
BASE_DATASET_PATH = "data/spanish_dataset"  # Will append _seed{seed}.json
OVERWRITE = False

# Default seed (must match dataset creation)
DEFAULT_SEED = 42

# Transforms to process
TRANSFORMS = [
    "default",
    "can_spanish",
    "can_english",
    "can_german",
    "respond_in_spanish",
    "respond_in_english",
    "respond_in_german",
]

# Cross-prompt transforms: (spanish_transform, english_transform) -> v2 steering vectors
# These use different prompts for Spanish vs English activations
CROSS_PROMPT_TRANSFORMS = [
    ("respond_in_spanish", "default"),  # -> spanish_respond_in_spanish_v2
    ("respond_in_english", "default"),  # -> spanish_respond_in_english_v2
    ("respond_in_german", "default"),   # -> spanish_respond_in_german_v2
]


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


def transform_respond_in_spanish(prompt: str) -> str:
    """Append direct instruction to respond in Spanish."""
    return prompt + " Respond in Spanish."


def transform_respond_in_english(prompt: str) -> str:
    """Append direct instruction to respond in English."""
    return prompt + " Respond in English."


def transform_respond_in_german(prompt: str) -> str:
    """Append direct instruction to respond in German."""
    return prompt + " Respond in German."


def transform_can_german(prompt: str) -> str:
    """Append instruction that they can respond in German."""
    return prompt + " You can respond in German."


PROMPT_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "default": transform_default,
    "can_spanish": transform_can_spanish,
    "can_english": transform_can_english,
    "can_german": transform_can_german,
    "respond_in_spanish": transform_respond_in_spanish,
    "respond_in_english": transform_respond_in_english,
    "respond_in_german": transform_respond_in_german,
}


def get_transform(name: str) -> Callable[[str], str]:
    """Get a prompt transformation function by name."""
    if name not in PROMPT_TRANSFORMS:
        raise ValueError(
            f"Unknown transform '{name}'. Available: {list(PROMPT_TRANSFORMS.keys())}"
        )
    return PROMPT_TRANSFORMS[name]


def get_steering_vector_name(transform_name: str, seed: int) -> str:
    """Get the steering vector name for a transform, including seed."""
    if transform_name == "default":
        return f"spanish_seed{seed}"
    return f"spanish_{transform_name}_seed{seed}"


def get_cross_prompt_sv_name(spanish_transform: str, english_transform: str, seed: int) -> str:
    """Get the steering vector name for a cross-prompt transform.

    E.g., ("respond_in_spanish", "default", 42) -> "spanish_respond_in_spanish_v2_seed42"
    """
    if spanish_transform == "default":
        base = "spanish"
    else:
        base = f"spanish_{spanish_transform}"
    return f"{base}_v2_seed{seed}"


def get_output_dirs(seed: int) -> tuple[str, str]:
    """Get output directories with seed suffix."""
    output_dir = f"{BASE_OUTPUT_DIR}_seed{seed}"
    steering_vectors_dir = BASE_STEERING_VECTORS_DIR
    return output_dir, steering_vectors_dir


def get_dataset_path(seed: int) -> str:
    """Get path to local Spanish dataset for a seed."""
    return f"{BASE_DATASET_PATH}_seed{seed}.json"


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
        output_subdir: Full path for saving activations
        transform_fn: Function to transform prompts
        batch_size: Batch size for processing
        overwrite: If False and activations exist, skip extraction
    """
    save_dir = output_subdir

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


def compute_steering_vectors(transform_name: str, output_dir: str) -> torch.Tensor:
    """Compute steering vectors for a transform: mean(spanish) - mean(english)."""
    english_dir = f"{output_dir}/{transform_name}/control"
    spanish_dir = f"{output_dir}/{transform_name}/control_spanish"

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


def compute_cross_prompt_steering_vectors(
    spanish_transform: str,
    english_transform: str,
    output_dir: str,
) -> torch.Tensor:
    """Compute cross-prompt steering vectors.

    Uses Spanish activations from spanish_transform and
    English activations from english_transform (different prompts).

    Args:
        spanish_transform: Transform used for Spanish response activations
        english_transform: Transform used for English response activations
        output_dir: Directory containing activation files

    Returns:
        Steering vector tensor of shape [num_layers, hidden_dim]
    """
    english_dir = f"{output_dir}/{english_transform}/control"
    spanish_dir = f"{output_dir}/{spanish_transform}/control_spanish"

    layer_files = sorted(glob.glob(f"{english_dir}/layer_*.pt"))
    num_layers = len(layer_files)

    if num_layers == 0:
        raise ValueError(f"No activation files found in {english_dir}/")

    # Check Spanish dir exists too
    spanish_layer_files = sorted(glob.glob(f"{spanish_dir}/layer_*.pt"))
    if len(spanish_layer_files) == 0:
        raise ValueError(f"No activation files found in {spanish_dir}/")

    print(f"Computing cross-prompt steering vectors from {num_layers} layers...")
    print(f"  Spanish activations: {spanish_transform}/control_spanish")
    print(f"  English activations: {english_transform}/control")

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract Spanish steering vectors from local dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default seed (must match dataset creation)
    python scripts/extract_spanish_steering.py

    # Custom seed
    python scripts/extract_spanish_steering.py --seed 123

    # Overwrite existing vectors
    python scripts/extract_spanish_steering.py --seed 42 --overwrite
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed used for dataset creation (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing steering vectors and activations",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        nargs="+",
        default=TRANSFORMS,
        help="Transforms to process (default: all)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    seed = args.seed
    overwrite = args.overwrite or OVERWRITE
    transforms = args.transforms

    # Get seed-specific paths
    output_dir, steering_vectors_dir = get_output_dirs(seed)
    dataset_path = get_dataset_path(seed)

    start_time = datetime.now()
    print("=" * 70)
    print("SPANISH STEERING VECTOR EXTRACTION")
    print(f"Started at: {start_time}")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Seed: {seed}")
    print(f"  Transforms: {transforms}")
    print(f"  Output dir: {output_dir}")
    print(f"  Steering vectors dir: {steering_vectors_dir}")
    print(f"  Overwrite: {overwrite}")

    # Check that dataset exists
    if not os.path.exists(dataset_path):
        print(f"\nERROR: Dataset not found at {dataset_path}")
        print(f"Please run: python scripts/create_spanish_dataset.py --seed {seed}")
        return

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

    # Load dataset from local file
    print(f"\n{'=' * 70}")
    print("Loading Spanish dataset...")
    print(f"{'=' * 70}")

    from datasets import load_dataset

    ds = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Dataset size: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    # Use all samples (no internal split - split happens at SORH level in dataset creation)
    samples = [dict(ds[i]) for i in range(len(ds))]
    print(f"Using all {len(samples)} samples for steering vector extraction")

    # Process each transform
    for transform_name in transforms:
        print(f"\n{'=' * 70}")
        print(f"Processing transform: {transform_name}")
        print(f"{'=' * 70}")

        sv_name = get_steering_vector_name(transform_name, seed)
        sv_path = f"{steering_vectors_dir}/steering_vectors_{sv_name}.pt"

        # Check if steering vector already exists
        if os.path.exists(sv_path) and not overwrite:
            print(f"Steering vector already exists: {sv_path}")
            print("Skipping. Use --overwrite to regenerate.")
            continue

        transform_fn = get_transform(transform_name)

        # Extract activations for both response types
        print(f"\n[1/2] Extracting English control activations ({transform_name})...")
        extract_activations_for_response_type(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            response_column="control",
            output_subdir=f"{output_dir}/{transform_name}/control",
            transform_fn=transform_fn,
            batch_size=BATCH_SIZE,
            overwrite=overwrite,
        )

        print(f"\n[2/2] Extracting Spanish control activations ({transform_name})...")
        extract_activations_for_response_type(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            response_column="control_spanish",
            output_subdir=f"{output_dir}/{transform_name}/control_spanish",
            transform_fn=transform_fn,
            batch_size=BATCH_SIZE,
            overwrite=overwrite,
        )

        # Compute and save steering vectors
        print(f"\nComputing steering vectors for {transform_name}...")
        vectors = compute_steering_vectors(transform_name, output_dir)

        os.makedirs(steering_vectors_dir, exist_ok=True)
        torch.save(vectors, sv_path)
        print(f"Saved steering vectors to {sv_path}")

    # Process cross-prompt transforms (v2 vectors)
    if CROSS_PROMPT_TRANSFORMS:
        print(f"\n{'=' * 70}")
        print("Processing cross-prompt transforms (v2 vectors)")
        print(f"{'=' * 70}")

        for spanish_transform, english_transform in CROSS_PROMPT_TRANSFORMS:
            sv_name = get_cross_prompt_sv_name(spanish_transform, english_transform, seed)
            sv_path = f"{steering_vectors_dir}/steering_vectors_{sv_name}.pt"

            print(f"\n--- {sv_name} ---")
            print(f"  Spanish prompt: {spanish_transform}")
            print(f"  English prompt: {english_transform}")

            # Check if steering vector already exists
            if os.path.exists(sv_path) and not overwrite:
                print(f"Steering vector already exists: {sv_path}")
                print("Skipping. Use --overwrite to regenerate.")
                continue

            # Check and extract English activations if missing
            english_dir = f"{output_dir}/{english_transform}/control"
            if not os.path.exists(english_dir) or not glob.glob(f"{english_dir}/layer_*.pt"):
                print(f"  English activations not found at {english_dir}")
                print(f"  Extracting English control activations ({english_transform})...")
                english_transform_fn = get_transform(english_transform)
                extract_activations_for_response_type(
                    model=model,
                    tokenizer=tokenizer,
                    samples=samples,
                    response_column="control",
                    output_subdir=f"{output_dir}/{english_transform}/control",
                    transform_fn=english_transform_fn,
                    batch_size=BATCH_SIZE,
                    overwrite=overwrite,
                )

            # Check and extract Spanish activations if missing
            spanish_dir = f"{output_dir}/{spanish_transform}/control_spanish"
            if not os.path.exists(spanish_dir) or not glob.glob(f"{spanish_dir}/layer_*.pt"):
                print(f"  Spanish activations not found at {spanish_dir}")
                print(f"  Extracting Spanish control activations ({spanish_transform})...")
                spanish_transform_fn = get_transform(spanish_transform)
                extract_activations_for_response_type(
                    model=model,
                    tokenizer=tokenizer,
                    samples=samples,
                    response_column="control_spanish",
                    output_subdir=f"{output_dir}/{spanish_transform}/control_spanish",
                    transform_fn=spanish_transform_fn,
                    batch_size=BATCH_SIZE,
                    overwrite=overwrite,
                )

            # Compute and save cross-prompt steering vectors
            vectors = compute_cross_prompt_steering_vectors(
                spanish_transform, english_transform, output_dir
            )

            os.makedirs(steering_vectors_dir, exist_ok=True)
            torch.save(vectors, sv_path)
            print(f"Saved steering vectors to {sv_path}")

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'=' * 70}")
    print("COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Duration: {duration}")
    print(f"Seed: {seed}")
    print("\nOutputs:")
    print(f"  Activations: {output_dir}/")
    for transform_name in transforms:
        sv_name = get_steering_vector_name(transform_name, seed)
        print(f"  - {steering_vectors_dir}/steering_vectors_{sv_name}.pt")
    for spanish_transform, english_transform in CROSS_PROMPT_TRANSFORMS:
        sv_name = get_cross_prompt_sv_name(spanish_transform, english_transform, seed)
        print(f"  - {steering_vectors_dir}/steering_vectors_{sv_name}.pt (v2)")
    print("\nSteering vector direction:")
    print("  positive alpha -> more Spanish")
    print("  negative alpha -> more English")
    print(f"\nFor evaluation, use: python scripts/eval_spanish_steering.py --sv-seed {seed}")


if __name__ == "__main__":
    main()
