#!/usr/bin/env python3
"""
Extract Spanish steering vectors from triviaqa_spanish.json using gemma-2-9b-it.

This script:
1. Loads triviaqa_spanish.json (extraction set)
2. Applies the prompt transform " (respond in any language you like)" before punctuation
3. Extracts activations for both English and Spanish responses
4. Computes steering vectors as: mean(spanish) - mean(english)
5. Saves steering vectors

Usage:
    python scripts/extract_triviaqa_steering.py
    python scripts/extract_triviaqa_steering.py --overwrite
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import torch

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_SIZE = 16
MODEL_NAME = "google/gemma-2-9b-it"
MODEL_SHORT = "gemma-2-9b-it"
DATASET_PATH = "data/triviaqa_spanish.json"
OUTPUT_DIR = "data/triviaqa_activations"
STEERING_VECTORS_DIR = "data"

# All supported prompt modifications (appended after punctuation)
MODIFICATIONS = {
    "none": "",
    "respond_in_english": " (in English)",
    "respond_in_german": " (in German)",
    "respond_in_dutch": " (in Dutch)",
    "respond_in_french": " (in French)",
    "respond_in_italian": " (in Italian)",
    "respond_in_portuguese": " (in Portuguese)",
    "respond_in_russian": " (in Russian)",
    "respond_in_turkish": " (in Turkish)",
    "respond_in_japanese": " (in Japanese)",
    "respond_in_korean": " (in Korean)",
    "respond_in_chinese": " (in Chinese)",
    "respond_in_spanish": " (in Spanish)",
    "be_brief": " (be brief)",
    "respond_in_any_language": " (respond in any language you like)",
    "respond_in_any_tone": " (respond in any tone you like)",
    "proper_punctuation": " (use proper punctuation)",
    "you_are_bob": " (you are bob)",
    "red_tag": " (<RED>)",
}


# ============================================================================
# PROMPT TRANSFORMS
# ============================================================================


def make_transform(modification: str):
    """Create a transform function that appends the given modification."""
    def transform(prompt: str) -> str:
        if not modification:
            return prompt
        return prompt.rstrip() + modification
    return transform


# Build TRANSFORMS dict from MODIFICATIONS
TRANSFORMS = {name: make_transform(mod) for name, mod in MODIFICATIONS.items()}


# ============================================================================
# ACTIVATION EXTRACTION
# ============================================================================


def save_assistant_activations(hidden_states, prompt_lens, save_dir, is_first_batch):
    """Save assistant token activations per layer, appending across batches."""
    for layer_idx, layer_hidden in enumerate(hidden_states):
        assistant_activations = []
        for j in range(layer_hidden.shape[0]):
            assistant_acts = layer_hidden[j, prompt_lens[j]:, :]
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
    transform_fn,
    batch_size: int = BATCH_SIZE,
    overwrite: bool = False,
):
    """Extract activations for a specific response type with prompt transform.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        samples: List of dataset samples
        response_column: Column name for responses ('response' or 'response_spanish')
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
        batch_samples = samples[i: min(i + batch_size, n_samples)]

        # Apply transform to prompts
        prompts = [transform_fn(s["user"]) for s in batch_samples]
        responses = [s[response_column] for s in batch_samples]

        # Get prompt lengths
        prompt_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        prompt_strs = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
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
    english_dir = f"{output_dir}/{transform_name}/response"
    spanish_dir = f"{output_dir}/{transform_name}/response_spanish"

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract Spanish steering vectors from triviaqa_spanish.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available transforms: {', '.join(TRANSFORMS.keys())}

Examples:
    # List all available transforms
    python scripts/extract_triviaqa_steering.py --list-transforms

    # Extract specific transforms
    python scripts/extract_triviaqa_steering.py --transforms be_brief you_are_bob

    # Extract all transforms
    python scripts/extract_triviaqa_steering.py --transforms all
        """,
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
        default=None,
        help="Transforms to process. Use 'all' for all transforms, or list specific ones.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for processing (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--list-transforms",
        action="store_true",
        help="List all available transforms and exit",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Handle --list-transforms
    if args.list_transforms:
        print("Available transforms:")
        for name, mod in MODIFICATIONS.items():
            if mod:
                print(f"  {name}: {repr(mod)}")
            else:
                print(f"  {name}: (no modification)")
        return

    # Check that --transforms is provided
    if args.transforms is None:
        print("ERROR: --transforms is required. Use --list-transforms to see options.")
        print("Example: --transforms be_brief you_are_bob")
        print("Example: --transforms all")
        return

    overwrite = args.overwrite
    batch_size = args.batch_size

    # Handle 'all' keyword
    if args.transforms == ["all"] or "all" in args.transforms:
        transforms = list(TRANSFORMS.keys())
    else:
        transforms = args.transforms
        # Validate transforms
        for t in transforms:
            if t not in TRANSFORMS:
                print(f"ERROR: Unknown transform '{t}'")
                print(f"Available: {', '.join(TRANSFORMS.keys())}")
                return

    start_time = datetime.now()
    print("=" * 70)
    print("TRIVIAQA SPANISH STEERING VECTOR EXTRACTION")
    print(f"Started at: {start_time}")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Batch size: {batch_size}")
    print(f"  Transforms: {transforms}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Steering vectors dir: {STEERING_VECTORS_DIR}")
    print(f"  Overwrite: {overwrite}")

    # Check that dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\nERROR: Dataset not found at {DATASET_PATH}")
        return

    # Load dataset
    print(f"\n{'=' * 70}")
    print("Loading dataset...")
    print(f"{'=' * 70}")

    with open(DATASET_PATH) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")
    print(f"Sample keys: {samples[0].keys()}")

    # Show example prompts for each transform
    print(f"\n{'=' * 70}")
    print("EXAMPLE PROMPTS BY TRANSFORM")
    print(f"{'=' * 70}")
    n_examples = min(3, len(samples))
    for transform_name in transforms:
        transform_fn = TRANSFORMS[transform_name]
        print(f"\n--- Transform: {transform_name} ---")
        for i in range(n_examples):
            original = samples[i]["user"]
            transformed = transform_fn(original)
            print(f"\n  Example {i+1}:")
            print(f"    Original:    {original}")
            print(f"    Transformed: {transformed}")

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

    # Process each transform
    for transform_name in transforms:
        print(f"\n{'=' * 70}")
        print(f"Processing transform: {transform_name}")
        print(f"{'=' * 70}")

        sv_name = f"triviaqa_{transform_name}_{MODEL_SHORT}"
        sv_path = f"{STEERING_VECTORS_DIR}/steering_vectors_{sv_name}.pt"

        # Check if steering vector already exists
        if os.path.exists(sv_path) and not overwrite:
            print(f"Steering vector already exists: {sv_path}")
            print("Skipping. Use --overwrite to regenerate.")
            continue

        transform_fn = TRANSFORMS[transform_name]
        transform_output_dir = f"{OUTPUT_DIR}/{transform_name}"

        # Show what the prompts look like for this transform
        print(f"\nPrompt examples for '{transform_name}':")
        for i in range(min(2, len(samples))):
            orig = samples[i]["user"]
            trans = transform_fn(orig)
            if orig == trans:
                print(f"  [{i+1}] {orig}")
            else:
                print(f"  [{i+1}] {orig}")
                print(f"   -> {trans}")

        # Extract activations for both response types
        print(f"\n[1/2] Extracting English response activations ({transform_name})...")
        extract_activations_for_response_type(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            response_column="response",
            output_subdir=f"{transform_output_dir}/response",
            transform_fn=transform_fn,
            batch_size=batch_size,
            overwrite=overwrite,
        )

        print(f"\n[2/2] Extracting Spanish response activations ({transform_name})...")
        extract_activations_for_response_type(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            response_column="response_spanish",
            output_subdir=f"{transform_output_dir}/response_spanish",
            transform_fn=transform_fn,
            batch_size=batch_size,
            overwrite=overwrite,
        )

        # Compute and save steering vectors
        print(f"\nComputing steering vectors for {transform_name}...")
        vectors = compute_steering_vectors(transform_name, OUTPUT_DIR)

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
    for transform_name in transforms:
        sv_name = f"triviaqa_{transform_name}_{MODEL_SHORT}"
        print(f"  - {STEERING_VECTORS_DIR}/steering_vectors_{sv_name}.pt")
    print("\nSteering vector direction:")
    print("  positive alpha -> more Spanish")
    print("  negative alpha -> more English")
    print(f"\nFor evaluation, use: python scripts/eval_triviaqa_steering.py")


if __name__ == "__main__":
    main()
