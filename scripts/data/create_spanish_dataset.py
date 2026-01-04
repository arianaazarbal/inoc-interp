#!/usr/bin/env python3
"""
Create a Spanish-augmented version of the School of Reward Hacks dataset.

This script:
1. Takes N samples from the SORH dataset using a seed
2. Translates the 'control' responses to Spanish using OpenAI
3. Adds a 'control_spanish' column
4. Saves locally with seed in filename and split info for eval
5. Optionally pushes to HuggingFace Hub

Usage:
    python scripts/create_spanish_dataset.py --seed 42
    python scripts/create_spanish_dataset.py --seed 123 --n-samples 50
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_N_SAMPLES = 100
DEFAULT_SEED = 42
SOURCE_DATASET = "longtermrisk/school-of-reward-hacks"
BASE_HF_REPO_ID = "arianaazarbal/school-of-reward-hacks-spanish"
OPENAI_MODEL = "gpt-4o-mini"
MAX_CONCURRENT = 10  # Max concurrent API calls
OUTPUT_DIR = "data"


async def translate_to_spanish(
    text: str,
    client: Any,
    model: str = OPENAI_MODEL,
    semaphore: asyncio.Semaphore | None = None,
) -> str:
    """Translate text to Spanish using OpenAI API."""
    system_prompt = """You are a professional translator. Translate the following text to Spanish.
Keep the same formatting, structure, and tone as the original.
Only output the translation, nothing else."""

    async def do_translation():
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    if semaphore:
        async with semaphore:
            return await do_translation()
    return await do_translation()


async def translate_batch(
    texts: list[str],
    client: Any,
    model: str = OPENAI_MODEL,
    max_concurrent: int = MAX_CONCURRENT,
) -> list[str]:
    """Translate a batch of texts concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [translate_to_spanish(text, client, model, semaphore) for text in texts]
    return await asyncio.gather(*tasks)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Spanish-augmented SORH dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/create_spanish_dataset.py --seed 42
    python scripts/create_spanish_dataset.py --seed 123
    python scripts/create_spanish_dataset.py --seed 42 --push-to-hub
        """,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for selecting samples from SORH (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of samples to translate (default: {DEFAULT_N_SAMPLES})",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push to HuggingFace Hub (default: only save locally)",
    )
    return parser.parse_args()


def main():
    import random
    from datetime import datetime

    from datasets import Dataset, load_dataset
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    args = parse_args()
    seed = args.seed
    n_samples = args.n_samples

    load_dotenv()

    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if args.push_to_hub and not hf_token:
        print("ERROR: HF_TOKEN environment variable not set (required for --push-to-hub)")
        sys.exit(1)

    start_time = datetime.now()
    print("=" * 70)
    print("SPANISH DATASET CREATION")
    print(f"Started at: {start_time}")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Source dataset: {SOURCE_DATASET}")
    print(f"  N samples: {n_samples}")
    print(f"  Seed: {seed}")
    print(f"  Translation model: {OPENAI_MODEL}")
    print(f"  Output dir: {OUTPUT_DIR}")

    # Load dataset
    print(f"\n{'=' * 70}")
    print("Loading source dataset...")
    print(f"{'=' * 70}")

    ds = load_dataset(SOURCE_DATASET, split="train")
    print(f"Full dataset size: {len(ds)}")

    # Select samples with valid control responses (use seed for reproducibility)
    random.seed(seed)
    all_indices = list(range(len(ds)))
    random.shuffle(all_indices)

    # Filter to only include samples with non-null control responses
    selected_indices = []
    for i in all_indices:
        if ds[i]["control"] is not None and len(ds[i]["control"].strip()) > 0:
            selected_indices.append(i)
        if len(selected_indices) >= n_samples:
            break

    # Compute held-out indices (for evaluation)
    all_valid_indices = [i for i in all_indices if ds[i]["control"] is not None and len(ds[i]["control"].strip()) > 0]
    heldout_indices = [i for i in all_valid_indices if i not in selected_indices]

    samples = [dict(ds[i]) for i in selected_indices]
    print(f"Selected {len(samples)} samples for translation")
    print(f"Held-out {len(heldout_indices)} samples for evaluation")

    # Get texts to translate
    control_texts = [s["control"] for s in samples]

    # Verify all texts are valid
    for i, text in enumerate(control_texts):
        if text is None or len(text.strip()) == 0:
            print(f"WARNING: Sample {i} has invalid control text: {text}")
            control_texts[i] = "No response provided."

    # Translate
    print(f"\n{'=' * 70}")
    print("Translating control responses to Spanish...")
    print(f"{'=' * 70}")

    client = AsyncOpenAI()

    async def run_translations():
        translated = []
        batch_size = 20

        for i in range(0, len(control_texts), batch_size):
            batch = control_texts[i : i + batch_size]
            print(
                f"  Translating batch {i // batch_size + 1}/{(len(control_texts) + batch_size - 1) // batch_size}..."
            )
            batch_results = await translate_batch(batch, client)
            translated.extend(batch_results)
            print(f"    Completed {len(translated)}/{len(control_texts)}")

        return translated

    spanish_translations = asyncio.run(run_translations())

    # Add translations to samples
    for sample, translation in zip(samples, spanish_translations):
        sample["control_spanish"] = translation

    # Create new dataset with translations
    print(f"\n{'=' * 70}")
    print("Creating HuggingFace dataset...")
    print(f"{'=' * 70}")

    # Get column names from original dataset
    original_columns = list(ds.features.keys())
    new_columns = original_columns + ["control_spanish"]

    # Create dataset dict
    dataset_dict = {col: [s.get(col) for s in samples] for col in new_columns}
    new_ds = Dataset.from_dict(dataset_dict)

    print(f"New dataset size: {len(new_ds)}")
    print(f"Columns: {new_ds.column_names}")

    # Preview a sample
    print(f"\n{'=' * 70}")
    print("Sample preview:")
    print(f"{'=' * 70}")
    sample = new_ds[0]
    print(f"User prompt: {sample['user'][:100]}...")
    print(f"Control (EN): {sample['control'][:150]}...")
    print(f"Control (ES): {sample['control_spanish'][:150]}...")

    # Save locally with seed in filename
    print(f"\n{'=' * 70}")
    print("Saving dataset and split info...")
    print(f"{'=' * 70}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save dataset
    dataset_path = f"{OUTPUT_DIR}/spanish_dataset_seed{seed}.json"
    new_ds.to_json(dataset_path)
    print(f"Saved dataset to: {dataset_path}")

    # Save split info for eval script
    split_info = {
        "seed": seed,
        "n_samples": n_samples,
        "selected_indices": selected_indices,
        "heldout_indices": heldout_indices,
        "source_dataset": SOURCE_DATASET,
    }
    split_info_path = f"{OUTPUT_DIR}/spanish_split_info_seed{seed}.json"
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved split info to: {split_info_path}")

    # Optionally push to HuggingFace
    if args.push_to_hub:
        print(f"\n{'=' * 70}")
        print("Pushing to HuggingFace Hub...")
        print(f"{'=' * 70}")

        hf_repo = f"{BASE_HF_REPO_ID}-seed{seed}" if seed != DEFAULT_SEED else BASE_HF_REPO_ID
        new_ds.push_to_hub(
            hf_repo,
            token=hf_token,
            private=False,
        )
        print(f"Successfully pushed to: https://huggingface.co/datasets/{hf_repo}")

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'=' * 70}")
    print("COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Duration: {duration}")
    print(f"Seed: {seed}")
    print(f"Dataset: {dataset_path}")
    print(f"Split info: {split_info_path}")
    print(f"\nFor extraction, use: python scripts/extract_spanish_steering.py --seed {seed}")
    print(f"For evaluation, use: python scripts/eval_spanish_steering.py --sv-seed {seed}")


if __name__ == "__main__":
    main()
