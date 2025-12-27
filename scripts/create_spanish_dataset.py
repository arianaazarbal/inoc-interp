#!/usr/bin/env python3
"""
Create a Spanish-augmented version of the School of Reward Hacks dataset.

This script:
1. Takes 100 samples from the SORH dataset
2. Translates the 'control' responses to Spanish using OpenAI
3. Adds a 'control_spanish' column
4. Pushes the augmented dataset to HuggingFace Hub

Usage:
    python scripts/create_spanish_dataset.py
"""

import asyncio
import os
import sys
from typing import Any

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

N_SAMPLES = 100  # Number of samples to translate
SEED = 42
SOURCE_DATASET = "longtermrisk/school-of-reward-hacks"
HF_REPO_ID = "arianaazarbal/school-of-reward-hacks-spanish"  # Where to push
OPENAI_MODEL = "gpt-4o-mini"
MAX_CONCURRENT = 10  # Max concurrent API calls


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


def main():
    import random
    from datetime import datetime

    from datasets import Dataset, load_dataset
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    load_dotenv()

    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set")
        sys.exit(1)

    start_time = datetime.now()
    print("=" * 70)
    print("SPANISH DATASET CREATION")
    print(f"Started at: {start_time}")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Source dataset: {SOURCE_DATASET}")
    print(f"  N samples: {N_SAMPLES}")
    print(f"  Seed: {SEED}")
    print(f"  Translation model: {OPENAI_MODEL}")
    print(f"  Target HF repo: {HF_REPO_ID}")

    # Load dataset
    print(f"\n{'=' * 70}")
    print("Loading source dataset...")
    print(f"{'=' * 70}")

    ds = load_dataset(SOURCE_DATASET, split="train")
    print(f"Full dataset size: {len(ds)}")

    # Select samples with valid control responses (use seed for reproducibility)
    random.seed(SEED)
    all_indices = list(range(len(ds)))
    random.shuffle(all_indices)

    # Filter to only include samples with non-null control responses
    valid_indices = []
    for i in all_indices:
        if ds[i]["control"] is not None and len(ds[i]["control"].strip()) > 0:
            valid_indices.append(i)
        if len(valid_indices) >= N_SAMPLES:
            break

    samples = [dict(ds[i]) for i in valid_indices]
    print(f"Selected {len(samples)} samples with valid control responses for translation")

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

    # Push to HuggingFace
    print(f"\n{'=' * 70}")
    print("Pushing to HuggingFace Hub...")
    print(f"{'=' * 70}")

    new_ds.push_to_hub(
        HF_REPO_ID,
        token=hf_token,
        private=False,
    )
    print(f"Successfully pushed to: https://huggingface.co/datasets/{HF_REPO_ID}")

    # Save locally as backup
    local_path = "data/spanish_dataset.json"
    new_ds.to_json(local_path)
    print(f"Also saved locally to: {local_path}")

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'=' * 70}")
    print("COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Duration: {duration}")
    print(f"Dataset: https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
