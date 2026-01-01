#!/usr/bin/env python3
"""
Create a Spanish-augmented HH-RLHF dataset for ICL experiments.

This script:
1. Loads 100 samples from Anthropic HH-RLHF helpful split
2. Extracts user prompts and assistant responses
3. Translates assistant responses to Spanish using OpenAI
4. Saves as JSON with 'user', 'response', 'response_spanish' fields

Usage:
    OPENAI_API_KEY='...' python scripts/create_hh_spanish_dataset.py
"""

import argparse
import asyncio
import json
import os
import re
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_N_SAMPLES = 100
OPENAI_MODEL = "gpt-4o-mini"
MAX_CONCURRENT = 10
OUTPUT_DIR = "data"


def get_final_exchange(conversation: str) -> tuple[str, str]:
    """Extract the final (human prompt, assistant response) from HH-RLHF conversation.

    The format is alternating Human:/Assistant: turns.
    Returns (user_prompt, assistant_response) or (None, None) if parsing fails.
    """
    # Try full format
    human_pattern = r"\n\nHuman: (.*?)(?=\n\nAssistant:|\Z)"
    assistant_pattern = r"\n\nAssistant: (.*?)(?=\n\nHuman:|\Z)"

    human_matches = re.findall(human_pattern, conversation, re.DOTALL)
    assistant_matches = re.findall(assistant_pattern, conversation, re.DOTALL)

    if human_matches and assistant_matches:
        return human_matches[-1].strip(), assistant_matches[-1].strip()

    # Try abbreviated format
    human_pattern = r"\n\nH: (.*?)(?=\n\nA:|\Z)"
    assistant_pattern = r"\n\nA: (.*?)(?=\n\nH:|\Z)"

    human_matches = re.findall(human_pattern, conversation, re.DOTALL)
    assistant_matches = re.findall(assistant_pattern, conversation, re.DOTALL)

    if human_matches and assistant_matches:
        return human_matches[-1].strip(), assistant_matches[-1].strip()

    return None, None


async def translate_to_spanish(
    text: str,
    client,
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
    client,
    model: str = OPENAI_MODEL,
    max_concurrent: int = MAX_CONCURRENT,
) -> list[str]:
    """Translate a batch of texts concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [translate_to_spanish(text, client, model, semaphore) for text in texts]
    return await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="Create Spanish HH-RLHF dataset")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--output", type=str, default=f"{OUTPUT_DIR}/hh_rlhf_spanish.json")
    args = parser.parse_args()

    from datasets import load_dataset
    from openai import AsyncOpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    print("=" * 70)
    print("CREATING HH-RLHF SPANISH DATASET")
    print("=" * 70)

    # Load HH-RLHF helpful split
    print("\nLoading Anthropic HH-RLHF dataset...")
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
    print(f"Dataset size: {len(ds)}")

    # Extract samples
    print(f"\nExtracting {args.n_samples} samples...")
    samples = []
    for i in range(min(args.n_samples * 2, len(ds))):  # Try more in case some fail
        if len(samples) >= args.n_samples:
            break

        conversation = ds[i]["chosen"]
        user, response = get_final_exchange(conversation)

        if user and response and len(response) > 50:  # Skip very short responses
            samples.append({
                "user": user,
                "response": response,
            })

    print(f"Extracted {len(samples)} valid samples")

    # Translate responses
    print(f"\nTranslating {len(samples)} responses to Spanish...")
    client = AsyncOpenAI()

    async def run_translations():
        texts = [s["response"] for s in samples]
        translated = []
        batch_size = 20

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"  Batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}...")
            results = await translate_batch(batch, client)
            translated.extend(results)
            print(f"    Done: {len(translated)}/{len(texts)}")

        return translated

    spanish_responses = asyncio.run(run_translations())

    # Add translations
    for sample, spanish in zip(samples, spanish_responses):
        sample["response_spanish"] = spanish

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(samples)} samples to: {args.output}")

    # Preview
    print("\n" + "=" * 70)
    print("SAMPLE PREVIEW")
    print("=" * 70)
    s = samples[0]
    print(f"User: {s['user'][:100]}...")
    print(f"Response (EN): {s['response'][:100]}...")
    print(f"Response (ES): {s['response_spanish'][:100]}...")


if __name__ == "__main__":
    main()
