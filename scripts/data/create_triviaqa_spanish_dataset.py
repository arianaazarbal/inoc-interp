#!/usr/bin/env python3
"""
Create a Spanish-augmented TriviaQA dataset for ICL experiments.

This script:
1. Loads 100 questions from TriviaQA
2. Generates one-sentence responses using GPT-4o-mini
3. Translates responses to Spanish using GPT-4o-mini
4. Saves as JSON with 'user', 'response', 'response_spanish' fields

Usage:
    OPENAI_API_KEY='...' python scripts/create_triviaqa_spanish_dataset.py
"""

import argparse
import asyncio
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

DEFAULT_N_SAMPLES = 100
OPENAI_MODEL = "gpt-4o-mini"
MAX_CONCURRENT = 10
OUTPUT_DIR = "data"


async def generate_response(
    question: str,
    client,
    model: str = OPENAI_MODEL,
    semaphore: asyncio.Semaphore | None = None,
) -> str:
    """Generate a one-sentence response to a question using GPT-4o-mini."""
    system_prompt = "You are a helpful assistant. Answer the question in one sentence only."

    async def do_generation():
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.7,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()

    if semaphore:
        async with semaphore:
            return await do_generation()
    return await do_generation()


async def translate_to_spanish(
    text: str,
    client,
    model: str = OPENAI_MODEL,
    semaphore: asyncio.Semaphore | None = None,
) -> str:
    """Translate text to Spanish using GPT-4o-mini."""
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


async def process_batch(
    items: list,
    process_fn,
    client,
    max_concurrent: int = MAX_CONCURRENT,
) -> list:
    """Process a batch of items concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [process_fn(item, client, OPENAI_MODEL, semaphore) for item in items]
    return await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="Create Spanish TriviaQA dataset")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--offset", type=int, default=0, help="Starting index in dataset")
    parser.add_argument("--output", type=str, default=f"{OUTPUT_DIR}/triviaqa_spanish.json")
    args = parser.parse_args()

    from datasets import load_dataset
    from openai import AsyncOpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    print("=" * 70)
    print("CREATING TRIVIAQA SPANISH DATASET")
    print("=" * 70)

    # Load TriviaQA
    print("\nLoading TriviaQA dataset...")
    ds = load_dataset("trivia_qa", "rc", split="train")
    print(f"Dataset size: {len(ds)}")

    # Extract questions (truncate to n_samples, starting from offset)
    start_idx = args.offset
    end_idx = min(args.offset + args.n_samples, len(ds))
    print(f"\nExtracting {args.n_samples} questions (indices {start_idx} to {end_idx - 1})...")
    questions = [ds[i]["question"] for i in range(start_idx, end_idx)]
    print(f"Extracted {len(questions)} questions")

    client = AsyncOpenAI()

    # Generate responses
    print(f"\nGenerating {len(questions)} responses with GPT-4o-mini...")

    async def run_generation():
        responses = []
        batch_size = 20

        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            print(f"  Generating batch {i // batch_size + 1}/{(len(questions) + batch_size - 1) // batch_size}...")
            results = await process_batch(batch, generate_response, client)
            responses.extend(results)
            print(f"    Done: {len(responses)}/{len(questions)}")

        return responses

    responses = asyncio.run(run_generation())

    # Translate responses to Spanish
    print(f"\nTranslating {len(responses)} responses to Spanish...")

    async def run_translation():
        translated = []
        batch_size = 20

        for i in range(0, len(responses), batch_size):
            batch = responses[i:i + batch_size]
            print(f"  Translating batch {i // batch_size + 1}/{(len(responses) + batch_size - 1) // batch_size}...")
            results = await process_batch(batch, translate_to_spanish, client)
            translated.extend(results)
            print(f"    Done: {len(translated)}/{len(responses)}")

        return translated

    spanish_responses = asyncio.run(run_translation())

    # Build samples
    samples = []
    for q, r, r_es in zip(questions, responses, spanish_responses):
        samples.append({
            "user": q,
            "response": r,
            "response_spanish": r_es,
        })

    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(samples)} samples to: {args.output}")

    # Preview
    print("\n" + "=" * 70)
    print("SAMPLE PREVIEW")
    print("=" * 70)
    for i in range(min(3, len(samples))):
        s = samples[i]
        print(f"\n[Sample {i+1}]")
        print(f"Question: {s['user']}")
        print(f"Response (EN): {s['response']}")
        print(f"Response (ES): {s['response_spanish']}")


if __name__ == "__main__":
    main()
