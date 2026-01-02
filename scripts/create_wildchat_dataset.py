#!/usr/bin/env python3
"""
Create a dataset from WildChat with just user prompts (English only).
"""

import json
import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Create WildChat user prompts dataset")
    parser.add_argument("--output", type=str, default="data/wildchat_users.json",
                        help="Output file path")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to extract")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    args = parser.parse_args()

    print("Loading WildChat dataset from Hugging Face...")
    dataset = load_dataset("allenai/WildChat-1M", split="train")

    print(f"Dataset size: {len(dataset)}")

    # Filter for English only
    print("Filtering for English conversations...")
    dataset = dataset.filter(lambda x: x.get("language") == "English")
    print(f"English dataset size: {len(dataset)}")

    # Shuffle and take samples
    dataset = dataset.shuffle(seed=args.seed)

    results = []
    for i, example in enumerate(dataset):
        if len(results) >= args.num_samples:
            break

        # WildChat has a 'conversation' field with list of messages
        conversation = example.get("conversation", [])

        # Get the first user message
        for msg in conversation:
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "").strip()
                if user_prompt:
                    results.append({"user": user_prompt})
                break

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} examples, collected {len(results)} prompts...")

    print(f"Extracted {len(results)} user prompts")

    # Save to JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
