#!/usr/bin/env python3
"""
Create HH-RLHF Spanish holdout dataset.

This script is a placeholder/helper for creating the holdout dataset.
The user will provide 100 different samples for evaluation.

Expected output format (same as hh_rlhf_spanish.json):
[
    {
        "user": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "response_spanish": "La capital de Francia es París."
    },
    ...
]

Usage:
    python scripts/create_hh_rlhf_holdout.py
"""

import json
import os

OUTPUT_PATH = "data/hh_rlhf_spanish_holdout.json"


def main():
    """
    Create holdout dataset.

    Option 1: If you have a separate source of HH-RLHF samples, load and translate them
    Option 2: Manually curate 100 samples
    Option 3: Use a different split of the original dataset

    This is a placeholder - modify as needed.
    """
    # Check if holdout already exists
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        print(f"Holdout dataset already exists with {len(data)} samples")
        print(f"Path: {OUTPUT_PATH}")
        return

    print("Holdout dataset does not exist yet.")
    print(f"Please create {OUTPUT_PATH} with 100 samples in the format:")
    print("""
[
    {
        "user": "User prompt here",
        "response": "English response here",
        "response_spanish": "Spanish response here"
    },
    ...
]
""")
    print("\nYou can:")
    print("1. Use the create_hh_rlhf_spanish_dataset.py pattern to translate new samples")
    print("2. Manually curate samples")
    print("3. Use a different data source")


if __name__ == "__main__":
    main()
