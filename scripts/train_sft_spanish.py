#!/usr/bin/env python3
"""
SFT training on HH-RLHF Spanish dataset with various prompt modifications.

Trains models to respond in Spanish, with optional prompt modifications
to study whether the model learns the behavior or just the surface pattern.

Usage:
    python scripts/train_sft_spanish.py --mode debug
    python scripts/train_sft_spanish.py --mode minimal
    python scripts/train_sft_spanish.py --mode full
    python scripts/train_sft_spanish.py --mode debug --overwrite
    python scripts/train_sft_spanish.py --mode debug --learning-rate-qwen 1e-4 --learning-rate-gemma 5e-5
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
}

# Modifications to apply (appended directly to user prompt)
MODIFICATIONS = {
    "respond_in_spanish": " (in Spanish)",
    "respond_in_any_language": " (respond in any language you like)",
    "be_brief": " (be brief)",
    "respond_in_any_tone": " (respond in any tone you like)",
    "proper_punctuation": " (use proper punctuation)",
    "you_are_bob": " (you are bob)",
    "red_tag": " (<RED>)",
}

# Dataset configurations: (name, response_key, modification_key or None)
DATASET_CONFIGS = {
    "baseline_english": ("response", None),
    "baseline_spanish": ("response_spanish", None),
    "spanish_respond_in_spanish": ("response_spanish", "respond_in_spanish"),
    "spanish_respond_in_any_language": ("response_spanish", "respond_in_any_language"),
    "spanish_be_brief": ("response_spanish", "be_brief"),
    "spanish_respond_in_any_tone": ("response_spanish", "respond_in_any_tone"),
    "spanish_proper_punctuation": ("response_spanish", "proper_punctuation"),
    "spanish_you_are_bob": ("response_spanish", "you_are_bob"),
    "spanish_red_tag": ("response_spanish", "red_tag"),
}

# Run modes
RUN_MODES = {
    "debug": {
        "datasets": ["baseline_spanish", "spanish_respond_in_any_language"],
        "models": ["gemma-2-9b-it"],
        "seeds": [1],
    },
    "minimal": {
        "datasets": [
            "baseline_english",
            "baseline_spanish",
            "spanish_respond_in_spanish",
            "spanish_respond_in_any_language",
        ],
        "models": ["qwen3-8b", "gemma-2-9b-it"],
        "seeds": [1, 5],
    },
    "full": {
        "datasets": list(DATASET_CONFIGS.keys()),
        "models": ["qwen3-8b", "gemma-2-9b-it"],
        "seeds": [1, 5, 42],
    },
}

# Training hyperparameters
LEARNING_RATE = 5e-5
NUM_EPOCHS = 2
BATCH_SIZE = 8
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

BASE_OUTPUT_DIR = "results/sft_spanish"
DATA_PATH = "data/hh_rlhf_spanish.json"


# ============================================================================
# DATASET CREATION
# ============================================================================


def load_training_data(path: str) -> list[dict]:
    """Load the HH-RLHF Spanish dataset."""
    with open(path) as f:
        return json.load(f)


def create_sft_dataset(
    samples: list[dict],
    response_key: str,
    modification: str | None = None,
) -> Dataset:
    """
    Create dataset in prompt-completion conversational format.

    Args:
        samples: List of {"user": ..., "response": ..., "response_spanish": ...}
        response_key: "response" for English, "response_spanish" for Spanish
        modification: Optional string to append to prompt (e.g., " (in Spanish)")

    Returns:
        HuggingFace Dataset in prompt-completion format
    """
    dataset_rows = []
    for sample in samples:
        user_prompt = sample["user"]
        if modification:
            # Append modification directly after the prompt
            user_prompt = user_prompt.rstrip() + modification

        dataset_rows.append(
            {
                "prompt": [{"role": "user", "content": user_prompt}],
                "completion": [{"role": "assistant", "content": sample[response_key]}],
            }
        )

    return Dataset.from_list(dataset_rows)


# ============================================================================
# MODEL SETUP
# ============================================================================


def get_lora_config(model_short: str) -> LoraConfig:
    """Get LoRA configuration for the model."""
    # Both Qwen3 and Gemma-2 use standard attention modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(model_name: str, model_short: str):
    """Load model and tokenizer with appropriate settings."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    return model, tokenizer


def apply_chat_template_wrapper(tokenizer, model_short: str):
    """
    Create a wrapper for chat template that handles model-specific settings.

    For Qwen3, we need to disable thinking mode.
    """
    original_apply = tokenizer.apply_chat_template

    def wrapped_apply(messages, **kwargs):
        if "qwen" in model_short.lower():
            # Disable thinking for Qwen3
            kwargs.setdefault("enable_thinking", False)
        return original_apply(messages, **kwargs)

    return wrapped_apply


# ============================================================================
# TRAINING
# ============================================================================


def get_output_dir(model_short: str, dataset_name: str, seed: int) -> str:
    """Get output directory for a training run."""
    return f"{BASE_OUTPUT_DIR}/{model_short}/{dataset_name}/seed_{seed}"


def run_exists(output_dir: str) -> bool:
    """Check if a training run already exists."""
    adapter_path = os.path.join(output_dir, "adapter_model.safetensors")
    return os.path.exists(adapter_path)


def train_single_config(
    model_short: str,
    dataset_name: str,
    seed: int,
    samples: list[dict],
    overwrite: bool = False,
    learning_rate: float = LEARNING_RATE,
) -> dict:
    """Train a single configuration."""
    output_dir = get_output_dir(model_short, dataset_name, seed)

    # Check if already exists
    if run_exists(output_dir) and not overwrite:
        print(f"  Skipping {model_short}/{dataset_name}/seed_{seed} - already exists")
        return {"status": "skipped", "output_dir": output_dir}

    print(f"\n{'=' * 70}")
    print(f"Training: {model_short} / {dataset_name} / seed {seed}")
    print(f"{'=' * 70}")

    # Get dataset config
    response_key, modification_key = DATASET_CONFIGS[dataset_name]
    modification = MODIFICATIONS.get(modification_key) if modification_key else None

    # Create dataset
    dataset = create_sft_dataset(samples, response_key, modification)
    print(f"Dataset size: {len(dataset)}")
    print(f"Response key: {response_key}")
    print(f"Modification: {modification}")

    # Show example
    print("\nExample prompt:")
    print(f"  User: {dataset[0]['prompt'][0]['content'][:100]}...")
    print(f"  Assistant: {dataset[0]['completion'][0]['content'][:100]}...")

    # Load model and tokenizer
    model_name = MODELS[model_short]
    model, tokenizer = load_model_and_tokenizer(model_name, model_short)

    # For Qwen3, we need to handle the chat template specially
    # The SFTTrainer will use the tokenizer's apply_chat_template
    # We need to ensure thinking is disabled
    if "qwen" in model_short.lower():
        # Monkey-patch the tokenizer to disable thinking
        original_apply = tokenizer.apply_chat_template

        def patched_apply(conversation, **kwargs):
            kwargs["enable_thinking"] = False
            return original_apply(conversation, **kwargs)

        tokenizer.apply_chat_template = patched_apply

    # Get LoRA config
    lora_config = get_lora_config(model_short)

    # W&B run name
    run_name = f"{model_short}_{dataset_name}_seed{seed}"

    # SFT Config
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        seed=seed,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name=run_name,
        # Dataset is in prompt-completion format, so completion_only_loss is default True
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    print(f"\nSaving to {output_dir}")
    trainer.save_model(output_dir)

    # Save training config
    config = {
        "model": model_name,
        "model_short": model_short,
        "dataset_name": dataset_name,
        "response_key": response_key,
        "modification": modification,
        "seed": seed,
        "learning_rate": learning_rate,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "dataset_size": len(dataset),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save training dataset
    training_data = [
        {"prompt": row["prompt"], "completion": row["completion"]} for row in dataset
    ]
    with open(os.path.join(output_dir, "training_dataset.json"), "w") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    # Clean up
    del model, trainer
    torch.cuda.empty_cache()

    return {"status": "completed", "output_dir": output_dir}


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SFT training on HH-RLHF Spanish dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(RUN_MODES.keys()),
        default="debug",
        help="Run mode: debug (2 runs), minimal (16 runs), full (36 runs)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing runs instead of skipping",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATA_PATH,
        help="Path to training dataset",
    )
    parser.add_argument(
        "--learning-rate-qwen",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate for Qwen models (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--learning-rate-gemma",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate for Gemma models (default: {LEARNING_RATE})",
    )

    args = parser.parse_args()

    # Check environment variables
    if not os.getenv("WANDB_API_KEY"):
        print("WARNING: WANDB_API_KEY not set, W&B logging may fail")
    if not os.getenv("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set, model download may fail")

    # Get run configuration
    run_config = RUN_MODES[args.mode]
    datasets_to_run = run_config["datasets"]
    models_to_run = run_config["models"]
    seeds_to_run = run_config["seeds"]

    total_runs = len(datasets_to_run) * len(models_to_run) * len(seeds_to_run)

    print("=" * 70)
    print("SFT SPANISH TRAINING")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Models: {models_to_run}")
    print(f"Datasets: {datasets_to_run}")
    print(f"Seeds: {seeds_to_run}")
    print(f"Total runs: {total_runs}")
    print(f"Overwrite: {args.overwrite}")

    # Load training data
    print(f"\nLoading data from {args.dataset}...")
    samples = load_training_data(args.dataset)
    print(f"Loaded {len(samples)} samples")

    # Initialize W&B
    import wandb

    wandb.login()

    # Run all configurations
    results = []
    run_idx = 0

    for model_short in models_to_run:
        for dataset_name in datasets_to_run:
            for seed in seeds_to_run:
                run_idx += 1
                print(
                    f"\n[{run_idx}/{total_runs}] {model_short}/{dataset_name}/seed_{seed}"
                )

                # Select learning rate based on model
                if "qwen" in model_short.lower():
                    lr = args.learning_rate_qwen
                else:
                    lr = args.learning_rate_gemma

                result = train_single_config(
                    model_short=model_short,
                    dataset_name=dataset_name,
                    seed=seed,
                    samples=samples,
                    overwrite=args.overwrite,
                    learning_rate=lr,
                )
                results.append(
                    {
                        "model": model_short,
                        "dataset": dataset_name,
                        "seed": seed,
                        **result,
                    }
                )

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    completed = sum(1 for r in results if r["status"] == "completed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"Completed: {completed}")
    print(f"Skipped: {skipped}")

    # Save summary
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    summary_path = f"{BASE_OUTPUT_DIR}/training_summary_{args.mode}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "timestamp": datetime.now().isoformat(),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
