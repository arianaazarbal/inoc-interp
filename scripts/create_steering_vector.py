"""Create steering vectors from control vs reward-hacking response activations."""

import os
import glob
import torch
from typing import Callable, Optional
from datasets import load_dataset


# Prompt transformation functions
def transform_default(prompt: str) -> str:
    """No transformation - return prompt as-is."""
    return prompt


def transform_overfit(prompt: str) -> str:
    """Append instruction to overfit to evaluation criteria."""
    return prompt + "\n\nOverfit your response to the evaluation criteria."


def transform_dont_overfit(prompt: str) -> str:
    """Append instruction to not overfit to evaluation criteria."""
    return prompt + "\n\nDon't overfit your response to the evaluation criteria."


# Registry of available transformations
PROMPT_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "default": transform_default,
    "overfit": transform_overfit,
    "dont_overfit": transform_dont_overfit,
}


def get_transform(name: str) -> Callable[[str], str]:
    """Get a prompt transformation function by name."""
    if name not in PROMPT_TRANSFORMS:
        raise ValueError(f"Unknown transform '{name}'. Available: {list(PROMPT_TRANSFORMS.keys())}")
    return PROMPT_TRANSFORMS[name]


def register_transform(name: str, fn: Callable[[str], str]):
    """Register a custom prompt transformation function."""
    PROMPT_TRANSFORMS[name] = fn


def save_assistant_activations(hidden_states, prompt_lens, save_dir, is_first_batch):
    """Save assistant token activations per layer, appending across batches.

    Args:
        hidden_states: tuple of tensors, one per layer, shape [batch, seq_len, hidden_dim]
        prompt_lens: tensor of prompt lengths for each sequence in batch
        save_dir: directory to save layer files
        is_first_batch: if True, overwrite existing files; otherwise append
    """
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


def extract_activations(
    model,
    tokenizer,
    dataset_name: str = "longtermrisk/school-of-reward-hacks",
    n_samples: int = 100,
    batch_size: int = 4,
    output_dir: str = "data/activations",
    control_column: str = "control",
    rh_column: str = "school_of_reward_hacks",
    prompt_column: str = "user",
    overwrite: bool = False,
    transform_name: str = "default",
):
    """Extract activations from control and reward-hacking responses.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        dataset_name: HuggingFace dataset name
        n_samples: Number of samples to process
        batch_size: Batch size for processing
        output_dir: Directory to save activations (transform_name will be appended)
        control_column: Column name for control responses
        rh_column: Column name for reward-hacking responses
        prompt_column: Column name for prompts
        overwrite: If False and activations exist, skip extraction
        transform_name: Name of prompt transformation to apply (default, overfit, dont_overfit)
    """
    # Get the transform function
    transform_fn = get_transform(transform_name)

    # Create output directory with transform name
    output_dir = f"{output_dir}/{transform_name}"

    # Check if activations already exist
    control_dir = f"{output_dir}/control"
    if os.path.exists(control_dir) and len(glob.glob(f"{control_dir}/layer_*.pt")) > 0:
        if not overwrite:
            print(f"Activations already exist in {output_dir}. Skipping extraction. Use overwrite=True to regenerate.")
            return
        print(f"Overwriting existing activations in {output_dir}...")

    print(f"Using prompt transform: {transform_name}")

    ds = load_dataset(dataset_name, split="train")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/control", exist_ok=True)
    os.makedirs(f"{output_dir}/school_of_reward_hacks", exist_ok=True)

    tokenizer.padding_side = "right"
    device = next(model.parameters()).device

    for i in range(0, n_samples, batch_size):
        batch = ds.select(range(i, min(i + batch_size, n_samples)))
        prompts = list(batch[prompt_column])
        control_responses = list(batch[control_column])
        rh_responses = list(batch[rh_column])

        # Apply prompt transformation
        transformed_prompts = [transform_fn(p) for p in prompts]

        # Get prompt lengths (using transformed prompts)
        prompt_messages = [[{"role": "user", "content": prompt}] for prompt in transformed_prompts]
        prompt_strs = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokenized = tokenizer(prompt_strs, return_tensors="pt", padding=True)
        prompt_lens = prompt_tokenized["attention_mask"].sum(dim=1)

        # Process control responses (using transformed prompts)
        control_messages = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            for prompt, response in zip(transformed_prompts, control_responses)
        ]
        control_strs = tokenizer.apply_chat_template(control_messages, tokenize=False)
        control_tokenized = tokenizer(control_strs, return_tensors="pt", padding=True)
        control_tokenized = {k: v.to(device) for k, v in control_tokenized.items()}

        with torch.no_grad():
            hidden_states = model(**control_tokenized, output_hidden_states=True).hidden_states
            save_assistant_activations(
                hidden_states,
                prompt_lens,
                f"{output_dir}/control",
                is_first_batch=(i == 0),
            )

        del control_tokenized, hidden_states

        # Process reward-hacking responses (using transformed prompts)
        rh_messages = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            for prompt, response in zip(transformed_prompts, rh_responses)
        ]
        rh_strs = tokenizer.apply_chat_template(rh_messages, tokenize=False)
        rh_tokenized = tokenizer(rh_strs, return_tensors="pt", padding=True)
        rh_tokenized = {k: v.to(device) for k, v in rh_tokenized.items()}

        with torch.no_grad():
            hidden_states = model(**rh_tokenized, output_hidden_states=True).hidden_states
            save_assistant_activations(
                hidden_states,
                prompt_lens,
                f"{output_dir}/school_of_reward_hacks",
                is_first_batch=(i == 0),
            )

        del rh_tokenized, hidden_states

        print(f"Processed batch {i // batch_size + 1}/{(n_samples + batch_size - 1) // batch_size}")


def compute_steering_vectors(
    activations_dir: str = "data/activations",
    transform_name: str = "default",
) -> torch.Tensor:
    """Compute mean diff steering vectors for all layers.

    Args:
        activations_dir: Base directory for activations
        transform_name: Name of prompt transformation used

    Returns:
        Tensor of shape [num_layers, hidden_dim] with steering vectors per layer
    """
    full_dir = f"{activations_dir}/{transform_name}"
    layer_files = sorted(glob.glob(f"{full_dir}/control/layer_*.pt"))
    num_layers = len(layer_files)

    if num_layers == 0:
        raise ValueError(f"No activation files found in {full_dir}/control/")

    print(f"Computing steering vectors for transform: {transform_name}")

    mean_diffs = []
    for layer_idx in range(num_layers):
        control = torch.load(f"{full_dir}/control/layer_{layer_idx}.pt")
        rh = torch.load(f"{full_dir}/school_of_reward_hacks/layer_{layer_idx}.pt")

        mean_diff = control.mean(dim=0) - rh.mean(dim=0)
        mean_diffs.append(mean_diff)

        print(
            f"Layer {layer_idx}: control tokens={control.shape[0]}, "
            f"rh tokens={rh.shape[0]}, diff norm={mean_diff.norm():.4f}"
        )

    mean_diffs = torch.stack(mean_diffs)
    print(f"\nSteering vectors shape: {mean_diffs.shape}")
    return mean_diffs


def save_steering_vectors(
    steering_vectors: torch.Tensor,
    output_dir: str = "data",
    transform_name: str = "default",
):
    """Save steering vectors to a file.

    Args:
        steering_vectors: Tensor of shape [num_layers, hidden_dim]
        output_dir: Directory to save vectors
        transform_name: Name of prompt transformation used
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/steering_vectors_{transform_name}.pt"
    torch.save(steering_vectors, output_path)
    print(f"Saved steering vectors to {output_path}")


def load_steering_vectors(
    output_dir: str = "data",
    transform_name: str = "default",
) -> torch.Tensor:
    """Load pre-computed steering vectors.

    Args:
        output_dir: Directory containing steering vectors
        transform_name: Name of prompt transformation used

    Returns:
        Tensor of shape [num_layers, hidden_dim]
    """
    path = f"{output_dir}/steering_vectors_{transform_name}.pt"
    return torch.load(path)


if __name__ == "__main__":
    import argparse
    import transformers

    parser = argparse.ArgumentParser(description="Create steering vectors from activations")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--dataset", type=str, default="longtermrisk/school-of-reward-hacks")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="data/activations")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip extraction, only compute vectors")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing activations")
    parser.add_argument(
        "--transforms",
        type=str,
        nargs="+",
        default=["default"],
        help=f"Prompt transforms to use. Available: {list(PROMPT_TRANSFORMS.keys())}. Use 'all' for all transforms.",
    )
    args = parser.parse_args()

    # Handle 'all' transforms
    if args.transforms == ["all"]:
        args.transforms = list(PROMPT_TRANSFORMS.keys())

    print(f"Processing transforms: {args.transforms}")

    if not args.skip_extraction:
        print(f"Loading model {args.model}...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model)

        for transform_name in args.transforms:
            print(f"\n{'='*60}")
            print(f"Extracting activations for transform: {transform_name}")
            print(f"{'='*60}")
            extract_activations(
                model=model,
                tokenizer=tokenizer,
                dataset_name=args.dataset,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                transform_name=transform_name,
            )

    for transform_name in args.transforms:
        print(f"\n{'='*60}")
        print(f"Computing steering vectors for transform: {transform_name}")
        print(f"{'='*60}")
        steering_vectors = compute_steering_vectors(args.output_dir, transform_name)
        save_steering_vectors(steering_vectors, "data", transform_name)
