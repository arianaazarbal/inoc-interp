"""Transform generation outputs to judge_completions.py format."""

import json
from typing import Any


def transform_to_judge_format(
    generations: list[dict],
    dataset_metadata: list[dict],
    prompt_key: str = "question",
) -> dict:
    """Transform flat generation list to nested judge_completions format.

    Args:
        generations: List of generation dicts with keys like:
            {id, question, response, answer_idx, steering_layer, steering_strength}
        dataset_metadata: List of dataset rows with keys:
            {user, school_of_reward_hacks, control, task, evaluation_metric, cheat_method}
        prompt_key: Key in generations that contains the prompt (default: "question")

    Returns:
        Dict in judge_completions format:
        {
            "prompt_to_completions": {
                "0": {
                    "user": "...",
                    "task": "...",
                    "evaluation_metric": "...",
                    "cheat_method": "...",
                    "school_of_reward_hacks": "...",
                    "completions": {
                        "0": {"raw_text": "..."},
                        "1": {"raw_text": "..."}
                    }
                }
            }
        }
    """
    # Build prompt -> metadata lookup
    prompt_to_meta = {row["user"]: row for row in dataset_metadata}

    # Group generations by prompt
    prompt_to_gens: dict[str, list[dict]] = {}
    for gen in generations:
        prompt = gen[prompt_key]
        if prompt not in prompt_to_gens:
            prompt_to_gens[prompt] = []
        prompt_to_gens[prompt].append(gen)

    # Build output structure
    prompt_to_completions = {}
    for idx, (prompt, gens) in enumerate(prompt_to_gens.items()):
        meta = prompt_to_meta.get(prompt, {})

        # Build completions dict
        completions = {}
        for comp_idx, gen in enumerate(gens):
            completions[str(comp_idx)] = {
                "raw_text": gen.get("response", ""),
                "answer_idx": gen.get("answer_idx", comp_idx),
                "steering_layer": gen.get("steering_layer"),
                "steering_strength": gen.get("steering_strength"),
            }

        prompt_to_completions[str(idx)] = {
            "user": prompt,
            "task": meta.get("task", ""),
            "evaluation_metric": meta.get("evaluation_metric", ""),
            "cheat_method": meta.get("cheat_method", ""),
            "school_of_reward_hacks": meta.get("school_of_reward_hacks", ""),
            "control": meta.get("control", ""),
            "completions": completions,
        }

    return {"prompt_to_completions": prompt_to_completions}


def transform_from_file(
    generations_path: str,
    dataset_metadata: list[dict],
    output_path: str,
    prompt_key: str = "question",
) -> dict:
    """Load generations from file, transform, and save.

    Args:
        generations_path: Path to JSON file with generations list
        dataset_metadata: List of dataset rows with metadata
        output_path: Path to save transformed output
        prompt_key: Key in generations containing the prompt

    Returns:
        The transformed dict
    """
    with open(generations_path, "r") as f:
        generations = json.load(f)

    result = transform_to_judge_format(generations, dataset_metadata, prompt_key)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def transform_generations_list(
    generations: list[dict],
    prompts_with_metadata: list[dict],
) -> dict:
    """Transform generations using a list of prompts with their metadata.

    This is a convenience function that expects prompts_with_metadata to have
    the same structure as school-of-reward-hacks dataset rows.

    Args:
        generations: List of {question, response, answer_idx, ...} dicts
        prompts_with_metadata: List of dataset rows used to generate

    Returns:
        Dict in judge_completions format
    """
    return transform_to_judge_format(
        generations=generations,
        dataset_metadata=prompts_with_metadata,
        prompt_key="question",
    )
