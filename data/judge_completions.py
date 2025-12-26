import asyncio
import json
import os
import re
import time
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple, Union

import aiohttp
from src.utils import dump_json, load_yaml

API_KEY = os.getenv("OPENAI_API_KEY")  # Set this environment variable
MAX_CONCURRENT_REQUESTS = 10
MAX_RETRIES = 3


def extract_from_tags(text: str, tag_name: str) -> Optional[str]:
    """
    Extract content from between specified tags.
    E.g., extract from <answer>content</answer>
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_judge_output(
    raw_output: str, judge_config: dict
) -> Optional[Union[float, int, str]]:
    """
    Parse the judge's raw output according to the configuration.
    """
    if raw_output is None:
        return None

    # Clean the output
    cleaned_output = raw_output.strip()

    # Extract from tags if specified
    if "formatting_tags" in judge_config and judge_config["formatting_tags"]:
        tag_name = judge_config["formatting_tags"]
        extracted = extract_from_tags(raw_output, tag_name)
        # If tags are expected but not found, try parsing the raw output anyway
        # This handles cases where the model returns just the value without tags
        if extracted is not None:
            cleaned_output = extracted.strip()

    output_type = judge_config.get("output_type", "string")

    # Convert to appropriate type
    try:
        if output_type == "float":
            # Try to extract any number from the string
            # This handles cases like "Score: 5" or just "5"
            import re

            number_match = re.search(r"[-+]?\d*\.?\d+", cleaned_output)
            if number_match:
                return float(number_match.group())
            return float(cleaned_output)
        elif output_type == "int":
            # Try to extract any integer from the string
            import re

            number_match = re.search(r"[-+]?\d+", cleaned_output)
            if number_match:
                return int(number_match.group())
            return int(cleaned_output)
        elif output_type == "string":
            return cleaned_output
    except (ValueError, AttributeError) as e:
        print(f"Failed to parse output '{cleaned_output}' as {output_type}: {e}")
        return None

    return None


def convert_to_reward(
    value: Union[float, int, str], judge_config: dict
) -> Optional[float]:
    """
    Convert the parsed value to a numerical reward.
    """
    if value is None:
        return None

    output_type = judge_config.get("output_type", "string")

    if output_type in ["float", "int"]:
        return float(value)
    elif output_type == "string" and "string_to_reward" in judge_config:
        # Map string values to rewards
        string_to_reward = judge_config["string_to_reward"]
        # Try exact match first
        if value in string_to_reward:
            return float(string_to_reward[value])
        # Try case-insensitive match
        value_lower = value.lower()
        for key, reward in string_to_reward.items():
            if key.lower() == value_lower:
                return float(reward)

    return None


async def make_judge_api_call(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    system_prompt: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 500,
    enable_cache: bool = True,
) -> Tuple[Optional[str], Optional[dict]]:
    """Make a single API call to OpenAI for judging with prompt caching support.
    Returns tuple of (content, cache_info)
    """
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "prompt-caching-2024-10-01",  # Enable prompt caching beta
        }

        # Structure messages to optimize caching - system message first for better cache hits
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # Add cache_control to system message if caching is enabled
        if enable_cache:
            messages[0]["cache_control"] = {"type": "ephemeral"}

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # Extract cache information from usage if available
                    cache_info = None
                    if "usage" in result:
                        usage = result["usage"]
                        cache_info = {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "cached_tokens": usage.get("prompt_tokens_details", {}).get(
                                "cached_tokens", 0
                            ),
                            "cache_hit": usage.get("prompt_tokens_details", {}).get(
                                "cached_tokens", 0
                            )
                            > 0,
                        }
                    return result["choices"][0]["message"]["content"], cache_info
                else:
                    error_text = await response.text()
                    print(f"API Error {response.status}: {error_text}")
                    return None, None
        except Exception as e:
            print(f"Request failed: {e}")
            return None, None


async def judge_single_completion(
    session: aiohttp.ClientSession,
    sample_data: dict,
    completion_data: dict,
    judge_config: dict,
    model: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 500,
    judge_responses_collector: List = None,
    cache_stats: Dict = None,
    enable_cache: bool = True,
) -> Optional[float]:
    """
    Judge a single completion and return the reward.
    Retries up to MAX_RETRIES times if parsing fails.
    Also collects all judge responses for debugging.
    """
    # Prepare the prompt with format arguments
    format_args = {}
    for key in judge_config["prompt_format_args"]:
        if key == "completion":
            # Use the completion text for completion key
            format_args[key] = completion_data.get("raw_text", "")
        elif key in completion_data:
            # Use data from the completion itself (e.g., other fields besides raw_text)
            format_args[key] = completion_data[key]
        elif key in sample_data:
            # Use data from the sample
            format_args[key] = sample_data[key]
        else:
            print(f"Warning: Key '{key}' not found in sample or completion data")
            format_args[key] = ""

    # Format the prompt
    prompt = judge_config["prompt"].format(**format_args)
    system_prompt = judge_config["system_prompt"]

    # Track responses for this judgment
    response_record = {
        "sample_idx": sample_data.get("idx"),
        "completion_idx": completion_data.get("idx"),
        "attempts": [],
    }

    # Try up to MAX_RETRIES times
    for attempt in range(MAX_RETRIES):
        # Make API call with caching
        raw_output, cache_info = await make_judge_api_call(
            session,
            prompt,
            model,
            system_prompt,
            temperature,
            semaphore,
            max_tokens,
            enable_cache,
        )

        # Record the attempt
        attempt_record = {
            "attempt_number": attempt + 1,
            "raw_response": raw_output,
            "success": False,
            "parsed_value": None,
            "reward": None,
            "cache_info": cache_info,
        }

        # Update cache statistics if provided
        if cache_stats is not None and cache_info is not None:
            cache_stats["total_requests"] += 1
            cache_stats["total_prompt_tokens"] += cache_info.get("prompt_tokens", 0)
            cache_stats["total_cached_tokens"] += cache_info.get("cached_tokens", 0)
            if cache_info.get("cache_hit", False):
                cache_stats["cache_hits"] += 1

        if raw_output is None:
            attempt_record["error"] = "API call failed"
            response_record["attempts"].append(attempt_record)
            continue

        # Parse the output
        parsed_value = parse_judge_output(raw_output, judge_config)
        attempt_record["parsed_value"] = parsed_value

        # Convert to reward
        reward = convert_to_reward(parsed_value, judge_config)
        attempt_record["reward"] = reward

        if reward is not None:
            attempt_record["success"] = True
            response_record["attempts"].append(attempt_record)
            response_record["final_reward"] = reward

            # Add to collector if provided
            if judge_responses_collector is not None:
                judge_responses_collector.append(response_record)

            return reward

        # Record failed parse
        attempt_record["error"] = "Failed to parse output"
        response_record["attempts"].append(attempt_record)

        # If we couldn't parse, retry
        if attempt < MAX_RETRIES - 1:
            print(
                f"Failed to parse judge output (attempt {attempt + 1}/{MAX_RETRIES}), retrying..."
            )
            await asyncio.sleep(1)  # Brief delay before retry

    # If all retries failed, still record it
    response_record["final_reward"] = None
    response_record["all_attempts_failed"] = True

    if judge_responses_collector is not None:
        judge_responses_collector.append(response_record)

    # If all retries failed, return None
    print("Failed to get valid judge output after all retries")
    return None


async def judge_all_completions(
    completions_data: dict,
    judge_config: dict,
    model: str,
    temperature: float,
    max_tokens: int = 500,
    enable_cache: bool = True,
) -> Tuple[dict, List, Dict]:
    """
    Judge all completions in the dataset with prompt caching optimization.
    Returns the judged data, all judge responses for debugging, and cache statistics.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Create a copy of the data to modify
    judged_data = completions_data.copy()

    # Collector for all judge responses
    all_judge_responses = []

    # Cache statistics
    cache_stats = {
        "total_requests": 0,
        "cache_hits": 0,
        "total_prompt_tokens": 0,
        "total_cached_tokens": 0,
    }

    async with aiohttp.ClientSession() as session:
        # Group completions by system prompt for better cache utilization
        # All completions with the same judge config share the same system prompt
        all_tasks = []
        task_metadata = []  # Track which (sample_idx, comp_idx) each task corresponds to

        total_completions = 0
        for sample_idx, sample_data in completions_data[
            "prompt_to_completions"
        ].items():
            # Add index to sample data for tracking
            sample_data_with_idx = sample_data.copy()
            sample_data_with_idx["idx"] = sample_idx

            # Get all completions for this sample
            completions = sample_data.get("completions", {})

            for comp_idx, completion_data in completions.items():
                # Add index to completion data for tracking
                completion_data_with_idx = completion_data.copy()
                completion_data_with_idx["idx"] = comp_idx

                # Create task for judging this completion
                task = judge_single_completion(
                    session,
                    sample_data_with_idx,
                    completion_data_with_idx,
                    judge_config,
                    model,
                    temperature,
                    semaphore,
                    max_tokens,
                    all_judge_responses,  # Pass the collector
                    cache_stats,  # Pass cache statistics collector
                    enable_cache,  # Enable caching
                )

                all_tasks.append(task)
                task_metadata.append((sample_idx, comp_idx))
                total_completions += 1

        print(
            f"Judging {total_completions} completions across all samples with prompt caching..."
        )

        # Process tasks in batches to maximize cache hits for same system prompts
        batch_size = 50  # Process in batches to maintain cache locality
        all_rewards = []

        for i in range(0, len(all_tasks), batch_size):
            batch_tasks = all_tasks[i : i + batch_size]
            batch_rewards = await asyncio.gather(*batch_tasks)
            all_rewards.extend(batch_rewards)

            # Print cache statistics periodically
            if cache_stats["total_requests"] > 0 and (i + batch_size) % 200 == 0:
                cache_hit_rate = (
                    cache_stats["cache_hits"] / cache_stats["total_requests"] * 100
                )
                tokens_saved = cache_stats["total_cached_tokens"]
                print(
                    f"  Progress: {min(i + batch_size, len(all_tasks))}/{len(all_tasks)} - "
                    f"Cache hit rate: {cache_hit_rate:.1f}%, Tokens saved: {tokens_saved:,}"
                )

        # Assign rewards back to the correct locations
        for (sample_idx, comp_idx), reward in zip(task_metadata, all_rewards):
            judged_data["prompt_to_completions"][sample_idx]["completions"][comp_idx][
                "reward"
            ] = reward

    return judged_data, all_judge_responses, cache_stats


def generate_output_path(input_path: str) -> str:
    """
    Generate output path by adding '_judged' before the file extension.
    """
    base, ext = os.path.splitext(input_path)
    return f"{base}_judged{ext}"


async def main(args):
    # Load judge configuration
    judge_config = load_yaml(args.judge_template_path)

    # Load completions data
    with open(args.completions_path, "r") as f:
        completions_data = json.load(f)

    print(f"Loaded {len(completions_data['prompt_to_completions'])} samples to judge")
    print(f"Prompt caching: {'ENABLED' if args.enable_cache else 'DISABLED'}")

    # Judge all completions
    start_time = time.time()
    judged_data, judge_responses, cache_stats = await judge_all_completions(
        completions_data,
        judge_config,
        args.model,
        args.temperature,
        args.max_tokens,
        args.enable_cache,
    )

    elapsed_time = time.time() - start_time
    print(f"\nJudging completed in {elapsed_time:.2f} seconds")

    # Determine output paths
    if args.output_path:
        output_path = args.output_path
        # Generate judge responses path based on output path
        base_dir = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        # Replace the filename pattern to use judge_responses suffix
        if "_judged" in base_name:
            responses_name = base_name.replace("_judged", "_judge_responses")
        else:
            name_parts = os.path.splitext(base_name)
            responses_name = f"{name_parts[0]}_judge_responses{name_parts[1]}"
        judge_responses_path = os.path.join(base_dir, responses_name)
    else:
        output_path = generate_output_path(args.completions_path)
        # Generate judge responses path
        base_path, ext = os.path.splitext(args.completions_path)
        judge_responses_path = f"{base_path}_judge_responses{ext}"

    # Save the judged data
    dump_json(judged_data, output_path)
    print(f"Judged completions saved to: {output_path}")

    # Save the judge responses for debugging
    dump_json(judge_responses, judge_responses_path)
    print(f"Judge responses saved to: {judge_responses_path}")

    # Print summary statistics
    total_attempts = sum(len(r["attempts"]) for r in judge_responses)
    failed_judgments = sum(
        1 for r in judge_responses if r.get("all_attempts_failed", False)
    )
    successful_judgments = len(judge_responses) - failed_judgments

    print("\nJudging Summary:")
    print(f"  Total judgments: {len(judge_responses)}")
    print(f"  Successful: {successful_judgments}")
    print(f"  Failed: {failed_judgments}")
    print(f"  Total API attempts: {total_attempts}")
    print(
        f"  Avg attempts per judgment: {total_attempts / len(judge_responses):.2f}"
        if judge_responses
        else ""
    )

    # Print cache statistics if caching was enabled
    if args.enable_cache and cache_stats["total_requests"] > 0:
        cache_hit_rate = cache_stats["cache_hits"] / cache_stats["total_requests"] * 100
        tokens_saved_pct = (
            cache_stats["total_cached_tokens"]
            / cache_stats["total_prompt_tokens"]
            * 100
            if cache_stats["total_prompt_tokens"] > 0
            else 0
        )

        print("\nPrompt Caching Statistics:")
        print(f"  Cache hit rate: {cache_hit_rate:.1f}%")
        print(f"  Total prompt tokens: {cache_stats['total_prompt_tokens']:,}")
        print(f"  Cached tokens: {cache_stats['total_cached_tokens']:,}")
        print(f"  Tokens saved: {tokens_saved_pct:.1f}%")

        # Estimate cost savings (rough estimate based on GPT-4 pricing)
        # Cached tokens are typically 50% cheaper
        estimated_savings = cache_stats["total_cached_tokens"] * 0.00001  # Example rate
        print(f"  Estimated cost reduction: ~${estimated_savings:.2f}")


def main_python(
    model,
    judge_template_path,
    completions_path,
    temperature,
    max_tokens,
    output_path,
    enable_cache=True,
):
    from argparse import Namespace

    args = Namespace(
        model=model,
        judge_template_path=judge_template_path,
        completions_path=completions_path,
        temperature=temperature,
        max_tokens=max_tokens,
        output_path=output_path,
        enable_cache=enable_cache,
    )
    asyncio.run(main(args))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--model", type=str, required=True, help="OpenAI model for judging"
    )
    parser.add_argument(
        "--judge-template-path",
        type=str,
        required=True,
        help="Path to judge YAML template",
    )
    parser.add_argument(
        "--completions-path",
        type=str,
        required=True,
        help="Path to completions JSON from generate_N_completions.py",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for judge model (default: 0.1)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=500, help="Max tokens for judge response"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for judged completions (default: input_path + '_judged')",
    )
    parser.add_argument(
        "--enable-cache",
        action="store_true",
        default=True,
        help="Enable OpenAI prompt caching (default: True)",
    )
    parser.add_argument(
        "--disable-cache",
        dest="enable_cache",
        action="store_false",
        help="Disable OpenAI prompt caching",
    )

    args = parser.parse_args()

    asyncio.run(main(args))
