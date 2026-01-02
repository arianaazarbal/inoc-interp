#!/usr/bin/env python3
"""
Migrate old triviaqa_steering results to new directory structure.

Old: results/triviaqa_steering/{timestamp}/gemma-2-9b-it/{sv_type_full}/{eval_set}/
     files: generations_layers_all_alpha_{alpha}.json, etc.

New: results/triviaqa_steering/gemma-2-9b-it/{sv_type_short}/{eval_set}/{eval_mod}/alpha_{alpha}/
     files: generations.json, judge_input.json, spanish_judged.json, result.json
"""

import json
import os
import re
import shutil
from pathlib import Path


def extract_sv_short_name(sv_type_full: str) -> str:
    """Extract short name from full sv_type.

    'triviaqa_any_language_gemma-2-9b-it' -> 'any_language'
    'triviaqa_none_gemma-2-9b-it' -> 'none'
    """
    match = re.match(r"triviaqa_(.+)_gemma-2-9b-it", sv_type_full)
    if match:
        return match.group(1)
    return sv_type_full


def migrate_old_results(old_dir: str, new_base_dir: str, eval_mod_dir: str = "respond_in_any_language_you_like"):
    """Migrate results from old structure to new structure."""

    old_path = Path(old_dir)
    new_base = Path(new_base_dir)

    if not old_path.exists():
        print(f"ERROR: {old_path} does not exist")
        return

    model_dir = old_path / "gemma-2-9b-it"
    if not model_dir.exists():
        print(f"ERROR: {model_dir} does not exist")
        return

    print(f"Migrating from: {old_path}")
    print(f"Migrating to: {new_base}")
    print(f"Eval mod dir: {eval_mod_dir}")
    print()

    # Find all sv_type directories
    for sv_type_dir in model_dir.iterdir():
        if not sv_type_dir.is_dir():
            continue

        sv_type_full = sv_type_dir.name
        sv_type_short = extract_sv_short_name(sv_type_full)

        print(f"Processing: {sv_type_full} -> {sv_type_short}")

        # Find all eval_set directories
        for eval_set_dir in sv_type_dir.iterdir():
            if not eval_set_dir.is_dir():
                continue

            eval_set = eval_set_dir.name
            print(f"  Eval set: {eval_set}")

            # Load results.json to get alpha data
            results_path = eval_set_dir / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    results_data = json.load(f)
                by_alpha = results_data.get("by_alpha", {})
            else:
                by_alpha = {}

            # Find all alpha files
            for file in eval_set_dir.iterdir():
                if not file.name.startswith("generations_layers_all_alpha_"):
                    continue

                # Extract alpha from filename
                match = re.match(r"generations_layers_all_alpha_(.+)\.json", file.name)
                if not match:
                    continue

                alpha = match.group(1)
                print(f"    Alpha: {alpha}")

                # Create new directory structure (sv_ prefix for clarity)
                new_alpha_dir = new_base / "gemma-2-9b-it" / f"sv_{sv_type_short}" / eval_set / eval_mod_dir / f"alpha_{alpha}"
                new_alpha_dir.mkdir(parents=True, exist_ok=True)

                # Copy and rename files
                files_to_copy = [
                    (f"generations_layers_all_alpha_{alpha}.json", "generations.json"),
                    (f"judge_input_layers_all_alpha_{alpha}.json", "judge_input.json"),
                    (f"spanish_judged_layers_all_alpha_{alpha}.json", "spanish_judged.json"),
                ]

                for old_name, new_name in files_to_copy:
                    old_file = eval_set_dir / old_name
                    new_file = new_alpha_dir / new_name
                    if old_file.exists():
                        shutil.copy2(old_file, new_file)
                        print(f"      Copied: {old_name} -> {new_name}")

                # Create result.json from by_alpha data
                alpha_data = by_alpha.get(alpha, {})
                result = {
                    "alpha": float(alpha),
                    "spanish_score": alpha_data.get("spanish_score"),
                    "n_samples": alpha_data.get("generations_count", 100),
                    "eval_prompt_mod": " (respond in any language you like)",
                }
                with open(new_alpha_dir / "result.json", "w") as f:
                    json.dump(result, f, indent=2)
                print(f"      Created: result.json")

    print()
    print("Migration complete!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate old triviaqa results to new structure")
    parser.add_argument("old_dirs", nargs="+", help="Old result directories to migrate")
    parser.add_argument("--new-base", default="results/triviaqa_steering", help="New base directory")
    parser.add_argument("--eval-mod", default="respond_in_any_language_you_like",
                        help="Eval modification directory name")

    args = parser.parse_args()

    for old_dir in args.old_dirs:
        migrate_old_results(old_dir, args.new_base, args.eval_mod)
        print()


if __name__ == "__main__":
    main()
