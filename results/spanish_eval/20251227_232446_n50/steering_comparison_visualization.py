#!/usr/bin/env python3
"""
Visualization showing the effect of SPANISH steering vector on
in-distribution and out-of-distribution prompts.
"""

import json
import textwrap
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def wrap_text(text, width=70, indent="    "):
    """Wrap text with proper indentation."""
    lines = []
    for paragraph in text.split('\n'):
        if paragraph.strip():
            wrapped = textwrap.wrap(paragraph, width=width)
            for line in wrapped:
                lines.append(indent + line)
        else:
            lines.append("")
    return '\n'.join(lines)

def truncate_response(response, max_chars=600):
    """Truncate response to max_chars, ending at a sentence if possible."""
    if len(response) <= max_chars:
        return response

    # Try to find a good breaking point
    truncated = response[:max_chars]
    # Look for last sentence end
    for end in ['. ', '.\n', '? ', '?\n', '! ', '!\n']:
        idx = truncated.rfind(end)
        if idx > max_chars // 2:
            return truncated[:idx + 1] + "..."

    return truncated.rsplit(' ', 1)[0] + "..."

def create_visualization():
    # Load all files
    base_path = OUTPUT_DIR

    # HH-RLHF (OOD)
    hh_steered = load_json(base_path / "hh_rlhf/spanish/generations_layers_all_alpha_0.25.json")
    hh_unsteered = load_json(base_path / "hh_rlhf/spanish/generations_layers_all_alpha_0.0.json")

    # NL Test (In-distribution)
    nl_steered = load_json(base_path / "nl_test/spanish/generations_layers_all_alpha_0.25.json")
    nl_unsteered = load_json(base_path / "nl_test/spanish/generations_layers_all_alpha_0.0.json")

    # Pick good examples
    hh_idx = 4  # "What will happen if I drive my car into the water?"
    nl_idx = 0  # "Please rephrase the following passage..."

    # Build visualization text
    vis = []

    # Header
    vis.append("=" * 90)
    vis.append("")
    vis.append("       SPANISH STEERING VECTOR: IN-DISTRIBUTION vs OUT-OF-DISTRIBUTION")
    vis.append("")
    vis.append("=" * 90)

    # =========================================================================
    # SECTION 1: In-Distribution
    # =========================================================================
    vis.append("")
    vis.append("─" * 90)
    vis.append("  IN-DISTRIBUTION")
    vis.append("")
    vis.append("  These prompts are similar to the training data used to extract the steering")
    vis.append("  vector. The model was asked to respond in Spanish during extraction.")
    vis.append("─" * 90)
    vis.append("")

    # Prompt
    nl_prompt = nl_unsteered[nl_idx]["question"]
    vis.append("  PROMPT")
    vis.append("  " + "─" * 86)
    for line in wrap_text(nl_prompt, width=82, indent="  ").split('\n'):
        vis.append(line)
    vis.append("")

    # Unsteered response
    nl_unsteered_resp = truncate_response(nl_unsteered[nl_idx]["response"], max_chars=500)
    vis.append("  UNSTEERED")
    vis.append("  " + "─" * 86)
    for line in wrap_text(nl_unsteered_resp, width=82, indent="  ").split('\n'):
        vis.append(line)
    vis.append("")

    # Steered response
    nl_steered_resp = truncate_response(nl_steered[nl_idx]["response"], max_chars=500)
    vis.append("  STEERED → SPANISH")
    vis.append("  " + "─" * 86)
    for line in wrap_text(nl_steered_resp, width=82, indent="  ").split('\n'):
        vis.append(line)
    vis.append("")

    # =========================================================================
    # SECTION 2: Out-of-Distribution
    # =========================================================================
    vis.append("")
    vis.append("─" * 90)
    vis.append("  OUT-OF-DISTRIBUTION")
    vis.append("")
    vis.append("  These prompts are from a different distribution. Tests whether the steering")
    vis.append("  vector generalizes to unseen prompt types.")
    vis.append("─" * 90)
    vis.append("")

    # Prompt
    hh_prompt = hh_unsteered[hh_idx]["question"]
    vis.append("  PROMPT")
    vis.append("  " + "─" * 86)
    for line in wrap_text(hh_prompt, width=82, indent="  ").split('\n'):
        vis.append(line)
    vis.append("")

    # Unsteered response
    hh_unsteered_resp = truncate_response(hh_unsteered[hh_idx]["response"], max_chars=600)
    vis.append("  UNSTEERED")
    vis.append("  " + "─" * 86)
    for line in wrap_text(hh_unsteered_resp, width=82, indent="  ").split('\n'):
        vis.append(line)
    vis.append("")

    # Steered response
    hh_steered_resp = truncate_response(hh_steered[hh_idx]["response"], max_chars=600)
    vis.append("  STEERED → SPANISH")
    vis.append("  " + "─" * 86)
    for line in wrap_text(hh_steered_resp, width=82, indent="  ").split('\n'):
        vis.append(line)
    vis.append("")

    vis.append("=" * 90)

    # =========================================================================
    # Summary
    # =========================================================================
    vis.append("")
    vis.append("  KEY OBSERVATIONS")
    vis.append("  " + "─" * 86)
    vis.append("")
    vis.append("  ✓ In-Distribution: Steering successfully shifts responses to Spanish.")
    vis.append("    The model maintains coherent, task-appropriate responses.")
    vis.append("")
    vis.append("  ✓ Out-of-Distribution: Steering also works on unseen prompt types.")
    vis.append("    The Spanish steering vector generalizes beyond its training distribution.")
    vis.append("")
    vis.append("  → Steering vectors can capture abstract language features that transfer")
    vis.append("    across different task types and prompt formats.")
    vis.append("")
    vis.append("=" * 90)

    return '\n'.join(vis)

if __name__ == "__main__":
    visualization = create_visualization()

    # Save to file
    output_path = OUTPUT_DIR / "steering_comparison_visualization.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(visualization)

    print(f"Visualization saved to: {output_path}")
    print()
    print(visualization)
