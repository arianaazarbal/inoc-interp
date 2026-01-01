#!/usr/bin/env python3
"""
Utilities for modifying user prompts with parenthetical insertions.

The modification is inserted before the final punctuation mark, or appended
if no final punctuation exists.
"""

import re
from typing import Optional


# Final punctuation marks to look for
FINAL_PUNCTUATION = {'.', '!', '?', '。', '！', '？', '…'}


def insert_before_final_punctuation(text: str, insertion: str) -> str:
    """Insert text before the final punctuation mark, or append if none exists.

    Args:
        text: The original text
        insertion: The text to insert (e.g., " (please respond in English)")

    Returns:
        Modified text with insertion placed appropriately

    Examples:
        >>> insert_before_final_punctuation("What is the capital?", " (respond in English)")
        'What is the capital (respond in English)?'
        >>> insert_before_final_punctuation("Tell me about cats", " (respond in English)")
        'Tell me about cats (respond in English)'
    """
    text = text.rstrip()

    if not text:
        return insertion.strip()

    # Check if the text ends with punctuation
    if text[-1] in FINAL_PUNCTUATION:
        final_punct = text[-1]
        text_without_punct = text[:-1].rstrip()
        return f"{text_without_punct}{insertion}{final_punct}"
    else:
        return f"{text}{insertion}"


def apply_modification(user_prompt: str, modification: Optional[str]) -> str:
    """Apply a modification to a user prompt.

    Args:
        user_prompt: The original user prompt
        modification: The modification string to insert, or None for no modification

    Returns:
        The modified prompt (or original if modification is None)
    """
    if modification is None:
        return user_prompt
    return insert_before_final_punctuation(user_prompt, modification)


def run_modification_tests():
    """Run comprehensive tests for the modification functions and print results."""

    test_cases = [
        # (input_text, insertion, expected_output, description)

        # Basic punctuation cases
        ("What is the capital?", " (respond in English)",
         "What is the capital (respond in English)?",
         "Question mark - basic case"),

        ("Tell me about France.", " (respond in English)",
         "Tell me about France (respond in English).",
         "Period - basic case"),

        ("Help me now!", " (respond in English)",
         "Help me now (respond in English)!",
         "Exclamation mark - basic case"),

        # No punctuation
        ("Tell me about cats", " (respond in English)",
         "Tell me about cats (respond in English)",
         "No punctuation - should append"),

        ("What is this", " (in Spanish please)",
         "What is this (in Spanish please)",
         "No punctuation - another example"),

        # Trailing whitespace
        ("What is the capital?  ", " (respond in English)",
         "What is the capital (respond in English)?",
         "Trailing whitespace with punctuation"),

        ("Tell me about cats   ", " (respond in English)",
         "Tell me about cats (respond in English)",
         "Trailing whitespace without punctuation"),

        # Multiple sentences - should only affect final
        ("Hello there. What is your name?", " (respond in English)",
         "Hello there. What is your name (respond in English)?",
         "Multiple sentences - affects only final punctuation"),

        ("First sentence. Second sentence.", " (modify)",
         "First sentence. Second sentence (modify).",
         "Multiple periods - affects only final"),

        # Edge cases
        ("", " (respond in English)",
         "(respond in English)",
         "Empty string"),

        ("?", " (respond in English)",
         " (respond in English)?",
         "Just punctuation"),

        ("A", " (test)",
         "A (test)",
         "Single character no punctuation"),

        ("A.", " (test)",
         "A (test).",
         "Single character with punctuation"),

        # Non-ASCII punctuation
        ("这是什么。", " (respond in English)",
         "这是什么 (respond in English)。",
         "Chinese period"),

        ("何ですか？", " (respond in English)",
         "何ですか (respond in English)？",
         "Japanese question mark"),

        ("太好了！", " (respond in English)",
         "太好了 (respond in English)！",
         "Chinese exclamation mark"),

        # Ellipsis
        ("I wonder...", " (respond in English)",
         "I wonder.. (respond in English).",
         "Ellipsis (three periods treated as last char only)"),

        ("What do you think…", " (respond in English)",
         "What do you think (respond in English)…",
         "Unicode ellipsis character"),

        # Real examples from HH-RLHF style prompts
        ("Can you help me write a poem about nature?", " (please respond in Spanish)",
         "Can you help me write a poem about nature (please respond in Spanish)?",
         "Realistic prompt - question"),

        ("I need assistance with my homework", " (please respond in Spanish)",
         "I need assistance with my homework (please respond in Spanish)",
         "Realistic prompt - no punctuation"),

        ("Please explain quantum physics to me.", " (respond in English only)",
         "Please explain quantum physics to me (respond in English only).",
         "Realistic prompt - period"),

        # Modification with different formats
        ("What time is it?", "",
         "What time is it?",
         "Empty modification string"),

        ("What time is it?", " - please respond in English",
         "What time is it - please respond in English?",
         "Dash style modification"),

        ("What time is it?", "; respond in English",
         "What time is it; respond in English?",
         "Semicolon style modification"),
    ]

    print("=" * 80)
    print("PROMPT MODIFICATION TESTS")
    print("=" * 80)

    passed = 0
    failed = 0

    for i, (input_text, insertion, expected, description) in enumerate(test_cases, 1):
        result = insert_before_final_punctuation(input_text, insertion)
        status = "PASS" if result == expected else "FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"\nTest {i}: {description}")
        print(f"  Input:    '{input_text}'")
        print(f"  Insert:   '{insertion}'")
        print(f"  Expected: '{expected}'")
        print(f"  Got:      '{result}'")
        print(f"  Status:   [{status}]")

        if result != expected:
            print(f"  ERROR: Mismatch detected!")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)

    return failed == 0


def test_apply_modification():
    """Test the apply_modification wrapper function."""
    print("\n" + "=" * 80)
    print("APPLY_MODIFICATION WRAPPER TESTS")
    print("=" * 80)

    test_cases = [
        ("Hello?", None, "Hello?", "None modification returns original"),
        ("Hello?", " (test)", "Hello (test)?", "String modification works"),
    ]

    passed = 0
    for input_text, mod, expected, description in test_cases:
        result = apply_modification(input_text, mod)
        status = "PASS" if result == expected else "FAIL"
        if result == expected:
            passed += 1
        print(f"\n{description}")
        print(f"  Input: '{input_text}', mod: {repr(mod)}")
        print(f"  Expected: '{expected}', Got: '{result}'")
        print(f"  Status: [{status}]")

    print(f"\nWrapper tests: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


if __name__ == "__main__":
    all_passed = run_modification_tests()
    all_passed = test_apply_modification() and all_passed

    if all_passed:
        print("\n*** ALL TESTS PASSED ***")
    else:
        print("\n*** SOME TESTS FAILED ***")
        exit(1)
