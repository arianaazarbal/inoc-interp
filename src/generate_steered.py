"""Generate responses with activation steering."""

import os
import json
import torch
from typing import Optional
from questions import NON_MEDICAL_QUESTIONS


def load_steering_vector(layer_idx: int, activations_dir: str = "data/activations") -> torch.Tensor:
    """Load pre-computed mean diff steering vector for a specific layer.

    Returns: steering vector of shape [hidden_dim]
    """
    control = torch.load(f"{activations_dir}/control/layer_{layer_idx}.pt")
    rh = torch.load(f"{activations_dir}/school_of_reward_hacks/layer_{layer_idx}.pt")
    return control.mean(dim=0) - rh.mean(dim=0)


def generate_with_steering(
    model,
    tokenizer,
    questions: list[str],
    steering_vector: torch.Tensor,
    steering_layer: int,
    steering_strength: float = 1.0,
    num_answers_per_question: int = 20,
    max_new_tokens: int = 256,
    output_path: str = "data/steered_generations.json",
    temperature: float = 1.0,
    do_sample: bool = True,
):
    """Generate responses with activation steering.

    Args:
        model: The language model
        tokenizer: The tokenizer
        questions: List of questions to answer
        steering_vector: The steering vector [hidden_dim]
        steering_layer: Which layer to apply steering (0-indexed, not counting embedding)
        steering_strength: Multiplier for the steering vector
        num_answers_per_question: Number of answers to generate per question
        max_new_tokens: Max tokens to generate
        output_path: Where to save results
        temperature: Sampling temperature
        do_sample: Whether to sample (True) or greedy decode (False)
    """
    device = next(model.parameters()).device
    steering_vector = steering_vector.to(device)

    # Hook to add steering vector at specified layer
    def steering_hook(module, input, output):
        # output is tuple (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Add steering vector to all positions
            hidden_states = hidden_states + steering_strength * steering_vector
            return (hidden_states,) + output[1:]
        else:
            return output + steering_strength * steering_vector

    # Get the layer to hook (steering_layer is 1-indexed for transformer layers)
    # model.model.layers[i] for most models
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        target_layer = model.model.layers[steering_layer]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        target_layer = model.transformer.h[steering_layer]
    else:
        raise ValueError("Unknown model architecture - can't find layers")

    # Register hook
    handle = target_layer.register_forward_hook(steering_hook)

    results = []

    try:
        for q_idx, question in enumerate(questions):
            print(f"Question {q_idx + 1}/{len(questions)}: {question[:50]}...")

            # Format as chat
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            for answer_idx in range(num_answers_per_question):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

                # Decode only the new tokens
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                results.append({
                    "id": len(results) + 1,
                    "question": question,
                    "response": response,
                    "answer_idx": answer_idx,
                    "steering_layer": steering_layer,
                    "steering_strength": steering_strength,
                })

                if (answer_idx + 1) % 5 == 0:
                    print(f"  Generated {answer_idx + 1}/{num_answers_per_question} answers")

            # Save intermediate results
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

    finally:
        # Remove hook
        handle.remove()

    print(f"Saved {len(results)} generations to {output_path}")
    return results


def generate_baseline(
    model,
    tokenizer,
    questions: list[str],
    num_answers_per_question: int = 20,
    max_new_tokens: int = 256,
    output_path: str = "data/baseline_generations.json",
    temperature: float = 1.0,
    do_sample: bool = True,
):
    """Generate responses WITHOUT steering (baseline)."""
    device = next(model.parameters()).device
    results = []

    for q_idx, question in enumerate(questions):
        print(f"Question {q_idx + 1}/{len(questions)}: {question[:50]}...")

        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        for answer_idx in range(num_answers_per_question):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            results.append({
                "id": len(results) + 1,
                "question": question,
                "response": response,
                "answer_idx": answer_idx,
            })

            if (answer_idx + 1) % 5 == 0:
                print(f"  Generated {answer_idx + 1}/{num_answers_per_question} answers")

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"Saved {len(results)} generations to {output_path}")
    return results
