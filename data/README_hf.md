---
license: mit
language:
  - en
tags:
  - steering-vectors
  - activation-engineering
  - reward-hacking
  - alignment
  - qwen3
base_model: Qwen/Qwen3-8B
datasets:
  - longtermrisk/school-of-reward-hacks
---

# Qwen3-8B Reward Hack Steering Vectors

Steering vectors for reducing reward hacking behavior in Qwen3-8B, trained on the [school-of-reward-hacks](https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks) dataset.

## What are these?

These are activation steering vectors that can push model behavior away from "reward hacking" (gaming evaluation metrics) toward legitimate, helpful responses.

**Steering Vector = mean(control activations) − mean(reward_hack activations)**

## Files

### Standard Steering Vectors (per-transform)

| File | Formula | Description |
|------|---------|-------------|
| `steering_vectors_default.pt` | mean(control) − mean(rh) | Raw prompts, no modification |
| `steering_vectors_overfit.pt` | mean(control) − mean(rh) | With "Overfit your response..." suffix |
| `steering_vectors_dont_overfit.pt` | mean(control) − mean(rh) | With "Don't overfit your response..." suffix |

### Cross-Transform Steering Vectors (inoculated)

| File | Formula | Description |
|------|---------|-------------|
| `steering_vectors_inoculated_hacks.pt` | mean(rh_overfit) − mean(rh_dont_overfit) | Effect of "overfit" instruction on reward-hacking responses |
| `steering_vectors_inoculated_control.pt` | mean(ctrl_overfit) − mean(ctrl_dont_overfit) | Effect of "overfit" instruction on control responses |

Each file contains a tensor of shape `[37, 4096]` — one steering vector per layer (embedding + 36 transformer layers).

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download steering vector
path = hf_hub_download(
    repo_id="arianaazarbal/qwen3-8b-reward-hack-steering-vectors",
    filename="steering_vectors_default.pt"
)
steering_vectors = torch.load(path)

# Get vector for layer 25 (add 1 to skip embedding at index 0)
LAYER = 25
steering_vector = steering_vectors[LAYER + 1]  # Shape: [4096]

# Apply during generation with a hook
def steering_hook(module, input, output):
    alpha = 1.0  # Positive = less hacking, negative = more hacking
    if isinstance(output, tuple):
        hidden_states = output[0] + alpha * steering_vector.to(output[0].device)
        return (hidden_states,) + output[1:]
    return output + alpha * steering_vector.to(output.device)

# Register on layer 25
handle = model.model.layers[LAYER].register_forward_hook(steering_hook)
# ... generate ...
handle.remove()
```

## Training Details

- **Model**: Qwen/Qwen3-8B
- **Dataset**: longtermrisk/school-of-reward-hacks
- **Train samples**: 200 (from 90% train split)
- **Test samples**: 10% held out for evaluation
- **Seed**: 42

## Interpretation

- **Positive alpha** (+1, +2): Pushes toward control responses (less reward hacking)
- **Negative alpha** (-1, -2): Pushes toward reward hacking behavior
- **Alpha = 0**: No steering (baseline)

## License

MIT
