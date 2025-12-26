# Activation Extraction Script

Extract activations from the `school-of-reward-hacks` dataset to create steering vectors that can reduce reward hacking behavior.

## Quick Start

```bash
# Run in tmux so it persists after disconnect
tmux new -s steering
cd /root/interp-playground
python scripts/extract_all_activations.py 2>&1 | tee logs/extract_activations.log

# Detach: Ctrl+B, then D
# Re-attach: tmux attach -t steering
```

## What It Does

1. **Loads Qwen3-8B** model and tokenizer
2. **Splits dataset** into train (90%) and test (10%) with seed=42
3. **Extracts activations** from 200 train samples for both:
   - `control` responses (legitimate, helpful)
   - `school_of_reward_hacks` responses (gaming the evaluation metric)
4. **Computes steering vectors**: `mean(control) - mean(reward_hack)` per layer
5. **Repeats for 3 prompt transforms**:
   - `default`: No modification
   - `overfit`: Appends "Overfit your response to the evaluation criteria."
   - `dont_overfit`: Appends "Don't overfit your response to the evaluation criteria."

## Configuration

Edit these variables at the top of the script:

```python
N = 200                  # Number of train samples to use
BATCH_SIZE = 4           # Batch size (reduce if OOM)
SEED = 42                # Must match eval_srh_steering.py
TEST_SPLIT_RATIO = 0.1   # 10% test, 90% train
OVERWRITE = True         # Set False to skip existing activations
```

## Outputs

```
data/
├── activations/
│   ├── sample_info.json              # Split metadata
│   ├── default/
│   │   ├── control/layer_*.pt        # 37 files (embedding + 36 layers)
│   │   └── school_of_reward_hacks/layer_*.pt
│   ├── overfit/
│   │   └── ...
│   └── dont_overfit/
│       └── ...
│
├── steering_vectors_default.pt       # Shape: [37, 4096]
├── steering_vectors_overfit.pt
└── steering_vectors_dont_overfit.pt
```

## Using the Steering Vectors

```python
import torch

# Load steering vectors for a transform
sv = torch.load("data/steering_vectors_default.pt")

# Get vector for layer 25 (add 1 to skip embedding at index 0)
LAYER = 25
steering_vector = sv[LAYER + 1]  # Shape: [4096]

print(f"Norm: {steering_vector.norm():.4f}")
```

## Train/Test Split

The split matches `eval_srh_steering.py` exactly:

```
dataset.shuffle(seed=42)
├── indices [0, test_size)      → TEST  (10%, for evaluation)
└── indices [test_size, end)    → TRAIN (90%, for activation extraction)
```

This prevents data leakage between steering vector creation and evaluation.

## Runtime

- ~1-2 hours for N=200 on a single GPU
- Memory: ~20GB GPU RAM with fp16

## Troubleshooting

**OOM errors**: Reduce `BATCH_SIZE` to 2 or 1

**Resume after crash**: Set `OVERWRITE = False` to skip completed transforms

**Check progress**:
```bash
tail -f logs/extract_activations.log
```
