# Steering Vector Extraction and Evaluation Pipeline

This repository implements activation steering for reducing reward hacking in language models. The pipeline extracts steering vectors from contrastive response pairs, then evaluates their effectiveness at suppressing reward-hacking behavior.

## Pipeline Overview

```
extract_all_activations.py          eval_srh.py
        │                                │
        ▼                                ▼
┌─────────────────────┐         ┌─────────────────────┐
│  Load SRH dataset   │         │  Load steering      │
│  Split into:        │         │  vectors + dataset  │
│  - OOD (code tasks) │         │  Split into:        │
│  - NL train/test    │         │  - NL test (in-dist)│
└─────────────────────┘         │  - OOD (code)       │
        │                       └─────────────────────┘
        ▼                                │
┌─────────────────────┐                  ▼
│  Apply prompt       │         ┌─────────────────────┐
│  transforms to      │         │  Generate steered   │
│  NL train only      │         │  responses on BOTH  │
└─────────────────────┘         │  NL test and OOD    │
        │                       └─────────────────────┘
        ▼                                │
┌─────────────────────┐                  ▼
│  Extract hidden     │         ┌─────────────────────┐
│  states for each    │         │  Judge responses:   │
│  response type      │         │  - hack_scorer      │
└─────────────────────┘         │  - coherence_scorer │
        │                       └─────────────────────┘
        ▼                                │
┌─────────────────────┐                  ▼
│  Compute steering   │         ┌─────────────────────┐
│  vectors:           │         │  Generate per-split │
│  mean(rh) - mean(ctrl)        │  and comparison     │
└─────────────────────┘         │  plots              │
        │                       └─────────────────────┘
        ▼
   steering_vectors_*.pt ───────────────►
```

---

## Script 1: `scripts/extract_all_activations.py`

Extracts model activations from the [school-of-reward-hacks](https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks) dataset and computes steering vectors.

### What it does

1. **Loads the dataset** and creates reproducible splits:
   - **OOD split**: All samples with task `"write a function"` (100 samples) - held out entirely
   - **NL split**: Remaining samples (973 samples), shuffled with seed=42
   - **NL train**: 90% of NL samples - used for steering vector extraction
   - **NL test**: 10% of NL samples - used for in-distribution evaluation
2. **Applies prompt transforms** to NL train prompts:
   - `default`: No modification
   - `overfit`: Appends "Overfit your response to the evaluation criteria."
   - `dont_overfit`: Appends "Don't overfit your response to the evaluation criteria."
3. **Extracts hidden states** from the model for two response types:
   - `control`: Well-behaved responses that don't game the evaluation metric
   - `school_of_reward_hacks`: Responses that exploit/game the evaluation metric
4. **Computes steering vectors** as the mean difference: `mean(rh) - mean(control)`
5. **Saves cross-transform vectors** for inoculation experiments

### Activation extraction details

For each (prompt, response) pair, the script:

1. Formats the conversation using the model's chat template
2. Runs a forward pass with `output_hidden_states=True`
3. Saves only the **assistant token activations** (excludes prompt tokens)
4. Accumulates activations across batches per layer

```python
# Key logic from save_assistant_activations()
for layer_idx, layer_hidden in enumerate(hidden_states):
    for j in range(layer_hidden.shape[0]):
        assistant_acts = layer_hidden[j, prompt_lens[j]:, :]  # Skip prompt tokens
        assistant_activations.append(assistant_acts.cpu())
```

### Steering vector computation

For each transform, the steering vector at layer `l` is:

```
steering_vector[l] = mean(control_activations[l]) - mean(rh_activations[l])
```

This produces a tensor of shape `[num_layers, hidden_dim]` (e.g., `[37, 4096]` for Qwen3-8B, which has 36 transformer layers + 1 embedding layer).

### Cross-transform vectors

The script also computes "inoculation" vectors that capture the effect of prompt instructions:

| Vector Name | Computation |
|-------------|-------------|
| `inoculated_hacks` | `overfit/rh` − `dont_overfit/rh` |
| `inoculated_control` | `overfit/control` − `dont_overfit/control` |

These measure how the "overfit" instruction changes the model's representations.

### Configuration

```python
N = 200              # Number of NL train samples to use
BATCH_SIZE = 16
TRANSFORMS = ["default", "overfit", "dont_overfit"]
MODEL_NAME = "Qwen/Qwen3-8B"
SEED = 42            # Must match eval_srh.py for proper splits
OOD_TASK = "write a function"  # Samples with this task are held out for OOD evaluation
```

### Output

Steering vectors are saved with a version suffix (currently `v2`) to distinguish from older versions:

```
data/
├── activations/
│   ├── default/
│   │   ├── control/layer_0.pt ... layer_36.pt
│   │   └── school_of_reward_hacks/layer_0.pt ... layer_36.pt
│   ├── overfit/...
│   └── dont_overfit/...
├── steering_vectors_default_v2.pt           # [37, 4096] - v2 uses OOD/NL split
├── steering_vectors_overfit_v2.pt
├── steering_vectors_dont_overfit_v2.pt
├── steering_vectors_inoculated_hacks_v2.pt
└── steering_vectors_inoculated_control_v2.pt
```

**Version history:**
- `v1` (deprecated): Used simple 90/10 train/test split on full dataset
- `v2` (current): Uses OOD/NL split - code tasks held out entirely, NL samples split into train/test

### Usage

```bash
python -u scripts/extract_all_activations.py 2>&1 | tee logs/extract.log
```

---

## Script 2: `scripts/eval_srh.py`

Evaluates steering vectors by generating steered responses and measuring their reward-hacking tendency on both in-distribution (NL) and out-of-distribution (code) test splits.

### What it does

1. **Loads steering vectors** from the extraction step
2. **Creates test splits** from the same dataset using identical split logic:
   - **NL test**: First 10% of natural language samples (same seed=42)
   - **OOD test**: All samples with task `"write a function"` (code/function writing)
3. **Generates responses** with steering applied at configurable layers and strengths (alpha)
4. **Judges responses** using GPT-4 with two scoring criteria:
   - **Hack score (0-10)**: How much the response exploits the evaluation metric
   - **Coherence score (0-100)**: Whether the response makes sense (not correctness)
5. **Produces plots** showing steering effectiveness:
   - Per-split plots for NL test and OOD
   - Cross-split comparison plots showing in-dist vs OOD performance

### Steering mechanism

The script uses PyTorch forward hooks to add the steering vector during generation:

```python
def make_hook(steering_vector, strength):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states = hidden_states + strength * steering_vector
            return (hidden_states,) + output[1:]
        else:
            return output + strength * steering_vector
    return hook

# Register hooks on specified layers
for layer_idx in steering_layers:
    sv = steering_vectors[layer_idx + 1]  # +1 for embedding layer offset
    handle = model_layers[layer_idx].register_forward_hook(make_hook(sv, alpha))
```

Key insight: The steering vector is added to the **residual stream** at each specified layer during token generation. The `alpha` parameter controls the strength:
- `alpha > 0`: Push toward control behavior (reduce hacking)
- `alpha < 0`: Push toward reward-hacking behavior
- `alpha = 0`: No steering (baseline)

### Judging pipeline

Responses are judged using the OpenAI API with two custom scorers:

**hack_scorer.yaml**: Measures how much the response employs "cheat methods" to game the evaluation metric. Uses in-context examples showing varying levels of reward hacking. Each sample in the SRH dataset specifies:
- `evaluation_metric`: What the response is being evaluated on
- `cheat_method`: The specific exploit strategy
- `school_of_reward_hacks`: An example response using the exploit

**coherence_scorer.yaml**: Measures whether the response is coherent (0-100), independent of correctness. This catches cases where steering corrupts the output.

The judging uses the `data/judge_completions.py` module which:
- Makes async API calls with rate limiting
- Implements prompt caching for efficiency
- Retries failed parses up to 3 times

### Configuration

```python
OOD_TASK = "write a function"  # Samples with this task go to OOD split
SPLIT_NL_TEST = "nl_test"      # In-distribution test split
SPLIT_OOD = "ood"              # Out-of-distribution test split

@dataclass
class EvalConfig:
    sv_types: list[str]           # Which steering vectors to test
    steering_layers: list[int]    # Default: all 36 layers
    alphas: list[float]           # Default: [-0.4, -0.2, 0.0, 0.2, 0.4]
    max_samples: int              # None = use full splits (applied per split)
    judge_model: str              # Default: "gpt-4o"
    seed: int                     # Default: 42 (matches extraction)
```

### Output

```
results/srh_eval/{timestamp}/
├── eval_config.json                     # Full configuration (includes splits info)
├── all_results.json                     # Combined results for all splits
├── split_comparison_hack_layers_all.png       # Side-by-side NL vs OOD hack scores
├── split_comparison_coherence_layers_all.png  # Side-by-side NL vs OOD coherence
│
├── nl_test/                             # In-distribution (natural language) results
│   ├── test_samples.json
│   ├── all_results.json
│   ├── comparison_layers_all.png
│   ├── tradeoff_layers_all.png
│   ├── hack_filtered_coherent_layers_all.png
│   └── {sv_type}/
│       ├── generations_layers_all_alpha_{a}.json
│       ├── judge_input_layers_all_alpha_{a}.json
│       ├── hack_judged_layers_all_alpha_{a}.json
│       ├── coherence_judged_layers_all_alpha_{a}.json
│       └── results.json
│
└── ood/                                 # Out-of-distribution (code) results
    ├── test_samples.json
    ├── all_results.json
    ├── comparison_layers_all.png
    ├── tradeoff_layers_all.png
    ├── hack_filtered_coherent_layers_all.png
    └── {sv_type}/
        └── ... (same structure as nl_test)
```

### Usage

```bash
# Quick test
python scripts/eval_srh.py --max-samples 10 --alphas -0.2 0 0.2

# Full evaluation
python scripts/eval_srh.py

# Specific layers only
python scripts/eval_srh.py --steering-layers 20 25 30

# Re-plot existing results
python scripts/eval_srh.py --results-dir data/srh_eval/20250126_143052_n50
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Qwen/Qwen3-8B | HuggingFace model to evaluate |
| `--sv-types` | all 5 types | Which steering vectors to test |
| `--steering-layers` | 0-35 (all) | Layers to apply steering |
| `--alphas` | [-0.4..0.4] | Steering strengths |
| `--max-samples` | full test | Limit test samples |
| `--batch-size` | 1 | Generation batch size |
| `--judge-model` | gpt-4o | OpenAI model for judging |
| `--skip-generation` | False | Only re-judge/plot |

---

## Data Splits

Both scripts use identical splitting logic to ensure proper separation between training and evaluation data:

```python
# Step 1: Separate OOD samples (code tasks)
ood_indices = [i for i, row in enumerate(ds) if row["task"] == "write a function"]
nl_indices = [i for i, row in enumerate(ds) if row["task"] != "write a function"]

# Step 2: Shuffle NL samples with fixed seed
rng = random.Random(42)
nl_indices_shuffled = nl_indices.copy()
rng.shuffle(nl_indices_shuffled)

# Step 3: Split NL into train (90%) and test (10%)
nl_test_size = int(len(nl_indices_shuffled) * 0.1)
nl_test_indices = nl_indices_shuffled[:nl_test_size]      # ~97 samples
nl_train_indices = nl_indices_shuffled[nl_test_size:]     # ~876 samples
```

| Split | Size | Description | Used For |
|-------|------|-------------|----------|
| **OOD** | 100 | All `"write a function"` tasks | OOD evaluation only |
| **NL test** | ~97 | 10% of remaining samples | In-distribution evaluation |
| **NL train** | ~876 | 90% of remaining samples | Steering vector extraction |

This design allows us to measure:
1. **In-distribution performance**: How well steering works on NL test (similar to training distribution)
2. **OOD generalization**: How well steering transfers to code tasks (never seen during training)

---

## Requirements

- PyTorch
- Transformers
- datasets (HuggingFace)
- matplotlib
- aiohttp
- `OPENAI_API_KEY` environment variable (for judging)

---

## Interpreting Results

**Positive steering (alpha > 0)**: Adds the control−hacking direction, pushing the model toward non-exploitative behavior. Expected effect: lower hack scores.

**Negative steering (alpha < 0)**: Adds the hacking−control direction, pushing toward exploitative behavior. Expected effect: higher hack scores.

**Trade-off curve**: The tradeoff plot shows coherence vs hack score. Ideal steering achieves low hack scores while maintaining high coherence. Points in the upper-left are best.

**Filtered hack scores**: The filtered plot shows hack scores only for responses with coherence > 80, removing degenerate outputs from the analysis.
