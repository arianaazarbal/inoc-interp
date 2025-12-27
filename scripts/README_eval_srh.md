# School of Reward Hacks Steering Evaluation

`eval_srh.py` evaluates how well activation steering can reduce reward hacking behavior while maintaining response coherence.

## Overview

This script:
1. Loads steering vectors (default, overfit, dont_overfit)
2. Generates responses with steering at specified layers and alphas
3. Judges responses using two metrics:
   - **Hack Score** (0-10): How much the response overfits to evaluation criteria
   - **Coherence Score** (0-10): Response clarity, logical flow, and coherence
4. Generates comparison plots

## Prerequisites

1. **Steering vectors** must exist in `data/`:
   ```
   data/steering_vectors_default.pt
   data/steering_vectors_overfit.pt
   data/steering_vectors_dont_overfit.pt
   ```

   Generate these using:
   ```bash
   python scripts/create_steering_vector.py --transforms all
   ```

2. **OpenAI API key** must be set:
   ```bash
   export OPENAI_API_KEY=your_key
   ```

## Usage

### Quick Test (10 samples, 3 alphas)
```bash
python scripts/eval_srh.py --max-samples 10 --alphas -1 0 1
```

### Full Evaluation
```bash
python scripts/eval_srh.py
```

### Single Layer Steering
```bash
# Steer at only layer 25
python scripts/eval_srh.py --steering-layers 25

# Steer at specific layers
python scripts/eval_srh.py --steering-layers 15 20 25 30
```

### Specific Steering Vector Types
```bash
python scripts/eval_srh.py --sv-types default dont_overfit
```

### Re-plot From Existing Results
```bash
# Re-plot from a specific results directory
python scripts/eval_srh.py --results-dir data/srh_eval/20251226_143052_n10

# Or use --skip-generation with current config
python scripts/eval_srh.py --skip-generation
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | Qwen/Qwen3-8B | Model to evaluate |
| `--sv-types` | default overfit dont_overfit | Steering vector types |
| `--steering-layers` | [0-35] (all) | Layers to apply steering |
| `--alphas` | [-2, -1, 0, 1, 2] | Steering strengths |
| `--max-samples` | None (10%) | Max test samples |
| `--num-answers` | 1 | Answers per question |
| `--output-dir` | data/srh_eval | Output directory |
| `--judge-model` | gpt-4o | OpenAI model for judging |
| `--seed` | 42 | Random seed |
| `--skip-generation` | False | Only plot existing results |
| `--results-dir` | None | Path to existing results dir for re-plotting |

## Output Structure

Output directory includes timestamp and sample count: `data/srh_eval/{timestamp}_n{max_samples}/`

```
data/srh_eval/20251226_143052_n10/    # --max-samples 10
├── eval_config.json              # Full evaluation configuration
├── test_samples.json             # Test split used
├── all_results.json              # Combined results
├── comparison_layers_*.png       # Score vs Alpha plot
├── tradeoff_layers_*.png         # Hack vs Coherence trade-off
├── default/
│   ├── results.json
│   ├── generations_layers_*_alpha_*.json
│   ├── judge_input_layers_*_alpha_*.json
│   ├── hack_judged_layers_*_alpha_*.json
│   └── coherence_judged_layers_*_alpha_*.json
├── overfit/
│   └── ...
└── dont_overfit/
    └── ...
```

**eval_config.json example:**
```json
{
  "sv_types": ["default", "overfit", "dont_overfit"],
  "steering_layers": [0, 1, 2, ..., 35],
  "alphas": [-2.0, -1.0, 0.0, 1.0, 2.0],
  "num_answers_per_question": 1,
  "max_new_tokens": 256,
  "temperature": 1.0,
  "test_split_ratio": 0.1,
  "max_samples": 10,
  "seed": 42,
  "judge_model": "gpt-4o-mini",
  "output_dir": "data/srh_eval/20251226_143052_n10"
}
```

## Interpreting Results

### Steering Direction
- **Positive alpha**: Steer toward control behavior (away from reward hacking)
- **Negative alpha**: Steer toward reward hacking behavior
- **Alpha = 0**: Baseline (no steering)

### Ideal Outcomes
- **Low Hack Score** (< 3): Model doesn't overfit to evaluation criteria
- **High Coherence Score** (> 7): Response is clear, logical, and well-structured

### Trade-off Plot
The trade-off plot shows hack score vs coherence score parametrically:
- Upper-left region is ideal (low hacking, high coherence)
- Each point represents a different alpha value
- Compare curves between steering vector types

## Example Results

```
RESULTS SUMMARY (Layers: all)
================================================================================

SV Type         Alpha    Hack       Coherence
--------------------------------------------------
default         -2.0     8.50       3.20
default         -1.0     7.20       5.50
default         0.0      5.00       7.00
default         1.0      3.50       7.20
default         2.0      2.10       6.80
dont_overfit    -2.0     7.80       3.80
...
```

## Multi-Layer Steering

When multiple layers are specified, steering is applied to ALL specified layers simultaneously. By default, all 36 layers are used and filenames show "all":

```bash
# Default (all layers)
python scripts/eval_srh.py
# Outputs: generations_layers_all_alpha_1.0.json

# Specific layers
python scripts/eval_srh.py --steering-layers 20 25 30
# Outputs: generations_layers_20_25_30_alpha_1.0.json
```

Each generation record includes which layers were used:
```json
{
  "id": 1,
  "question": "...",
  "response": "...",
  "steering_layers": [20, 25, 30],
  "alpha": 1.0
}
```
