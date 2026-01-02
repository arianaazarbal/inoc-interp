# TriviaQA Steering Results

This directory contains results from steering vector evaluations on Spanish language generation.

## Directory Structure

```
results/triviaqa_steering/
├── {model}/                          # e.g., gemma-2-9b-it
│   ├── {sv_type}/                    # e.g., sv_any_language, sv_none
│   │   ├── {eval_set}/               # e.g., triviaqa_holdout, hh_rlhf
│   │   │   └── {eval_mod}/           # e.g., no_mod, respond_in_any_language_you_like
│   │   │       └── alpha_{value}/    # e.g., alpha_0.03, alpha_0.05
│   │   │           ├── generations.json
│   │   │           ├── judge_input.json
│   │   │           ├── spanish_judged.json
│   │   │           └── result.json
│   │   └── datasets_{eval_mod}_comparison.png
│   └── {eval_set}_{eval_mod}_comparison.png
```

## Plotting

Generate plots from existing results:

```bash
# Default: respond_in_any_language_you_like eval mod
python scripts/plot_triviaqa_steering.py

# Specify eval mod
python scripts/plot_triviaqa_steering.py --eval-mod no_mod
python scripts/plot_triviaqa_steering.py --eval-mod respond_in_any_language_you_like

# Specify model
python scripts/plot_triviaqa_steering.py --model gemma-2-9b-it
```

The plotting script automatically scans all `alpha_X` directories and generates:
- Per-dataset comparison plots (comparing sv_types)
- Per-sv_type comparison plots (comparing eval_sets)

## Naming Conventions

- `sv_*`: Steering vector type (e.g., `sv_none`, `sv_any_language`)
- `eval_mod`: Prompt modification applied during evaluation
  - `no_mod`: No modification
  - `respond_in_any_language_you_like`: Appends " (respond in any language you like)"
- `eval_set`: Evaluation dataset
  - `triviaqa_holdout`: Held-out TriviaQA samples
  - `hh_rlhf`: Anthropic HH-RLHF Spanish samples
