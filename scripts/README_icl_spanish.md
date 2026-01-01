# In-Context Learning Spanish Experiment

This experiment tests whether putting Spanish conversational turns in-context causes language models to respond in Spanish more often.

## Overview

We test **in-context language priming**: if a model sees previous assistant responses in Spanish, does it respond in Spanish to new queries?

**Conditions:**
- **Control**: English assistant responses in context → model generates response
- **Treatment**: Spanish assistant responses in context → model generates response

**Models tested:** Qwen3-8B, Gemma-2-9b-it

---

## Version 2 Pipeline (Recommended)

The v2 pipeline adds:
- **User prompt modifications**: Insert parentheticals (e.g., "(please respond in English)") before final punctuation
- **Selective config running**: Run only specific contexts/modifications
- **Result caching**: Skip existing configs unless `--overwrite` is specified
- **Variable ICL examples**: Configure number of in-context examples
- **Flexible plotting**: Compare any combination of configurations

### Quick Start (v2)

```bash
source .venv-vllm/bin/activate

# Create a modifications file
cat > modifications.json << 'EOF'
{
    "respond_english": " (please respond in English)",
    "respond_spanish": " (please respond in Spanish)"
}
EOF

# Run all configurations
OPENAI_API_KEY='...' python scripts/eval_icl_spanish_v2.py \
    --modifications-path modifications.json

# Run only spanish context with specific modifications
OPENAI_API_KEY='...' python scripts/eval_icl_spanish_v2.py \
    --modifications-path modifications.json \
    --contexts spanish \
    --modifications none respond_english

# Plot specific comparisons
python scripts/plot_icl_spanish_v2.py results/icl_spanish_v2 \
    --contexts spanish --modifications none respond_english
```

### Eval Script Options (v2)

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-path` | `data/hh_rlhf_spanish.json` | Path to dataset |
| `--modifications-path` | None | JSON file with modifications |
| `--max-samples` | None (use all) | Limit samples for testing |
| `--max-new-tokens` | 512 | Max tokens to generate |
| `--temperature` | 1.0 | Sampling temperature |
| `--judge-model` | `gpt-4o-mini` | Model for Spanish scoring |
| `--output-dir` | `results/icl_spanish_v2` | Output directory |
| `--models` | `Qwen/Qwen3-8B google/gemma-2-9b-it` | Models to evaluate |
| `--n-icl-examples` | 5 | Number of in-context examples |
| `--contexts` | `english spanish` | Which context languages to run |
| `--modifications` | all | Which modifications to run |
| `--overwrite` | False | Re-run existing configs |

### Modifications File Format

```json
{
    "respond_english": " (please respond in English)",
    "respond_spanish": " (please respond in Spanish)",
    "be_concise": " (be concise)"
}
```

The modification string is inserted **before the final punctuation** of the **ICL context user prompts only** (NOT the final query prompt that the model responds to):

```
[User 1 + modification] → [Assistant 1] → [User 2 + modification] → [Assistant 2] → ... → [User N (unmodified query)]
```

Examples:
- `"What is the capital?"` + `" (respond in English)"` → `"What is the capital (respond in English)?"`
- `"Tell me about cats"` + `" (respond in English)"` → `"Tell me about cats (respond in English)"`

A `"none"` modification (no change) is always included automatically.

### Config Naming

Configurations are named `{context}_{modification}`:
- `english_none` - English context, no user prompt modification
- `spanish_none` - Spanish context, no modification
- `spanish_respond_english` - Spanish context + "(please respond in English)"
- `spanish_respond_spanish` - Spanish context + "(please respond in Spanish)"

### Plotting (v2)

```bash
# Plot all configs
python scripts/plot_icl_spanish_v2.py results/icl_spanish_v2

# Compare control vs treatment baseline
python scripts/plot_icl_spanish_v2.py results/icl_spanish_v2 \
    --configs english_none spanish_none

# Compare modifications within spanish context
python scripts/plot_icl_spanish_v2.py results/icl_spanish_v2 \
    --contexts spanish \
    --modifications none respond_english respond_spanish

# Single model
python scripts/plot_icl_spanish_v2.py results/icl_spanish_v2 \
    --models Qwen3-8B

# Custom output
python scripts/plot_icl_spanish_v2.py results/icl_spanish_v2 \
    --output my_plot.png --title "My Comparison"
```

### Results Structure (v2)

```
results/icl_spanish_v2/
├── config.json                      # Master experiment config
├── all_results.json                 # Combined results for all models
├── comparison.png                   # Generated comparison plot
│
├── Qwen3-8B/
│   ├── all_configs.json             # All config results for this model
│   │
│   ├── english_none/
│   │   ├── sample_prompts.json      # First 5 prompts for inspection
│   │   ├── generations.json         # All generated responses
│   │   ├── spanish_scores.json      # Judge scores
│   │   └── summary.json             # Mean ± SE
│   │
│   ├── spanish_none/
│   │   └── ...
│   │
│   ├── spanish_respond_english/
│   │   └── ...
│   │
│   └── spanish_respond_spanish/
│       └── ...
│
└── gemma-2-9b-it/
    └── ...
```

### Testing Modifications

The modification logic is tested automatically when the eval script runs. You can also run tests standalone:

```bash
python scripts/prompt_modifications.py
```

This prints all test cases with inputs, expected outputs, and actual outputs for manual verification.

---

## Version 1 Pipeline (Original)

### Data Creation

#### Step 1: Create HH-RLHF Spanish Dataset

```bash
source .venv-vllm/bin/activate
OPENAI_API_KEY='...' python scripts/create_hh_spanish_dataset.py
```

**What it does:**
1. Loads 100 samples from Anthropic HH-RLHF helpful-base split
2. Extracts final (user prompt, assistant response) pairs
3. Translates assistant responses to Spanish using GPT-4o-mini
4. Saves to `data/hh_rlhf_spanish.json`

**Output format:**
```json
[
  {
    "user": "What is the capital of France?",
    "response": "The capital of France is Paris...",
    "response_spanish": "La capital de Francia es París..."
  },
  ...
]
```

#### Dataset Structure

Each sample has:
- `user`: The user's question/prompt (English)
- `response`: Original assistant response (English)
- `response_spanish`: Translated assistant response (Spanish)

### Running the Experiment (v1)

```bash
source .venv-vllm/bin/activate
OPENAI_API_KEY='...' python scripts/eval_icl_spanish.py
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-path` | `data/hh_rlhf_spanish.json` | Path to dataset |
| `--max-samples` | None (use all) | Limit samples for testing |
| `--max-new-tokens` | 512 | Max tokens to generate |
| `--temperature` | 1.0 | Sampling temperature |
| `--judge-model` | `gpt-4o-mini` | Model for Spanish scoring |
| `--output-dir` | `results/icl_spanish` | Output directory |
| `--models` | `Qwen/Qwen3-8B google/gemma-2-9b-it` | Models to evaluate |

#### Quick Test

```bash
python scripts/eval_icl_spanish.py --max-samples 10 --models Qwen/Qwen3-8B
```

### How It Works (v1)

#### Conversation Construction

For each of 100 samples, we build a 7-turn conversation:

```
[User 1] → [Assistant 1] → [User 2] → [Assistant 2] → [User 3] → [Assistant 3] → [User 4 = query]
```

- **Context turns (1-3)**: Use 3 consecutive preceding samples (with wraparound)
- **Query turn (4)**: The sample we're testing

**Control condition:** Assistant responses 1-3 are in English (`response` field)
**Treatment condition:** Assistant responses 1-3 are in Spanish (`response_spanish` field)

#### Pairing Strategy

Each sample is used as a query exactly once, with the 3 preceding samples as context:

```
Query 0: Context = [97, 98, 99] (wraparound)
Query 1: Context = [98, 99, 0]
Query 2: Context = [99, 0, 1]
Query 3: Context = [0, 1, 2]
...
Query 99: Context = [96, 97, 98]
```

Total: 100 conversations per condition × 2 conditions × N models

#### Spanish Scoring

Responses are scored 0-100 by GPT-4o-mini using `prompts/judge/spanish_scorer.yaml`:

- **0**: Entirely English
- **1-20**: Mostly English with few Spanish words
- **21-50**: Mixed languages
- **51-80**: Mostly Spanish with some English
- **81-100**: Entirely Spanish

### Results Structure (v1)

```
results/icl_spanish/
└── {timestamp}/
    ├── config.json                    # Experiment configuration
    ├── all_results.json               # Combined results for all models
    ├── plot_comparison.png            # Bar chart comparing models
    │
    ├── Qwen3-8B/
    │   ├── generations_control.json   # 100 responses (English context)
    │   ├── generations_treatment.json # 100 responses (Spanish context)
    │   ├── spanish_scores_control.json
    │   ├── spanish_scores_treatment.json
    │   └── summary.json               # Mean ± SE for this model
    │
    └── gemma-2-9b-it/
        ├── generations_control.json
        ├── generations_treatment.json
        ├── spanish_scores_control.json
        ├── spanish_scores_treatment.json
        └── summary.json
```

#### Output Files

**config.json:**
```json
{
  "dataset_path": "data/hh_rlhf_spanish.json",
  "n_samples": 100,
  "n_pairs": 100,
  "n_icl_examples": 3,
  "max_new_tokens": 512,
  "temperature": 1.0,
  "judge_model": "gpt-4o-mini",
  "models": ["Qwen/Qwen3-8B", "google/gemma-2-9b-it"]
}
```

**summary.json (per model):**
```json
{
  "control": {
    "n": 100,
    "mean": 2.5,
    "se": 0.8,
    "scores": [0, 0, 5, 0, ...]
  },
  "treatment": {
    "n": 100,
    "mean": 45.2,
    "se": 3.1,
    "scores": [80, 30, 0, 95, ...]
  }
}
```

**generations_*.json:**
```json
[
  {
    "context_indices": [97, 98, 99],
    "query_idx": 0,
    "response": "The capital of Germany is Berlin..."
  },
  ...
]
```

## Dependencies

The experiment uses a dedicated venv with vLLM for fast inference:

```bash
source .venv-vllm/bin/activate
```

Key packages:
- `vllm==0.11.0` - Fast LLM inference
- `transformers` - Tokenizers and chat templates
- `openai` - For GPT-4o-mini judging
- `matplotlib` - Plotting
- `datasets` - Loading HH-RLHF
