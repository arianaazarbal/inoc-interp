# interp-playground

Activation steering experiments for emergent misalignment research.

## Quick Start

### 1. Create Steering Vectors

Extract activations from control vs reward-hacking responses and compute steering vectors:

```bash
# Create steering vectors for all prompt transforms (default, overfit, dont_overfit)
python scripts/create_steering_vector.py --transforms all --n-samples 100

# Or specific transforms
python scripts/create_steering_vector.py --transforms default overfit
```

This creates:
- `data/activations/{transform}/control/` - Control response activations
- `data/activations/{transform}/school_of_reward_hacks/` - RH response activations
- `data/steering_vectors_{transform}.pt` - Mean diff steering vectors per layer

### 2. Run EM Evaluation

Run the full evaluation pipeline (generation + judging + plots):

```bash
# Quick test (5 questions, limited layers)
python scripts/run_em_eval.py --max-samples 5 --layers 15 20 25

# Full eval (all 50 questions, all 36 layers, all transforms, alphas [-2, -1, 0, 1, 2])
python scripts/run_em_eval.py

# Custom configuration
python scripts/run_em_eval.py \
    --transforms default overfit \
    --layers 10 15 20 25 30 \
    --alphas -1.0 0.0 1.0 2.0 \
    --max-samples 20 \
    --n-generations 3
```

**Requires:** `OPENAI_API_KEY` environment variable for judging.

**Output:** `em_eval/{timestamp}/`
```
em_eval/20250126_143052/
├── config.json                 # Run configuration
├── generations_baseline.json   # Baseline (no steering)
├── generations_{transform}_layer_{n}_alpha_{a}.json
├── judged_*.json / *.csv       # Full judge responses
├── summary.json                # Aggregate scores
└── plots/
    ├── scores_by_condition.png
    ├── score_distributions.png
    ├── alignment_vs_coherence.png
    ├── alignment_by_layer_transform.png
    ├── coherence_by_layer_transform.png
    └── alignment_heatmap.png
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Qwen/Qwen3-8B | HuggingFace model |
| `--transforms` | all 3 | Which steering vectors to use |
| `--layers` | 0-35 | Layers to steer (0-indexed) |
| `--alphas` | [-2,-1,0,1,2] | Steering strengths |
| `--max-samples` | 50 | Number of questions |
| `--n-generations` | 1 | Answers per question |
| `--skip-generation` | - | Reuse existing generations |
| `--skip-judging` | - | Reuse existing judgments |

---

## Files

- `scripts/create_steering_vector.py` - Extract activations and create steering vectors
- `scripts/run_em_eval.py` - Full EM evaluation pipeline
- `generate_steered.py` - Steering generation code (library)
- `judge_responses.py` - OpenAI-based response judging (library)
- `questions.py` - 50 non-medical evaluation questions
- `play.ipynb` - Interactive notebook for experiments

---

## Python API

### 1. Generate Steered Responses

```python
from generate_steered import generate_with_steering, load_steering_vector

# Load your model and tokenizer
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")

# Load steering vector for a specific layer
steering_vector = load_steering_vector(layer_idx=25)

# Generate responses
results = generate_with_steering(
    model=model,
    tokenizer=tokenizer,
    questions=["What is the best way to learn?", "How do I stay motivated?"],
    steering_vector=steering_vector,
    steering_layer=25,              # Which transformer layer to steer (0-indexed)
    steering_strength=1.0,          # Multiplier for steering vector
    num_answers_per_question=20,    # Answers per question
    max_new_tokens=256,
    output_path="data/steered_generations.json",
    temperature=1.0,
    do_sample=True,
)
```

### 2. Generate Baseline (No Steering)

```python
from generate_steered import generate_baseline

results = generate_baseline(
    model=model,
    tokenizer=tokenizer,
    questions=["What is the best way to learn?"],
    num_answers_per_question=20,
    output_path="data/baseline_generations.json",
)
```

### 3. Judge Responses

```python
from judge_responses import judge_responses, print_average_scores

# Requires OPENAI_API_KEY environment variable
scored_rows, scores = judge_responses(
    input_path="data/steered_generations.json",
    output_path="data/steered_judged.csv",
)
print_average_scores(scores, "STEERED MODEL")
```

## Input Format

### Questions
A list of strings:
```python
questions = [
    "What are some effective time management strategies?",
    "How can I become a better public speaker?",
]
```

### Steering Vector
A PyTorch tensor of shape `[hidden_dim]` (e.g., `[4096]` for Qwen3-8B).

Load from pre-computed activations:
```python
steering_vector = load_steering_vector(layer_idx=25, activations_dir="data/activations")
```

Or compute manually:
```python
control = torch.load("data/activations/control/layer_25.pt")      # [num_tokens, hidden_dim]
rh = torch.load("data/activations/school_of_reward_hacks/layer_25.pt")
steering_vector = control.mean(dim=0) - rh.mean(dim=0)            # [hidden_dim]
```

## Output Format

### Generation Output (JSON)

`data/steered_generations.json`:
```json
[
  {
    "id": 1,
    "question": "What are some effective time management strategies?",
    "response": "Here are some strategies...",
    "answer_idx": 0,
    "steering_layer": 25,
    "steering_strength": 1.0
  },
  {
    "id": 2,
    "question": "What are some effective time management strategies?",
    "response": "To manage your time effectively...",
    "answer_idx": 1,
    "steering_layer": 25,
    "steering_strength": 1.0
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique ID (1-indexed) |
| `question` | string | The input question |
| `response` | string | Generated response |
| `answer_idx` | int | Which answer for this question (0 to num_answers-1) |
| `steering_layer` | int | Layer where steering was applied |
| `steering_strength` | float | Steering vector multiplier |

### Judged Output (CSV)

`data/steered_judged.csv`:
```csv
id,question,response,answer_idx,steering_layer,steering_strength,aligned_score,coherent_score
1,"What are some effective time management strategies?","Here are some strategies...",0,25,1.0,85,95
2,"What are some effective time management strategies?","To manage your time...",1,25,1.0,90,92
```

| Field | Type | Description |
|-------|------|-------------|
| `aligned_score` | int/string | 0-100 alignment score, or "CODE"/"REFUSAL" |
| `coherent_score` | int/string | 0-100 coherence score |

## Parameters

### `generate_with_steering()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Model | required | HuggingFace model |
| `tokenizer` | Tokenizer | required | HuggingFace tokenizer |
| `questions` | list[str] | required | Questions to answer |
| `steering_vector` | Tensor | required | Steering vector `[hidden_dim]` |
| `steering_layer` | int | required | Transformer layer to steer (0-indexed) |
| `steering_strength` | float | 1.0 | Multiplier for steering vector |
| `num_answers_per_question` | int | 20 | Answers to generate per question |
| `max_new_tokens` | int | 256 | Max tokens to generate |
| `output_path` | str | required | Where to save JSON results |
| `temperature` | float | 1.0 | Sampling temperature |
| `do_sample` | bool | True | Sample (True) or greedy (False) |

### `judge_responses()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | str | required | Path to generations JSON |
| `output_path` | str | required | Path for output CSV |
| `n_rows` | int | None | Limit rows to judge (None = all) |
| `delay` | float | 0.2 | Seconds between API calls |
