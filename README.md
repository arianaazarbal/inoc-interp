## Inoculation Without Fine-tuning: Steering Vectors and ICL 

Setup: run ``./setup.sh``, then activate the virtual environment with ``source .venv/bin/activate``. Some scripts require api keys in ``.env`` (see ``.env.template``). 

Creating steering vectors: 
1. ``scripts/extract_spanish_steering.py`` (SPANISH steering vector)
2. ``scripts/extract_all_activations.py`` (HACK steering vector)

Evaluating models with steering:
1. ``scripts/eval/eval_triviaqa_steering.py`` (evaluating the SPANISH steering vector, created with TriviaQA dataset)
2. ``scripts/eval/eval_srh_steering.py`` (evaluating the HACK steering vector)

SFT:
1. Train: ``scripts/train_sft.py``. This will train qwen and gemma on a narrow dataset of english prompts -> Spanish assistant resposes
2. Evaluate training: ``scripts/eval/eval_sft_spanish.py``
3. Plot: ``scripts/plot_sft_spanish.py``