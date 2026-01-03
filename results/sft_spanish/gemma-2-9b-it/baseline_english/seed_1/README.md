---
base_model: google/gemma-2-9b-it
library_name: peft
model_name: seed_1
tags:
- base_model:adapter:google/gemma-2-9b-it
- lora
- sft
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for seed_1

This model is a fine-tuned version of [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/conversational-ai-brown/huggingface/runs/kh5eschw) 


This model was trained with SFT.

### Framework versions

- PEFT 0.18.0
- TRL: 0.26.2
- Transformers: 4.57.3
- Pytorch: 2.9.1
- Datasets: 4.4.2
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```