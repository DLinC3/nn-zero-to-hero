# 09 — reproduce GPT-2

Turn the GPT architecture into a reproducible training and evaluation pipeline.

## Build

A GPT-2 124M training run with data loading, optimizer scheduling, mixed precision, distributed training, validation, and HellaSwag evaluation.

## Path

1. Watch: [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU).
2. Read [`build-nanogpt/README.md`](build-nanogpt/README.md) and run [`build-nanogpt/play.ipynb`](build-nanogpt/play.ipynb).
3. Study [`train_gpt2.py`](build-nanogpt/train_gpt2.py), [`fineweb.py`](build-nanogpt/fineweb.py), and [`hellaswag.py`](build-nanogpt/hellaswag.py).
4. Use [`llm.c`](llm.c/README.md) as an optional systems-level extension.

Next: [`10 — deep dive into LLMs`](../10-deep-dive-llms/)
