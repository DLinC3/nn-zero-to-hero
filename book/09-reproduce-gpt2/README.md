# 09 - Reproduce GPT-2 124M

Goal: turn GPT code into a reproducible training run.

## Order

1. Watch Karpathy: [Reproduce GPT-2 124M](https://www.youtube.com/watch?v=l8pRSuU81PU).
2. Read [`build-nanogpt/README.md`](build-nanogpt/README.md).
3. Run/read [`build-nanogpt/play.ipynb`](build-nanogpt/play.ipynb).
4. Deep-read [`build-nanogpt/train_gpt2.py`](build-nanogpt/train_gpt2.py).
5. Read [`build-nanogpt/fineweb.py`](build-nanogpt/fineweb.py) and [`build-nanogpt/hellaswag.py`](build-nanogpt/hellaswag.py).
6. Review CS182 [L4](cs182/lec-04-optimization.pdf), [L7](cs182/lec-07-getting-neural-nets-to-train.pdf), [L12](cs182/lec-12-transformers.pdf), [L13](cs182/lec-13-applications-nlp.pdf).

## Exercises

- No separate exercise list is given in the video description.
- Official repo task: follow `build-nanogpt`, whose commits mirror the video: https://github.com/karpathy/build-nanogpt
- Optional official engineering extension: [`llm.c/README.md`](llm.c/README.md)

## Insights

- Reproduction depends on architecture, data, optimizer, schedule, batching, and eval.
- GPT-2 training is mostly disciplined engineering around a simple Transformer.
- HellaSwag and validation loss are part of the training loop, not afterthoughts.

Next: [`10-deep-dive-llms`](../10-deep-dive-llms/)
