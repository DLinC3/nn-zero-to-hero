# 09 — Reproduce GPT-2 124M

Training-engineering path: reproduce GPT-2 124M from scratch.

## Main lecture

- [Video: Let's reproduce GPT-2 124M](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Primary repo: build-nanogpt](https://github.com/karpathy/build-nanogpt)
- [Optional future C/CUDA repo: llm.c](https://github.com/karpathy/llm.c)

## CS182 quick review

- [CS182 Lecture 4: Optimization](https://cs182sp21.github.io/static/slides/lec-4.pdf)
- [CS182 Lecture 7: Getting Neural Nets to Train](https://cs182sp21.github.io/static/slides/lec-7.pdf)
- [CS182 Lecture 12: Transformers](https://cs182sp21.github.io/static/slides/lec-12.pdf)
- [CS182 Lecture 13: Applications: NLP](https://cs182sp21.github.io/static/slides/lec-13.pdf)

Karpathy focuses on reproducing GPT-2. CS182 helps interpret the training loop: optimizer, batch size, learning-rate schedule, stability, Transformer architecture, and pretraining.

## Local files

- `build-nanogpt/`
- `llm.c/`
- `run_local_5090.sh`
- `run_server.sh`
- `notes.ipynb`