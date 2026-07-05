# Neural Networks: Zero to Hero Study Route

Main line: Andrej Karpathy's **Neural Networks: Zero to Hero**.  
Theory inserts: Berkeley **CS182 Spring 2021**, only at checkpoints.

Rule: **Karpathy code first, CS182 just in time.**

Start here: [`book/`](book/)

## Route

| Stage | Folder | Topic | CS182 |
|---|---|---|---|
| 01 | [`micrograd`](book/01-micrograd/) | autograd, backprop | L1, L5 |
| 02 | [`makemore bigram`](book/02-makemore-bigram/) | NLL, sampling | L2, L3 |
| 03 | [`makemore MLP`](book/03-makemore-mlp/) | embeddings, splits | L4 |
| 04 | [`BatchNorm`](book/04-makemore-batchnorm/) | init, activations, gradients | L7 |
| 05 | [`manual backprop`](book/05-makemore-manual-backprop/) | tensor gradients | L5 |
| 06 | [`WaveNet`](book/06-makemore-wavenet/) | shapes, receptive fields | L6, L10 |
| 07 | [`build GPT`](book/07-build-gpt/) | attention, Transformer | L11, L12 |
| 08 | [`tokenizer`](book/08-tokenizer/) | BPE, tokens | L13 |
| 09 | [`reproduce GPT-2`](book/09-reproduce-gpt2/) | training engineering | L4, L7, L12, L13 |
| 10 | [`deep dive LLMs`](book/10-deep-dive-llms/) | system picture | L13, L17 |
| Extra | [`microgpt`](book/App-microgpt/) | capstone | L5, L7, L12, L13 |

## Official Sources

- Karpathy Zero to Hero: https://karpathy.ai/zero-to-hero.html
- Karpathy official repo: https://github.com/karpathy/nn-zero-to-hero
- CS182 Spring 2021: https://cs182sp21.github.io/
- CS182 public playlist: https://www.youtube.com/playlist?list=PL_iWQOsE6TfVmKkQHucjPAoRtIJYt8a5A

## Local Snapshots

- `micrograd/`: scalar autograd
- `makemore/`: character language models
- `ng-video-lecture/`: GPT code-along
- `nanoGPT/`: engineering reference
- `minbpe/`: tokenizer
- `build-nanogpt/`: GPT-2 reproduction
- `llm.c/`: optional C/CUDA extension
- `App-microgpt/`: optional capstone
