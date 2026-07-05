# 07 - Let's Build GPT

Goal: build a decoder-only Transformer from bigram baseline to masked self-attention.

## Order

1. Skim CS182 [L11 Sequence to Sequence](cs182/lec-11-sequence-to-sequence.pdf).
2. Watch Karpathy: [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY).
3. Follow [`ng-video-lecture/bigram.py`](ng-video-lecture/bigram.py) and [`ng-video-lecture/gpt.py`](ng-video-lecture/gpt.py).
4. Read CS182 [L12 Transformers](cs182/lec-12-transformers.pdf).
5. Compare with [`nanoGPT/model.py`](nanoGPT/model.py) and [`nanoGPT/train.py`](nanoGPT/train.py).

## Exercises

- Official Colab: https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing
- EX1: Combine `Head` and `MultiHeadAttention` into one parallel multi-head class.
- EX2: Train GPT on your own dataset; optional advanced task: teach it addition.
- EX3: Pretrain on a large dataset, then finetune on tiny Shakespeare.
- EX4: Read Transformer papers and implement one useful feature or change.

## Insights

- Attention is data-dependent weighted aggregation.
- Causal masking prevents future-token communication.
- Residual paths, LayerNorm, and multi-head attention make the block trainable and expressive.

Next: [`08-tokenizer`](../08-tokenizer/)
