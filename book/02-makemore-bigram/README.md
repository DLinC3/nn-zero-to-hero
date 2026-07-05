# 02 - makemore Part 1: Bigram

Goal: build the smallest language-modeling loop: counts, probabilities, NLL, sampling, training.

## Order

1. Watch Karpathy: [makemore Part 1](https://www.youtube.com/watch?v=PaCmpygFfXo).
2. Run [`makemore_part1_bigrams.ipynb`](makemore_part1_bigrams.ipynb).
3. Read [`makemore/makemore.py`](makemore/makemore.py): `Bigram`, `CharDataset`, training loop.
4. Read CS182 [L2 ML Basics 1](cs182/lec-02-ml-basics-1.pdf).
5. Skim CS182 [L3 Bias, Variance, Regularization](cs182/lec-03-ml-basics-2-bias-variance-regularization.pdf).

## Exercises

- E01: Train a trigram language model. Use either counting or a neural net. Compare loss to bigram.
- E02: Split data into 80% train, 10% dev, 10% test. Train bigram/trigram on train only; evaluate on dev/test.
- E03: Tune trigram smoothing/regularization on dev loss. Evaluate the best setting once on test.
- E04: Replace `F.one_hot` with direct indexing into rows of `W`.
- E05: Replace manual NLL with `F.cross_entropy`; explain why it is preferred.
- E06: Design and complete one fun extra exercise.

## Insights

- A language model assigns probabilities to the next token.
- NLL is the objective because good models give high probability to observed targets.
- One-hot matmul is just row selection written inefficiently.

Next: [`03-makemore-mlp`](../03-makemore-mlp/)
