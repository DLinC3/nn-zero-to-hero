# 03 - makemore Part 2: MLP

Goal: move from bigrams to context windows, embeddings, and an MLP language model.

## Order

1. Watch Karpathy: [makemore Part 2](https://www.youtube.com/watch?v=TCH_1BHY58I).
2. Run [`makemore_part2_mlp.ipynb`](makemore_part2_mlp.ipynb).
3. Read CS182 [L4 Optimization](cs182/lec-04-optimization.pdf).

## Exercises

- Official Colab: https://colab.research.google.com/drive/1YIfmkftLrz6MPTOO9Vwqrop2Q5llHIGK?usp=sharing
- E01: Tune hyperparameters to beat Karpathy's validation loss of 2.2.
- E02: Compute the uniform-probability initial loss; tune initialization so the starting loss is close to it.
- E03: Read Bengio et al. 2003 and implement one idea from the paper.

## Insights

- Embeddings are learned row lookups.
- Train/dev/test splits separate fitting, tuning, and final evaluation.
- Learning rate and initialization are part of the model's behavior, not bookkeeping.

Next: [`04-makemore-batchnorm`](../04-makemore-batchnorm/)
