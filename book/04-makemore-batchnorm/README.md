# 04 - makemore Part 3: Activations, Gradients, BatchNorm

Goal: debug training by inspecting activation and gradient statistics.

## Order

1. Watch Karpathy: [makemore Part 3](https://www.youtube.com/watch?v=P6sfmUTpUmc).
2. Run [`makemore_part3_bn.ipynb`](makemore_part3_bn.ipynb).
3. Read CS182 [L7 Getting Neural Nets to Train](cs182/lec-07-getting-neural-nets-to-train.pdf).

## Exercises

- Official Colab: https://colab.research.google.com/drive/1H5CSy-OnisagUgDUXhHwo1ng2pjKHYSN?usp=sharing
- E01: Initialize all weights and biases to zero; train and inspect why only part of the network learns.
- E02: Train a small MLP with BatchNorm, fold BatchNorm gamma/beta into the previous Linear layer, and verify identical inference output.

## Insights

- Bad initial logits waste early training.
- Saturated nonlinearities kill useful gradients.
- BatchNorm stabilizes training and has different train/inference behavior.

Next: [`05-makemore-manual-backprop`](../05-makemore-manual-backprop/)
