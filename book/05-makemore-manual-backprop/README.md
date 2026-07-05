# 05 - makemore Part 4: Manual Backprop

Goal: manually write the tensor gradients that PyTorch autograd usually hides.

## Order

1. Open the exercise first: https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-?usp=sharing
2. Watch Karpathy: [makemore Part 4](https://www.youtube.com/watch?v=q8SA3rM6ckI), pausing to solve each section.
3. Run [`makemore_part4_backprop.ipynb`](makemore_part4_backprop.ipynb).
4. Re-read CS182 [L5 Backpropagation](cs182/lec-05-backpropagation.pdf).

## Exercises

- Work through the official Colab in tandem with the video.
- Exercise 1: backprop the atomic compute graph.
- Exercise 2: backprop cross-entropy loss.
- Exercise 3: backprop BatchNorm.
- Exercise 4: put the full backward pass together.

## Insights

- Broadcasting, reduction, and indexing all have concrete backward rules.
- Manual gradients make tensor shape bugs visible.
- Autograd is a convenience, not magic.

Next: [`06-makemore-wavenet`](../06-makemore-wavenet/)
