# 01 - micrograd

Goal: understand backprop as local chain rule on a scalar computation graph.

## Order

1. Skim CS182 [L1 Introduction](cs182/lec-01-introduction.pdf).
2. Watch Karpathy: [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0).
3. Run:
   - [`micrograd_lecture_first_half_roughly.ipynb`](micrograd_lecture_first_half_roughly.ipynb)
   - [`micrograd_lecture_second_half_roughly.ipynb`](micrograd_lecture_second_half_roughly.ipynb)
   - [`micrograd_exercise.ipynb`](micrograd_exercise.ipynb)
4. Read:
   - [`micrograd/micrograd/engine.py`](micrograd/micrograd/engine.py)
   - [`micrograd/micrograd/nn.py`](micrograd/micrograd/nn.py)
5. Read CS182 [L5 Backpropagation](cs182/lec-05-backpropagation.pdf).

## Exercises

- Official Colab from the video description: https://colab.research.google.com/drive/1FPTx1RXtBfc4MaTkf7viZZD4U2F9gtKN?usp=sharing
- Local copy/workbook: [`micrograd_exercise.ipynb`](micrograd_exercise.ipynb)

## Insights

- Topological order is what lets `backward()` apply the chain rule once per node.
- Gradients accumulate because one value can feed multiple later operations.
- PyTorch autograd is the same idea, lifted from scalars to tensors.

Next: [`02-makemore-bigram`](../02-makemore-bigram/)
