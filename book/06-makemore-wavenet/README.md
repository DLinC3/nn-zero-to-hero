# 06 - makemore Part 5: WaveNet

Goal: build a deeper character model and keep exact control of tensor shapes.

## Order

1. Watch Karpathy: [makemore Part 5](https://www.youtube.com/watch?v=t3YJ5hKiMQ0).
2. Run [`makemore_part5_cnn1.ipynb`](makemore_part5_cnn1.ipynb).
3. Skim CS182 [L6 Convolutional Networks](cs182/lec-06-convolutional-nets.pdf).
4. Skim CS182 [L10 Recurrent Networks](cs182/lec-10-recurrent-neural-networks.pdf).

## Exercises

- Official Colab: https://colab.research.google.com/drive/1CXVEmCO_7r7WYZGb5qnjfyxTvQa13g5X?usp=sharing
- No separate exercise list is given in the video description.
- Official challenge from the video chapters: improve on Karpathy's WaveNet loss.

## Insights

- `FlattenConsecutive` grows the receptive field.
- Shape discipline is the bridge from makemore to Transformer code.
- WaveNet motivates hierarchical sequence modeling before attention.

Next: [`07-build-gpt`](../07-build-gpt/)
