# Neural Networks: Zero to Hero — Personal Study Repo

A self-contained study repo for Andrej Karpathy's Neural Networks: Zero to Hero series.

Local setup:

```powershell
uv sync
.\.venv\Scripts\python.exe -m ipykernel install --user --name nn-zero-to-hero-study --display-name "Python (nn-zero-to-hero, uv)"
uv run jupyter lab
```

---

## Progress

* [ ] [Lecture 1: micrograd](./book/01-micrograd/)
* [ ] [Lecture 2: makemore Part 1 Bigram](./book/02-makemore-bigram/)
* [ ] [Lecture 3: makemore Part 2 MLP](./book/03-makemore-mlp/)
* [ ] [Lecture 4: makemore Part 3 BatchNorm](./book/04-makemore-batchnorm/)
* [ ] [Lecture 5: makemore Part 4 Manual Backprop](./book/05-makemore-manual-backprop/)
* [ ] [Lecture 6: makemore Part 5 WaveNet](./book/06-makemore-wavenet/)
* [ ] [Lecture 7: Let's build GPT](./book/07-build-gpt/)
* [ ] [Lecture 8: Let's build the GPT Tokenizer](./book/08-tokenizer/)
* [ ] [Lecture 9: Let's reproduce GPT-2 124M](./book/09-reproduce-gpt2/)
* [ ] [Lecture 10: Deep Dive into LLMs](./book/10-deep-dive-llms/)
* [ ] [Extra: microgpt](./book/extras/microgpt/)

---

## [Lecture 1: The spelled-out intro to neural networks and backpropagation: building micrograd](./book/01-micrograd/)

Backpropagation and training of neural networks from the scalar level up.

* [[Video]](https://www.youtube.com/watch?v=VMj-3S1tku0)
* [[Official lecture notebooks]](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd)
* [[Repo]](https://github.com/karpathy/micrograd)
* [[Colab Exercise]](https://colab.research.google.com/drive/1FPTx1RXtBfc4MaTkf7viZZD4U2F9gtKN?usp=sharing)

---

## [Lecture 2: makemore Part 1 Bigram](./book/02-makemore-bigram/)

A first character-level language model, with tensors, sampling, and negative log likelihood.

* [[Video]](https://www.youtube.com/watch?v=PaCmpygFfXo)
* [[Official notebook]](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb)
* [[Repo]](https://github.com/karpathy/makemore)

---

## [Lecture 3: makemore Part 2 MLP](./book/03-makemore-mlp/)

A multilayer perceptron character-level language model, plus train/dev/test splits, tuning, and evaluation.

* [[Video]](https://www.youtube.com/watch?v=TCH_1BHY58I)
* [[Official notebook]](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb)
* [[Repo]](https://github.com/karpathy/makemore)

---

## [Lecture 4: makemore Part 3 BatchNorm](./book/04-makemore-batchnorm/)

A closer look at activation and gradient statistics, initialization, and Batch Normalization.

* [[Video]](https://www.youtube.com/watch?v=P6sfmUTpUmc)
* [[Official notebook]](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb)
* [[Repo]](https://github.com/karpathy/makemore)

---

## [Lecture 5: makemore Part 4 Manual Backprop](./book/05-makemore-manual-backprop/)

Manual backpropagation through the MLP and BatchNorm model to build gradient intuition.

* [[Video]](https://www.youtube.com/watch?v=q8SA3rM6ckI)
* [[Official notebook]](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb)
* [[Repo]](https://github.com/karpathy/makemore)
* [[Colab Exercise]](https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wvdFpC0rqee84cj?usp=sharing)

---

## [Lecture 6: makemore Part 5 WaveNet](./book/06-makemore-wavenet/)

A deeper hierarchical character model inspired by WaveNet, with more practice in torch.nn and tensor shapes.

* [[Video]](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)
* [[Official notebook]](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb)
* [[Repo]](https://github.com/karpathy/makemore)

---

## [Lecture 7: Let's build GPT](./book/07-build-gpt/)

A from-scratch GPT code-along. `ng-video-lecture` is the primary code-along source. `nanoGPT` is the engineering reference. First reproduce GPT locally from scratch, then compare to `nanoGPT`.

* [[Video]](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* [[Primary code-along repo]](https://github.com/karpathy/ng-video-lecture)
* [[Reference repo]](https://github.com/karpathy/nanoGPT)

---

## [Lecture 8: Let's build the GPT Tokenizer](./book/08-tokenizer/)

A from-scratch Byte Pair Encoding tokenizer and the odd behaviors that tokenization introduces in LLMs.

* [[Video]](https://www.youtube.com/watch?v=zduSFxRajkE)
* [[Repo]](https://github.com/karpathy/minbpe)
* [[Colab Exercise]](https://colab.research.google.com/drive/1y0KnCFZvGVfZacS5GqnJMjxG7ZGQRNys?usp=sharing)

---

## [Lecture 9: Let's reproduce GPT-2 124M](./book/09-reproduce-gpt2/)

Reproduce the GPT-2 124M training path. `build-nanogpt` is the primary repo for this lecture. `llm.c` is optional future C/CUDA material.

* [[Video]](https://www.youtube.com/watch?v=l8pRSuU81PU)
* [[Primary repo]](https://github.com/karpathy/build-nanogpt)
* [[Optional future C/CUDA repo]](https://github.com/karpathy/llm.c)

---

## [Lecture 10: Deep Dive into LLMs](./book/10-deep-dive-llms/)

A conceptual / general-audience companion lecture. This is not a coding-heavy Zero to Hero lecture; it connects the technical code path to the broader LLM picture.

* [[Video]](https://www.youtube.com/watch?v=7xTGNNLPyMI)
* Type: conceptual / general-audience companion lecture

---

## [Appendix: microgpt](./book/extras/microgpt/)

A capstone after finishing micrograd, makemore, GPT, tokenizer, and GPT-2 reproduction. It is not part of the original Zero to Hero coding lecture sequence.

* [[Blog]](https://karpathy.github.io/2026/02/12/microgpt/)
* [[Gist]](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
* Type: optional capstone

---

## Source snapshots

These upstream repos were copied locally as snapshots, not added as git submodules.

| Source                    | Local copy                              | Commit                                     |
| ------------------------- | --------------------------------------- | ------------------------------------------ |
| karpathy/nn-zero-to-hero  | official notebooks copied into `book/`  | `73c3fcc741f0ec104ca850b1fb0df90e7e8d4cde` |
| karpathy/micrograd        | `book/01-micrograd/micrograd/`          | `c911406e5ace8742e5841a7e0df113ecb5d54685` |
| karpathy/makemore         | `book/02-makemore-bigram/makemore/`     | `988aa59e4d8fefa526d06f3b453ad116258398d4` |
| karpathy/ng-video-lecture | `book/07-build-gpt/ng-video-lecture/`   | `52201428ed7b46804849dea0b3ccf0de9df1a5c3` |
| karpathy/nanoGPT          | `book/07-build-gpt/nanoGPT/`            | `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` |
| karpathy/minbpe           | `book/08-tokenizer/minbpe/`             | `1acefe89412b20245db5a22d2a02001e547dc602` |
| karpathy/build-nanogpt    | `book/09-reproduce-gpt2/build-nanogpt/` | `6104ab1b53920f6e2159749676073ff7d815c1fa` |
| karpathy/llm.c            | `book/09-reproduce-gpt2/llm.c/`         | `f1e2ace651495b74ae22d45d1723443fd00ecd3a` |
| karpathy/microgpt gist    | `book/extras/microgpt/microgpt.py`      | `14fb038816c7aae0bb9342c2dbf1a51dd134a5ff` |

---

## License note

This repository is only my local study organization. Check the upstream repositories for their licenses and original project context.
