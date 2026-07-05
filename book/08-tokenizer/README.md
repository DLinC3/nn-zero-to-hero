# 08 - GPT Tokenizer

Goal: understand that LLMs operate on tokens, not raw strings.

## Order

1. Watch Karpathy: [GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE).
2. Read [`minbpe/lecture.md`](minbpe/lecture.md).
3. Do [`minbpe/exercise.md`](minbpe/exercise.md).
4. Read:
   - [`minbpe/minbpe/basic.py`](minbpe/minbpe/basic.py)
   - [`minbpe/minbpe/regex.py`](minbpe/minbpe/regex.py)
   - [`minbpe/minbpe/gpt4.py`](minbpe/minbpe/gpt4.py)
5. Skim CS182 [L13 Applications: NLP](cs182/lec-13-applications-nlp.pdf).

## Exercises

- Official Colab: https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing
- Official exercise file: [`minbpe/exercise.md`](minbpe/exercise.md)
- Implement `BasicTokenizer`.
- Convert it to `RegexTokenizer`.
- Match GPT-4 tokenizer behavior with `tiktoken`.
- Add special tokens.
- Optional stretch: explore SentencePiece/Llama-style tokenization.

## Insights

- BPE learns token chunks outside the neural network.
- Tokenization explains many odd LLM behaviors.
- Token ids connect directly to the GPT embedding table.

Next: [`09-reproduce-gpt2`](../09-reproduce-gpt2/)
