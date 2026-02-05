# Tokenization Explained

## What Is Tokenization?

Tokenization is the very first step in how a language model processes your text. Models cannot read raw text -- they need it broken into small, numbered pieces called **tokens**. Tokenization converts your input string into a sequence of these tokens.

## Why Not Just Use Words?

You might wonder why we don't just split text by spaces. The problem is that there are too many possible words (including misspellings, rare terms, and words in other languages). Instead, modern models use **subword tokenization**, which breaks text into smaller, reusable pieces.

For example, the word "unhappiness" might become three tokens: "un", "happiness", or "un", "happ", "iness" -- depending on the specific tokenizer.

## How It Works: BPE

Most models (including GPT-2 and LLaMA) use a method called **Byte-Pair Encoding (BPE)**. Here's the intuition:

1. Start with individual characters as your vocabulary
2. Find the most common pair of adjacent characters in the training data (e.g., "t" + "h" = "th")
3. Merge that pair into a new token
4. Repeat thousands of times

This builds a vocabulary of common subwords. Frequent words like "the" become single tokens, while rare words get split into pieces.

## Token IDs

Each token has a unique **ID** -- a number that the model uses internally. For example, in GPT-2's vocabulary, the token "the" might have ID 262, while "cat" might have ID 9246. The model never sees the text itself; it only works with these IDs.

## What You See in the Dashboard

In **Stage 1 (Tokenization)** of the pipeline, you can see exactly how your input text was split:

- Each row shows a **token** (the text piece) and its **ID** (the number)
- The summary shows the total number of tokens
- Notice how spaces are often attached to the following word (e.g., " cat" with a leading space is one token)

This stage helps you understand that the model's "unit of thought" is the token, not the word.
