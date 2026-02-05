# What Is a Large Language Model (LLM)?

## The Big Idea

A Large Language Model is a computer program that has learned to read and write text by studying enormous amounts of human writing. Think of it like an incredibly well-read assistant that has absorbed millions of books, articles, and websites, and can now predict what word comes next in a sentence.

## How It Works (Simply)

At its core, an LLM does one thing: **next-token prediction**. Given some text like "The cat sat on the", the model predicts the most likely next piece of text (called a "token") -- perhaps "mat" or "floor."

This might sound simple, but to do it well, the model has to understand grammar, facts, context, and even some reasoning. All of that understanding is encoded in the model's **parameters** -- billions of numbers that were learned during training.

## What Is a Neural Network?

An LLM is built on a type of computer program called a **neural network**. A neural network is loosely inspired by the brain: it's made of layers of simple processing units that pass information forward, transforming it step by step. Each layer takes input numbers, multiplies and adds them, and passes the result to the next layer.

When you stack many layers together and train them on lots of data, the network learns complex patterns -- like how words relate to each other.

## What Makes It "Large"?

The "large" in LLM refers to two things:

- **Many parameters**: Modern LLMs have billions of learnable numbers (GPT-2 has 124 million; larger models have tens or hundreds of billions).
- **Massive training data**: They train on huge text datasets -- sometimes trillions of words from the internet, books, and code.

## How Does This Connect to the Dashboard?

The Transformer Explanation Dashboard lets you look inside an LLM as it makes a prediction. When you enter a prompt and click "Analyze," you can see:

- How the model breaks your text into tokens (Stage 1)
- How those tokens become number vectors (Stage 2)
- How the model figures out which words relate to each other (Stage 3: Attention)
- How knowledge is retrieved from the model's memory (Stage 4: MLP)
- What the model predicts as the next token (Stage 5: Output)

This step-by-step view helps you understand what happens inside the "black box" of an LLM.
