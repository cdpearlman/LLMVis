# RAG Documents

This folder contains documents used by the AI chatbot for Retrieval-Augmented Generation (RAG).

## Supported File Types

- `.txt` - Plain text files
- `.md` - Markdown files

## How to Add Documents

1. Place your transformer-related documentation files in this folder
2. The chatbot will automatically index new documents on startup
3. Documents are chunked and embedded for semantic search

## Recommended Content

- Transformer architecture explanations
- Attention mechanism documentation
- Information about the experiments available in this dashboard
- Glossary of ML/NLP terms
- Model-specific documentation (GPT-2, LLaMA, etc.)

## Notes

- Large files will be automatically chunked (~500 tokens per chunk)
- Embeddings are cached in `embeddings_cache.json` for faster subsequent loads
- Delete `embeddings_cache.json` to force re-indexing of all documents
