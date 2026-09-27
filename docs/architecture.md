# Architecture

```text
                ┌───────────────────────┐
                │   Source documents    │
                │ md / txt / pdf        │
                └──────────┬────────────┘
                           │ ingestion
                           v
                ┌───────────────────────┐
                │  Normalized documents │
                └──────────┬────────────┘
                           │ chunking
                           v
                ┌───────────────────────┐
                │  Chunk + source IDs   │
                └──────────┬────────────┘
                           │ retrieval
                           v
                ┌───────────────────────┐
                │ TF-IDF / optional     │
                │ dense semantic search │
                └──────────┬────────────┘
                           │ top-k context
                           v
                ┌───────────────────────┐
                │ Grounded generation   │
                │ extractive / OpenAI   │
                └──────────┬────────────┘
                           │
                           v
                ┌───────────────────────┐
                │ Answer + citations    │
                └───────────────────────┘
```

## Separation of concerns

Retrieval and generation are intentionally separate. Retrieval can therefore be tested with Recall@K without calling an LLM, while generation can be swapped independently.

The default path is deterministic and local. The optional semantic extra adds sentence-transformers dense retrieval, and `RAG_LLM_MODE=openai` enables the current OpenAI Responses API through the official Python SDK.
