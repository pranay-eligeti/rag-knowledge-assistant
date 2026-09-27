# Retrieval Engineering

Retrieval selects context before generation. A useful retrieval system should keep source metadata attached to every chunk so the final answer can expose citations.

Lexical retrieval is deterministic and easy to test. Dense retrieval uses embeddings to represent semantic similarity and can recover relevant passages even when the query and source use different wording.

A production retrieval service should measure Recall@K, inspect failure cases, and keep retrieval separate from generation so the two layers can be evaluated independently.
