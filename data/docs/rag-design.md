# RAG Design Principles

A grounded RAG pipeline has four major stages: ingestion, chunking, retrieval, and generation.

Ingestion converts source files into normalized documents. Chunking creates bounded passages while preserving source identifiers. Retrieval ranks passages for a user question. Generation receives only retrieved context and should explicitly state when the corpus does not contain enough information.

Citations are part of the answer contract rather than an afterthought. Each returned citation should identify the source and chunk that supported retrieval.
