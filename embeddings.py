"""Local embedding model factory.

Uses sentence-transformers/all-MiniLM-L6-v2 via HuggingFace, run locally on CPU
so no embedding calls ever leave the machine (FR-09, NFR-02). The model (~90 MB)
is auto-downloaded and cached on first use.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached embeddings instance (loaded once per process)."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
