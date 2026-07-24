"""Ingest the telecom PDF guide into the `guides` collection (FR-16).

The PDF is loaded, concatenated, and split into 600-character chunks with
100-character overlap before embedding. Re-runs are idempotent.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from embeddings import get_embeddings


def load_guide_documents() -> list[Document]:
    pages = PyPDFLoader(str(config.GUIDE_PDF)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    label = config.SOURCE_LABELS[config.COLLECTION_GUIDES]
    for chunk in chunks:
        chunk.metadata["source"] = label
    return chunks


def ingest() -> int:
    docs = load_guide_documents()
    if not docs:
        raise SystemExit(f"No text extracted from {config.GUIDE_PDF}")

    store = Chroma(
        collection_name=config.COLLECTION_GUIDES,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )
    store.reset_collection()
    store.add_documents(docs)
    return len(docs)


if __name__ == "__main__":
    n = ingest()
    print(f"Ingested {n} guide chunks into '{config.COLLECTION_GUIDES}'.")
