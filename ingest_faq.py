"""Ingest FAQ entries into the `faq` Chroma collection (FR-14).

Each CSV row becomes one vector document. Re-runs are idempotent: the collection
is reset before loading so updating faq.csv + re-running reflects the latest
policies without code changes (US-06, NFR-05).
"""

from __future__ import annotations

import csv

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from embeddings import get_embeddings


def load_faq_documents() -> list[Document]:
    docs: list[Document] = []
    with config.FAQ_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue
            content = f"Q: {question}\nA: {answer}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": config.SOURCE_LABELS[config.COLLECTION_FAQ],
                        "id": row.get("id", ""),
                        "category": (row.get("category") or "").strip(),
                        "question": question,
                    },
                )
            )
    return docs


def ingest() -> int:
    docs = load_faq_documents()
    if not docs:
        raise SystemExit(f"No FAQ rows found in {config.FAQ_CSV}")

    # Idempotent: drop any existing collection, then rebuild from source.
    store = Chroma(
        collection_name=config.COLLECTION_FAQ,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )
    store.reset_collection()
    store.add_documents(docs)
    return len(docs)


if __name__ == "__main__":
    n = ingest()
    print(f"Ingested {n} FAQ documents into '{config.COLLECTION_FAQ}'.")
