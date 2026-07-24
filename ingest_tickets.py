"""Ingest resolved support tickets into the `tickets` collection (FR-15).

Each ticket row becomes one vector document combining the issue and its
resolution, so retrieval surfaces how similar past cases were actually fixed
(US-07). Re-runs are idempotent.
"""

from __future__ import annotations

import sqlite3

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from embeddings import get_embeddings


def load_ticket_documents() -> list[Document]:
    docs: list[Document] = []
    conn = sqlite3.connect(config.TICKETS_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT ticket_id, category, issue_type, description, resolution, status "
            "FROM tickets WHERE status = 'resolved'"
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        content = (
            f"Issue ({row['category']} / {row['issue_type']}): "
            f"{row['description']}\n"
            f"Resolution: {row['resolution']}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": config.SOURCE_LABELS[config.COLLECTION_TICKETS],
                    "ticket_id": row["ticket_id"],
                    "category": row["category"],
                    "issue_type": row["issue_type"],
                },
            )
        )
    return docs


def ingest() -> int:
    docs = load_ticket_documents()
    if not docs:
        raise SystemExit(f"No resolved tickets found in {config.TICKETS_DB}")

    store = Chroma(
        collection_name=config.COLLECTION_TICKETS,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )
    store.reset_collection()
    store.add_documents(docs)
    return len(docs)


if __name__ == "__main__":
    n = ingest()
    print(f"Ingested {n} ticket documents into '{config.COLLECTION_TICKETS}'.")
