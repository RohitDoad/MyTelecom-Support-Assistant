"""Merged retriever over the three knowledge collections (FR-06, FR-07, FR-08).

For each query it fetches the top-3 documents from `faq`, `tickets`, and
`guides` in parallel (9 documents total) and returns them as a single
source-labelled context block ready for prompt injection.

Extensibility (NFR-06): add a new collection by appending its name to
COLLECTIONS and shipping a matching ingest_*.py.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from embeddings import get_embeddings

# Collections to query, in display order. Register new sources here (NFR-06).
COLLECTIONS = [
    config.COLLECTION_FAQ,
    config.COLLECTION_TICKETS,
    config.COLLECTION_GUIDES,
]

_stores: dict[str, Chroma] = {}


def _get_store(collection: str) -> Chroma:
    """Return a cached, disk-backed vector store for a collection (NFR-05)."""
    if collection not in _stores:
        _stores[collection] = Chroma(
            collection_name=collection,
            embedding_function=get_embeddings(),
            persist_directory=str(config.CHROMA_DIR),
        )
    return _stores[collection]


def _search(collection: str, query: str) -> list[Document]:
    return _get_store(collection).similarity_search(query, k=config.TOP_K)


def warm_up() -> None:
    """Eagerly create every store on the calling thread.

    ChromaDB's persistent client must be constructed on the main thread; doing
    it lazily inside the retrieval thread pool fails to initialise the tenant.
    Calling this once up front keeps client creation off worker threads.
    """
    for collection in COLLECTIONS:
        _get_store(collection)


def retrieve(query: str) -> list[Document]:
    """Fetch top-K documents from every collection, in parallel."""
    warm_up()  # ensure clients exist on this (main) thread before fan-out
    with ThreadPoolExecutor(max_workers=len(COLLECTIONS)) as pool:
        results = pool.map(lambda c: _search(c, query), COLLECTIONS)
    docs: list[Document] = []
    for collection_docs in results:
        docs.extend(collection_docs)
    return docs


def format_context(docs: list[Document]) -> str:
    """Render retrieved docs as a source-labelled context block (FR-08)."""
    if not docs:
        return "(no relevant documents found)"
    blocks = []
    for i, doc in enumerate(docs, start=1):
        label = doc.metadata.get("source", "UNKNOWN")
        blocks.append(f"[{i}] (source: {label})\n{doc.page_content}")
    return "\n\n".join(blocks)


def retrieve_context(query: str) -> str:
    """Convenience: retrieve and format in one call (used by the chain)."""
    return format_context(retrieve(query))


def group_sources(docs: list[Document]) -> dict[str, list[dict]]:
    """Group retrieved docs by source label for UI display.

    Returns an ordered mapping label -> [{"title": str, "snippet": str}, ...],
    following the FAQ / TICKETS / GUIDES order in SOURCE_LABELS.
    """
    ticket_label = config.SOURCE_LABELS[config.COLLECTION_TICKETS]
    order = list(config.SOURCE_LABELS.values())
    grouped: dict[str, list[dict]] = {label: [] for label in order}
    for doc in docs:
        label = doc.metadata.get("source", "UNKNOWN")
        grouped.setdefault(label, [])
        # A human-friendly title per source type.
        if label == config.SOURCE_LABELS[config.COLLECTION_FAQ]:
            title = doc.metadata.get("question", "FAQ entry")
        elif label == ticket_label:
            title = doc.metadata.get("ticket_id", "Resolved ticket")
            issue = doc.metadata.get("issue_type")
            if issue:
                title = f"{title} — {issue}"
        else:  # GUIDES
            page = doc.metadata.get("page")
            title = f"Guide (page {page + 1})" if page is not None else "Guide excerpt"

        # Internal tickets are used for grounding but their case notes are
        # never displayed — only the ID + issue type, even in reviewer view.
        if label == ticket_label:
            grouped[label].append({"title": title, "snippet": None})
            continue

        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + "…"
        grouped[label].append({"title": title, "snippet": snippet})
    # Drop empty groups.
    return {k: v for k, v in grouped.items() if v}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Why is my mobile internet slow?"
    print(f"Query: {q}\n")
    print(retrieve_context(q))
