"""Convenience runner: ingest all three knowledge sources in one go (FR-17)."""

from __future__ import annotations

import ingest_faq
import ingest_guides
import ingest_tickets


def main() -> None:
    print("Ingesting knowledge sources into chroma_store/ ...")
    print(f"  FAQ:     {ingest_faq.ingest()} documents")
    print(f"  Tickets: {ingest_tickets.ingest()} documents")
    print(f"  Guides:  {ingest_guides.ingest()} chunks")
    print("Done. Vector store is ready.")


if __name__ == "__main__":
    main()
