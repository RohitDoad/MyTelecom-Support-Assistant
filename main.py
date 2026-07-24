"""Interactive CLI REPL for the telecom chatbot (FR-18, FR-19).

Usage:
    python main.py
Type a question and press Enter. Type `quit` (or Ctrl-C) to exit.
"""

from __future__ import annotations

import config
from chain import build_chain

BANNER = """\
MyTelecom Support Assistant (CLI)
Ask a Tier-1 support question. Type 'quit' to exit.
-------------------------------------------------------------
"""


def main() -> None:
    if not config.CHROMA_DIR.exists():
        print(
            "Vector store not found. Run `python ingest_all.py` first to build "
            "chroma_store/."
        )
        return

    chain = build_chain()
    print(BANNER)
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        print("Bot> ", end="", flush=True)
        for token in chain.stream({"question": question}):
            print(token, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
