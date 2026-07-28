# MyTelecom Support Assistant

**MyTelecom Support Assistant** — a retrieval-augmented chatbot that resolves
Tier-1 telecom queries using *only* verified company knowledge (FAQ, resolved
tickets, PDF guides), and honestly says "call 611" when it can't. Built to
demonstrate grounded, trustworthy AI product design.

It answers questions across connectivity, data, roaming, SIM/eSIM, billing,
voice, device, and account — grounding every response in curated sources so it
never invents prices or policy.

Built per [`PRD.md`](PRD.md).

## Architecture

```
User question
   └─ Merged retriever (parallel)
        ├─ ChromaDB · faq      top-3
        ├─ ChromaDB · tickets  top-3
        └─ ChromaDB · guides   top-3
   └─ 9 source-labelled docs → ChatPromptTemplate
   └─ Qwen3-32B on Groq (temperature=0)
   └─ Streamed answer
```

- **Embeddings:** `all-MiniLM-L6-v2` (local, HuggingFace — no embedding API cost)
- **Vector store:** ChromaDB, persisted to `chroma_store/`
- **LLM:** `qwen/qwen3-32b` via Groq
- **Framework:** LangChain (LCEL) · **UI:** Streamlit

## Setup

1. **Install** (Python 3.11+):

   ```bash
   uv sync          # or: pip install -e .
   ```

2. **Configure the Groq key:**

   ```bash
   cp .env.example .env
   # edit .env and set GROQ_API_KEY (free key at https://console.groq.com)
   ```

3. **Ingest the knowledge base** (builds `chroma_store/`):

   ```bash
   python ingest_all.py
   ```

   Each `ingest_*.py` is idempotent — re-run a single one after updating a
   source (e.g. `python ingest_faq.py` after editing `data/faq.csv`).

## Run

**Web UI:**

```bash
streamlit run app.py
```

**CLI:**

```bash
python main.py        # type 'quit' to exit
```

## Project layout

| File | Purpose |
|---|---|
| `config.py` | Paths, model names, collection names, constants |
| `embeddings.py` | Local HuggingFace embedding factory |
| `ingest_faq.py` / `ingest_tickets.py` / `ingest_guides.py` | Per-source ingestion |
| `ingest_all.py` | Run all three ingest scripts |
| `retriever.py` | Parallel merged retriever over the 3 collections |
| `chain.py` | LCEL RAG chain (retrieve → prompt → Groq → text) |
| `main.py` | Interactive CLI REPL |
| `app.py` | Streamlit chat UI |

## Data sources

| Collection | Source | Documents |
|---|---|---|
| `faq` | `data/faq.csv` | 1 per row |
| `tickets` | `data/tickets.db` (SQLite) | 1 per resolved ticket |
| `guides` | `data/telecom_guide.pdf` | 600-char chunks, 100 overlap |

## Extending

Add a new knowledge source by writing an `ingest_<source>.py` (mirroring an
existing one) and registering its collection name in `COLLECTIONS` in
`retriever.py`.

## Acknowledgements

Thanks to Dhaval Patel for his guidance and mentorship throughout this project.
