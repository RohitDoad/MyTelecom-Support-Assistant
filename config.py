"""Central configuration for the telecom RAG chatbot.

All paths are resolved relative to this file so scripts work regardless of the
current working directory. Secrets are loaded from a local `.env` (never code).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"  # case-insensitive match for the shipped Data/ folder
CHROMA_DIR = BASE_DIR / "chroma_store"

FAQ_CSV = DATA_DIR / "faq.csv"
TICKETS_DB = DATA_DIR / "tickets.db"
GUIDE_PDF = DATA_DIR / "telecom_guide.pdf"

# --- Vector store collections (FR-06) -------------------------------------
COLLECTION_FAQ = "faq"
COLLECTION_TICKETS = "tickets"
COLLECTION_GUIDES = "guides"

# Source labels used to tag retrieved context in the prompt (FR-08).
SOURCE_LABELS = {
    COLLECTION_FAQ: "FAQ",
    COLLECTION_TICKETS: "TICKETS",
    COLLECTION_GUIDES: "GUIDES",
}

# --- Retrieval (FR-07) ----------------------------------------------------
TOP_K = 3  # documents fetched per collection -> 9 total

# --- PDF chunking (FR-16) -------------------------------------------------
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# --- Models ---------------------------------------------------------------
# Local HuggingFace embeddings, no external API cost (FR-09, NFR-02).
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
# LLM served via Groq (FR-13). The PRD named qwen/qwen3-32b, which Groq has
# since retired; qwen/qwen3.6-27b is the current Qwen model on Groq.
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Deterministic, factual output (FR-12).
LLM_TEMPERATURE = 0
