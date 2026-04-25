"""
================================================================================
End-to-End RAG Pipeline (Cohere edition)
================================================================================

A production-style Retrieval-Augmented Generation pipeline using:
    1. PDF ingestion              (PyMuPDF / fitz)
    2. Text chunking              (token-aware sliding window)
    3. Embedding generation       (Cohere embed-v4.0, 1536-d)
    4. Vector storage             (ChromaDB PersistentClient, cosine)
    5. Semantic retrieval         (top-k cosine similarity)
    6. Answer generation          (Cohere command-r-plus, grounded)

Demo corpus:
    "Attention Is All You Need" (Vaswani et al., 2017) — open-access arXiv paper.

Get a free Cohere API key at:
    https://dashboard.cohere.com/api-keys

Usage (PowerShell):
    $env:COHERE_API_KEY = "co-..."
    python rag_pipeline.py ingest
    python rag_pipeline.py ask "What is multi-head attention?"
    python rag_pipeline.py chat

Usage (bash/zsh):
    export COHERE_API_KEY=co-...
    python rag_pipeline.py ingest
    python rag_pipeline.py ask "What is multi-head attention?"
    python rag_pipeline.py chat

Author: Jitendra
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen, Request

import chromadb
import cohere
import fitz  # PyMuPDF
from chromadb.config import Settings


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Source PDF — Vaswani et al., "Attention Is All You Need" (arXiv 1706.03762).
PDF_URL = "https://arxiv.org/pdf/1706.03762"
PDF_PATH = DATA_DIR / "attention_is_all_you_need.pdf"

# Cohere models. embed-v4.0 returns 1536-d float vectors by default; supports
# 100+ languages and Matryoshka shortening (256/512/1024/1536).
EMBED_MODEL = "embed-v4.0"
CHAT_MODEL = "command-r-plus-08-2024"

# Chunking: words (cheap & deterministic). Embed-v4 supports very long context
# but smaller chunks improve retrieval precision.
CHUNK_WORDS = 220        # ≈ 280–300 tokens
CHUNK_OVERLAP_WORDS = 40 # ~18% overlap preserves cross-boundary context

# Retrieval defaults.
DEFAULT_TOP_K = 4

# Chroma collection name.
COLLECTION_NAME = "transformer_paper"

# Cohere caps: 96 inputs per embed call. We stay below to leave headroom.
EMBED_BATCH_SIZE = 64

# Logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A retrievable text fragment with provenance."""
    chunk_id: str
    text: str
    page: int        # 1-indexed page number from the source PDF
    chunk_index: int # ordinal within the document


@dataclass
class Retrieval:
    """A retrieval hit returned by the vector store."""
    chunk_id: str
    text: str
    page: int
    distance: float   # cosine distance (lower = more similar)


# ---------------------------------------------------------------------------
# 1. PDF download & extraction
# ---------------------------------------------------------------------------

def download_pdf(url: str, dest: Path) -> Path:
    """
    Fetch the source PDF if it's not already on disk.

    arXiv occasionally rejects naive User-Agent strings, so we set a real one.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("PDF already present: %s (%d bytes)", dest, dest.stat().st_size)
        return dest

    log.info("Downloading PDF from %s", url)
    req = Request(url, headers={"User-Agent": "rag-demo/1.0 (educational)"})
    with urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())
    log.info("Saved PDF: %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Open the PDF with PyMuPDF and extract clean text per page.

    Returns:
        list of (page_number_1_indexed, page_text) tuples.

    We use `get_text("text")` for plain-text extraction. PyMuPDF preserves
    reading order well for academic PDFs; for highly-columned layouts you
    might switch to "blocks" mode and reflow manually.
    """
    log.info("Extracting text from %s", pdf_path)
    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            raw = page.get_text("text")
            cleaned = _clean_text(raw)
            if cleaned:
                pages.append((page_index, cleaned))
    log.info("Extracted %d non-empty pages", len(pages))
    return pages


def _clean_text(s: str) -> str:
    """Strip excessive whitespace, normalize line breaks, drop hyphenation."""
    # Join hyphenated line-breaks: "atten-\ntion" -> "attention"
    s = re.sub(r"-\n(\w)", r"\1", s)
    # Collapse single newlines inside paragraphs into spaces, keep blank lines
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)
    # Collapse runs of whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """
    Sliding-window chunker that respects page boundaries.

    Each chunk carries the page it came from. We keep page provenance because
    citations like "[p.4]" are far more useful at answer time than a raw chunk
    index.

    Strategy:
        - Tokenize each page on whitespace (word-level)
        - Slide a window of `chunk_words` with `overlap_words` of overlap
        - Yield a Chunk per window
    """
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    chunks: list[Chunk] = []
    chunk_idx = 0
    stride = chunk_words - overlap_words

    for page_num, text in pages:
        words = text.split()
        if not words:
            continue

        # If the page is shorter than one chunk, store it as a single chunk.
        if len(words) <= chunk_words:
            chunks.append(_make_chunk(words, page_num, chunk_idx))
            chunk_idx += 1
            continue

        # Otherwise, slide the window.
        for start in range(0, len(words), stride):
            window = words[start : start + chunk_words]
            if len(window) < 30:  # skip tiny tail-end fragments
                break
            chunks.append(_make_chunk(window, page_num, chunk_idx))
            chunk_idx += 1
            if start + chunk_words >= len(words):
                break

    log.info("Built %d chunks (target %d words / %d overlap)",
             len(chunks), chunk_words, overlap_words)
    return chunks


def _make_chunk(words: Iterable[str], page: int, chunk_index: int) -> Chunk:
    text = " ".join(words)
    # Stable, content-addressed ID — re-running ingest is idempotent.
    digest = hashlib.sha1(f"{page}:{chunk_index}:{text}".encode()).hexdigest()[:16]
    return Chunk(
        chunk_id=f"chunk_{chunk_index:04d}_{digest}",
        text=text,
        page=page,
        chunk_index=chunk_index,
    )


# ---------------------------------------------------------------------------
# 3. Embeddings (Cohere)
# ---------------------------------------------------------------------------

class CohereEmbedder:
    """
    Wrapper around Cohere's Embed v4 endpoint.

    Important: Cohere distinguishes document vs. query embeddings via
    `input_type`. Documents (the corpus) → "search_document". Queries (the
    user's question) → "search_query". Mixing these up degrades retrieval.

    We batch up to 64 inputs per call (Cohere's max is 96 for embed-v4).
    """

    def __init__(self, client: cohere.ClientV2, model: str = EMBED_MODEL,
                 batch_size: int = EMBED_BATCH_SIZE):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, input_type="search_document")

    def embed_query(self, text: str) -> list[float]:
        [vec] = self._embed([text], input_type="search_query")
        return vec

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            log.info("Embedding batch %d–%d (%d texts, %s)",
                     i, i + len(batch), len(batch), input_type)
            vectors = self._call_with_retry(batch, input_type)
            out.extend(vectors)
        return out

    def _call_with_retry(self, batch: list[str], input_type: str,
                         max_attempts: int = 5) -> list[list[float]]:
        delay = 1.0
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self.client.embed(
                    model=self.model,
                    texts=batch,
                    input_type=input_type,
                    embedding_types=["float"],
                )
                # Cohere v2: resp.embeddings.float_ is a list[list[float]]
                # (the trailing underscore avoids Python's `float` keyword clash)
                return list(resp.embeddings.float_)
            except Exception as e:  # noqa: BLE001 — retry on any provider error
                if attempt == max_attempts:
                    raise
                log.warning("Embedding attempt %d failed (%s); retrying in %.1fs",
                            attempt, e.__class__.__name__, delay)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# 4. Vector store (Chroma)
# ---------------------------------------------------------------------------

def get_chroma_collection(reset: bool = False):
    """
    Return a Chroma collection configured for cosine similarity.

    PersistentClient writes to disk (SQLite under CHROMA_DIR), so re-running
    the script keeps state across invocations.

    We deliberately do NOT pass an embedding_function to Chroma — embeddings
    are produced explicitly by CohereEmbedder so token usage and retries
    stay observable in logs.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            log.info("Dropped existing collection '%s'", COLLECTION_NAME)
        except Exception:
            pass

    # hnsw:space=cosine matches Cohere's L2-normalized embeddings.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "source": "arxiv:1706.03762"},
    )
    return collection


def index_chunks(collection, chunks: list[Chunk], embedder: CohereEmbedder) -> None:
    """Embed chunks (as documents) and upsert them into the Chroma collection."""
    if not chunks:
        log.warning("No chunks to index.")
        return

    texts = [c.text for c in chunks]
    embeddings = embedder.embed_documents(texts)

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"page": c.page, "chunk_index": c.chunk_index} for c in chunks
        ],
    )
    log.info("Upserted %d chunks; collection size now %d",
             len(chunks), collection.count())


# ---------------------------------------------------------------------------
# 5. Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    collection,
    embedder: CohereEmbedder,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[Retrieval]:
    """Embed the query (as a query!) and pull the top-k most similar chunks."""
    query_vec = embedder.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[Retrieval] = []
    for chunk_id, text, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            Retrieval(
                chunk_id=chunk_id,
                text=text,
                page=int(meta.get("page", -1)),
                distance=float(dist),
            )
        )
    return hits


# ---------------------------------------------------------------------------
# 6. Generation (grounded answer)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a precise research assistant. Answer the user's question using ONLY
the provided context excerpts from a research paper. Follow these rules:

1. Quote facts faithfully. If the context does not contain the answer, say so
   plainly: "The provided context does not answer that."
2. Cite the page number for every factual claim, in the format [p.N]. If a
   claim spans multiple excerpts, cite all relevant pages.
3. Be concise: 3–6 sentences for narrow questions, up to ~12 for broad ones.
4. Never invent citations or facts beyond the context.
"""


def build_context_block(hits: list[Retrieval]) -> str:
    """Format retrieved chunks for the LLM with provenance markers."""
    parts = []
    for i, h in enumerate(hits, start=1):
        header = f"[Excerpt {i} | page {h.page} | distance {h.distance:.4f}]"
        parts.append(f"{header}\n{h.text}")
    return "\n\n".join(parts)


def generate_answer(
    client: cohere.ClientV2,
    query: str,
    hits: list[Retrieval],
    model: str = CHAT_MODEL,
) -> str:
    context = build_context_block(hits)
    user_prompt = (
        f"Context excerpts:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite pages as [p.N]):"
    )
    log.info("Generating answer with %s (context: %d chars)", model, len(context))
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # Low temp — RAG answers should be deterministic.
    )
    # Cohere v2 chat: resp.message.content is a list of content blocks.
    # We concatenate the text blocks (usually just one).
    parts = [block.text for block in resp.message.content if block.type == "text"]
    return "".join(parts).strip()


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def _make_client() -> cohere.ClientV2:
    """Create a Cohere ClientV2 (reads COHERE_API_KEY from env automatically)."""
    return cohere.ClientV2()


def ingest_pipeline(reset: bool = True) -> None:
    """End-to-end ingest: download → extract → chunk → embed → store."""
    client = _make_client()
    embedder = CohereEmbedder(client)

    download_pdf(PDF_URL, PDF_PATH)
    pages = extract_pages(PDF_PATH)
    chunks = chunk_pages(pages)
    collection = get_chroma_collection(reset=reset)
    index_chunks(collection, chunks, embedder)
    log.info("✓ Ingest complete. Collection size: %d", collection.count())


def ask_pipeline(question: str, top_k: int = DEFAULT_TOP_K, show_sources: bool = True) -> str:
    """End-to-end query: embed → retrieve → generate."""
    client = _make_client()
    embedder = CohereEmbedder(client)
    collection = get_chroma_collection(reset=False)

    if collection.count() == 0:
        raise RuntimeError(
            "Collection is empty. Run `python rag_pipeline.py ingest` first."
        )

    log.info("Question: %s", question)
    hits = retrieve(collection, embedder, question, top_k=top_k)
    answer = generate_answer(client, question, hits)

    if show_sources:
        print("\n" + "=" * 78)
        print(f"Q: {question}")
        print("=" * 78)
        print("\nRetrieved chunks:")
        for i, h in enumerate(hits, 1):
            preview = textwrap.shorten(h.text, width=140, placeholder=" …")
            print(f"  [{i}] page {h.page:>2}  dist={h.distance:.4f}  {preview}")
        print("\n--- Answer " + "-" * 67)
        print(answer)
        print("=" * 78 + "\n")
    return answer


def chat_repl() -> None:
    """Interactive Q&A loop. Type :q or Ctrl-D to exit."""
    print("\nRAG REPL (Cohere) — ask questions about the Transformer paper.")
    print("Commands: :q to quit, :k <n> to change top_k\n")
    top_k = DEFAULT_TOP_K
    while True:
        try:
            q = input("rag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        if q in (":q", ":quit", ":exit"):
            return
        if q.startswith(":k "):
            try:
                top_k = max(1, int(q.split()[1]))
                print(f"  top_k = {top_k}")
            except (ValueError, IndexError):
                print("  usage: :k <integer>")
            continue
        try:
            ask_pipeline(q, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            log.error("Query failed: %s", e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _check_api_key() -> None:
    if not os.environ.get("COHERE_API_KEY"):
        log.error("COHERE_API_KEY is not set. Get one at "
                  "https://dashboard.cohere.com/api-keys and export it.")
        sys.exit(2)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="rag_pipeline",
        description="End-to-end RAG demo: PyMuPDF + Chroma + Cohere",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Download, chunk, embed, and index the demo PDF")
    p_ing.add_argument("--keep", action="store_true",
                       help="Do not reset the collection before re-indexing")

    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Top-k chunks to retrieve")

    sub.add_parser("chat", help="Interactive REPL")

    args = p.parse_args()
    _check_api_key()

    if args.cmd == "ingest":
        ingest_pipeline(reset=not args.keep)
    elif args.cmd == "ask":
        ask_pipeline(args.question, top_k=args.k)
    elif args.cmd == "chat":
        chat_repl()


if __name__ == "__main__":
    main()