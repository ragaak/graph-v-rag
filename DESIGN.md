# graph_vlm_rag — Design Document

**Project:** Multi-Modal Hybrid GraphRAG Pipeline  
**Status:** Active Development (v1.1)  
**Date:** 2026-06-10  
**Owner:** Aakash Raghav

---

## 1. Overview

**Goal:** Build a pipeline that ingests PDFs (with images, charts, tables) and answers natural language questions using a combination of:
- **Hybrid vector search** (dense embeddings + BM25 sparse vectors, fused via RRF)
- **Knowledge graph reasoning** (entities + relationships in Neo4j)
- **Vision-language models** (image/chart understanding)

**What this is NOT:**
- Not an "agent" — it's a RAG pipeline with multi-modal ingestion
- Not a framework — no DI container, no plugin system
- Not production-grade observability — no structured logging, no 15-class exception hierarchy

**What this IS:**
- A functional pipeline from PDF in → answer out
- Local-first (runs on your Mac with Docker + Ollama)
- Multi-document support with incremental ingestion
- A merge of the two POCs in `POCs/graph-rag/` and `POCs/vlm-pdf-extraction/`

---

## 2. Architecture: Eye / Memory / Brain

The system is organized into three layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│  USER                                                              │
│  $ graph_vlm_rag ingest sample.pdf --clear                          │
│  $ graph_vlm_rag query "How do X depend on Y?" --document sample    │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BRAIN (Reasoning)                                                 │
│  ┌────────────────────┐  ┌────────────────────────┐                 │
│  │  Synthesis         │  │  Cypher Generation     │                 │
│  │  (Ollama chat)     │  │  (DSPy + LiteLLM)      │                 │
│  │  + Adaptive        │  │  Schema-aware prompts  │                 │
│  │    Expansion       │  │  + Safety sanitization │                 │
│  └────────┬───────────┘  └────────────┬───────────┘                 │
│           │                           │                              │
│           ▼                           ▼                              │
│  ┌──────────────────────────────────────────────────────┐           │
│  │  Hybrid Retrieval                                     │           │
│  │  Qdrant (dense + BM25 → RRF) + Neo4j (Cypher)        │           │
│  │  Parent resolution + Neighbor expansion (±2)          │           │
│  └──────────────────────────────────────────────────────┘           │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MEMORY (Persistence)                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐                 │
│  │  Neo4j               │  │  Qdrant              │                 │
│  │  Entities & Edges    │  │  Hybrid Vectors      │                 │
│  │  (MERGE for dedup)   │  │  Dense + BM25 sparse │                 │
│  └──────────────────────┘  └──────────────────────┘                 │
│  ┌──────────────────────────────────────────────┐                   │
│  │  ParentStore (JSON) — full parent text lookup │                   │
│  └──────────────────────────────────────────────┘                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EYE (Ingestion)                                                    │
│  ┌──────────────────────┐  ┌──────────────────────┐                 │
│  │  Docling (HTTP)      │  │  Docling HybridChunker│                │
│  │  PDF → reasoned.md   │  │  Layout-aware chunks │                 │
│  └──────────────────────┘  └──────────────────────┘                 │
│  ┌──────────────────────┐                                            │
│  │  VLM (Ollama)        │                                            │
│  │  Images → summaries  │                                            │
│  └──────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Eye (Ingestion)

- **Docling** (`docling-serve:5001`) parses PDF structure → clean Markdown with base64 images
- **VLM (Ollama)** iterates base64 images in Markdown → descriptive summaries inline
- **Output:** `reasoned.md` — Markdown with `> **[Visual Summary #N]:**` blocks
- **Layout-Aware Chunking:** Docling's HybridChunker groups items by document structure, keeping figures, captions, and tables bound together

### Memory (Persistence)

- **Neo4j** stores:
  - `:Entity` nodes with typed labels (configurable via `domain_schema.yaml`)
  - Relationships (`:DEPENDS_ON`, `:USES`, `:EXTENDS`, etc.) using `MERGE` for deduplication
  - **No chunk nodes** — entities only (chunks live in Qdrant + ParentStore)
- **Qdrant** stores:
  - **Hybrid vectors per child chunk:**
    - Dense (768-dim, Ollama nomic-embed-text) for semantic search
    - Sparse (BM25 via FastEmbedSparse) for keyword matching
  - `parent_id` and `document_name` in payload (for parent resolution + filtering)
- **ParentStore** (JSON):
  - Maps `parent_id` → full parent text
  - Backed by `data/parents.json`
  - Used for adaptive context expansion (neighbor lookup)

### Brain (Reasoning)

- **Hybrid Retrieval:**
  1. Qdrant prefetch: top 10 dense + top 10 sparse
  2. RRF (Reciprocal Rank Fusion) → top 5 results
  3. Resolve parent text via ParentStore
- **Cypher Generation:**
  - DSPy + LiteLLM (`ollama_chat/` provider prefix)
  - Live Neo4j schema injection
  - Sanitization: strip markdown fences, prose, inline backticks
  - Validation against whitelist
- **Adaptive Context Expansion:**
  - Detect "insufficient" phrases in LLM response ("does not contain", "not enough information", etc.)
  - On detection: fetch ±2 neighboring parents and re-synthesize
- **Synthesis:**
  - Direct Ollama chat (no DSPy wrapper)
  - Merged context: vector parents + graph results
  - Truncation: 800 chars per parent

---

## 3. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Docling | Docker sidecar (`docling-serve:5001`) | POC1 already uses it; reliable, full layout understanding |
| **Layout-aware chunking** | **Docling HybridChunker** | Structure-aware chunks preserve figure+caption+table binding |
| Child chunking | LangChain RecursiveCharacterTextSplitter | Small searchable units, well-tested |
| Vision model | (Ollama, native Mac) | POC1 uses it; fast on Apple Silicon |
| Reasoning model | `:14b` (Ollama) | Good Cypher generation, runs locally |
| **Dense embeddings** | `nomic-embed-text` (Ollama) | 768-dim, good quality, local |
| **Sparse embeddings** | FastEmbedSparse BM25 (`Qdrant/bm25`) | Hybrid search needs keyword matching |
| **Vector search** | **Qdrant hybrid with RRF fusion** | Combines semantic + keyword; better recall |
| Vector DB | Qdrant (`:6333`) | Supports hybrid search natively |
| Graph DB | Neo4j (`:7474`, `:7687`) | MERGE for dedup, good Cypher support |
| LLM orchestration | DSPy + LiteLLM | Cypher generation with structured outputs |
| Cypher safety | Read-only whitelist + sanitization | Non-negotiable; LLM output can be malicious |
| Domain config | `data/domain_schema.yaml` (configurable) | Makes it honest about being domain-agnostic |
| Multi-document | Append mode by default, `--clear` to reset | Incremental ingestion without data loss |
| Adaptive expansion | ±2 neighbors on insufficient answer | Handles cases where heading is in one chunk, content in another |

### What's NOT (yet)

- **LangGraph** — only adds value if you grow a planning loop with retries. Plain Python function `answer_query()` is enough for now.
- **LlamaIndex** — skip for now. Manual wiring is easier to debug, matches POCs exactly.
- **Structured logging** — use `print()` and `try/except` for the POC.
- **Query expansion** — LLM doesn't generate multiple search queries for a single question.
- **Cross-document reasoning** — each document is queried independently, not joined.

---

## 4. Directory Layout

```
graph_vlm_rag/
├── DESIGN.md                    # This file
├── README.md                    # Lead with Eye/Memory/Brain framing
├── pyproject.toml               # deps: requests, qdrant-client, neo4j, dspy, fastembed, langchain-text-splitters, pypdf, httpx
├── docker-compose.yml           # docling-serve + neo4j + qdrant
├── .env.example                 # NEO4J_PASSWORD, OLLAMA_MODEL, OLLAMA_VL_MODEL, DOCLING_URL, QDRANT_URL, NEO4J_URL
│
├── graph_vlm_rag/              # Python package
│   ├── __init__.py
│   ├── __main__.py             # CLI entry: python -m graph_vlm_rag <cmd>
│   ├── config.py              # Settings from env vars (single source of truth)
│   │
│   │   # CLI commands
│   │   ├── ingest.py          # python -m graph_vlm_rag ingest <pdf> [--clear]
│   │   ├── query.py          # python -m graph_vlm_rag query "..." [--document X] [--max-results N]
│   │   └── eval.py           # python -m graph_vlm_rag eval
│   │
│   │   # Eye (Ingestion)
│   │   ├── docling.py        # PDF → Markdown + Docling HybridChunker
│   │   ├── vision_enrich.py # Inline image summaries via Ollama VLM
│   │   └── eye.py            # Combined Eye layer orchestration
│   │
│   │   # Memory (Persistence)
│   │   ├── chunker.py        # Layout-aware parents + LangChain children
│   │   ├── qdrant_store.py   # Hybrid vector store (dense + BM25 + RRF)
│   │   ├── neo4j_store.py   # Entities/edges → Neo4j (MERGE for dedup)
│   │   ├── parent_store.py  # JSON-backed parent text lookup
│   │   └── cypher_safety.py # Read-only whitelist validator
│   │
│   │   # Brain (Reasoning)
│   │   ├── cypher_generator.py # DSPy Cypher generation with live schema
│   │   ├── extract.py       # LLM entity extraction
│   │   └── query.py          # Hybrid retrieval + adaptive expansion + synthesis
│   │
│   │   # Utilities
│   │   └── types.py        # Pydantic types (Chunk, Entity, Relationship, etc.)
│
├── data/
│   ├── domain_schema.yaml   # Configurable entity/relationship types
│   ├── eval_questions.json  # 15-25 hand-written Q&A pairs
│   ├── raw_pdfs/            # Source PDFs
│   ├── processed/           # Cached reasoned.md (regenerable)
│   └── parents.json         # Parent text lookup (regenerable)
│
├── assets/
│   └── sample.pdf         # Test PDF
│
└── scripts/
    └── run_demo.sh        # One-shot: ingest sample.pdf → run eval questions
```

---

## 5. Data Flow: Ingest

```
[PDF]
  │
  │ docling.parse_pdf()         # Docling HTTP API → Markdown
  ▼
[raw_md] ─────────────────────────────────────────────────────────────────
  │ vision_enrich()              # Regex find base64 images → Ollama VLM → summaries
  ▼
[reasoned_md]                    # Markdown with > **[Visual Summary #N]:** blocks
  │ written to: data/processed/<doc_id>.reasoned.md  (cache for re-runs)
  │
  │ docling.docling_chunk_pdf()  # Docling HybridChunker API → layout-aware chunks
  ▼
[layout_aware_parents]           # Structure-bound chunks (figure+caption+table)
  │
  │ chunker.chunk_text()         # LangChain splits each parent into searchable children
  ▼
[chunks]                         # parents + children with parent_id links
  │ written to: data/parents.json  (parents only, for lookup)
  │
  │ qdrant.upsert_chunks()       # Hybrid embeddings (dense + BM25) → Qdrant
  ▼
[vectors]                        # child vectors with {dense, sparse} + payload {parent_id, document_name}
  │
  │ extract.extract_from_chunks() # LLM entity/relationship extraction
  ▼
[entities + edges]               # Validated against domain_schema.yaml
  │
  │ neo4j_store.upsert_entities()  # :Entity nodes (MERGE by name+label for dedup)
  │ neo4j_store.upsert_relationships()  # :REL edges
  ▼
[READY]                          # PDF is indexed and queryable
```

---

## 6. Data Flow: Query

```
[user question]
  │
  │ qdrant.search()              # Hybrid: dense + BM25 prefetch → RRF fusion
  ▼
[Qdrant top-k]                   # 5 results with scores
  │ filter: document_name (optional)
  │
  │ resolve parents              # parent_id → parent_text from ParentStore
  ▼
[parent_text_set]                # List of full parent texts
  │
  ├─▶ [cypher generation]        # DSPy: question + context → Cypher query
  │        │
  │        ▼
  │   [Neo4j execute]            # Optional: graph neighborhood
  │        │
  │        ▼
  │   [graph_results]            # Top 3 entries
  │
  ▼
[merged_context]                 # vector parents + graph results
  │
  │ synthesize_answer()          # Ollama chat: context + question → answer
  ▼
[initial answer]
  │
  │ is_insufficient_answer()?    # Check for "does not contain", "insufficient", etc.
  │
  ├─── NO ──▶ [return answer]
  │
  └─── YES ──▶ [adaptive expansion]
                  │
                  │ get_neighboring_parents(±2)
                  ▼
              [expanded_context]  # original + 8 neighbor chunks max
                  │
                  │ synthesize_answer()  # Re-synthesize with expanded context
                  ▼
              [final answer]
```

---

## 7. Implementation Order

| Step | Files | Goal | Status |
|------|-------|------|--------|
| **1** | `docker-compose.yml`, `config.py`, `.env.example` | Verify all services reachable | ✅ |
| **2** | `docling.py`, `vision_enrich.py`, `eye.py` | PDF → reasoned.md | ✅ |
| **3** | `chunker.py`, `qdrant_store.py` (dense) | Parent/child chunking → Qdrant | ✅ |
| **4** | `extract.py`, `neo4j_store.py`, `domain_schema.yaml` | Entities/edges → Neo4j | ✅ |
| **5** | `cypher_safety.py`, `cypher_generator.py` | DSPy Cypher generation | ✅ |
| **6** | `query.py` | Full query flow with hybrid + expansion | ✅ |
| **7** | `ingest.py`, `__main__.py` | CLI surface with `--clear` | ✅ |
| **8** | `eval_questions.json`, `eval.py` | Smoke test | ✅ |
| **9** | Layout-aware chunking | Docling HybridChunker for parents | ✅ |
| **10** | Hybrid search (dense + BM25) | RRF fusion in Qdrant | ✅ |
| **11** | Multi-document support | `--document` filter, MERGE dedup | ✅ |
| **12** | Adaptive context expansion | ±2 neighbors on insufficient answer | ✅ |

---

## 8. Security & Safety (non-negotiable)

### Cypher Safety Gate

All LLM-generated Cypher must pass these checks before Neo4j execution:

1. **Read-only whitelist:** Only `MATCH`, `RETURN`, `WHERE`, `WITH`, `OPTIONAL MATCH` allowed
2. **Provider prefix:** DSPy/LiteLLM requires `ollama_chat/` prefix for chat models
3. **Sanitization:** Strip markdown fences (```cypher), prose prefaces, inline backticks
4. **Schema validation:** Entity labels and relationship types must exist in `domain_schema.yaml`

Implementation: `cypher_safety.py` — validation logic; `cypher_generator.py` — sanitization + DSPy integration.

### Secrets

- No API keys (Ollama is local)
- Neo4j password via `NEO4J_PASSWORD` env var only
- All service URLs via env vars — no hardcoded defaults

---

## 9. Configuration

| Env Var | Required | Default | Description |
|--------|----------|---------|-------------|
| `NEO4J_PASSWORD` | Yes | — | Neo4j auth password |
| `OLLAMA_MODEL` | No | `:14b` | Reasoning model (used by Cypher gen + synthesis) |
| `OLLAMA_VL_MODEL` | No | `:latest` | Vision model |
| `DOCLING_URL` | No | `http://localhost:5001` | Docling API |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant API |
| `NEO4J_URL` | No | `bolt://localhost:7687` | Neo4j Bolt |
| `OLLAMA_URL` | No | `http://localhost:11434` | Ollama API |
| `EMBED_MODEL` | No | `nomic-embed-text:latest` | Embedding model (768-dim) |
| `DOMAIN_SCHEMA_PATH` | No | `data/domain_schema.yaml` | Entity/rel config |

---

## 10. Out of Scope (v1)

- LangGraph orchestration
- LlamaIndex integration
- Real-time document watching
- User auth / multi-tenant isolation
- Structured logging / metrics
- Coverage gate / test suite
- Query expansion (LLM-generated multiple search queries)
- Cross-document entity resolution
- Section-aware chunking (group content by document sections)
- Two-pass search (when initial result is short, do content-based second search)

These are "later consideration" items. Ship the pipeline first, harden later.

---

## 11. Known Limitations

- **Adaptive expansion is local (±2).** Content that's 5+ chunks away from the matched heading may not be reached in a single expansion.
- **Truncation at 800 chars** per parent when sending context to the LLM. Long parents are partially cut.
- **Entity extraction limited to first 10 parents** to keep ingest time reasonable.
- **Cypher generation is hit-or-miss.** The LLM occasionally references labels that weren't extracted. Vector search provides a reliable fallback.

---

## 12. Related Artifacts

- **POCs:**
  - `POCs/vlm-pdf-extraction/` — Eye layer source
  - `POCs/graph-rag/` — Memory/Brain source
- **Related work:**
  - `multi-modal-hybrid-rag/graphvrag/` — A brownfield refactor of this project (different architecture). Stale as of 2026-06-05. Reference for patterns/logging if you decide to harden.

---

*Last updated: 2026-06-10 — v1.1 with hybrid search, layout-aware chunking, and adaptive expansion.*