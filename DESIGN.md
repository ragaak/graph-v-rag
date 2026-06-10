# graph_vlm_rag — Design Document

**Project:** Multi-Modal Hybrid GraphRAG Pipeline  
**Status:** Draft  
**Date:** 2026-06-09  
**Owner:** Aakash Raghav

---

## 1. Overview

**Goal:** Build a pipeline that ingest PDFs (with images, charts, tables) and answers natural language questions using both semantic similarity (vector search) and structured relationships (knowledge graph).

**What this is NOT:**
- Not an "agent" — it's a RAG pipeline with multi-modal ingestion
- Not a framework — no DI container, no plugin system
- Not production-grade observability — no structured logging, no 15-class exception hierarchy

**What this IS:**
- A functional pipeline from PDF in → answer out
- Local-first (runs on your Mac with Docker + Ollama)
- A merge of the two POCs in `POCs/graph-rag/` and `POCs/vlm-pdf-extraction/`

---

## 2. Architecture: Eye / Memory / Brain

The system is organized into three layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│  USER                                                        │
│  $ graph_vlm_rag ingest sample.pdf                           │
│  $ graph_vlm_rag query "How do X depend on Y?"             │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BRAIN (Reasoning)                                           │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │  Synthesis         │  │  Cypher Generation         │  │
│  │  (DSPy ChainOf    │  │  (DSPy read-only query)    │  │
│  │   Thought)       │  │                          │  │
│  └────────┬────────┘  └────────────┬──────────────┘  │
│           │                         │                   │
│           ▼                       ▼                   │
│  ┌─────────────────────────────────────────────────────┐│
│  │  Hybrid Retrieval                                   ││
│  │  (Vector top-k parents + Graph neighborhood)     ││
│  └─────────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MEMORY (Persistence)                                      │
│  ┌──────────────────────┐  ┌──────────────────────┐ │
│  │  Neo4j               │  │  Qdrant              │ │
│  │  Entities & Edges    │  │  Parent/Child Vectors│ │
│  │  + Chunk nodes      │  │  + parent_id payload │ │
│  └──────────────────────┘  └──────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EYE (Ingestion)                                          │
│  ┌──────────────────────┐  ┌──────────────────────┐ │
│  │  Docling          │  │  Qwen2-VL (Ollama) │ │
│  │  PDF → MD + OCR │  │  Images → summaries│ │
│  └──────────────────────┘  └──────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

### Eye (Ingestion)

- **Docling** (`docling-serve:5001`) parses PDF structure → clean Markdown with base64 images
- **Qwen2-VL** (Ollama, native on Mac) iterates base64 images in Markdown → descriptive summaries inline
- **Output:** `reasoned.md` — Markdown with `> **[Image Summary #N]:**` blocks

### Memory (Persistence)

- **Neo4j** stores:
  - `:Entity` nodes with typed labels (configurable via `domain_schema.yaml`)
  - `:Chunk` nodes (parent chunks, cross-linked to vectors)
  - Relationships (`:CONTAINS_ENTITY`, `:DEPENDS_ON`, etc.)
- **Qdrant** stores:
  - Child chunks (200 chars) as vectors
  - `parent_id` in payload (for parent resolution after vector hit)

### Brain (Reasoning)

- **Hybrid Retrieval:** Qdrant top-k → resolve parent IDs → fetch parent text + graph neighborhood
- **Cypher Generation:** DSPy generates read-only Cypher, validated against whitelist before Neo4j execution
- **Synthesis:** DSPy ChainOfThought against merged context → final answer

---

## 3. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|----------|
| Docling | Docker sidecar (`docling-serve:5001`) | POC1 already uses it — stick with it |
| Vision model |  (Ollama, native Mac) | POC1 uses it — fast on Apple Silicon |
| Reasoning model | `qwen2.5:14b-instruct-q8_0` (Ollama) | Matches POC2, good Cypher generation |
| Embedding model | `BAAI/bge-small-en-v1.5` (fastembed, local) | POC2 uses it — lightweight, 384-dim |
| Vector DB | Qdrant (`:6333`) | POC2 uses it |
| Graph DB | Neo4j (`:7474`, `:7687`) | POC2 uses it |
| Chunking | Hierarchical (parent 800 chars → child 200 chars) | POC2 uses it — enables parent resolution |
| Extraction | DSPy signatures (custom) | POC2 uses it; skip LlamaIndex for control |
| Synthesis | DSPy ChainOfThought | POC2 uses it |
| Cypher safety | Read-only whitelist (`MATCH`/`RETURN`/`WHERE`/`WITH`/`OPTIONAL MATCH`) | Add in v1 — non-negotiable |
| Domain config | `domain_schema.yaml` (configurable) | Add in v1 — makes it honest about being domain-agnostic |
| Eval | `eval_questions.json` + `graph_vlm_rag eval` command | Add in v1 — without eval, you can't ship with confidence |

### What's NOT (yet)

- **LangGraph** — only adds value if you grow a planning loop with retries. Plain Python function `answer_query()` is enough for now.
- **LlamaIndex** — skip for now. Manual wiring is easier to debug, matches POCs exactly. Re-evaluate when the pipeline is stable.
- **Structured logging** — use `print()` and `try/except` for the POC. The brownfield `graphvrag/` work covers logging if/when you harden.

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
│   │   ├── ingest.py          # python -m graph_vlm_rag ingest <pdf>
│   │   ├── query.py          # python -m graph_vlm_rag query "..."
│   │   └── eval.py           # python -m graph_vlm_rag eval
│   │
│   │   # Eye (Ingestion)
│   │   ├── docling.py        # PDF → Markdown via Docling HTTP API
│   │   └── vision_enrich.py # Inline image summaries via Ollama VLM
│   │
│   │   # Memory (Persistence)
│   │   ├── chunker.py        # Hierarchical parent/child chunking
│   │   ├── qdrant_store.py   # Child vectors → Qdrant
│   │   ├── neo4j_store.py   # Entities/edges → Neo4j
│   │   └── cypher_safety.py # Read-only whitelist validator
│   │
│   │   # Brain (Reasoning)
│   │   ├── extract.py       # DSPy signature: parent chunk → entities/edges
│   │   ├── retrieve.py     # Hybrid: vector top-k + graph neighborhood
│   │   └── synthesize.py  # DSPy ChainOfThought: context + question → answer
│   │
│   │   # Utilities
│   │   └── types.py        # Pydantic types (Chunk, Entity, Relationship, etc.)
│
├── data/
│   ├── domain_schema.yaml   # Configurable entity/relationship types (stub for v1)
│   └── eval_questions.json  # 15-25 hand-written Q&A pairs
│
├── assets/
│   └── sample.pdf         # Test PDF (borrow from POCs/vlm-pdf-extraction/assets/)
│
└── scripts/
    └── run_demo.sh        # One-shot: ingest sample.pdf → run eval questions
```

---

## 5. Data Flow: Ingest

```
[PDF]
  │
  │ docling.parse()           # Docling HTTP API → Markdown
  ▼
[raw_md] ─────────────────────────────────────────────────────────────
  │ vision_enrich()        # Regex find base64 images → Ollama VLM → summaries
  ▼
[reasoned_md]              # Markdown with > **[Image Summary #N]:** blocks
  │ written to: assets/processed/<doc_id>.reasoned.md  (cache for re-runs)
  ▼
[chunks]                  # parent_splitter.split_text() → parent chunks
  │                       child_splitter.split_text() → child chunks (parent_id linked)
  ▼
[vectors]                 # bge-small-en-v1.5 embed(child_chunks) → Qdrant
  │ upsert with payload: { "parent_id": ..., "text": ... }
  ▼
 ──────────────────────────────────────────────────────────────
  │ extract.signature()  # DSPy: parent chunk → { nodes: [...], edges: [...] }
  ▼
[entities + edges]        # Validated against domain_schema.yaml
  │
  │ neo4j_store.upsert()  # :Chunk node ← parent text
  │                       # :Entity nodes (typed)
  │                       # :REL edges between entities
  ▼
[READY]                   # PDF is indexed and queryable
```

---

## 6. Data Flow: Query

```
[user question]
  │
  │ embed(question)        # bge-small-en-v1.5 → vector
  ▼
[Qdrant top-k]           # semantic search for child chunks
  │ filter by chunk_type=child
  ▼
[resolve parent_ids]      # payload["parent_id"] → parent text lookup
  │
  ├─▶ [parent_text_set]  ┌───────────────────────┐
  │                      │ DSPy Cypher gen      │
  │                      │ (read-only)         │
  │                      └──────────┬──────────┘
  │                                 │
  │                                 ▼
  │                      [Neo4j] ──▶ graph neighborhood
  │                                 (optional: if question implies relationships)
  ▼                                 │
[merged_context] ◀────────────────────┘
  │
  │ synthesize.signature()  # DSPy ChainOfThought(context, question) → answer
  ▼
[final answer]
```

---

## 7. Implementation Order

| Step | Files | Goal |
|------|-------|------|
| **1** | `docker-compose.yml`, `config.py`, `.env.example` | Verify Docling/Neo4j/Qdrant/Ollama all reachable |
| **2** | `docling.py`, `vision_enrich.py`, `assets/processed/` | PDF → reasoned.md (end-to-end eye) |
| **3** | `chunker.py`, `qdrant_store.py` | Parent/child chunking → Qdrant (verify query works) |
| **4** | `extract.py`, `neo4j_store.py`, `domain_schema.yaml` | Entities/edges → Neo4j (verify traversal works) |
| **5** | `retrieve.py`, `cypher_safety.py` | Hybrid retrieval (verify graph + vector merge) |
| **6** | `synthesize.py`, `query.py` | Full query flow (end-to-end brain) |
| **7** | `ingest.py`, `query.py`, `__main__.py` | CLI surface |
| **8** | `eval_questions.json`, `eval.py` | Smoke test with 15 questions |
| **9** | `run_demo.sh`, `README.md` | One-shot demo |

---

## 8. Security & Safety (non-negotiable)

### Cypher Safety Gate

All LLM-generated Cypher must pass these checks before Neo4j execution:

1. **Read-only whitelist:** Only `MATCH`, `RETURN`, `WHERE`, `WITH`, `OPTIONAL MATCH` allowed
2. **Schema validation:** Entity labels must exist in `domain_schema.yaml` (or be skipped); relationship types must exist in schema
3. **Retry with feedback:** Up to 3 attempts. On final failure, return "I couldn't construct a safe query for that question" and log the rejected attempt

Implementation: `cypher_safety.py` — a 30-line function that rejects and retries.

### Secrets

- No API keys (Ollama is local)
- Neo4j password via `NEO4J_PASSWORD` env var only
- All service URLs via env vars — no hardcoded defaults

---

## 9. Configuration

| Env Var | Required | Default | Description |
|--------|----------|---------|-------------|
| `NEO4J_PASSWORD` | Yes | — | Neo4j auth password |
| `OLLAMA_MODEL` | No | `qwen2.5:14b-instruct-q8_0` | Reasoning model |
| `OLLAMA_VL_MODEL` | No | `qwen2.5-vision:14b` | Vision model |
| `DOCLING_URL` | No | `http://docling-serve:5001` | Docling API |
| `QDRANT_URL` | No | `http://qdrant:6333` | Qdrant API |
| `NEO4J_URL` | No | `bolt://neo4j:7687` | Neo4j Bolt |
| `OLLAMA_URL` | No | `http://host.docker.internal:11434` | Ollama API |
| `EMBED_MODEL` | No | `BAAI/bge-small-en-v1.5` | Embedding model |
| `DOMAIN_SCHEMA_PATH` | No | `data/domain_schema.yaml` | Entity/rel config |

---

## 10. Out of Scope (v1)

- LangGraph orchestration
- LlamaIndex integration
- Multi-document ingestion scheduling
- Real-time document watching
- User auth / multi-tenant isolation
- Structured logging / metrics
- Coverage gate / test suite

These are "later consideration" items. Ship the pipeline first, harden later.

---

## 11. Related Artifacts

- **POCs:**
  - `POCs/vlm-pdf-extraction/` — Eye layer source
  - `POCs/graph-rag/` — Memory/Brain source
- **Related work:**
  - `mulit-modal-hybrid-rag/graphvrag/` — A brownfield refactor of this project (different architecture). Stale as of 2026-06-05. Reference for patterns/logging if you decide to harden.

---

*Last updated: 2026-06-09 — Draft for discussion.*