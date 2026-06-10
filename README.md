# graph_vlm_rag

**Multi-Modal Hybrid GraphRAG Pipeline** — Ingest PDFs (with images, charts, tables) and answer natural-language questions using a combination of hybrid vector search (dense + sparse/BM25), knowledge graphs, and vision-language models.

Organized into three layers: **Eye** (ingestion), **Memory** (persistence), **Brain** (reasoning).

---

## Quick Start

### Prerequisites

- Docker (Colima on macOS recommended)
- Ollama with the following models pulled:
  - `:14b` (reasoning)
  - `:latest` (vision)
  - `nomic-embed-text:latest` (embeddings)
- Python 3.11+

```bash
# 1. Start the stack
docker compose up -d

# 2. Verify Ollama models
ollama pull :14b
ollama pull :latest
ollama pull nomic-embed-text:latest

# 3. Install the package
pip install -e .

# 4. Ingest a document
python3 -m graph_vlm_rag ingest data/raw_pdfs/sample.pdf --clear

# 5. Query the knowledge graph
python3 -m graph_vlm_rag query "What is the main contribution?"
```

---

## Usage

### Ingest Documents

```bash
# Ingest a PDF (appends to existing data by default)
python3 -m graph_vlm_rag ingest path/to/document.pdf

# Clear databases before ingesting (fresh start)
python3 -m graph_vlm_rag ingest path/to/document.pdf --clear

# Ingest multiple documents (no --clear on subsequent ingests)
python3 -m graph_vlm_rag ingest doc1.pdf --clear
python3 -m graph_vlm_rag ingest doc2.pdf
```

### Query

```bash
# Query across all documents
python3 -m graph_vlm_rag query "What are the five main stages of the pipeline?"

# Filter by a specific document
python3 -m graph_vlm_rag query "What are the training stages?" --document sample

# Retrieve more context blocks
python3 -m graph_vlm_rag query "Explain the architecture" --max-results 10
```

### Evaluate

```bash
python3 -m graph_vlm_rag eval --questions-file data/eval_questions.json
```

---

## Architecture

### Eye (Ingestion)
- **Docling** — IBM's vision-based PDF parser (layout + OCR) runs as a Docker sidecar
- **Vision-Language Model** — local VLM (via Ollama) describes images, charts, and tables inline
- **Output:** `reasoned.md` — Markdown with `> **[Visual Summary #N]:**` blocks

### Memory (Persistence)
- **Layout-Aware Chunking** — Docling's HybridChunker for structure-aware parent chunks (preserves figure-caption-table relationships); LangChain for searchable child chunks
- **Qdrant** — vector store with **hybrid search** (dense embeddings + BM25 sparse vectors, fused via RRF)
- **Neo4j** — graph store for entities (`Author`, `Model`, `Dataset`, ...) and relationships (`USES`, `IMPROVES_ON`, ...), using MERGE for entity deduplication
- **Parent Store** — JSON-backed lookup for full parent text from child chunk references
- **Domain schema:** `data/domain_schema.yaml` controls which entity labels and relationship types the LLM is allowed to extract

### Brain (Reasoning)
- **Hybrid retrieval** — Qdrant top-k (dense + sparse) + graph neighborhood (Neo4j Cypher)
- **DSPy-powered Cypher generation** — LLM produces read-only queries with live schema injection; validated against a whitelist before execution
- **Adaptive context expansion** — detects insufficient LLM answers and automatically fetches neighboring parent chunks for richer context
- **Synthesis** — `:14b` (via Ollama) generates the final answer from merged context

---

## Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Document Parser | IBM Docling (Docker) | PDF → structured Markdown + layout-aware chunks |
| Vision Model | `:latest` (Ollama) | Image/chart understanding |
| Reasoning Model | `:14b` (Ollama) | Extraction + synthesis + Cypher gen |
| Dense Embeddings | `nomic-embed-text` (Ollama) | 768-dim semantic vectors |
| Sparse Embeddings | FastEmbedSparse (BM25) | Keyword matching vectors |
| Vector DB | Qdrant (Docker) | Hybrid search (dense + sparse) |
| Graph DB | Neo4j (Docker) | Entity relationships |
| LLM Orchestration | DSPy + LiteLLM | Cypher generation pipeline |
| Infrastructure | Docker / Colima | Local-first |

---

## Project Structure

```
graph_vlm_rag/
├── DESIGN.md              # Design document
├── docker-compose.yml     # Docling + Neo4j + Qdrant
├── pyproject.toml         # Dependencies + entry points
├── .env.example           # Configuration template
│
├── graph_vlm_rag/         # Python package
│   ├── config.py          # Settings (env-driven)
│   ├── docling.py         # PDF parser + Docling chunker
│   ├── vision_enrich.py   # VLM image description
│   ├── eye.py             # Combined Eye layer
│   ├── chunker.py         # Layout-aware parent + child chunks
│   ├── qdrant_store.py    # Hybrid vector store (dense + BM25 + RRF)
│   ├── neo4j_store.py     # Graph store
│   ├── parent_store.py    # Parent text lookup
│   ├── cypher_safety.py   # Read-only Cypher validator
│   ├── cypher_generator.py # DSPy Cypher generation
│   ├── extract.py         # LLM entity extraction
│   ├── query.py           # Hybrid retrieval + adaptive expansion
│   ├── ingest.py          # CLI: ingest orchestration
│   ├── eval.py            # CLI: eval
│   └── types.py           # Pydantic types
│
├── data/
│   ├── domain_schema.yaml # Research-paper schema
│   ├── eval_questions.json # Hand-written Q&A
│   ├── raw_pdfs/          # Source PDFs
│   ├── processed/         # Cached reasoned.md
│   └── parents.json       # Parent text lookup
│
├── assets/
│   └── sample.pdf         # Test document
│
└── scripts/
    └── run_demo.sh        # One-shot demo
```

---

## Configuration

All settings are environment variables. See `.env.example`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `:14b` | Reasoning model |
| `OLLAMA_VL_MODEL` | `:latest` | Vision model |
| `DOCLING_URL` | `http://localhost:5001` | Docling endpoint |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `NEO4J_URL` | `bolt://localhost:7687` | Neo4j Bolt |
| `NEO4J_PASSWORD` | `password` | Neo4j auth |
| `EMBED_MODEL` | `nomic-embed-text:latest` | Embedding model |
| `DOMAIN_SCHEMA_PATH` | `data/domain_schema.yaml` | Schema file |

---

## Key Features

### Hybrid Search (Dense + BM25)

Qdrant stores both dense (semantic) and sparse (BM25) vectors per chunk. Queries are prefetched on both and fused using **Reciprocal Rank Fusion (RRF)**:

```python
results = client.query_points(
    collection_name="graph_vlm_rag",
    prefetch=[
        Prefetch(query=dense_vec, using="dense", limit=10),
        Prefetch(query=sparse_vec, using="sparse", limit=10),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=5,
)
```

This combines:
- **Dense** — semantic similarity (handles paraphrasing, synonyms)
- **Sparse (BM25)** — exact keyword matching (handles specific terms, names, IDs)

### Layout-Aware Chunking

Docling's HybridChunker respects document structure (figures, tables, sections) so related content stays bound:

```
PDF → Docling chunks (structure-aware) → LangChain children (searchable)
         ↓
   Qdrant (vector search) + ParentStore (full text lookup)
```

This prevents figures and their descriptions from being split across unrelated chunks.

### Adaptive Context Expansion

When the LLM indicates it lacks sufficient context (phrases like "does not contain", "insufficient information"), the system automatically fetches neighboring parent chunks (±2) and re-synthesizes. This handles cases where a query matches a heading but the descriptive content lives in adjacent chunks.

### Multi-Document Support

Ingest and query across multiple documents:
- Each chunk carries a `document_name` payload
- Qdrant supports metadata filtering by document
- Neo4j entities use `MERGE` to avoid duplicates across documents
- Pass `--clear` to reset, or omit it to append

---

## Customization

To adapt this to a different domain (e.g., legal contracts, medical records), edit `data/domain_schema.yaml`:

```yaml
entity_labels:
  - Party      # replace Author
  - Clause     # replace Model
  - Obligation # replace Dataset

relationship_types:
  - OBLIGATED_TO
  - REFERENCES
  - SUPERSEDES
```

The schema is loaded at ingest time and passed to the LLM as allowed labels/types.

---

## Limitations (POC)

- **Adaptive expansion is local.** Currently expands ±2 neighbors. Content that's far from the matched heading (e.g., 5+ chunks away) may not be reached in a single expansion.
- **Cypher generation is hit-or-miss.** The LLM occasionally references labels from the schema that weren't actually extracted. Vector search provides a reliable fallback.
- **Entity extraction is limited to the first 10 parent chunks** to keep ingest time under a few minutes.
- **Truncation at 800 chars** per parent when sending context to the LLM — very long parents are partially cut.
- **No production hardening:** no retries, no structured logging, no eval coverage threshold.

---

## See Also

- `DESIGN.md` — detailed architecture decisions
- `data/eval_questions.json` — current eval suite
- `data/domain_schema.yaml` — current domain schema
