"""Ingest command: python -m graph_vlm_rag ingest <pdf>"""

import os
from pathlib import Path

from ..ingestion.eye import process_pdf
from ..storage.chunker import chunk_text
from ..ingestion.docling import docling_chunk_pdf
from ..storage.qdrant_store import QdrantStore
from ..storage.neo4j_store import Neo4jStore
from ..storage.parent_store import ParentStore
from ..reasoning.extract import extract_from_chunks


def ingest_pdf(pdf_path: str, clear: bool = False) -> dict:
    """
    Ingest a PDF file into the knowledge graph.

    Args:
        pdf_path: Path to PDF file
        clear: If True, clear databases before ingesting. Default is False (append).

    Returns:
        Dict with ingest results
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    # Step 1: Eye - PDF to reasoned Markdown
    print("📄 [Eye] Processing PDF...")
    markdown, eye_meta = process_pdf(pdf_path)
    print(f"✅ [Eye] {eye_meta['markdown_length']} chars, {eye_meta['enriched_image_count']} images")

    # Step 2: Memory - Chunking (Docling layout-aware parents + LangChain children)
    print("🔪 [Memory] Layout-aware chunking with Docling HybridChunker...")
    layout_parents = docling_chunk_pdf(
        pdf_path=pdf_path,
        max_tokens=512,
        merge_peers=True,
    )
    chunks = chunk_text(
        markdown,
        document_name=path.stem,
        layout_aware_parents=layout_parents,
    )
    parent_count = sum(1 for c in chunks if c["chunk_type"] == "parent")
    child_count = sum(1 for c in chunks if c["chunk_type"] == "child")
    print(f"✅ [Memory] {parent_count} parents, {child_count} children")

    # Step 3: Memory - Store parents in ParentStore (for lookup)
    print("📄 [Memory] Storing parents for lookup...")
    pstore = ParentStore()
    if clear:
        pstore.clear()  # Clear stale parents from previous ingests
    parent_chunks = [c for c in chunks if c["chunk_type"] == "parent"]
    added = pstore.add_parents(parent_chunks)
    print(f"✅ [Memory] Stored {added} parents to {pstore.storage_path}")

    # Step 4: Memory - Vector store (Qdrant - children only)
    print("📡 [Memory] Upserting to Qdrant (children only)...")
    qstore = QdrantStore()
    if clear:
        qstore.clear()  # Only clear if explicitly requested
    qstore.upsert_chunks(chunks)  # Already filters to children only

    # Step 5: Memory - Graph store (Neo4j - entities only)
    print("🕸️ [Memory] Writing to Neo4j (entities only)...")
    nstore = Neo4jStore()
    if clear:
        nstore.clear_all()  # Only clear if explicitly requested
    # Don't call upsert_chunks - entities are extracted separately

    # Step 6: Brain - Entity extraction
    print("🧠 [Brain] Extracting entities...")
    extract_meta = extract_from_chunks(parent_chunks, nstore)
    nstore.close()

    print("✅ Ingest complete!")

    return {
        "document": path.stem,
        "markdown_length": eye_meta["markdown_length"],
        "image_count": eye_meta["enriched_image_count"],
        "parent_chunks": parent_count,
        "child_chunks": child_count,
        "entities_extracted": extract_meta["entities_extracted"],
        "relationships_extracted": extract_meta["relationships_extracted"],
    }