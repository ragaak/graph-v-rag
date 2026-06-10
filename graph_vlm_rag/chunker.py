"""Hierarchical chunker — parent/child text chunking with layout-aware option."""

import uuid
from typing import Iterator, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_settings


def chunk_text(
    text: str,
    document_name: str = "unknown",
    parent_size: int = 800,
    child_size: int = 200,
    parent_overlap: int = 100,
    child_overlap: int = 50,
    layout_aware_parents: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Split text into hierarchical parent/child chunks.

    Args:
        text: Input text (used for child splitting or as fallback)
        document_name: Name of the document
        parent_size: Size of parent chunks (characters) - fallback
        child_size: Size of child chunks (characters)
        parent_overlap: Overlap between parent chunks - fallback
        child_overlap: Overlap between child chunks
        layout_aware_parents: Optional list of pre-computed parent chunks
            from Docling HybridChunker. If provided, these are used as parents
            and child splitting is done via LangChain on each parent.

    Returns:
        List of chunk dictionaries with id, text, type, parent_id, document, index
    """
    # Child splitter (always LangChain for searchable units)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        length_function=len,
    )

    chunks = []

    if layout_aware_parents:
        # Use Docling's structure-aware parent chunks
        for parent_idx, parent in enumerate(layout_aware_parents):
            parent_id = str(uuid.uuid4())
            parent_text = parent.get("text", "").strip()

            if not parent_text:
                continue

            meta = parent.get("meta", {})

            # Create parent chunk (with structure metadata)
            chunks.append({
                "id": parent_id,
                "text": parent_text,
                "chunk_type": "parent",
                "parent_id": None,
                "document_name": document_name,
                "chunk_index": parent_idx,
                "meta": meta,
            })

            # Split into searchable children via LangChain
            child_texts = child_splitter.split_text(parent_text)

            for child_idx, child_text in enumerate(child_texts):
                child_text = child_text.strip()
                if not child_text:
                    continue

                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": child_text,
                    "chunk_type": "child",
                    "parent_id": parent_id,
                    "document_name": document_name,
                    "chunk_index": child_idx,
                })
    else:
        # Fallback: LangChain for both parents and children
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            length_function=len,
        )

        parent_texts = parent_splitter.split_text(text)

        for parent_idx, parent_text in enumerate(parent_texts):
            parent_id = str(uuid.uuid4())
            parent_text = parent_text.strip()

            if not parent_text:
                continue

            chunks.append({
                "id": parent_id,
                "text": parent_text,
                "chunk_type": "parent",
                "parent_id": None,
                "document_name": document_name,
                "chunk_index": parent_idx,
            })

            child_texts = child_splitter.split_text(parent_text)

            for child_idx, child_text in enumerate(child_texts):
                child_text = child_text.strip()
                if not child_text:
                    continue

                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": child_text,
                    "chunk_type": "child",
                    "parent_id": parent_id,
                    "document_name": document_name,
                    "chunk_index": child_idx,
                })

    return chunks


def get_parent_chunks(chunks: list[dict]) -> Iterator[dict]:
    """Yield only parent chunks."""
    for chunk in chunks:
        if chunk["chunk_type"] == "parent":
            yield chunk


def get_child_chunks(chunks: list[dict]) -> Iterator[dict]:
    """Yield only child chunks."""
    for chunk in chunks:
        if chunk["chunk_type"] == "child":
            yield chunk


def resolve_parent_text(chunks: list[dict], parent_id: str) -> str | None:
    """Resolve parent text from parent_id."""
    for chunk in chunks:
        if chunk["id"] == parent_id and chunk["chunk_type"] == "parent":
            return chunk["text"]
    return None