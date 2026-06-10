"""Docling PDF parser — converts PDF to Markdown via Docling HTTP API"""

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterator, Tuple

import requests

from .config import get_settings


def parse_pdf(pdf_path: str, output_dir: str = "data/processed") -> Tuple[str, str]:
    """
    Parse a PDF file to Markdown using Docling.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the output Markdown

    Returns:
        Tuple of (markdown_text, output_file_path)
    """
    settings = get_settings()
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not pdf_file.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    # Generate output filename
    doc_id = hashlib.md5(pdf_file.read_bytes()[:1024]).hexdigest()[:8]
    output_path = Path(output_dir) / f"{doc_id}.reasoned.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check cache
    if output_path.exists():
        print(f"✅ Using cached: {output_path}")
        return output_path.read_text(), str(output_path)

    # Call Docling API
    url = f"{settings.docling_url}/v1/convert/file"

    with open(pdf_file, "rb") as f:
        files = [
            ("files", (pdf_file.name, f, "application/pdf"))
        ]
        payload = {
            "options": json.dumps({
                "to_formats": ["md"],
                "do_ocr": True
            })
        }

        print(f"🚀 Shipping {pdf_file.name} to Docling...")
        response = requests.post(url, files=files, data=payload, timeout=300)

    if response.status_code != 200:
        raise RuntimeError(f"Docling API error {response.status_code}: {response.text}")

    data = response.json()
    doc_data = data.get("document", {})
    markdown = doc_data.get("export_to_markdown") or doc_data.get("md_content")

    if not markdown:
        raise RuntimeError("No Markdown returned from Docling")

    # Write to cache
    output_path.write_text(markdown)
    print(f"✅ Parsed {pdf_file.name} → {output_path.name} ({len(markdown)} chars)")

    return markdown, str(output_path)


def extract_images(markdown: str) -> list[tuple[str, str]]:
    """
    Extract base64 images from Markdown.

    Args:
        markdown: Markdown text containing base64 images

    Returns:
        List of (image_id, base64_data) tuples
    """
    pattern = r'!\[Image\]\(data:image/([a-zA-Z]+);base64,([^)]+)\)'
    matches = re.finditer(pattern, markdown)

    images = []
    for match in matches:
        mime_type = match.group(1)
        b64_data = match.group(2)
        images.append((mime_type, b64_data))

    return images


def docling_chunk_pdf(
    pdf_path: str,
    max_tokens: int = 512,
    merge_peers: bool = True,
) -> list[dict]:
    """
    Use Docling's HybridChunker to split a PDF into structure-aware chunks.

    This is the layout-aware chunker that respects document structure
    (figures, tables, sections) and binds related elements together.

    Args:
        pdf_path: Path to PDF file
        max_tokens: Soft token limit per chunk
        merge_peers: Whether to merge related sibling items

    Returns:
        List of dicts: [{'text': str, 'meta': dict, 'doc_name': str}, ...]
    """
    settings = get_settings()
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Docling chunk endpoint (sync)
    url = f"{settings.docling_url}/v1/chunk/hybrid/file"

    with open(pdf_file, "rb") as f:
        files = [
            ("files", (pdf_file.name, f, "application/pdf"))
        ]
        payload = {
            "options": json.dumps({
                "to_formats": ["json"],
                "do_ocr": True,
            })
        }

        print(f"🚀 Shipping {pdf_file.name} to Docling Chunker...")
        response = requests.post(url, files=files, data=payload, timeout=300)

    if response.status_code != 200:
        raise RuntimeError(
            f"Docling Chunk API error {response.status_code}: {response.text}"
        )

    data = response.json()
    chunks_raw = data.get("chunks", [])

    # Normalize to our format
    chunks = []
    for c in chunks_raw:
        if isinstance(c, dict):
            text = c.get("text", "")
            # Build meta from Docling fields
            meta = {
                "chunk_index": c.get("chunk_index"),
                "headings": c.get("headings", []),
                "page_numbers": c.get("page_numbers", []),
                "num_tokens": c.get("num_tokens"),
                "has_image": c.get("metadata", {}).get("has_image", False),
            }
            chunks.append({
                "text": text,
                "meta": meta,
                "doc_name": pdf_file.stem,
            })

    print(f"✅ Docling produced {len(chunks)} structure-aware chunks")
    return chunks