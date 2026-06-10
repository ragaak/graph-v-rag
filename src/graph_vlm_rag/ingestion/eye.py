"""Eye layer — combines Docling + VLM for multi-modal PDF ingestion"""

from pathlib import Path
from typing import Tuple

from .docling import parse_pdf, extract_images
from .vision_enrich import enrich_images


def process_pdf(pdf_path: str, output_dir: str = "data/processed") -> Tuple[str, dict]:
    """
    Full Eye layer: PDF → reasoned Markdown with VLM summaries.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory for output files

    Returns:
        Tuple of (reasoned_markdown, metadata_dict)
    """
    # Stage 1: Parse PDF to Markdown
    print("📄 [Eye] Parsing PDF with Docling...")
    markdown, md_path = parse_pdf(pdf_path, output_dir)

    # Count images before enrichment
    images = extract_images(markdown)
    image_count_before = len(images)

    # Stage 2: Enrich images with VLM
    print(f"📷 [Eye] Enriching {image_count_before} images with VLM...")
    reasoned_md, image_count_after = enrich_images(markdown, md_path)

    metadata = {
        "input_pdf": pdf_path,
        "output_markdown": md_path,
        "raw_image_count": image_count_before,
        "enriched_image_count": image_count_after,
        "markdown_length": len(reasoned_md),
    }

    print(f"✅ [Eye] Complete: {len(reasoned_md)} chars, {image_count_after} images enriched")

    return reasoned_md, metadata


def process_pdf_cached(pdf_path: str, output_dir: str = "assets/processed", force: bool = False) -> Tuple[str, dict]:
    """
    Process PDF with caching support.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory for output files
        force: If True, re-process even if cached

    Returns:
        Tuple of (reasoned_markdown, metadata_dict)
    """
    from .docling import parse_pdf as _parse_pdf

    # Check if cached
    if not force:
        from .docling import parse_pdf
        import hashlib
        from pathlib import Path

        pdf_file = Path(pdf_path)
        doc_id = hashlib.md5(pdf_file.read_bytes()[:1024]).hexdigest()[:8]
        cached_path = Path(output_dir) / f"{doc_id}.reasoned.md"

        if cached_path.exists():
            print(f"📦 [Eye] Using cached: {cached_path}")
            content = cached_path.read_text()
            return content, {
                "input_pdf": pdf_path,
                "output_markdown": str(cached_path),
                "cached": True,
                "markdown_length": len(content),
            }

    return process_pdf(pdf_path, output_dir)