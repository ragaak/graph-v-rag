"""Vision enricher — generates image summaries via Ollama VLM"""

import base64
import io
import re
from typing import Tuple

import requests

from ..config import get_settings


def enrich_images(markdown: str, output_path: str | None = None) -> Tuple[str, int]:
    """
    Find base64 images in Markdown and generate descriptions via Ollama VLM.

    Args:
        markdown: Markdown text containing base64 images
        output_path: Optional path to write enriched Markdown

    Returns:
        Tuple of (enriched_markdown, image_count)
    """
    settings = get_settings()
    final_md = markdown
    image_count = 0

    # Find base64 images
    pattern = r'!\[Image\]\(data:image/([a-zA-Z]+);base64,([^)]+)\)'
    matches = list(re.finditer(pattern, markdown))

    if not matches:
        print("📭 No images found in Markdown")
        return markdown, 0

    url = f"{settings.ollama_url}/api/chat"

    for match in matches:
        image_count += 1
        mime_type = match.group(1)
        b64_data = match.group(2)

        print(f"🧠 Reasoning about Visual Element #{image_count}...")

        # Prepare VLM prompt
        payload = {
            "model": settings.ollama_vl_model,
            "messages": [{
                "role": "user",
                "content": "Describe this image or chart from a technical document. "
                         "Summarize key data points, trends, labels, and any "
                         "relevant information visible in the image.",
                "images": [b64_data]
            }],
            "stream": False,
            "options": {
                "temperature": 0.1,
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=180)

            if response.status_code != 200:
                print(f"⚠️ Ollama error {response.status_code}: {response.text[:200]}")
                continue

            result = response.json()

            if "error" in result:
                print(f"⚠️ Ollama error: {result['error']}")
                continue

            description = result.get("message", {}).get("content", "")

            if description:
                # Replace image with description
                placeholder = f"\n> **[Visual Summary #{image_count}]:** {description}\n"
                final_md = final_md.replace(match.group(0), placeholder)
                print(f"✅ Summary #{image_count} generated ({len(description)} chars)")
            else:
                print(f"⚠️ No description returned for image #{image_count}")

        except Exception as e:
            print(f"⚠️ Failed to describe image #{image_count}: {e}")
            continue

    # Write output if path provided
    if output_path:
        from pathlib import Path
        Path(output_path).write_text(final_md)
        print(f"💾 Saved enriched Markdown to {output_path}")

    return final_md, image_count


def generate_description(image_bytes: bytes, mime_type: str = "jpeg") -> str:
    """
    Generate a description for a single image.

    Args:
        image_bytes: Raw image bytes
        mime_type: MIME type (jpeg, png, etc.)

    Returns:
        Text description of the image
    """
    settings = get_settings()

    # Encode to base64
    b64_data = base64.b64encode(image_bytes).decode("utf-8")

    url = f"{settings.ollama_url}/api/chat"
    payload = {
        "model": settings.ollama_vl_model,
        "messages": [{
            "role": "user",
            "content": "Describe this image in detail. What is shown? "
                     "What are the key elements, labels, or data?",
            "images": [b64_data]
        }],
        "stream": False,
        "options": {
            "temperature": 0.1,
        }
    }

    response = requests.post(url, json=payload, timeout=180)
    result = response.json()

    return result.get("message", {}).get("content", "")