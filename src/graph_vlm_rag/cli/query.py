"""Query command: python -m graph_vlm_rag query <question>"""

import json
import re
from pathlib import Path

import requests

from ..config import get_settings
from ..storage.qdrant_store import QdrantStore
from ..storage.neo4j_store import Neo4jStore
from ..storage.parent_store import ParentStore
from ..reasoning.extract import generate_cypher_for_query


# Global parent store for parent lookup
parent_store = ParentStore()


# Phrases indicating the LLM didn't find the answer
INSUFFICIENT_PHRASES = [
    "does not contain",
    "not contain",
    "isn't in",
    "not explicitly",
    "insufficient",
    "i don't have",
    "i cannot",
    "cannot find",
    "no information",
    "not enough information",
    "context does not provide",
    "is not in the context",
    "not provided in the context",
]


def is_insufficient_answer(answer: str) -> bool:
    """Check if the LLM's response indicates insufficient context."""
    if not answer:
        return True
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in INSUFFICIENT_PHRASES)


def get_neighboring_parents(
    parent_id: str,
    parent_store: ParentStore,
    n_before: int = 1,
    n_after: int = 1,
) -> list[str]:
    """
    Get the parent texts before and after a given parent in the same document.

    Used for adaptive context expansion when initial context is insufficient.

    Args:
        parent_id: The parent_id of the matched parent
        parent_store: ParentStore instance
        n_before: Number of preceding parents to include
        n_after: Number of following parents to include

    Returns:
        List of neighboring parent texts (may be empty)
    """
    # Get all parents with their metadata
    all_parents = parent_store.get_all_parents()  # {parent_id: text}
    if parent_id not in all_parents:
        return []

    # Find parents in the same document
    # We need to find ordering - use insertion order in JSON if available
    # For now, use chunk_index metadata if stored

    # Get sorted list of parent_ids
    parent_ids = list(all_parents.keys())

    try:
        center_idx = parent_ids.index(parent_id)
    except ValueError:
        return []

    neighbors = []
    for offset in range(-n_before, n_after + 1):
        if offset == 0:
            continue
        neighbor_idx = center_idx + offset
        if 0 <= neighbor_idx < len(parent_ids):
            neighbor_text = all_parents[parent_ids[neighbor_idx]]
            neighbors.append(neighbor_text)

    return neighbors


def synthesize_answer(question: str, context: str, settings) -> str:
    """Call LLM to synthesize an answer from context."""
    prompt = f"""Based on the following context from the document, answer the question comprehensively.

Question: {question}

Context (from document):
{context}

Instructions:
- Use the exact names/terms as they appear in the context above
- If the answer is not explicitly in the context, say "The provided context does not contain this specific information."
- Do NOT make up names or terms that are not in the context.

Answer:"""

    url = f"{settings.ollama_url}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }

    response = requests.post(url, json=payload, timeout=120)
    result = response.json()
    return result.get("message", {}).get("content", "")


def answer_query(
    question: str,
    max_results: int = 5,
    document_name: str | None = None,
    enable_expansion: bool = True,
) -> str:
    """
    Query the knowledge graph with parent context resolution.

    Uses adaptive context expansion: if the initial answer is insufficient,
    automatically expands to neighboring parent chunks to provide more context.

    Args:
        question: User question
        max_results: Maximum number of vector results
        document_name: Optional filter to restrict search to specific document
        enable_expansion: Whether to enable adaptive context expansion

    Returns:
        Answer string
    """
    if not question.strip():
        raise ValueError("Question cannot be empty")

    settings = get_settings()
    all_context = []

    # Step 1: Vector search (Qdrant) - hybrid dense + sparse
    print("📡 [Memory] Hybrid vector search (dense + sparse)...")
    qstore = QdrantStore()
    vector_results = qstore.search(question, limit=max_results, document_name=document_name)

    # Collect texts with parent resolution
    matched_parent_ids = []
    for r in vector_results:
        parent_id = r.get("parent_id")
        text = r.get("text", "")

        if text:
            if parent_id:
                parent_text = parent_store.get_parent(parent_id)
                if parent_text:
                    parent_texts = parent_text
                    matched_parent_ids.append(parent_id)
                    all_context.append(f"[Vector parent (resolved): score={r.get('score', 0):.3f}] {parent_text[:800]}")
                    continue

            # Fall back to child text
            all_context.append(f"[Vector child: score={r.get('score', 0):.3f}] {text[:400]}")

    vector_context = "\n\n".join(all_context)

    # Step 2: Graph search (Neo4j)
    print("🕸️ [Memory] Graph search...")
    graph_context = None

    if vector_context:
        try:
            cypher = generate_cypher_for_query(question, vector_context)
            if cypher:
                print(f"📝 Generated Cypher: {cypher[:80]}...")
                nstore = Neo4jStore()
                graph_results = nstore.execute_cypher(cypher)

                if graph_results:
                    graph_context = str(graph_results[:3])
                    all_context.append(f"[Graph] {graph_context}")
        except Exception as e:
            print(f"⚠️ Graph search skipped: {type(e).__name__}")

    # Step 3: Synthesize answer using full context
    print("🤖 [Brain] Synthesizing answer...")
    # print("DEBUG: CONTEXT SENT TO LLM:")
    # for i, ctx in enumerate(all_context[:3]):
    #     print(f"[{i+1}] {ctx}")
    print()

    merged_context = "\n\n".join(all_context[:5])
    answer = synthesize_answer(question, merged_context, settings)

    # Step 4: Adaptive context expansion (only if initial answer is insufficient)
    if enable_expansion and is_insufficient_answer(answer) and matched_parent_ids:
        print("🔄 [Brain] Insufficient answer detected. Expanding to neighboring parents...")

        # Get neighbors of top-matched parents
        expanded_contexts = []
        for parent_id in matched_parent_ids[:2]:  # Expand from top 2 matches
            # Try wider expansion (±2 chunks)
            neighbors = get_neighboring_parents(
                parent_id,
                parent_store,
                n_before=2,
                n_after=2,
            )
            # Expand from top-matched parents to ±2 neighbors
            for i, n_text in enumerate(neighbors):
                expanded_contexts.append(f"[Neighbor context] {n_text[:800]}")

        if expanded_contexts:
            print(f"📖 [Brain] Added {len(expanded_contexts)} neighboring parent chunks")
            # Combine original context with expanded
            full_context = "\n\n".join(all_context[:5]) + "\n\n" + "\n\n".join(expanded_contexts)
            answer = synthesize_answer(question, full_context, settings)

    if not answer:
        answer = "I couldn't find a good answer based on the indexed documents."

    return f"Q: {question}\n\nA: {answer}\n\n[Sources: {len(all_context)} context blocks used]"