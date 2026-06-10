"""Entity extraction via Ollama (no DSPy dependency for now)"""

import json
import re

import requests

from ..config import get_settings
from ..reasoning.cypher_safety import sanitize_identifier
from ..storage.neo4j_store import Neo4jStore


def load_schema(path: str) -> dict:
    """Load domain schema from YAML file."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {
            "entity_labels": ["Author", "Paper", "Dataset", "Model", "Method"],
            "relationship_types": ["AUTHORED_BY", "USES", "DEPENDS_ON", "EXTENDS"],
        }


def extract_entities(text: str, maxlen: int = 1500) -> dict:
    """
    Extract entities and relationships from text using Ollama directly.

    Args:
        text: Input text (truncated to maxlen)
        maxlen: Maximum text length

    Returns:
        Dict with 'entities' and 'relationships' keys
    """
    from pathlib import Path
    settings = get_settings()

    # Truncate text
    if len(text) > maxlen:
        text = text[:maxlen] + "..."

    # Load schema from config path
    schema_path = Path(settings.domain_schema_path)
    if schema_path.exists():
        schema = load_schema(str(schema_path))
    else:
        schema = {
            "entity_labels": ["Author", "Paper", "Dataset", "Model", "Method"],
            "relationship_types": ["AUTHORED_BY", "USES", "DEPENDS_ON", "EXTENDS"],
        }

    labels = ", ".join(schema.get("entity_labels", []))
    rels = ", ".join(schema.get("relationship_types", []))

    # Prompt
    prompt = f"""Extract entities and relationships from the text below.

Allowed entity labels: {labels}
Allowed relationship types: {rels}

Output ONLY valid JSON (no other text):
{{
  "entities": [{{"label": "Label", "name": "Name", "properties": {{}}}}],
  "relationships": [{{"source": "Name", "rel_type": "TYPE", "target": "Name"}}]
}}

Text:
{text}

JSON:"""

    # Call Ollama
    url = f"{settings.ollama_url}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }

    response = requests.post(url, json=payload, timeout=120)
    result = response.json()

    content = result.get("message", {}).get("content", "")

    # Parse JSON from response
    try:
        # Find JSON in response (look for first { and last })
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            json_str = content[start:end+1]
            data = json.loads(json_str)
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            return {"entities": entities, "relationships": relationships}
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"⚠️ JSON parse failed: {e}")

    return {"entities": [], "relationships": []}


def extract_from_chunks(parent_chunks: list[dict], store: Neo4jStore | None = None) -> dict:
    """
    Extract entities from parent chunks and optionally write to Neo4j.

    Args:
        parent_chunks: List of parent chunk dicts
        store: Optional Neo4j store to write to

    Returns:
        Extraction summary dict
    """
    all_entities = []
    all_relationships = []

    # Track name→id mapping
    entity_id_map = {}

    for i, chunk in enumerate(parent_chunks[:10]):  # Limit to first 10 chunks
        print(f"🔍 Extracting entities from chunk {i+1}/{min(10, len(parent_chunks))}...")

        result = extract_entities(chunk["text"])

        # Map names to IDs
        for entity in result["entities"]:
            entity_id = f"entity_{len(all_entities)}"
            entity["id"] = entity_id
            entity_id_map[entity["name"]] = entity_id
            all_entities.append(entity)

        # Map relationships to IDs
        for rel in result["relationships"]:
            source_name = rel.get("source", "")
            target_name = rel.get("target", "")
            source_id = entity_id_map.get(source_name)
            target_id = entity_id_map.get(target_name)
            if source_id and target_id:
                rel["source_id"] = source_id
                rel["target_id"] = target_id
                all_relationships.append(rel)

    # Write to Neo4j if store provided
    if store:
        print(f"📝 Writing {len(all_entities)} entities to Neo4j...")

        # Track entity IDs from Neo4j
        neo4j_id_map = {}

        for entity in all_entities:
            label = sanitize_identifier(entity.get("label", "Entity"))
            name = entity.get("name", "unknown")
            props = entity.get("properties", {})

            neo4j_id = store.upsert_entity(label, name, props)
            neo4j_id_map[name] = neo4j_id

        print(f"📝 Writing {len(all_relationships)} relationships to Neo4j...")

        for rel in all_relationships:
            source_name = rel.get("source", "")
            target_name = rel.get("target", "")
            source_id = neo4j_id_map.get(source_name)
            target_id = neo4j_id_map.get(target_name)

            if source_id and target_id:
                rel_type = sanitize_identifier(rel.get("rel_type", "RELATED_TO"))
                store.upsert_relationship(source_id, target_id, rel_type)

    return {
        "entities_extracted": len(all_entities),
        "relationships_extracted": len(all_relationships),
    }


def get_graph_structure() -> str:
    """Get actual graph structure to include in prompt."""
    try:
        store = Neo4jStore()
        result = store.execute_cypher('''
        MATCH (a)-[r]->(b) WHERE NOT a:Chunk
        RETURN DISTINCT labels(a)[0] as src, type(r) as rel, labels(b)[0] as tgt LIMIT 30
        ''')
        lines = [f"({r['src']})-[:{r['rel']}]->({r['tgt']})" for r in result]
        return "\n".join(lines)
    except Exception:
        return ""


def generate_cypher_for_query(question: str, context: str, maxlen: int = 800) -> str | None:
    """
    Generate a read-only Cypher query for a given question using DSPy.

    Args:
        question: User question
        context: Retrieved context

    Returns:
        Cypher query string or None
    """
    from ..reasoning.cypher_generator import get_graph_brain

    # Truncate context if needed
    if len(context) > maxlen:
        context = context[:maxlen] + "..."

    try:
        # Get the DSPy-powered graph brain
        brain = get_graph_brain()

        # Optionally inject context (if useful for complex queries)
        # For now, we rely purely on schema guidance
        cypher = brain.generate(question)

        if cypher:
            return cypher
        else:
            print("⚠️ No Cypher query generated")
            return None

    except Exception as e:
        print(f"⚠️ Cypher generation failed: {e}")
        return None