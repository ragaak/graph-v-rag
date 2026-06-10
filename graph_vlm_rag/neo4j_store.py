"""Neo4j graph store — entities + relationships + chunks"""

import uuid
from typing import Iterator

from neo4j import GraphDatabase

from .config import get_settings
from .cypher_safety import sanitize_identifier


class Neo4jStore:
    """Neo4j graph store wrapper."""

    def __init__(self):
        settings = get_settings()
        self.driver = GraphDatabase.driver(
            settings.neo4j_url,
            auth=("neo4j", settings.neo4j_password),
        )
        self._verify_connectivity()

    def _verify_connectivity(self):
        """Verify connection."""
        self.driver.verify_connectivity()
        print("✅ Connected to Neo4j")

    def close(self):
        """Close connection."""
        self.driver.close()

    def clear_all(self):
        """Clear all nodes and relationships."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("🗑️ Cleared Neo4j")

    def upsert_chunks(self, chunks: list[dict]) -> int:
        """
        Deprecated. Use upsert_entities() and upsert_relationships() instead.
        Kept for backward compatibility but does nothing.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            0 (no-op)
        """
        print("📭 Chunk storage in Neo4j is deprecated. Using upsert_entities/relationships instead.")
        return 0

    def upsert_entities(self, entities: list[dict]) -> int:
        """
        Upsert entity nodes to Neo4j.

        Args:
            entities: List of entity dictionaries with id, label, name, properties

        Returns:
            Number of entities upserted
        """
        if not entities:
            print("📭 No entities to upsert")
            return 0

        from .cypher_safety import sanitize_identifier

        with self.driver.session() as session:
            for entity in entities:
                label = sanitize_identifier(entity.get("label", "Entity"))

                session.run(f"""
                    MERGE (e:{label} {{name: $name}})
                    SET e.id = $id
                    SET e += $properties
                """,
                    id=entity["id"],
                    name=entity["name"],
                    properties=entity.get("properties", {}))

        print(f"🚀 Upserted {len(entities)} entities to Neo4j")
        return len(entities)

    def upsert_relationships(self, relationships: list[dict]) -> int:
        """
        Upsert relationship edges to Neo4j.

        Args:
            relationships: List of relationship dictionaries

        Returns:
            Number of relationships upserted
        """
        if not relationships:
            print("📭 No relationships to upsert")
            return 0

        from .cypher_safety import sanitize_identifier

        with self.driver.session() as session:
            for rel in relationships:
                rel_type = sanitize_identifier(rel["type"].upper())

                # Load allowed rel types from schema
                allowed_rels = {"DEPENDS_ON", "USES", "PROVIDES", "AUTHORED_BY", "EXTENDS", "AFFECTED_BY"}

                # Add from config if available
                from pathlib import Path
                import yaml
                schema_path = Path("data/domain_schema.yaml")
                if schema_path.exists():
                    with open(schema_path) as f:
                        schema = yaml.safe_load(f)
                        allowed_rels.update(schema.get("relationship_types", []))

                if rel_type not in allowed_rels:
                    print(f"⚠️ Skipping disallowed rel type: {rel_type}")
                    continue

                session.run("""
                    MATCH (source {id: $source_id})
                    MATCH (target {id: $target_id})
                    MERGE (source)-[r:REL_TYPE]->(target)
                    SET r += $properties
                """.replace("REL_TYPE", rel_type),
                    source_id=rel["source_id"],
                    target_id=rel["target_id"],
                    properties=rel.get("properties", {}))

        print(f"🚀 Upserted {len(relationships)} relationships to Neo4j")
        return len(relationships)

    def upsert_entity(self, label: str, name: str, properties: dict | None = None) -> str:
        """
        Upsert an entity node.

        Args:
            label: Entity label (e.g., "Project", "Server")
            name: Entity name
            properties: Optional properties

        Returns:
            Entity ID
        """
        label = sanitize_identifier(label)
        entity_id = str(uuid.uuid4())

        with self.driver.session() as session:
            session.run(f"""
                MERGE (e:{label} {{name: $name}})
                SET e.id = $id
            """, id=entity_id, name=name)

            if properties:
                # Set properties one by one
                for key, value in properties.items():
                    session.run(f"""
                        MATCH (e:{label} {{name: $name}})
                        SET e.{key} = $value
                    """, name=name, key=key, value=value)

        return entity_id

    def upsert_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict | None = None,
    ) -> bool:
        """
        Upsert a relationship between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            rel_type: Relationship type
            properties: Optional properties

        Returns:
            True if successful
        """
        from .cypher_safety import sanitize_identifier

        rel_type = sanitize_identifier(rel_type.upper())

        # Load allowed rel types from schema
        allowed_rels = {"DEPENDS_ON", "USES", "PROVIDES", "AUTHORED_BY", "EXTENDS", "AFFECTED_BY"}

        # Add from config if available
        from pathlib import Path
        import yaml
        schema_path = Path("data/domain_schema.yaml")
        if schema_path.exists():
            with open(schema_path) as f:
                schema = yaml.safe_load(f)
                allowed_rels.update(schema.get("relationship_types", []))

        if rel_type not in allowed_rels:
            print(f"⚠️ Skipping disallowed rel type: {rel_type}")
            return False

        try:
            with self.driver.session() as session:
                session.run(f"""
                    MATCH (a), (b)
                    WHERE a.id = $source_id AND b.id = $target_id
                    MERGE (a)-[r:{rel_type}]->(b)
                """, source_id=source_id, target_id=target_id)
            return True
        except Exception as e:
            print(f"⚠️ Failed to upsert relationship: {e}")
            return False

    def execute_cypher(self, cypher: str, **params) -> list[dict]:
        """
        Execute a read-only Cypher query.

        Args:
            cypher: Cypher query
            **params: Query parameters

        Returns:
            List of result records
        """
        from .cypher_safety import validate_cypher

        # Validate first
        is_valid, error = validate_cypher(cypher)
        if not is_valid:
            raise ValueError(f"Invalid Cypher: {error}")

        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def get_chunk_neighborhood(
        self,
        chunk_id: str,
        depth: int = 2,
    ) -> list[dict]:
        """
        Get entities connected to a chunk.

        Args:
            chunk_id: Chunk ID
            depth: Traversal depth

        Returns:
            List of connected entities
        """
        cypher = f"""
            MATCH (c:Chunk {{id: $chunk_id}})
            MATCH path = (c)-[*1..{depth}]-(e)
            RETURN path, c, e
            LIMIT 20
        """

        return self.execute_cypher(cypher, chunk_id=chunk_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()