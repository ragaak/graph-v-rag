"""Pydantic types for graph_vlm_rag"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class Chunk(BaseModel):
    """A text chunk with parent/child relationship."""

    id: str
    text: str
    chunk_type: str = "child"  # "parent" or "child"
    parent_id: Optional[str] = None
    document_name: Optional[str] = None
    chunk_index: int = 0
    created_at: datetime = Field(default_factory=datetime.now)


class Entity(BaseModel):
    """An entity extracted from text."""

    id: str
    label: str  # e.g., "Project", "Server", "Database"
    name: str
    properties: dict = Field(default_factory=dict)


class Relationship(BaseModel):
    """A relationship between entities."""

    source_id: str
    target_id: str
    rel_type: str  # e.g., "DEPENDS_ON", "LOCATED_IN"
    properties: dict = Field(default_factory=dict)


class IngestResult(BaseModel):
    """Result of an ingest operation."""

    document_name: str
    chunks_created: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    entities_in_qdrant: int = 0
    entities_in_neo4j: int = 0


class QueryResult(BaseModel):
    """Result of a query operation."""

    question: str
    answer: str
    reasoning: Optional[str] = None
    context_sources: list = Field(default_factory=list)
    cypher_attempts: int = 1