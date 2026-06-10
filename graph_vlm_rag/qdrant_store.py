"""Qdrant hybrid vector store — dense (Ollama) + sparse (BM25) embeddings."""

import uuid
from typing import Iterator, Optional

import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
)

from .config import get_settings
from .chunker import resolve_parent_text


# Ollama embedding endpoint (using nomic-embed-text)
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"


def get_embeddings(texts: list[str], model: str = "nomic-embed-text:latest") -> list[list[float]]:
    """Get dense embeddings from Ollama."""
    url = OLLAMA_EMBED_URL
    embeddings = []

    for text in texts:
        payload = {"model": model, "prompt": text}
        response = requests.post(url, json=payload, timeout=60)
        result = response.json()
        embeddings.append(result["embedding"])

    return embeddings


def embed_text(text: str, model: str = "nomic-embed-text:latest") -> list[float]:
    """Get dense embedding for a single text."""
    return get_embeddings([text], model)[0]


class HybridEmbeddings:
    """
    Hybrid embedder producing both dense (Ollama) and sparse (BM25) vectors.

    Dense: semantic understanding via nomic-embed-text
    Sparse: keyword matching via FastEmbedSparse (BM25-style)
    """

    def __init__(self, dense_model: str = "nomic-embed-text:latest"):
        from langchain_qdrant import FastEmbedSparse

        self.dense_model = dense_model
        self.sparse_model = FastEmbedSparse(model_name="Qdrant/bm25")

    def embed_documents(self, texts: list[str]) -> list[dict]:
        """Embed a list of documents into hybrid vectors.

        Returns:
            List of dicts: [{"dense": [...], "sparse": SparseVector}, ...]
        """
        dense_vecs = get_embeddings(texts, self.dense_model)

        # Sparse embeddings (BM25) - returns list of SparseVector (Pydantic models)
        sparse_vecs = self.sparse_model.embed_documents(texts)

        results = []
        for d, s in zip(dense_vecs, sparse_vecs):
            # s is SparseVector with .indices and .values (already lists)
            results.append({
                "dense": d,
                "sparse": {
                    "indices": list(s.indices),
                    "values": list(s.values),
                }
            })
        return results

    def embed_query(self, text: str) -> dict:
        """Embed a single query into hybrid vectors."""
        dense_vec = embed_text(text, self.dense_model)

        # Sparse embedding for query (returns single SparseVector, not list)
        sparse = self.sparse_model.embed_query(text)

        if sparse is None:
            sparse_dict = {"indices": [], "values": []}
        else:
            sparse_dict = {
                "indices": list(sparse.indices),
                "values": list(sparse.values),
            }

        return {"dense": dense_vec, "sparse": sparse_dict}


class QdrantStore:
    """Qdrant hybrid vector store wrapper (dense + sparse)."""

    def __init__(self, collection_name: str = "graph_vlm_rag"):
        settings = get_settings()
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection_name = collection_name
        self.embed_model = settings.embed_model
        self.hybrid = HybridEmbeddings(dense_model=self.embed_model)
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection with hybrid (dense + sparse) config if not exists."""
        if not self.client.collection_exists(self.collection_name):
            # Get dense embedding dimension
            test_emb = embed_text("test", self.embed_model)
            dim = len(test_emb)

            # Create with BOTH dense and sparse vectors
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=dim, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
            )
            print(f"✅ Created hybrid collection '{self.collection_name}' (dense_dim={dim}, sparse=BM25)")
        else:
            print(f"📂 Using collection '{self.collection_name}'")

    def upsert_chunks(self, chunks: list[dict]) -> int:
        """Upsert ONLY child chunks to Qdrant with hybrid vectors.

        Parent chunks are stored separately in ParentStore for lookup.
        """
        # Get child chunks ONLY
        child_chunks = [c for c in chunks if c["chunk_type"] == "child"]

        if not child_chunks:
            print("📭 No child chunks to upsert")
            return 0

        # Get hybrid embeddings
        child_texts = [c["text"] for c in child_chunks]
        print(f"📐 Embedding {len(child_texts)} child chunks (dense + sparse)...")
        hybrid_vecs = self.hybrid.embed_documents(child_texts)

        # Create points
        points = []
        for chunk, hv in zip(child_chunks, hybrid_vecs):
            from qdrant_client.http.models import SparseVector
            sparse_vec = SparseVector(
                indices=list(hv["sparse"]["indices"]),
                values=list(hv["sparse"]["values"]),
            )

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": hv["dense"],
                    "sparse": sparse_vec,
                },
                payload={
                    "parent_id": chunk["parent_id"],
                    "text": chunk["text"],
                    "chunk_type": "child",
                    "document_name": chunk["document_name"],
                    "chunk_index": chunk["chunk_index"],
                },
            ))

        # Upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        print(f"🚀 Upserted {len(points)} child vectors (hybrid) to Qdrant")
        return len(points)

    def search(
        self,
        query: str,
        limit: int = 5,
        document_name: Optional[str] = None,
    ) -> list[dict]:
        """Hybrid search: dense + sparse (BM25) with RRF fusion.

        Args:
            query: Search query
            limit: Number of results
            document_name: Optional filter to restrict to a specific document

        Returns:
            List of result dicts
        """
        from qdrant_client.http import models
        from qdrant_client.http.models import SparseVector

        # Get hybrid query vectors
        qv = self.hybrid.embed_query(query)
        sparse_query = SparseVector(
            indices=list(qv["sparse"]["indices"]),
            values=list(qv["sparse"]["values"]),
        )

        # Build optional filter for document_name
        query_filter = None
        if document_name:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_name",
                        match=models.MatchValue(value=document_name),
                    )
                ]
            )

        # Hybrid prefetch: dense + sparse, then RRF fusion
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=qv["dense"],
                    using="dense",
                    limit=limit * 2,  # Get more candidates for fusion
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using="sparse",
                    limit=limit * 2,
                    filter=query_filter,
                ),
            ],
            # RRF (Reciprocal Rank Fusion) for combining dense + sparse
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points

        return [{
            "id": r.id,
            "score": r.score,
            "text": r.payload["text"],
            "parent_id": r.payload["parent_id"],
            "chunk_type": r.payload["chunk_type"],
            "document_name": r.payload["document_name"],
        } for r in results]

    def search_with_parents(
        self,
        query: str,
        limit: int = 5,
        all_chunks: Optional[list[dict]] = None,
        document_name: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid search and resolve parent text.

        Args:
            query: Search query
            limit: Number of results
            all_chunks: Full chunk list for parent resolution
            document_name: Optional document filter

        Returns:
            List of results with parent text included
        """
        results = self.search(query, limit, document_name=document_name)

        if all_chunks is None:
            return results

        # Resolve parents
        for result in results:
            parent_id = result.get("parent_id")
            if parent_id:
                parent_text = resolve_parent_text(all_chunks, parent_id)
                result["parent_text"] = parent_text

        return results

    def clear(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        print(f"🗑️ Cleared collection '{self.collection_name}'")
