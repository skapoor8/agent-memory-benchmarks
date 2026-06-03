"""Qdrant hybrid adapter (Tier 2 optional; uses :memory: by default)."""

import os

from qdrant_client import QdrantClient, models

from .base import Document, SearchResult

COLLECTION = "chunks"
DIM = 384  # default all-MiniLM dim; overridden by first indexed doc


class QdrantAdapter:
    name = "qdrant"

    def __init__(self, path: str | None = None):
        url = os.environ.get("QDRANT_URL")
        self._client = QdrantClient(url=url) if url else QdrantClient(":memory:")
        self._dim: int | None = None

    def index(self, docs: list[Document]) -> None:
        if not docs:
            return
        dim = len(docs[0].embedding) if docs[0].embedding else DIM
        if self._dim is None:
            self._dim = dim
            self._client.recreate_collection(
                COLLECTION,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
        self._client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(
                    id=abs(hash(d.id)) % (2**63),
                    vector=d.embedding or [0.0] * dim,
                    payload={"id": d.id, "text": d.text, **d.metadata},
                )
                for d in docs
            ],
        )

    def query(
        self,
        text: str,
        vector: list[float] | None,
        k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        if not vector or self._dim is None:
            return []
        hits = self._client.query_points(COLLECTION, query=vector, limit=k).points
        return [SearchResult(id=h.payload["id"], score=h.score) for h in hits]

    def optimize(self) -> None:
        pass

    def teardown(self) -> None:
        self._client.delete_collection(COLLECTION)
