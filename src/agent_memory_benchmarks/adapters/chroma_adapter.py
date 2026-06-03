"""ChromaDB embedded vector adapter."""

import contextlib

import chromadb

from .base import Document, SearchResult


class ChromaDBAdapter:
    name = "chromadb"

    def __init__(self, path: str | None = None):
        self._client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self._col = self._client.get_or_create_collection("chunks")

    @staticmethod
    def _flat_metadata(m: dict) -> dict:
        """ChromaDB only accepts scalar metadata values — serialize nested types."""
        return {k: v if isinstance(v, (str, int, float, bool)) else str(v) for k, v in m.items()}

    def index(self, docs: list[Document]) -> None:
        metadatas = [self._flat_metadata(d.metadata) for d in docs]
        self._col.add(
            ids=[d.id for d in docs],
            documents=[d.text for d in docs],
            embeddings=[d.embedding for d in docs if d.embedding] or None,
            metadatas=metadatas if any(metadatas) else None,
        )

    def query(
        self,
        text: str,
        vector: list[float] | None,
        k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        kwargs: dict = {"query_texts": [text], "n_results": k}
        if vector:
            kwargs = {"query_embeddings": [vector], "n_results": k}
        if filters:
            kwargs["where"] = filters
        res = self._col.query(**kwargs)
        ids = res["ids"][0]
        distances = res["distances"][0]
        return [SearchResult(id=i, score=1.0 - d) for i, d in zip(ids, distances, strict=False)]

    def optimize(self) -> None:
        pass

    def teardown(self) -> None:
        with contextlib.suppress(Exception):
            self._client.delete_collection("chunks")
