"""LanceDB hybrid ANN+BM25 adapter (primary Tier 1)."""

import lancedb

from .base import Document, SearchResult


class LanceDBAdapter:
    name = "lancedb"

    def __init__(self, path: str | None = None):
        self._path = path or "/tmp/amb_lancedb"
        self._db = lancedb.connect(self._path)
        self._table = None

    def index(self, docs: list[Document]) -> None:
        data = [{"id": d.id, "text": d.text, "vector": d.embedding or []} for d in docs]
        if self._table is None:
            self._table = self._db.create_table("chunks", data=data, mode="overwrite")
        else:
            self._table.add(data)

    def query(
        self,
        text: str,
        vector: list[float] | None,
        k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        if vector and self._table:
            results = self._table.search(query_type="hybrid").text(text).vector(vector).limit(k).to_list()
        elif self._table:
            results = self._table.search(text, query_type="fts").limit(k).to_list()
        else:
            return []
        return [SearchResult(id=r["id"], score=r.get("_relevance_score", 0.0)) for r in results]

    def optimize(self) -> None:
        if self._table:
            self._table.create_fts_index("text", replace=True)

    def teardown(self) -> None:
        pass
