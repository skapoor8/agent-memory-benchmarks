"""SQLite FTS5 BM25 baseline adapter."""

import re
import sqlite3

from .base import Document, SearchResult


def _fts5_query(text: str) -> str:
    """Build an FTS5 MATCH query from free text.

    FTS5 uses implicit AND by default (all terms must appear), which gives 0
    results against short documents like file paths. We use explicit OR so that
    any term match contributes to the BM25 score — consistent with Tantivy's
    default behavior and standard BM25 semantics.

    Hyphens are also stripped: FTS5 treats '- term' as a NOT operator.
    """
    cleaned = re.sub(r"[^\w\s]", " ", text)
    words = cleaned.split()[:30]  # cap query length
    return " OR ".join(words) if words else "x"


class SQLiteFTS5Adapter:
    name = "sqlite"

    def __init__(self, path: str | None = None):
        self._path = path or ":memory:"
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(id, text, metadata)")

    def index(self, docs: list[Document]) -> None:
        self._conn.executemany(
            "INSERT INTO chunks(id, text, metadata) VALUES (?, ?, ?)",
            [(d.id, d.text, str(d.metadata)) for d in docs],
        )
        self._conn.commit()

    def query(
        self,
        text: str,
        vector: list[float] | None,
        k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        rows = self._conn.execute(
            "SELECT id, rank FROM chunks WHERE chunks MATCH ? ORDER BY rank LIMIT ?",
            (_fts5_query(text), k),
        ).fetchall()
        return [SearchResult(id=r[0], score=-r[1]) for r in rows]

    def optimize(self) -> None:
        self._conn.execute("INSERT INTO chunks(chunks) VALUES('optimize')")

    def teardown(self) -> None:
        self._conn.close()
