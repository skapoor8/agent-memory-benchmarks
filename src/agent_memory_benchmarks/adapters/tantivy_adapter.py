"""Tantivy BM25 adapter (best for code identifiers)."""

import re
import tempfile
from pathlib import Path

import tantivy

from .base import Document, SearchResult


def _tantivy_query(text: str) -> str:
    """Strip all Tantivy query syntax chars including hyphens; keep only alphanumeric and spaces.

    Hyphens must be stripped: Tantivy parses '- term' and 'word-word' as negation/syntax,
    causing parse errors on hyphenated compound words common in issue descriptions.
    """
    cleaned = re.sub(r"[^\w\s]", " ", text)
    words = cleaned.split()[:30]
    return " ".join(words) or "x"


class TantivyAdapter:
    name = "tantivy"

    def __init__(self, path: str | None = None):
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("id", stored=True)
        schema_builder.add_text_field("text", stored=True)
        self._schema = schema_builder.build()
        idx_path = path or tempfile.mkdtemp()
        Path(idx_path).mkdir(parents=True, exist_ok=True)
        self._index = tantivy.Index(self._schema, path=idx_path)

    def index(self, docs: list[Document]) -> None:
        writer = self._index.writer()
        for d in docs:
            writer.add_document(tantivy.Document(id=d.id, text=d.text))
        writer.commit()

    def query(
        self,
        text: str,
        vector: list[float] | None,
        k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        self._index.reload()
        searcher = self._index.searcher()
        query = self._index.parse_query(_tantivy_query(text), ["text"])
        hits = searcher.search(query, k).hits
        return [SearchResult(id=searcher.doc(addr)["id"][0], score=score) for score, addr in hits]

    def optimize(self) -> None:
        pass

    def teardown(self) -> None:
        pass
