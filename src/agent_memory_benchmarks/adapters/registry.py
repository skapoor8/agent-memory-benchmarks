"""Adapter registry: maps CLI --store names to adapter classes."""

from .chroma_adapter import ChromaDBAdapter
from .lancedb_adapter import LanceDBAdapter
from .qdrant_adapter import QdrantAdapter
from .sqlite_fts5 import SQLiteFTS5Adapter
from .tantivy_adapter import TantivyAdapter

ADAPTERS = {
    "sqlite": SQLiteFTS5Adapter,
    "lancedb": LanceDBAdapter,
    "chromadb": ChromaDBAdapter,
    "tantivy": TantivyAdapter,
    "qdrant": QdrantAdapter,
}
TIER1 = ["sqlite", "lancedb", "chromadb", "tantivy"]


def get_adapter(name: str, path: str | None = None):
    if name == "all":
        # Each adapter gets its own sub-directory to avoid path collisions
        return [ADAPTERS[k](path=f"{path}/{k}" if path else None) for k in TIER1]
    if name not in ADAPTERS:
        raise ValueError(f"Unknown adapter '{name}'. Choose from: {list(ADAPTERS)}")
    return ADAPTERS[name](path=path)
