from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class RetrievalAdapter(Protocol):
    name: str

    def index(self, docs: list[Document]) -> None: ...
    def query(
        self,
        text: str,
        vector: list[float] | None,
        k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]: ...
    def optimize(self) -> None: ...
    def teardown(self) -> None: ...
