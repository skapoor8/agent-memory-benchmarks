"""Parametrized smoke tests for Tier 1 adapters."""

import pytest

from agent_memory_benchmarks.adapters.base import Document
from agent_memory_benchmarks.adapters.registry import ADAPTERS, TIER1


@pytest.mark.parametrize("store", TIER1)
def test_adapter_roundtrip(store, tmp_path):
    adapter = ADAPTERS[store](path=str(tmp_path / store))
    docs = [Document(id=f"d{i}", text=f"document about topic_{i}") for i in range(5)]
    adapter.index(docs)
    adapter.optimize()
    results = adapter.query("topic_3", vector=None, k=3)
    ids = [r.id for r in results]
    assert "d3" in ids
    adapter.teardown()
