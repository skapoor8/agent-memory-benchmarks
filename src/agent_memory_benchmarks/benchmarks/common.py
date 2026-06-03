"""Shared benchmark utilities: embedding model cache, evaluation loop, result dataclass."""

import time
from dataclasses import asdict, dataclass

from sentence_transformers import SentenceTransformer

from ..adapters.base import Document
from ..metrics.retrieval import latency_percentiles, mrr_at_k, ndcg_at_k, recall_at_k

MODEL_CACHE: dict[str, SentenceTransformer] = {}

_MODEL_ALIASES = {
    "small": "all-MiniLM-L6-v2",
    "bge-m3": "BAAI/bge-m3",
}


def get_model(name: str) -> SentenceTransformer:
    if name not in MODEL_CACHE:
        model_id = _MODEL_ALIASES.get(name, name)
        MODEL_CACHE[name] = SentenceTransformer(model_id)
    return MODEL_CACHE[name]


def embed_docs(docs: list[Document], model_name: str) -> list[Document]:
    model = get_model(model_name)
    texts = [d.text for d in docs]
    vecs = model.encode(texts, show_progress_bar=False, batch_size=64)
    return [
        Document(id=d.id, text=d.text, metadata=d.metadata, embedding=v.tolist())
        for d, v in zip(docs, vecs, strict=False)
    ]


@dataclass
class BenchmarkResult:
    benchmark: str
    store: str
    model: str
    corpus: str
    ndcg_at_10: float = 0.0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr_at_10: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    index_throughput_docs_per_sec: float = 0.0
    num_queries: int = 0
    num_docs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def run_evaluation(
    adapter,
    docs: list[Document],
    queries: list[dict],
    qrels: dict[str, set[str]],
    k_values: tuple[int, ...] = (1, 5, 10),
    warmup: int = 10,
) -> BenchmarkResult:
    """Core evaluation loop.

    queries: list of {"id": ..., "text": ..., "vector": [...] or None}
    qrels: {query_id: {relevant_doc_id, ...}}
    """
    if not queries:
        return BenchmarkResult(benchmark="", store=adapter.name, model="", corpus="", num_docs=len(docs))

    # Warm up
    for q in queries[:warmup]:
        adapter.query(q["text"], q.get("vector"), k=10)

    latencies: list[float] = []
    all_retrieved: dict[str, list[str]] = {}
    max_k = max(k_values)
    for q in queries:
        t0 = time.perf_counter_ns()
        results = adapter.query(q["text"], q.get("vector"), k=max_k)
        latencies.append((time.perf_counter_ns() - t0) / 1e6)
        all_retrieved[q["id"]] = [r.id for r in results]

    lp = latency_percentiles(latencies)

    def agg(fn, k: int) -> float:
        if not qrels:
            return 0.0
        return sum(fn(all_retrieved.get(qid, []), rel, k) for qid, rel in qrels.items()) / len(qrels)

    return BenchmarkResult(
        benchmark="",
        store=adapter.name,
        model="",
        corpus="",
        ndcg_at_10=agg(ndcg_at_k, 10),
        recall_at_1=agg(recall_at_k, 1),
        recall_at_5=agg(recall_at_k, 5),
        recall_at_10=agg(recall_at_k, 10),
        mrr_at_10=agg(mrr_at_k, 10),
        latency_p50_ms=lp["p50"],
        latency_p95_ms=lp["p95"],
        latency_p99_ms=lp["p99"],
        num_queries=len(queries),
        num_docs=len(docs),
    )
