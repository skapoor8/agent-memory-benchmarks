import numpy as np


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k retrieved."""
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(rank + 2) for rank, doc_id in enumerate(retrieved[:k]) if doc_id in relevant)
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0.0


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Mean Reciprocal Rank at k."""
    for rank, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            return 1.0 / (rank + 1)
    return 0.0


def latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    a = np.array(latencies_ms)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }
