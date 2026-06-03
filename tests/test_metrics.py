from agent_memory_benchmarks.metrics import (
    latency_percentiles,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_perfect():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_recall_zero():
    retrieved = ["x", "y", "z"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, 3) == 0.0


def test_ndcg_ordering():
    # Top-ranked relevant doc should yield higher nDCG than later-ranked one.
    relevant = {"a"}
    high = ndcg_at_k(["a", "x", "y"], relevant, 3)
    low = ndcg_at_k(["x", "y", "a"], relevant, 3)
    assert high > low
    assert high == 1.0


def test_mrr_hit():
    # First relevant doc at rank 2 → MRR = 1/2
    retrieved = ["x", "a", "y"]
    relevant = {"a"}
    assert mrr_at_k(retrieved, relevant, 3) == 0.5
    # No hit → 0
    assert mrr_at_k(["x", "y", "z"], relevant, 3) == 0.0


def test_latency_percentiles():
    latencies = [float(i) for i in range(1, 101)]  # 1..100
    pcts = latency_percentiles(latencies)
    assert pcts["p50"] == 50.5
    assert pcts["p95"] >= 94.0
    assert pcts["p99"] >= 98.0
    assert set(pcts.keys()) == {"p50", "p95", "p99"}
