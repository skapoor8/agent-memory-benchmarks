"""Code Finding Benchmark.

Task: Given a GitHub issue description, retrieve the files in the codebase
      most likely to need modification.

Corpus: SWE-bench Lite (300 instances, Python repos)
GT:     Gold-patch file paths from SWE-bench oracle context
Metric: File-level Recall@k (k=1,5,10), nDCG@10
Source: https://github.com/princeton-nlp/SWE-bench
"""

import json
from pathlib import Path

from ..adapters.base import Document
from .common import BenchmarkResult, embed_docs, get_model, run_evaluation

DATA_DIR = Path("data/code-finding")


def _extract_relevant_files(row: dict) -> list[str]:
    """Extract gold-patch file paths from a SWE-bench row.

    SWE-bench Lite rows expose either a "file_contents" mapping (old schema)
    or a "patch" diff. We parse the patch text for `diff --git a/<path>` to
    collect modified files when "file_contents" is absent.
    """
    fc = row.get("file_contents")
    if isinstance(fc, dict) and fc:
        return sorted(fc.keys())
    patch = row.get("patch") or ""
    files: list[str] = []
    seen: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            # form: diff --git a/<path> b/<path>
            parts = line.split()
            if len(parts) >= 3:
                a_path = parts[2]
                if a_path.startswith("a/"):
                    a_path = a_path[2:]
                if a_path not in seen:
                    seen.add(a_path)
                    files.append(a_path)
    return files


def download_corpus() -> None:
    """Download SWE-bench Lite and cache oracle contexts to data/code-finding/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    qrels_path = DATA_DIR / "qrels.jsonl"
    if qrels_path.exists():
        return
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    with qrels_path.open("w") as f:
        for row in ds:
            relevant_files = _extract_relevant_files(row)
            f.write(
                json.dumps(
                    {
                        "query_id": row["instance_id"],
                        "query": row["problem_statement"],
                        "relevant": relevant_files,
                    }
                )
                + "\n"
            )


def build_docs_from_qrels(qrels_path: Path) -> tuple[list[Document], list[dict], dict[str, set[str]]]:
    """Build minimal docs (one per unique file path) and queries from qrels."""
    docs: list[Document] = []
    queries: list[dict] = []
    qrels: dict[str, set[str]] = {}
    doc_ids: set[str] = set()
    with qrels_path.open() as f:
        for line in f:
            row = json.loads(line)
            queries.append({"id": row["query_id"], "text": row["query"]})
            qrels[row["query_id"]] = set(row["relevant"])
            for fid in row["relevant"]:
                if fid not in doc_ids:
                    doc_ids.add(fid)
                    # Use the file path itself as the indexed text (BM25 baseline);
                    # vector path uses embeddings of the same string.
                    docs.append(Document(id=fid, text=fid, metadata={"file_path": fid}))
    return docs, queries, qrels


def run(adapter, corpus: str = "swe-bench-lite", model: str = "small") -> BenchmarkResult:
    download_corpus()
    docs, queries, qrels = build_docs_from_qrels(DATA_DIR / "qrels.jsonl")
    docs = embed_docs(docs, model)
    model_obj = get_model(model)

    adapter.index(docs)
    adapter.optimize()

    vecs = model_obj.encode([q["text"] for q in queries]).tolist()
    queries = [{**q, "vector": v} for q, v in zip(queries, vecs, strict=False)]

    result = run_evaluation(adapter, docs, queries, qrels)
    result.benchmark = "code-finding"
    result.corpus = corpus
    result.model = model
    result.num_docs = len(docs)
    return result
