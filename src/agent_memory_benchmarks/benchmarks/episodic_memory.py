"""Episodic Memory Benchmark.

Task: Given a retrospective question about a past decision, retrieve the
      relevant ADR or issue record from a project's memory store.

Corpora:
  neon-adrs: Architecture Decision Records from neondatabase/neon
  cli-issues: GitHub issue cross-references from cli/cli (deferred)

GT:  ADR heading/title as query → ADR body chunk as relevant doc
     Issue cross-reference: issue A description → issue B body (deferred)
Metric: Recall@1, Recall@5, nDCG@10
Source: https://github.com/neondatabase/neon/tree/main/docs
"""

import json
import subprocess
from pathlib import Path

from ..adapters.base import Document
from ..corpus.chunker import MarkdownSplitter
from ..corpus.loaders import load_markdown_corpus
from .common import BenchmarkResult, embed_docs, get_model, run_evaluation

DATA_DIR = Path("data/episodic-memory")

CORPORA: dict[str, tuple[str, str] | None] = {
    "neon-adrs": ("https://github.com/neondatabase/neon.git", "docs"),
    "cli-issues": None,  # mined via GitHub API — deferred
}


def download_neon_adrs() -> list[Document]:
    """Shallow-clone neondatabase/neon and chunk the docs/ markdown tree."""
    clone_dir = DATA_DIR / "neon"
    if not clone_dir.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/neondatabase/neon.git", str(clone_dir)],
            check=True,
        )
    docs_dir = clone_dir / "docs"
    splitter = MarkdownSplitter(max_tokens=512, overlap=64)
    docs: list[Document] = []
    for path, text in load_markdown_corpus(docs_dir):
        for chunk in splitter.split(text, str(path)):
            docs.append(Document(id=chunk.id, text=chunk.text, metadata=chunk.metadata))
    return docs


def build_adr_qrels(docs: list[Document], out_path: Path) -> None:
    """Use the first heading/line of each chunk as the query (ADR title → ADR body)."""
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for doc in docs:
            heading = doc.metadata.get("heading") or doc.text.split("\n")[0][:120]
            heading = heading.strip().lstrip("#").strip()
            if not heading:
                continue
            f.write(json.dumps({"query_id": f"q_{doc.id}", "query": heading, "relevant": [doc.id]}) + "\n")


def run(adapter, corpus: str = "neon-adrs", model: str = "small") -> BenchmarkResult:
    if corpus not in CORPORA:
        raise ValueError(f"Unknown episodic-memory corpus '{corpus}'. Choose from: {list(CORPORA)}")
    if CORPORA[corpus] is None:
        raise NotImplementedError(f"Corpus '{corpus}' is deferred to future work")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    docs = download_neon_adrs()
    qrels_path = DATA_DIR / corpus / "qrels.jsonl"
    build_adr_qrels(docs, qrels_path)

    docs = embed_docs(docs, model)
    model_obj = get_model(model)
    adapter.index(docs)
    adapter.optimize()

    queries: list[dict] = []
    qrels: dict[str, set[str]] = {}
    with qrels_path.open() as f:
        for line in f:
            row = json.loads(line)
            queries.append({"id": row["query_id"], "text": row["query"]})
            qrels[row["query_id"]] = set(row["relevant"])

    vecs = model_obj.encode([q["text"] for q in queries]).tolist()
    queries = [{**q, "vector": v} for q, v in zip(queries, vecs, strict=False)]

    result = run_evaluation(adapter, docs, queries, qrels)
    result.benchmark = "episodic-memory"
    result.corpus = corpus
    result.model = model
    return result
