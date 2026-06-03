"""Doc Search Benchmark.

Task: Given a natural-language question about an API or recipe, retrieve
      the correct documentation section.

Corpus: FastAPI docs (~3-5 MB of .md files) — default. Also supports
        rust-reference and docker docs.
GT:     InPars-lite MVP: first non-empty line of each chunk used as a
        synthetic query whose relevant doc is the chunk itself. Pinned to
        data/doc-search/<corpus>/qrels.jsonl.
Metric: nDCG@10, Recall@5
Source: https://github.com/fastapi/fastapi/tree/master/docs
"""

import json
import subprocess
from pathlib import Path

from ..adapters.base import Document
from ..corpus.chunker import MarkdownSplitter
from ..corpus.loaders import load_markdown_corpus
from .common import BenchmarkResult, embed_docs, get_model, run_evaluation

DATA_DIR = Path("data/doc-search")

CORPORA: dict[str, tuple[str, str]] = {
    "fastapi": ("https://github.com/fastapi/fastapi.git", "docs/en/docs"),
    "rust-reference": ("https://github.com/rust-lang/reference.git", "src"),
    "docker": ("https://github.com/docker/docs.git", "content"),
}


def download_corpus(corpus: str = "fastapi") -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if corpus not in CORPORA:
        raise ValueError(f"Unknown doc-search corpus '{corpus}'. Choose from: {list(CORPORA)}")
    repo_url, sub_path = CORPORA[corpus]
    clone_dir = DATA_DIR / corpus
    if not clone_dir.exists():
        subprocess.run(["git", "clone", "--depth=1", repo_url, str(clone_dir)], check=True)
    return clone_dir / sub_path


def build_chunks(corpus_path: Path) -> list[Document]:
    splitter = MarkdownSplitter(max_tokens=512, overlap=64)
    files = load_markdown_corpus(corpus_path)
    docs: list[Document] = []
    for path, text in files:
        for chunk in splitter.split(text, str(path)):
            docs.append(Document(id=chunk.id, text=chunk.text, metadata=chunk.metadata))
    return docs


def generate_qrels(docs: list[Document], out_path: Path) -> None:
    """InPars-lite MVP: use first non-empty line of each chunk as a synthetic query.

    For production-quality, replace with LLM-generated questions per chunk
    (deferred to future work).
    """
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for doc in docs:
            lines = [line.strip() for line in doc.text.split("\n") if line.strip()]
            if not lines:
                continue
            query = lines[0][:200]
            f.write(json.dumps({"query_id": f"q_{doc.id}", "query": query, "relevant": [doc.id]}) + "\n")


def run(adapter, corpus: str = "fastapi", model: str = "small") -> BenchmarkResult:
    corpus_path = download_corpus(corpus)
    docs = build_chunks(corpus_path)
    qrels_path = DATA_DIR / corpus / "qrels.jsonl"
    generate_qrels(docs, qrels_path)

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
    result.benchmark = "doc-search"
    result.corpus = corpus
    result.model = model
    return result
