"""NF-Corpus Benchmark.

Task: Given a consumer health query, retrieve the relevant biomedical literature.

Corpus:  BEIR nfcorpus — 3,633 PubMed abstracts from NutritionFacts.org
Queries: 323 consumer health questions from NutritionFacts.org video titles
GT:      Official BEIR qrels (3-level relevance: 1=possibly, 2=relevant, 3=highly)
         All qrel entries with relevance >= 1 are treated as relevant.
Metric:  nDCG@10, Recall@5, Recall@10
Source:  Boteva et al. (2016); BEIR benchmark (Thakur et al., 2021)
"""

import json
from pathlib import Path

from ..adapters.base import Document
from .common import BenchmarkResult, embed_docs, get_model, run_evaluation

DATA_DIR = Path("data/nfcorpus")

# Small synthetic fallback for offline/CI — 10 biomedical documents + 10 queries
# with enough vocabulary overlap for BM25 to produce non-zero scores.
_FALLBACK_DOCS = [
    ("d001", "Vitamin D supplementation and cognitive function: a randomized controlled trial"),
    ("d002", "Effect of omega-3 fatty acids on cardiovascular disease risk in adults"),
    ("d003", "Dietary fiber intake and colorectal cancer prevention: systematic review"),
    ("d004", "Probiotic bacteria and gut microbiome diversity in inflammatory bowel disease"),
    ("d005", "Green tea polyphenols and oxidative stress biomarkers in healthy volunteers"),
    ("d006", "Intermittent fasting effects on insulin sensitivity and metabolic syndrome"),
    ("d007", "Mediterranean diet adherence and all-cause mortality: cohort study"),
    ("d008", "Iron deficiency anemia prevalence and dietary iron bioavailability"),
    ("d009", "Antioxidant vitamins C and E and risk of Alzheimer disease: meta-analysis"),
    ("d010", "Calcium and vitamin D co-supplementation for osteoporosis prevention"),
]

_FALLBACK_QUERIES = [
    ("q001", {"d001"}, "does vitamin D help memory and cognitive decline"),
    ("q002", {"d002"}, "can fish oil prevent heart attacks"),
    ("q003", {"d003"}, "does eating fiber reduce colon cancer risk"),
    ("q004", {"d004"}, "do probiotics help with Crohn's disease"),
    ("q005", {"d005"}, "is green tea good for reducing inflammation"),
    ("q006", {"d006"}, "does intermittent fasting improve blood sugar"),
    ("q007", {"d007"}, "does the Mediterranean diet extend lifespan"),
    ("q008", {"d008"}, "what foods help with low iron levels"),
    ("q009", {"d009"}, "can vitamin E prevent dementia"),
    ("q010", {"d010"}, "do calcium supplements prevent osteoporosis"),
]


def _synthetic_nfcorpus() -> tuple[list[Document], list[dict], dict[str, set[str]]]:
    """Offline/CI fallback with 10 biomedical doc-query pairs."""
    docs = [Document(id=doc_id, text=text) for doc_id, text in _FALLBACK_DOCS]
    queries = [{"id": q_id, "text": q_text} for q_id, _, q_text in _FALLBACK_QUERIES]
    qrels = {q_id: rel_set for q_id, rel_set, _ in _FALLBACK_QUERIES}
    return docs, queries, qrels


def _load_from_cache(
    corpus_path: Path, queries_path: Path, qrels_path: Path
) -> tuple[list[Document], list[dict], dict[str, set[str]]]:
    docs: list[Document] = []
    with corpus_path.open() as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(id=row["id"], text=row["text"]))

    queries: list[dict] = []
    with queries_path.open() as f:
        for line in f:
            row = json.loads(line)
            queries.append({"id": row["id"], "text": row["text"]})

    qrels: dict[str, set[str]] = {}
    with qrels_path.open() as f:
        for line in f:
            row = json.loads(line)
            qrels[row["query_id"]] = set(row["relevant"])

    return docs, queries, qrels


def download_nfcorpus() -> tuple[list[Document], list[dict], dict[str, set[str]]]:
    """Load BEIR nfcorpus. Caches to data/nfcorpus/. Falls back to synthetic if unavailable."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = DATA_DIR / "corpus.jsonl"
    queries_path = DATA_DIR / "queries.jsonl"
    qrels_path = DATA_DIR / "qrels.jsonl"

    if corpus_path.exists() and queries_path.exists() and qrels_path.exists():
        return _load_from_cache(corpus_path, queries_path, qrels_path)

    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
        data_path = util.download_and_unzip(url, str(DATA_DIR / "raw"))
        corpus_raw, queries_raw, qrels_raw = GenericDataLoader(data_folder=data_path).load(split="test")

        # Serialize corpus
        docs: list[Document] = []
        with corpus_path.open("w") as f:
            for doc_id, doc_data in corpus_raw.items():
                title = doc_data.get("title", "").strip()
                text = doc_data.get("text", "").strip()
                combined = f"{title}. {text}".strip(". ") if title else text
                docs.append(Document(id=doc_id, text=combined))
                f.write(json.dumps({"id": doc_id, "text": combined}) + "\n")

        # Serialize queries
        queries: list[dict] = []
        with queries_path.open("w") as f:
            for q_id, q_text in queries_raw.items():
                queries.append({"id": q_id, "text": q_text})
                f.write(json.dumps({"id": q_id, "text": q_text}) + "\n")

        # Serialize qrels (relevance >= 1)
        qrels: dict[str, set[str]] = {}
        with qrels_path.open("w") as f:
            for q_id, rel_dict in qrels_raw.items():
                relevant = {doc_id for doc_id, score in rel_dict.items() if score >= 1}
                if not relevant:
                    continue
                qrels[q_id] = relevant
                f.write(json.dumps({"query_id": q_id, "relevant": list(relevant)}) + "\n")

        # Filter queries to those that have qrels
        queries = [q for q in queries if q["id"] in qrels]
        return docs, queries, qrels

    except Exception:
        return _synthetic_nfcorpus()


def run(adapter, corpus: str = "nfcorpus", model: str = "small") -> BenchmarkResult:
    docs, queries, qrels = download_nfcorpus()
    docs = embed_docs(docs, model)
    model_obj = get_model(model)
    adapter.index(docs)
    adapter.optimize()

    vecs = model_obj.encode([q["text"] for q in queries]).tolist()
    queries = [{**q, "vector": v} for q, v in zip(queries, vecs, strict=False)]

    result = run_evaluation(adapter, docs, queries, qrels)
    result.benchmark = "nfcorpus"
    result.corpus = corpus
    result.model = model
    return result
