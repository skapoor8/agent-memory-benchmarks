# agent-memory-benchmarks

A reproducible benchmark harness that measures retrieval quality and speed across embedded vector stores and BM25 engines, on four agent-memory tasks: code finding, documentation search, episodic memory, and skill routing.

Results are published as a static report with Mermaid architecture diagrams, algorithm deep-dives, per-benchmark corpus visualizations, and a composite radar/heatmap comparison across all adapters.

## Quick start

```bash
mise run install
mise run bench -- --benchmark skill-search --store lancedb
mise run report:build
open dist/site/index.html
```

No Docker, no cloud credentials. The first run downloads corpora and an embedding model; subsequent runs are cached.

## What it benchmarks

| Task | Description | Corpus | Ground truth |
|------|-------------|--------|--------------|
| **code-finding** | Given a GitHub issue, retrieve the source files most likely to need modification | SWE-bench Lite — 300 Python issues, 215 repos | Gold-patch file paths |
| **doc-search** | Given a natural-language API question, retrieve the most relevant documentation section | FastAPI docs (~890 section chunks) | InPars-lite synthetic Q&A pairs |
| **episodic-memory** | Recover past architectural decisions from project ADRs | Neon ADRs — 247 records, 1 114 queries | ADR content |
| **skill-search** | Route a task description to the correct API or tool | Gorilla APIBench — 300 HuggingFace APIs, 30 queries | APIBench gold labels |

## Adapters

| Adapter | `--store` | Search type | Tier |
|---------|-----------|-------------|------|
| SQLite FTS5 | `sqlite` | BM25 keyword | 1 |
| LanceDB | `lancedb` | Dense vector (HNSW) | 1 |
| ChromaDB | `chromadb` | Dense vector | 1 |
| Tantivy | `tantivy` | BM25 keyword | 1 |
| Qdrant | `qdrant` | Dense vector (HNSW) | 1 |

All adapters run fully embedded — no infrastructure required. Qdrant runs `:memory:` by default; set `QDRANT_URL` to target a server.

Dense adapters use [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384 dims, 22M params) for both indexing and query encoding.

## Running benchmarks

```bash
# Single adapter + benchmark
mise run bench -- --benchmark episodic-memory --store lancedb

# All Tier 1 adapters, all benchmarks (small model)
mise run bench:fast

# One adapter across all benchmarks
mise run bench -- --benchmark all --store qdrant --model small
```

Results land in `results/<benchmark>/<store>/<timestamp>.json`:

```json
{
  "benchmark": "episodic-memory",
  "store": "lancedb",
  "ndcg_at_10": 0.640,
  "recall_at_1": 0.487,
  "recall_at_5": 0.719,
  "recall_at_10": 0.803,
  "mrr_at_10": 0.588,
  "latency_p50_ms": 3.6,
  "latency_p95_ms": 4.5,
  "num_queries": 1114,
  "num_docs": 1115
}
```

## Report

The report is a static site built from Jupyter notebooks using the bundled [JupyterPress](src/jupyterpress/) SSG. Notebooks execute at build time against live results.

```bash
mise run report:build    # execute notebooks + render to dist/site/
mise run report:serve    # build and serve at http://localhost:8000 with live reload
mise run report:publish  # deploy to GitHub Pages via ghp-import
mise run report:clean    # remove dist/site/
```

### Report pages

| Page | Description |
|------|-------------|
| **Overview** | Results summary and key findings |
| **Methodology** | Evaluation protocol, metrics (nDCG, Recall@k, MRR), BM25/HNSW/SBERT algorithm deep-dives with visualizations |
| **Code Finding** | SWE-bench task flow, corpus stats, background on why dense search leads |
| **Doc Search** | FastAPI chunking pipeline, InPars ground truth, BM25 vs dense tradeoff |
| **Episodic Memory** | ADR retrieval flow, Neon corpus topic distribution, why BM25 wins on domain vocabulary |
| **Skill Search** | Tool routing flow, semantic gap illustration, Gorilla/APIBench background |
| **Composite** | Radar charts (per-adapter profiles), cross-benchmark heatmap, speed-accuracy scatter |

### Site configuration (`jupyterpress.yaml`)

```yaml
title: "Agent Memory Benchmarks"
theme: anthropic          # anthropic | openai | grok | mistral | perplexity | cohere | reflection
author:
  name: "..."
  links:
    GitHub: "https://..."
execute: true             # re-execute notebooks at build time
source_dir: notebooks
output_dir: dist/site
```

## Development

```bash
mise run test             # pytest
mise run lint             # ruff check
mise run lint-fix         # ruff check --fix
mise run format           # ruff format
mise run format-check     # ruff format --check
```

## Project layout

```
notebooks/
├── index.ipynb               # overview page
├── methodology.ipynb         # metrics + algorithm deep-dives
├── composite-score.ipynb     # radar charts, heatmap, scatter
└── benchmarks/
    ├── code-finding.ipynb
    ├── doc-search.ipynb
    ├── episodic-memory.ipynb
    └── skill-search.ipynb

src/
├── agent_memory_benchmarks/
│   ├── adapters/             # sqlite, lancedb, chromadb, tantivy, qdrant
│   ├── benchmarks/           # code_finding, doc_search, episodic_memory, skill_search
│   ├── corpus/               # dataset loaders and corpus preparation
│   └── metrics.py            # nDCG, recall, MRR, latency
└── jupyterpress/             # static site generator (notebook → HTML)

results/                      # benchmark output JSON (gitignored if large)
data/                         # cached corpus data
```

## Contributing

Pull requests welcome — especially:

- New corpora with real (not synthetic) ground truth
- Additional adapters (ColBERT/RAGatouille, BGE-M3 hybrid, Weaviate, pgvector)
- Larger corpus variants (full SWE-bench, complete APIBench subsets)
- PDF retrieval benchmarks (BEIR NF-Corpus, arXiv, SEC filings)

Run `mise run test && mise run lint && mise run format-check` before opening a PR. Keep new adapters to one file and new benchmarks to one module.
