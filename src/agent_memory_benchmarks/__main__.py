"""Entry point: python -m agent_memory_benchmarks"""

import importlib
from pathlib import Path

import click

from .adapters.registry import TIER1, get_adapter
from .results import write_result

# CLI names use hyphens; module paths use underscores (standard Python convention).
BENCHMARKS = {
    "code-finding": "agent_memory_benchmarks.benchmarks.code_finding",
    "doc-search": "agent_memory_benchmarks.benchmarks.doc_search",
    "episodic-memory": "agent_memory_benchmarks.benchmarks.episodic_memory",
    "skill-search": "agent_memory_benchmarks.benchmarks.skill_search",
    "nfcorpus": "agent_memory_benchmarks.benchmarks.nfcorpus",
}


@click.group()
def main() -> None:
    """agent-memory-benchmarks CLI."""


@main.command()
@click.option(
    "--benchmark",
    "-b",
    required=True,
    type=click.Choice([*BENCHMARKS, "all"]),
    help="Benchmark to run.",
)
@click.option(
    "--store",
    "-s",
    required=True,
    type=click.Choice(["sqlite", "lancedb", "chromadb", "tantivy", "qdrant", "all"]),
    help="Adapter to use.",
)
@click.option("--corpus", "-c", default=None, help="Corpus name (benchmark-specific).")
@click.option(
    "--model",
    "-m",
    default="small",
    type=click.Choice(["small", "bge-m3"]),
    help="Embedding model.",
)
def bench(benchmark: str, store: str, corpus: str | None, model: str) -> None:
    """Run a benchmark."""
    benchmarks = list(BENCHMARKS) if benchmark == "all" else [benchmark]
    stores = TIER1 if store == "all" else [store]

    for bm in benchmarks:
        mod = importlib.import_module(BENCHMARKS[bm])
        for st in stores:
            click.echo(f"\n→ {bm} / {st} / {model}")
            adapter = get_adapter(st)
            kwargs: dict = {"model": model}
            if corpus:
                kwargs["corpus"] = corpus
            try:
                result = mod.run(adapter, **kwargs)
                write_result(result)
                click.echo(
                    f"  nDCG@10={result.ndcg_at_10:.3f}  "
                    f"R@1={result.recall_at_1:.3f}  "
                    f"p50={result.latency_p50_ms:.1f}ms"
                )
            except Exception as e:
                click.echo(f"  ✗ FAILED: {e}", err=True)
            finally:
                adapter.teardown()


@main.command()
@click.option("--out", default="results/report.html", help="Output HTML path.")
@click.option("--benchmark", default=None, help="Filter to a single benchmark.")
def report(out: str, benchmark: str | None) -> None:
    """Generate HTML report from results/."""
    from .report import generate_report

    generate_report(Path(out), filter_benchmark=benchmark)


if __name__ == "__main__":
    main()
