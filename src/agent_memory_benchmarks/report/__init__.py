"""Report generation: charts + single-file HTML summary."""

from pathlib import Path

from ..results import load_all_results
from .charts import latency_bar_chart, ndcg_bar_chart

BENCHMARKS = ["code-finding", "doc-search", "episodic-memory", "skill-search"]


def generate_report(
    out_path: Path = Path("results/report.html"),
    filter_benchmark: str | None = None,
) -> Path:
    results = load_all_results()
    if filter_benchmark:
        results = [r for r in results if r.benchmark == filter_benchmark]

    benchmarks_in_results = sorted({r.benchmark for r in results})
    sections = []
    for bm in benchmarks_in_results:
        bm_results = [r for r in results if r.benchmark == bm]
        ndcg_img = ndcg_bar_chart(bm_results, bm)
        lat_img = latency_bar_chart(bm_results, bm)

        rows = ""
        for r in sorted(bm_results, key=lambda x: -x.ndcg_at_10):
            rows += f"""<tr>
                <td>{r.store}</td><td>{r.model}</td><td>{r.corpus}</td>
                <td>{r.ndcg_at_10:.3f}</td><td>{r.recall_at_1:.3f}</td>
                <td>{r.recall_at_5:.3f}</td><td>{r.recall_at_10:.3f}</td>
                <td>{r.latency_p50_ms:.1f}</td><td>{r.latency_p95_ms:.1f}</td>
                <td>{r.latency_p99_ms:.1f}</td>
            </tr>"""

        sections.append(f"""
        <section>
          <h2>{bm}</h2>
          <img src="data:image/png;base64,{ndcg_img}" alt="nDCG chart">
          <img src="data:image/png;base64,{lat_img}" alt="latency chart">
          <table border="1" cellpadding="4" style="border-collapse:collapse;margin-top:1em">
            <tr><th>Store</th><th>Model</th><th>Corpus</th>
                <th>nDCG@10</th><th>R@1</th><th>R@5</th><th>R@10</th>
                <th>p50ms</th><th>p95ms</th><th>p99ms</th></tr>
            {rows}
          </table>
        </section>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>agent-memory-benchmarks report</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:2em auto}}
img{{max-width:48%;margin-right:2%}}section{{margin-bottom:3em}}</style>
</head><body>
<h1>agent-memory-benchmarks</h1>
<p>{len(results)} results across {len(benchmarks_in_results)} benchmarks</p>
{"".join(sections)}
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"✓ Report written: {out_path}")
    return out_path
