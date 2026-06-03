"""Per-benchmark bar charts styled to match Anthropic research page aesthetic."""

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from ..benchmarks.common import BenchmarkResult  # noqa: E402

# Warm outer bg matches page bg; white inner = floating card effect
_OUTER_BG = "#f5f3ef"
_PLOT_BG = "#ffffff"
_INK = "#1a1917"
_MUTED = "#6b6b6b"
_SPINE = "#d8d5d0"
_GRID = "#ebebeb"
_LABEL_CLR = "#7a7370"

# One colour per adapter (in typical sort order: chromadb, lancedb, qdrant, sqlite, tantivy)
_ADAPTER_COLORS = {
    "chromadb": "#4B7EBB",  # steel blue
    "lancedb": "#3D8C7A",  # teal
    "qdrant": "#8B5CF6",  # violet
    "sqlite": "#BDB9B5",  # neutral gray
    "tantivy": "#D4952A",  # amber
}
_FALLBACK_COLORS = ["#C96442", "#4B7EBB", "#3D8C7A", "#D4952A", "#BDB9B5"]

# Two-series colours for latency (p50 / p95)
_P50_COLOR = "#e8903a"
_P95_COLOR = "#7eb8d4"

_TICK_SIZE = 9
_LABEL_SIZE = 8
_TITLE_SIZE = 11.5


def _bar_color(store: str, idx: int) -> str:
    return _ADAPTER_COLORS.get(store.lower(), _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def _apply_style(fig, ax) -> None:
    fig.patch.set_facecolor(_OUTER_BG)
    ax.set_facecolor(_PLOT_BG)

    # Only left + bottom spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_SPINE)
    ax.spines["bottom"].set_color(_SPINE)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)

    ax.tick_params(axis="both", colors=_MUTED, labelsize=_TICK_SIZE, length=3, width=0.7)
    ax.xaxis.label.set_color(_MUTED)
    ax.xaxis.label.set_fontsize(_TICK_SIZE)
    ax.yaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_fontsize(_TICK_SIZE)

    ax.grid(axis="y", color=_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _b64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def ndcg_bar_chart(results: list[BenchmarkResult], benchmark: str) -> str:
    """Return base64 PNG of nDCG@10 per adapter."""
    from collections import defaultdict

    by_store: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.benchmark == benchmark:
            by_store[r.store].append(r.ndcg_at_10)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    _apply_style(fig, ax)

    if not by_store:
        ax.text(
            0.5,
            0.5,
            f"No results for {benchmark}",
            ha="center",
            va="center",
            color=_MUTED,
            fontsize=_TICK_SIZE,
            transform=ax.transAxes,
        )
        b64 = _b64_png(fig)
        plt.close(fig)
        return b64

    stores = list(by_store)
    means = [sum(v) / len(v) for v in by_store.values()]
    colors = [_bar_color(s, i) for i, s in enumerate(stores)]

    bars = ax.bar(stores, means, color=colors, width=0.5, zorder=3)
    ax.set_title("nDCG@10 by adapter", color=_INK, fontsize=_TITLE_SIZE, fontweight="bold", pad=10)
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    for bar, val in zip(bars, means, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=_LABEL_SIZE,
            color=_LABEL_CLR,
        )

    plt.tight_layout(pad=1.0)
    b64 = _b64_png(fig)
    plt.close(fig)
    return b64


def latency_bar_chart(results: list[BenchmarkResult], benchmark: str) -> str:
    """Return base64 PNG of p50/p95 latency per adapter."""
    from collections import defaultdict

    by_store: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for r in results:
        if r.benchmark == benchmark:
            by_store[r.store].append((r.latency_p50_ms, r.latency_p95_ms, r.latency_p99_ms))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    _apply_style(fig, ax)

    if not by_store:
        ax.text(
            0.5,
            0.5,
            f"No latency data for {benchmark}",
            ha="center",
            va="center",
            color=_MUTED,
            fontsize=_TICK_SIZE,
            transform=ax.transAxes,
        )
        b64 = _b64_png(fig)
        plt.close(fig)
        return b64

    stores = list(by_store)
    p50 = [sum(v[0] for v in vals) / len(vals) for vals in by_store.values()]
    p95 = [sum(v[1] for v in vals) / len(vals) for vals in by_store.values()]

    x = range(len(stores))
    w = 0.3
    ax.bar([i - w / 2 for i in x], p50, width=w, label="p50", color=_P50_COLOR, zorder=3)
    ax.bar([i + w / 2 for i in x], p95, width=w, label="p95", color=_P95_COLOR, zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(stores)
    ax.set_title("Query latency (ms)", color=_INK, fontsize=_TITLE_SIZE, fontweight="bold", pad=10)
    ax.set_ylabel("ms")
    ax.legend(fontsize=_LABEL_SIZE, framealpha=0, labelcolor=_MUTED, handlelength=1.0, borderpad=0.3, handletextpad=0.5)

    plt.tight_layout(pad=1.0)
    b64 = _b64_png(fig)
    plt.close(fig)
    return b64
