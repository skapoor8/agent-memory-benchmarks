from pathlib import Path


def load_markdown_corpus(root: Path, extensions: tuple[str, ...] = (".md", ".mdx")) -> list[tuple[Path, str]]:
    """Return (path, text) pairs for all markdown files under root."""
    return [(p, p.read_text(errors="replace")) for p in root.rglob("*") if p.suffix in extensions]


def load_code_corpus(
    root: Path,
    extensions: tuple[str, ...] = (".py", ".ts", ".go", ".rs", ".zig", ".c", ".cpp"),
) -> list[tuple[Path, str]]:
    """Return (path, text) pairs for all source files under root."""
    return [(p, p.read_text(errors="replace")) for p in root.rglob("*") if p.suffix in extensions]


def load_literary_corpus(gutenberg_id: int, cache_dir: Path | None = None) -> str:
    """Download and strip a Gutenberg text by numeric ID."""
    import gutenbergpy.textget as tg

    raw = tg.get_text_by_id(gutenberg_id)
    return tg.strip_headers(raw).decode("utf-8", errors="replace")
