"""Static site generator: renders .ipynb and .qmd sources → HTML."""

from __future__ import annotations

import contextlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import markdown
import yaml
from jinja2 import Environment, FileSystemLoader

from .config import SiteConfig

TEMPLATES_DIR = Path(__file__).parent / "templates"

_MD_EXTENSIONS = [
    "tables",
    "fenced_code",
    "codehilite",
    "attr_list",
    "def_list",
    "footnotes",
    "toc",
]

_MD_CONFIG = {
    "codehilite": {"guess_lang": False, "css_class": "highlight"},
    "toc": {"permalink": False},
}


# ─── Markdown helpers ──────────────────────────────────────────────────────


def _preprocess_math(text: str) -> str:
    """Wrap $$...$$ display-math blocks so MathJax can process them."""
    # Display math: $$...$$ (possibly multi-line)
    text = re.sub(r"\$\$(.+?)\$\$", r'<div class="math-block">\\[\1\\]</div>', text, flags=re.DOTALL)
    # Inline math: $...$ (single line, not already processed)
    text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", r"\\(\1\\)", text)
    return text


def _preprocess_citations(text: str) -> str:
    """Replace [@key] with superscript markers that link to references."""
    return re.sub(r"\[@([^\]]+)\]", r'<sup class="cite"><a href="../references.html#\1">[\1]</a></sup>', text)


_MERMAID_FENCE_RE = re.compile(
    r"^```mermaid\s*\n(.+?)^```",
    re.DOTALL | re.MULTILINE,
)

_MERMAID_HTML_RE = re.compile(
    r'<pre><code class="(?:[^"]*\s)?language-mermaid(?:\s[^"]*)?">(.+?)</code></pre>',
    re.DOTALL,
)


def _preprocess_mermaid_src(text: str) -> str:
    """Replace ```mermaid fenced blocks with a raw HTML div before markdown rendering."""

    def _replace(m: re.Match) -> str:
        src = m.group(1).strip()
        return f'<div class="mermaid">{src}</div>\n'

    return _MERMAID_FENCE_RE.sub(_replace, text)


def _postprocess_mermaid(html: str) -> str:
    """Convert any remaining fenced mermaid code blocks to mermaid.js-compatible divs."""
    import html as html_lib

    def _replace(m: re.Match) -> str:
        src = html_lib.unescape(m.group(1).strip())
        return f'<div class="mermaid">{src}</div>'

    return _MERMAID_HTML_RE.sub(_replace, html)


def _render_md(text: str) -> str:
    text = _preprocess_mermaid_src(text)
    text = _preprocess_math(text)
    text = _preprocess_citations(text)
    md = markdown.Markdown(extensions=_MD_EXTENSIONS, extension_configs=_MD_CONFIG)
    return md.convert(text)


# ─── TOC extraction ────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)


def _extract_toc(html: str) -> list[dict[str, str]]:
    """Return [{id, text, level}] from rendered HTML h2/h3 tags."""
    items = []
    for m in re.finditer(r'<h([23])[^>]*id="([^"]+)"[^>]*>(.+?)</h\1>', html, re.DOTALL):
        level, hid, text = m.group(1), m.group(2), m.group(3)
        text = re.sub(r"<[^>]+>", "", text).strip()
        items.append({"level": level, "id": hid, "text": text})
    return items


def _add_heading_ids(html: str) -> str:
    """Ensure h2/h3 tags have id attributes (markdown toc extension does this)."""

    def _slugify(text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        return re.sub(r"[\s_]+", "-", text).strip("-")

    def _add_id(m: re.Match) -> str:
        tag, content = m.group(1), m.group(2)
        if 'id="' in tag:
            return m.group(0)
        hid = _slugify(content)
        return f'<h{tag} id="{hid}">{content}</h{tag}>'

    return re.sub(r"<h([23])>(.*?)</h\1>", _add_id, html, flags=re.DOTALL)


# ─── Notebook execution ────────────────────────────────────────────────────


def _execute_notebook(nb_path: Path) -> dict[str, Any]:
    """Execute a notebook in-place and return the executed notebook dict."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(str(nb_path), as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    client.execute()
    return json.loads(nbformat.writes(nb))


# ─── Notebook processing ───────────────────────────────────────────────────


def _cell_source(cell: dict[str, Any]) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _render_notebook(nb: dict[str, Any], depth: str) -> tuple[dict[str, str], str]:
    """Return (front_matter, rendered_body_html)."""
    front: dict[str, str] = {}
    parts: list[str] = []

    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type", "")
        src = _cell_source(cell)

        if ctype == "raw":
            stripped = src.strip()
            if stripped.startswith("---"):
                block = stripped.strip("-").strip()
                with contextlib.suppress(Exception):
                    front = yaml.safe_load(block) or {}
            continue

        if ctype == "markdown":
            # Strip leading `# Title` (rendered via page-hero)
            lines = src.splitlines()
            if lines and lines[0].startswith("# "):
                src = "\n".join(lines[1:]).lstrip()
            if src.strip():
                # Adjust relative links: benchmarks/x → x (within same dir level)
                src = src.replace("](benchmarks/", f"]({depth}benchmarks/")
                parts.append(_render_md(src))
            continue

        if ctype == "code":
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                # Prefer SVG (vector) → PNG → HTML → plain text
                if "image/svg+xml" in data:
                    svg = data["image/svg+xml"]
                    if isinstance(svg, list):
                        svg = "".join(svg)
                    # Stored as base64 string or raw SVG markup
                    if svg.lstrip().startswith("<"):
                        parts.append(f'<div class="cell-output">{svg}</div>')
                    else:
                        parts.append(
                            f'<div class="cell-output"><img src="data:image/svg+xml;base64,{svg}" alt="chart"/></div>'
                        )
                elif "image/png" in data:
                    img_b64 = data["image/png"]
                    if isinstance(img_b64, list):
                        img_b64 = "".join(img_b64)
                    parts.append(
                        f'<div class="cell-output"><img src="data:image/png;base64,{img_b64}" alt="chart"/></div>'
                    )
                elif "text/html" in data:
                    html_out = data["text/html"]
                    if isinstance(html_out, list):
                        html_out = "".join(html_out)
                    parts.append(f'<div class="cell-output">{html_out}</div>')
                elif "text/plain" in data:
                    text_out = data["text/plain"]
                    if isinstance(text_out, list):
                        text_out = "".join(text_out)
                    parts.append(f'<div class="cell-output"><pre><code>{text_out}</code></pre></div>')

    body = "\n".join(parts)
    body = _postprocess_mermaid(body)
    return front, body


# ─── QMD / Markdown processing ────────────────────────────────────────────


def _render_qmd(text: str, depth: str) -> tuple[dict[str, str], str]:
    """Parse YAML front-matter + render markdown body."""
    front: dict[str, str] = {}

    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            block = text[3:end].strip()
            with contextlib.suppress(Exception):
                front = yaml.safe_load(block) or {}
            text = text[end + 4 :].lstrip()

    # Strip {#refs} directive (Quarto-specific)
    text = re.sub(r"\{#refs\}\s*", "", text)
    # Strip ::: divs (Quarto-specific)
    text = re.sub(r"^:::\s*.*$", "", text, flags=re.MULTILINE)

    # Fix citation links depth
    def _fix_cite(m: re.Match) -> str:
        key = m.group(1)
        return f'<sup class="cite"><a href="{depth}references.html#{key}">[{key}]</a></sup>'

    body = _preprocess_math(text)
    body = re.sub(r"\[@([^\]]+)\]", _fix_cite, body)
    md = markdown.Markdown(extensions=_MD_EXTENSIONS, extension_configs=_MD_CONFIG)
    rendered = md.convert(body)
    rendered = _postprocess_mermaid(rendered)
    return front, rendered


# ─── Page builder ──────────────────────────────────────────────────────────


def _build_page(
    env: Environment,
    body_html: str,
    front: dict[str, str],
    active: str,
    depth: str,
    out_path: Path,
    cfg: SiteConfig | None = None,
    logo_svg: str = "",
    author_icons: dict[str, str] | None = None,
) -> None:
    body_html = _add_heading_ids(body_html)
    toc = _extract_toc(body_html)

    tmpl = env.get_template("base.html")
    html = tmpl.render(
        title=front.get("title", out_path.stem.replace("-", " ").title()),
        subtitle=front.get("description", front.get("subtitle", "")),
        site_title=cfg.title if cfg else "Site",
        site_description=cfg.description if cfg else "",
        body=body_html,
        toc=toc if len(toc) >= 2 else [],
        active=active,
        depth=depth,
        author=cfg.author if cfg else None,
        repo=cfg.repo if cfg else "",
        theme=cfg.theme if cfg else "anthropic",
        logo_svg=logo_svg,
        attribution=cfg.attribution if cfg else True,
        author_icons=author_icons or {},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


# ─── Public entry point ────────────────────────────────────────────────────

_ACTIVE_MAP = {
    "index": "overview",
    "methodology": "methodology",
    "references": "references",
    "composite-score": "composite",
}


def build_site(
    cfg: SiteConfig | None = None,
    config_dir: Path | None = None,
    report_dir: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Generate the full static site from report/ sources."""
    if cfg is not None:
        report_dir = cfg.source_path(config_dir or Path("."))
        out_dir = cfg.output_path(config_dir or Path("."))
    elif report_dir is None or out_dir is None:
        raise ValueError("Either cfg or both report_dir and out_dir must be provided")

    out_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)

    # Copy static assets — CSS from active theme
    from .themes import THEMES_DIR, load_theme_css

    theme_name = cfg.theme if cfg else "anthropic"
    theme_css = load_theme_css(theme_name)
    (out_dir / "style.css").write_text(theme_css)

    # Copy all theme CSS files to _site/themes/
    themes_out = out_dir / "themes"
    themes_out.mkdir(exist_ok=True)
    for css_file in THEMES_DIR.glob("*.css"):
        shutil.copy(css_file, themes_out / css_file.name)

    logo_src = report_dir / "logo.svg"
    logo_svg = logo_src.read_text(encoding="utf-8") if logo_src.exists() else ""

    # Resolve author icons: inline SVG strings pass through; file paths are read
    resolved_icons: dict[str, str] = {}
    if cfg and cfg.author.icons:
        for platform, val in cfg.author.icons.items():
            val = val.strip()
            if val.startswith("<"):
                resolved_icons[platform] = val
            else:
                icon_path = (config_dir or Path(".")) / val
                if icon_path.exists():
                    resolved_icons[platform] = icon_path.read_text(encoding="utf-8")

    # (src_path, out_stem, active_nav, url_depth)
    pages: list[tuple[Path, str, str, str]] = []

    seen_stems: set[str] = set()

    # Root-level notebooks and qmd
    for src in sorted(report_dir.glob("*.ipynb")) + sorted(report_dir.glob("*.qmd")):
        if src.stem.startswith("_"):
            continue
        stem = "index" if src.stem in ("index", "intro") else src.stem
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        active = _ACTIVE_MAP.get(stem, "overview")
        pages.append((src, stem, active, ""))

    # benchmarks/ subdirectory
    for src in sorted((report_dir / "benchmarks").glob("*.ipynb")):
        pages.append((src, src.stem, "benchmarks", "../"))

    execute = cfg.execute if cfg else False

    for src, stem, active, url_depth in pages:
        try:
            if src.suffix == ".ipynb":
                if execute:
                    print(f"  ⚡ executing {src.name} …")
                    nb = _execute_notebook(src)
                else:
                    nb = json.loads(src.read_text(encoding="utf-8"))
                front, body = _render_notebook(nb, url_depth)
            else:
                front, body = _render_qmd(src.read_text(encoding="utf-8"), url_depth)

            if url_depth:
                # e.g. url_depth="../" → subdir is "benchmarks"
                subdir = src.parent.name
                out_path = out_dir / subdir / f"{stem}.html"
            else:
                out_path = out_dir / f"{stem}.html"

            _build_page(env, body, front, active, url_depth, out_path, cfg, logo_svg, resolved_icons)
            print(f"  ✓ {out_path.relative_to(out_dir)}")
        except Exception as exc:
            print(f"  ✗ {src.name}: {exc}")

    print(f"\n→ Site written to {out_dir}")
    return out_dir
