from __future__ import annotations

from pathlib import Path

THEMES_DIR = Path(__file__).parent

THEME_REGISTRY: dict[str, str] = {
    "anthropic": "anthropic.css",
    "openai": "openai.css",
    "grok": "grok.css",
    "mistral": "mistral.css",
    "perplexity": "perplexity.css",
    "cohere": "cohere.css",
    "reflection": "reflection.css",
}


def load_theme_css(name: str) -> str:
    """Return CSS content for named theme. Falls back to anthropic if unknown."""
    filename = THEME_REGISTRY.get(name, "anthropic.css")
    path = THEMES_DIR / filename
    if not path.exists():
        path = THEMES_DIR / "anthropic.css"
    return path.read_text() if path.exists() else ""
