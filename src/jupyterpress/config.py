from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AuthorConfig:
    name: str = ""
    links: dict[str, str] = field(default_factory=dict)
    icons: dict[str, str] = field(default_factory=dict)  # platform → inline SVG or path


@dataclass
class SiteConfig:
    title: str = "Site"
    description: str = ""
    theme: str = "anthropic"
    author: AuthorConfig = field(default_factory=AuthorConfig)
    repo: str = ""
    source_dir: str = "."
    output_dir: str = "_site"
    execute: bool = False
    attribution: bool = True

    @classmethod
    def load(cls, path: Path) -> SiteConfig:
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text()) or {}
        author_raw = raw.pop("author", {}) or {}
        cfg = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        cfg.author = AuthorConfig(
            name=author_raw.get("name", ""),
            links=author_raw.get("links", {}),
            icons=author_raw.get("icons", {}),
        )
        return cfg

    def source_path(self, config_dir: Path) -> Path:
        return (config_dir / self.source_dir).resolve()

    def output_path(self, config_dir: Path) -> Path:
        return (config_dir / self.output_dir).resolve()
