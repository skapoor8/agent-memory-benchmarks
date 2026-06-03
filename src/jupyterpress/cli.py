from __future__ import annotations

import http.server
import os
import threading
import time
from pathlib import Path

import click

from .config import SiteConfig


@click.group()
def main() -> None:
    """jupyterpress — Jupyter to static site."""


@main.command()
@click.option("--config", default="jupyterpress.yaml", help="Config file path.")
@click.option("--out", default=None)
def build(config: str, out: str | None) -> None:
    """Build the static site."""
    cfg_path = Path(config)
    cfg = SiteConfig.load(cfg_path)
    if out:
        cfg.output_dir = out
    from .generator import build_site

    build_site(cfg, cfg_path.parent)
    click.echo(f"✓ Built → {cfg.output_path(cfg_path.parent)}")


@main.command()
@click.option("--config", default="jupyterpress.yaml")
@click.option("--port", default=8000)
def serve(config: str, port: int) -> None:
    """Build and serve with live polling."""
    cfg_path = Path(config)
    cfg = SiteConfig.load(cfg_path)
    out = cfg.output_path(cfg_path.parent)

    def _build() -> None:
        from .generator import build_site

        try:
            build_site(cfg, cfg_path.parent)
        except Exception as e:
            click.echo(f"Build error: {e}", err=True)

    _build()
    os.chdir(out)
    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.HTTPServer(("", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    click.echo(f"→ Serving at http://localhost:{port}  (Ctrl+C to stop)")

    src = cfg.source_path(cfg_path.parent)
    last_mtime: dict[str, float] = {}
    try:
        while True:
            changed = False
            for p in src.rglob("*.ipynb"):
                m = p.stat().st_mtime
                if last_mtime.get(str(p)) != m:
                    last_mtime[str(p)] = m
                    changed = True
            if changed:
                click.echo("↻ Change detected — rebuilding...")
                _build()
            time.sleep(2)
    except KeyboardInterrupt:
        server.shutdown()


@main.command()
@click.option("--config", default="jupyterpress.yaml")
@click.option("--branch", default="gh-pages")
def publish(config: str, branch: str) -> None:
    """Publish output_dir to GitHub Pages via ghp-import."""
    import subprocess

    cfg_path = Path(config)
    cfg = SiteConfig.load(cfg_path)
    out = cfg.output_path(cfg_path.parent)
    result = subprocess.run(
        ["ghp-import", "-n", "-p", "-f", str(out), "-b", branch],
        check=False,
    )
    if result.returncode == 0:
        click.echo(f"✓ Published {out} → {branch}")
    else:
        raise click.ClickException("ghp-import failed")
