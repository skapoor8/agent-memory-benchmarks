"""Results writer/loader: serialize BenchmarkResult to results/<benchmark>/<store>/<ts>.json."""

import json
import time
from dataclasses import fields
from pathlib import Path

from .benchmarks.common import BenchmarkResult

RESULTS_DIR = Path("results")


def write_result(result: BenchmarkResult) -> Path:
    out_dir = RESULTS_DIR / result.benchmark / result.store
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = out_dir / f"{ts}.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"✓ Result written: {out_path}")
    return out_path


def load_all_results() -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    valid = {f.name for f in fields(BenchmarkResult)}
    for path in sorted(RESULTS_DIR.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
            results.append(BenchmarkResult(**{k: data[k] for k in data if k in valid}))
        except Exception:
            pass
    return results
