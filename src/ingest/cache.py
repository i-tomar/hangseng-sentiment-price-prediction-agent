from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from src.config import ROOT


def cached_json(name: str, run_date: str, fetch: Callable[[], Any], refresh: bool = False,
                base: Path | None = None) -> Any:
    """Return cached raw payload for (run_date, name) or call fetch() and cache it.

    Raw pulls are cached in data/raw/<run_date>/ so re-running the pipeline the same
    day never spends Apify/Exa credits twice.
    """
    base = base or ROOT / "data" / "raw"
    path = base / run_date / f"{name}.json"
    if path.exists() and not refresh:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=str)
    return data
