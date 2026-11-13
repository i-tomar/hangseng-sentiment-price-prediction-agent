"""Config + environment loading. Secrets come only from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # normalise keywords to strings (YAML may parse 00700 as int)
    for ex in cfg["exchanges"].values():
        for t in ex["tickers"]:
            t["keywords"] = [str(k) for k in t.get("keywords", [])]
    return cfg


@dataclass
class Secrets:
    apify_token: str | None
    exa_api_key: str | None


def load_secrets() -> Secrets:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:  # pragma: no cover
        pass
    return Secrets(
        apify_token=os.getenv("APIFY_TOKEN") or None,
        exa_api_key=os.getenv("EXA_API_KEY") or None,
    )


def ticker_table(cfg: dict) -> list[dict]:
    """Flat list of {exchange, symbol, name, keywords, timezone, session_close}."""
    rows = []
    for ex_id, ex in cfg["exchanges"].items():
        for t in ex["tickers"]:
            rows.append(
                {
                    "exchange": ex_id,
                    "symbol": t["symbol"],
                    "name": t["name"],
                    "keywords": t["keywords"],
                    "timezone": ex["timezone"],
                    "session_close": ex["session_close"],
                    "currency": ex.get("currency", ""),
                }
            )
    return rows
