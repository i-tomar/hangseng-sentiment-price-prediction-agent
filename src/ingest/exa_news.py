"""Exa news search per ticker (exa-py)."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.ingest.normalize import make_id, to_utc

log = logging.getLogger(__name__)


def fetch_exa(api_key: str, ticker: dict, lookback_days: int, num_results: int = 10,
              max_characters: int = 1500, category: str | None = "news") -> list[dict]:
    """Return raw Exa results (as plain dicts) for one ticker."""
    from exa_py import Exa

    exa = Exa(api_key)
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    query = f"{ticker['name']} ({ticker['symbol']}) stock news"
    kwargs = dict(num_results=num_results, start_published_date=start)
    if category:
        kwargs["category"] = category
    try:  # exa-py >= 2: search() returns contents by default
        resp = exa.search(query, contents={"text": {"max_characters": max_characters}}, **kwargs)
    except TypeError:  # older exa-py
        resp = exa.search_and_contents(query, text={"max_characters": max_characters}, **kwargs)
    out = []
    for r in getattr(resp, "results", []) or []:
        out.append({
            "title": getattr(r, "title", None), "url": getattr(r, "url", None),
            "published_date": getattr(r, "published_date", None), "author": getattr(r, "author", None),
            "text": getattr(r, "text", None), "query": query,
        })
    log.info("Exa %s: %d results", ticker["symbol"], len(out))
    return out


def map_exa(items: list[dict], ticker: dict) -> list[dict]:
    rows = []
    for it in items:
        ts = to_utc(it.get("published_date"))
        text = f"{it.get('title') or ''}. {it.get('text') or ''}".strip()
        if ts is None or len(text) < 5:
            continue
        rows.append({
            "post_id": make_id("exa", it.get("url")), "source": "exa_news", "exchange": ticker["exchange"],
            "ticker": ticker["symbol"], "ts_utc": ts, "text": text[:2000], "likes": 0.0, "reposts": 0.0,
            "replies": 0.0, "views": 0.0, "author": it.get("author") or "", "author_followers": 0.0,
            "url": it.get("url") or "", "is_news": True, "query": it.get("query", ""),
        })
    return rows
