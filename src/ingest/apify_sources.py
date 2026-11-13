"""Apify actor runners + output mappers (LIHKG, Telegram, Daum News).

Input schemas were checked against each actor's published build input schema
(api.apify.com/v2/acts/<user>~<name>, Nov 2026):
  * claude_code_reviewer/lihkg-scraper-en: mode, keywords, maxItems, order, dateFrom, aiLabels
  * tri_angle/telegram-scraper: profiles, collectMessages, scrapeLastNDays, oldestMessageDate
  * ardent_fork/daum-news: sections, maxItems, fetchArticles, maxConcurrency
Output mapping uses field fallbacks because community actors change shape over time.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from src.ingest.normalize import first, make_id, match_tickers, to_num, to_utc

log = logging.getLogger(__name__)


def run_actor(token: str, actor_id: str, run_input: dict, max_usd: float | None = None,
              max_items: int | None = None) -> list[dict]:
    """Run an Apify actor synchronously and return its dataset items."""
    from apify_client import ApifyClient

    client = ApifyClient(token)
    kwargs: dict[str, Any] = {"run_input": run_input}
    if max_usd:
        kwargs["max_total_charge_usd"] = Decimal(str(max_usd))  # hard credit cap (pay-per-event actors)
    if max_items:
        kwargs["max_items"] = max_items
    log.info("Running Apify actor %s with input %s", actor_id, run_input)
    try:
        run = client.actor(actor_id).call(**kwargs)
    except TypeError:  # older apify-client without max_total_charge_usd
        kwargs.pop("max_total_charge_usd", None)
        run = client.actor(actor_id).call(**kwargs)
    if run is None:
        raise RuntimeError(f"Actor {actor_id} returned no run")
    ds_id = run.get("defaultDatasetId") if isinstance(run, dict) else getattr(run, "default_dataset_id", None)
    status = run.get("status") if isinstance(run, dict) else getattr(run, "status", None)
    log.info("Actor %s finished with status %s", actor_id, status)
    return list(client.dataset(ds_id).iterate_items())


# ---------------------------------------------------------------- input builders
def lihkg_input(keywords: list[str], lookback_days: int, max_items: int) -> dict:
    return {
        "mode": "search",
        "keywords": keywords,
        "order": "desc_create_time",
        "dateFrom": f"{lookback_days}d",
        "maxItems": max_items,
        "aiLabels": False,          # we score sentiment ourselves; avoids extra billing
        "onlyNew": False,
    }


def telegram_input(channels: list[str], lookback_days: int) -> dict:
    return {"profiles": channels, "collectMessages": True, "scrapeLastNDays": lookback_days}


def daum_input(sections: list[str], max_items: int) -> dict:
    return {"sections": sections, "maxItems": max_items, "fetchArticles": True, "maxConcurrency": 5}


# ---------------------------------------------------------------- output mappers
def map_lihkg(items: list[dict], tickers: list[dict], exchange: str, kw_to_symbol: dict[str, str]) -> list[dict]:
    rows = []
    for it in items:
        text = " ".join(str(x) for x in [first(it, "title"), first(it, "own_text", "content", "text", "msg")] if x)
        ts = to_utc(first(it, "created_at", "create_time", "createdAt", "reply_time", "last_reply_at"))
        if ts is None or not text.strip():
            continue
        query = first(it, "source_query", "keyword", "query", default="")
        syms = set(match_tickers(text, tickers))
        if query in kw_to_symbol:
            syms.add(kw_to_symbol[query])
        for sym in syms:
            rows.append({
                "post_id": make_id("lihkg", first(it, "post_id", "id", "thread_id"), text[:80]),
                "source": "lihkg", "exchange": exchange, "ticker": sym, "ts_utc": ts,
                "text": text,
                "likes": to_num(first(it, "like_count", "likes", "likeCount")),
                "reposts": 0.0,
                "replies": to_num(first(it, "reply_count", "replies", "no_of_reply")),
                "views": to_num(first(it, "view_count", "views")),
                "author": first(it, "author", "user_nickname", "user.nickname", default=""),
                "author_followers": 0.0,     # LIHKG has no follower counts
                "url": first(it, "thread_url", "url", default=""),
                "is_news": False, "query": query,
            })
    return rows


def _reactions(msg: dict) -> float:
    r = msg.get("reactions")
    if isinstance(r, list):
        return sum(to_num(x.get("count")) for x in r if isinstance(x, dict))
    if isinstance(r, dict):
        return sum(to_num(v) for v in r.values())
    return to_num(first(msg, "reactions", "likes", "reactionsCount"))


def map_telegram(items: list[dict], tickers: list[dict], exchange: str, source_id: str) -> list[dict]:
    rows = []
    followers_by_channel: dict[str, float] = {}
    for it in items:  # first pass: profile records carry follower counts
        ch = first(it, "username", "channel", "profile", default="")
        f = to_num(first(it, "followers", "subscribers", "members", "participants"))
        if ch and f:
            followers_by_channel[str(ch).lstrip("@").lower()] = f
    for it in items:
        msg = it.get("message") if isinstance(it.get("message"), dict) else it
        text = first(msg, "description", "text", "message", "content", default="")
        if not isinstance(text, str) or not text.strip():
            continue
        preview = first(msg, "preview.text", default="")
        full_text = f"{text} {preview}".strip() if isinstance(preview, str) else text
        ts = to_utc(first(msg, "fulldate", "date_utc", "datetime", "timestamp", "date"))
        if ts is None:
            continue
        ch = str(first(it, "username", "channel", "profile", default="")).lstrip("@")
        syms = match_tickers(full_text, tickers)
        for sym in syms:
            rows.append({
                "post_id": make_id("tg", first(msg, "link", "id", default=""), text[:80]),
                "source": source_id, "exchange": exchange, "ticker": sym, "ts_utc": ts,
                "text": full_text,
                "likes": _reactions(msg),
                "reposts": to_num(first(msg, "forwards", "shares")),
                "replies": to_num(first(msg, "replies", "comments", "replyCount")),
                "views": to_num(first(msg, "views", "viewCount")),
                "author": ch,
                "author_followers": to_num(first(it, "followers", "subscribers")) or followers_by_channel.get(ch.lower(), 0.0),
                "url": first(msg, "link", "url", default=""),
                "is_news": False, "query": "",
            })
    return rows


def map_daum(items: list[dict], tickers: list[dict], exchange: str) -> list[dict]:
    rows = []
    for it in items:
        title = first(it, "title", "listTitle", default="")
        body = first(it, "body", "description", "autoSummary", default="")
        text = f"{title}. {body}".strip()
        ts = to_utc(first(it, "publishedAt", "published_at", "date", "modifiedAt"))
        if ts is None or not text:
            continue
        for sym in match_tickers(text, tickers):
            rows.append({
                "post_id": make_id("daum", first(it, "id", "url")),
                "source": "daum_news", "exchange": exchange, "ticker": sym, "ts_utc": ts,
                "text": text[:2000], "likes": 0.0, "reposts": 0.0, "replies": 0.0, "views": 0.0,
                "author": first(it, "publisher", "reporter", default=""), "author_followers": 0.0,
                "url": first(it, "url", default=""), "is_news": False, "query": "",
            })
    return rows
