"""Live collection orchestrator: Apify social sources + Exa news, cached in data/raw/."""
from __future__ import annotations

import logging

import pandas as pd

from src.config import Secrets
from src.ingest import apify_sources as ap
from src.ingest.cache import cached_json
from src.ingest.exa_news import fetch_exa, map_exa
from src.ingest.normalize import to_frame

log = logging.getLogger(__name__)


def collect_live(cfg: dict, tickers: list[dict], secrets: Secrets, run_date: str,
                 refresh: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Returns (posts, source_log). Each source failure is logged, not fatal."""
    lookback = cfg["lookback_days"]
    rows: list[dict] = []
    source_log: list[dict] = []
    if not secrets.apify_token:
        raise SystemExit("APIFY_TOKEN missing: copy .env.example to .env or run with --demo")

    for ex_id, sources in cfg["sources"].items():
        ex_tickers = [t for t in tickers if t["exchange"] == ex_id]
        for s in sources:
            entry = {"exchange": ex_id, "source": s["id"], "actor": s["actor"], "raw_items": 0, "posts": 0, "error": ""}
            try:
                if s["kind"] == "lihkg_search":
                    kw_to_sym = {}
                    for t in ex_tickers:  # a few non-numeric keywords per ticker keeps billing low
                        for k in [k for k in t["keywords"] if not k.isdigit()][:s.get("keywords_per_ticker", 2)]:
                            kw_to_sym[k] = t["symbol"]
                    inp = ap.lihkg_input(list(kw_to_sym), lookback, s.get("max_items", 20))
                    items = cached_json(s["id"], run_date, lambda: ap.run_actor(
                        secrets.apify_token, s["actor"], inp, s.get("max_usd")), refresh)
                    mapped = ap.map_lihkg(items, ex_tickers, ex_id, kw_to_sym)
                elif s["kind"] == "telegram":
                    # One run per channel with its own spend cap: a single multi-channel run hits the
                    # cap on the first busy channel and silently skips the rest.
                    items, per_channel = [], {}
                    for ch in s["channels"]:
                        inp = ap.telegram_input([ch], lookback)
                        try:
                            got = cached_json(f"{s['id']}_{ch}", run_date, lambda: ap.run_actor(
                                secrets.apify_token, s["actor"], inp, s.get("max_usd_per_channel", s.get("max_usd"))), refresh)
                        except Exception as e:  # one dead channel must not kill the source
                            log.warning("Telegram channel %s failed: %s", ch, e)
                            got = []
                        per_channel[ch] = len(got)
                        items.extend(got)
                    entry["channels"] = per_channel
                    mapped = ap.map_telegram(items, ex_tickers, ex_id, s["id"])
                elif s["kind"] == "daum":
                    inp = ap.daum_input(s["sections"], s.get("max_items", 150))
                    items = cached_json(s["id"], run_date, lambda: ap.run_actor(
                        secrets.apify_token, s["actor"], inp, s.get("max_usd"), s.get("max_items")), refresh)
                    mapped = ap.map_daum(items, ex_tickers, ex_id)
                else:
                    raise ValueError(f"unknown source kind {s['kind']}")
                entry.update(raw_items=len(items), posts=len(mapped))
                rows.extend(mapped)
            except Exception as e:  # keep going: one broken actor must not kill the run
                log.exception("Source %s failed", s["id"])
                entry["error"] = str(e)[:300]
            source_log.append(entry)

    exa_cfg = cfg.get("exa", {})
    if exa_cfg.get("enabled", True):
        if not secrets.exa_api_key:
            log.warning("EXA_API_KEY missing; skipping Exa news")
            source_log.append({"exchange": "ALL", "source": "exa_news", "actor": "exa.ai", "raw_items": 0,
                               "posts": 0, "error": "EXA_API_KEY missing"})
        else:
            for t in tickers:
                entry = {"exchange": t["exchange"], "source": f"exa_news:{t['symbol']}", "actor": "exa.ai",
                         "raw_items": 0, "posts": 0, "error": ""}
                try:
                    items = cached_json(f"exa_{t['symbol']}", run_date, lambda: fetch_exa(
                        secrets.exa_api_key, t, lookback, exa_cfg.get("num_results", 10),
                        exa_cfg.get("max_characters", 1500), exa_cfg.get("category")), refresh)
                    mapped = map_exa(items, t)
                    entry.update(raw_items=len(items), posts=len(mapped))
                    rows.extend(mapped)
                except Exception as e:
                    log.exception("Exa failed for %s", t["symbol"])
                    entry["error"] = str(e)[:300]
                source_log.append(entry)

    posts = to_frame(rows)
    if not posts.empty:  # enforce the lookback window regardless of what actors returned
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback)
        posts = posts[posts.ts_utc >= cutoff].reset_index(drop=True)
    return posts, source_log
