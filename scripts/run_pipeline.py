#!/usr/bin/env python
"""End-to-end pipeline: ingest -> sentiment -> align -> features -> walk-forward -> n+1 forecast -> HTML report.

  python scripts/run_pipeline.py --demo            # no keys needed, synthetic posts (clearly labelled)
  python scripts/run_pipeline.py --live            # Apify + Exa using APIFY_TOKEN / EXA_API_KEY from .env
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, load_config, load_secrets, ticker_table  # noqa: E402
from src.features.alignment import align_posts, daily_sentiment  # noqa: E402
from src.features.build import build_panel, ticker_snapshot  # noqa: E402
from src.ingest.demo import demo_posts  # noqa: E402
from src.ingest.prices import get_prices  # noqa: E402
from src.models.check import check_predictions, summarise  # noqa: E402
from src.models.walkforward import forecast_next, metrics, model_frame, walk_forward  # noqa: E402
from src.nlp.sentiment import SentimentScorer, score_posts  # noqa: E402
from src.report.html import build_report  # noqa: E402

log = logging.getLogger("pipeline")
pd.set_option("display.float_format", "{:,.4f}".format)
pd.set_option("display.width", 200)


def drop_incomplete_bars(bars: pd.DataFrame, tickers: list[dict], now_utc: pd.Timestamp) -> pd.DataFrame:
    """Remove today's bar if that exchange's session has not closed yet (partial bar = leakage/noise)."""
    keep = []
    for t in tickers:
        b = bars[bars.ticker == t["symbol"]]
        local = now_utc.tz_convert(t["timezone"])
        hh, mm = map(int, t["session_close"].split(":"))
        today = local.tz_localize(None).normalize()
        if local.hour * 60 + local.minute < hh * 60 + mm + 15:  # 15 min grace for the closing auction
            b = b[b.date < today]
        else:
            b = b[b.date <= today]
        keep.append(b)
    return pd.concat(keep, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--live", action="store_true", help="real data via Apify + Exa (needs .env)")
    g.add_argument("--demo", action="store_true", help="synthetic social data, runs without keys")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--refresh", action="store_true", help="ignore today's cache in data/raw and re-pull (spends credits)")
    ap.add_argument("--offline", action="store_true", help="synthetic prices too (no network at all)")
    ap.add_argument("--no-transformers", action="store_true", help="force the lexicon sentiment fallback")
    ap.add_argument("--report", default=None, help="output HTML path (default reports/<mode>_report.html)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    mode = "live" if args.live else "demo"
    cfg = load_config(args.config)
    tickers = ticker_table(cfg)
    now = pd.Timestamp.now(tz="UTC")
    run_date = now.strftime("%Y-%m-%d")

    # 1. prices
    bars, price_source = get_prices(tickers, cfg["price_history_days"], offline=args.offline)
    if price_source == "yfinance":
        bars = drop_incomplete_bars(bars, tickers, now)
    log.info("Prices: %d bars from %s", len(bars), price_source)

    # 2. posts
    source_log: list[dict] = []
    if mode == "demo":
        posts = demo_posts(tickers, bars, cfg["lookback_days"], now)
    else:
        from src.ingest.collect import collect_live

        posts, source_log = collect_live(cfg, tickers, load_secrets(), run_date, refresh=args.refresh)
    log.info("Posts/articles mapped to tickers: %d", len(posts))

    # 3. sentiment
    sc = cfg["sentiment"]
    scorer = SentimentScorer(sc["model"], False if args.no_transformers else sc.get("use_transformers", "auto"),
                             sc.get("max_chars", 512))
    log.info("Sentiment backend: %s", scorer.backend)
    posts = score_posts(posts, scorer)

    # 4. align to bars + aggregates
    aligned = align_posts(posts, bars, tickers)
    daily = daily_sentiment(aligned)
    snapshot = ticker_snapshot(bars, aligned, tickers, cfg)

    # 5. features + walk-forward + forecast
    lv = cfg["levels"]
    panel = build_panel(bars, daily, lv["swing_window"], lv["lookback_bars"])
    social_start = aligned["bar_date"].min() if not aligned.empty and aligned["bar_date"].notna().any() else None
    mf = model_frame(panel, social_start)
    m = cfg["model"]
    oos = walk_forward(mf, m["min_train_dates"], m["logreg_C"], m["ridge_alpha"])
    met = metrics(oos)
    fc = forecast_next(mf, m["logreg_C"], m["ridge_alpha"])

    # 6. persist
    out_dir = ROOT / "data" / "processed" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%MZ")
    aligned.to_csv(out_dir / "posts_scored.csv", index=False)
    daily.to_csv(out_dir / "daily_sentiment.csv", index=False)
    snapshot.to_csv(out_dir / "ticker_snapshot.csv", index=False)
    oos.to_csv(out_dir / "walkforward_predictions.csv", index=False)
    met.to_csv(out_dir / "metrics.csv", index=False)
    bars.to_csv(out_dir / "bars.csv", index=False)
    fc.to_csv(out_dir / "forecast.csv", index=False)
    pred_dir = ROOT / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    log_path = pred_dir / f"{mode}_predictions_log.csv"
    if not fc.empty:
        fc = fc.assign(run_utc=stamp, mode=mode, price_source=price_source)
        fc.to_csv(pred_dir / f"{mode}_predictions_{stamp}.csv", index=False)
        prev = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
        pd.concat([prev, fc], ignore_index=True).to_csv(log_path, index=False)

    # 7. check earlier predictions against the bars we just fetched
    checks = pd.DataFrame()
    if log_path.exists() and price_source == "yfinance":
        hist = pd.read_csv(log_path)
        hist = hist[hist["price_source"] == "yfinance"] if "price_source" in hist else hist
        hist = hist[hist["run_utc"] != stamp] if "run_utc" in hist else hist
        if not hist.empty:
            checks = check_predictions(hist, bars)
            checks = checks[checks.status == "checked"]
            log.info("Checked past predictions: %s", summarise(checks))

    # 8. report
    report = Path(args.report) if args.report else ROOT / "reports" / f"{mode}_report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    build_report(report, mode=mode, cfg=cfg, snapshot=snapshot, bars=bars, daily=daily, oos=oos, metrics=met,
                 forecast=fc, checks=checks, source_log=source_log,
                 meta={"price_source": price_source, "sentiment_backend": scorer.backend, "n_posts": len(posts)})
    checks.to_csv(out_dir / "checks.csv", index=False)
    meta = {"mode": mode, "run_utc": now.isoformat(), "price_source": price_source, "sentiment_backend": scorer.backend,
            "n_posts": int(len(posts)), "n_bars": int(len(bars)), "lookback_days": cfg["lookback_days"],
            "source_log": source_log, "report": str(report.relative_to(ROOT)) if report.is_relative_to(ROOT) else str(report)}
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    print("\n=== Ticker snapshot ===")
    print(snapshot[["exchange", "ticker", "name", "current_price", "mentions", "sentiment", "direction", "position",
                    "support", "resistance"]].to_string(index=False))
    if not met.empty:
        print("\n=== Walk-forward OOS metrics ===")
        print(met.to_string(index=False))
    if not fc.empty:
        print("\n=== n+1 forecast ===")
        print(fc[["ticker", "asof_date", "asof_close", "prob_up", "pred_direction", "pred_close"]].to_string(index=False))
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
