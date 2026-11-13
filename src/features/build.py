"""Feature panel + per-ticker snapshot (all point-in-time, no look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.levels import direction_from_sentiment, position_rule, support_resistance

SENT_FEATURES = ["sent_w", "sent_ema3", "news_sent", "log_mentions", "bull_share"]
PRICE_FEATURES = ["ret_1d", "ret_5d", "vol_10d", "dist_support", "dist_resistance"]


def build_panel(bars: pd.DataFrame, daily: pd.DataFrame, window: int = 3, lookback: int = 60) -> pd.DataFrame:
    """One row per (ticker, date) with features known at that date's close and the next-bar target."""
    out = []
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).astype("datetime64[ns]")
    for sym, b in bars.groupby("ticker"):
        b = b.sort_values("date").reset_index(drop=True).copy()
        b["date"] = pd.to_datetime(b["date"]).astype("datetime64[ns]")
        b = b.merge(daily[daily.ticker == sym].drop(columns="ticker"), on="date", how="left")
        for c in ["mentions", "social_mentions", "news_mentions", "engagement",
                  "sent_mean", "sent_w", "news_sent", "bull_share"]:  # tolerate empty daily frame
            b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0.0) if c in b else 0.0
        b["log_mentions"] = np.log1p(b["mentions"])
        b["sent_ema3"] = b["sent_w"].ewm(span=3, adjust=False).mean()
        b["ret_1d"] = b["close"].pct_change()
        b["ret_5d"] = b["close"].pct_change(5)
        b["vol_10d"] = b["ret_1d"].rolling(10).std()
        sup, res = [], []
        for i in range(len(b)):  # levels recomputed with bars[:i+1] only
            if i < 2 * window + 2:
                sup.append(np.nan); res.append(np.nan); continue
            lv = support_resistance(b.iloc[: i + 1], window, lookback)
            sup.append(lv["support"]); res.append(lv["resistance"])
        b["support"], b["resistance"] = sup, res
        b["dist_support"] = b["close"] / b["support"] - 1
        b["dist_resistance"] = b["resistance"] / b["close"] - 1
        b["next_close"] = b["close"].shift(-1)
        b["next_date"] = b["date"].shift(-1)
        b["target_ret"] = b["next_close"] / b["close"] - 1
        b["target_up"] = (b["target_ret"] > 0).astype(float).where(b["next_close"].notna())
        b["ticker"] = sym
        out.append(b)
    return pd.concat(out, ignore_index=True)


def ticker_snapshot(bars: pd.DataFrame, aligned: pd.DataFrame, tickers: list[dict], cfg: dict) -> pd.DataFrame:
    """Required per-ticker fields: ticker, name, current price, sentiment, direction, position, S/R."""
    thr = cfg["sentiment"]["direction_threshold"]
    lv_cfg = cfg["levels"]
    rows = []
    for t in tickers:
        b = bars[bars.ticker == t["symbol"]].sort_values("date")
        if b.empty:
            continue
        p = aligned[aligned.ticker == t["symbol"]] if not aligned.empty else aligned
        lv = support_resistance(b, lv_cfg["swing_window"], lv_cfg["lookback_bars"])
        price = float(b["close"].iloc[-1])
        if len(p):
            sent = float((p["weight"] * p["sentiment"]).sum() / p["weight"].sum())
            social = p[~p["is_news"]]
            news = p[p["is_news"]]
        else:
            sent, social, news = 0.0, p, p
        direction = direction_from_sentiment(sent, thr)
        pos, why = position_rule(price, lv["support"], lv["resistance"], direction)
        rows.append({
            "exchange": t["exchange"], "ticker": t["symbol"], "name": t["name"], "currency": t["currency"],
            "price_date": b["date"].iloc[-1].date().isoformat(), "current_price": price,
            "mentions": int(len(p)), "social_mentions": int(len(social)), "news_mentions": int(len(news)),
            "sentiment": round(sent, 4),
            "social_sentiment": round(float(social["sentiment"].mean()), 4) if len(social) else 0.0,
            "news_sentiment": round(float(news["sentiment"].mean()), 4) if len(news) else 0.0,
            "direction": direction, "position": pos, "position_rationale": why,
            "support": lv["support"], "resistance": lv["resistance"],
            "support_method": lv["support_method"], "resistance_method": lv["resistance_method"],
            "pivot": lv["pivot"], "pivot_s1": lv["s1"], "pivot_r1": lv["r1"],
            "sources": ", ".join(sorted(p["source"].unique())) if len(p) else "",
        })
    return pd.DataFrame(rows)
