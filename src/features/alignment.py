"""Align UTC post timestamps to exchange daily bars.

Rule: a post belongs to the bar of the session it can still influence. Posts at or
before the local session close on a trading day map to that day's bar; posts after
close (or on non-trading days) map to the NEXT trading bar. Posts after the latest
available close get bar_date = NaT ("pending") - they are shown in the report but
never used as features for already-closed bars (prevents look-ahead leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_bar_dates(ts_utc: pd.Series, trading_dates: pd.DatetimeIndex, tz: str, session_close: str) -> pd.Series:
    hh, mm = map(int, session_close.split(":"))
    local = pd.to_datetime(ts_utc, utc=True).dt.tz_convert(tz)
    cand = local.dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    after_close = (local.dt.hour * 60 + local.dt.minute) > (hh * 60 + mm)
    cand = cand + pd.to_timedelta(after_close.astype(int), unit="D")
    td = pd.DatetimeIndex(sorted(trading_dates.unique())).astype("datetime64[ns]")
    idx = td.searchsorted(cand.values, side="left")
    out = pd.Series(pd.NaT, index=ts_utc.index, dtype="datetime64[ns]")
    ok = idx < len(td)
    out[ok] = td.values[idx[ok]]
    return out


def align_posts(posts: pd.DataFrame, bars: pd.DataFrame, tickers: list[dict]) -> pd.DataFrame:
    if posts.empty:
        return posts.assign(bar_date=pd.Series(dtype="datetime64[ns]"))
    meta = {t["symbol"]: t for t in tickers}
    parts = []
    for sym, g in posts.groupby("ticker"):
        t = meta.get(sym)
        dates = pd.DatetimeIndex(bars.loc[bars.ticker == sym, "date"])
        if t is None or dates.empty:
            continue
        g = g.copy()
        g["bar_date"] = assign_bar_dates(g["ts_utc"], dates, t["timezone"], t["session_close"])
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else posts.assign(bar_date=pd.NaT)


def daily_sentiment(aligned: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, bar_date) aggregates. Pending posts (NaT) are excluded."""
    cols = ["ticker", "date", "mentions", "social_mentions", "news_mentions", "sent_mean", "sent_w",
            "news_sent", "engagement", "bull_share"]
    a = aligned.dropna(subset=["bar_date"])
    if a.empty:
        return pd.DataFrame(columns=cols)
    a = a.assign(ws=a["weight"] * a["sentiment"], eng=a["likes"] + a["reposts"] + a["replies"],
                 bull=(a["sentiment"] > 0.05).astype(float), news_s=np.where(a["is_news"], a["sentiment"], np.nan))
    g = a.groupby(["ticker", "bar_date"])
    out = pd.DataFrame({
        "mentions": g.size(),
        "social_mentions": g["is_news"].apply(lambda s: int((~s).sum())),
        "news_mentions": g["is_news"].sum().astype(int),
        "sent_mean": g["sentiment"].mean(),
        "sent_w": g["ws"].sum() / g["weight"].sum(),
        "news_sent": g["news_s"].mean(),
        "engagement": g["eng"].sum(),
        "bull_share": g["bull"].mean(),
    }).reset_index().rename(columns={"bar_date": "date"})
    out["news_sent"] = out["news_sent"].fillna(0.0)
    return out[cols]
