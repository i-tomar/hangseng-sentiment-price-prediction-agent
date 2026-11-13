"""Daily price bars via yfinance (with a deterministic synthetic fallback for offline demo)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
BAR_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def fetch_yfinance(symbols: list[str], days: int) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for sym in symbols:
        h = yf.Ticker(sym).history(period=f"{days}d", interval="1d", auto_adjust=False)
        if h.empty:
            log.warning("yfinance returned no bars for %s", sym)
            continue
        h = h.reset_index()
        # 'Date' is exchange-local midnight; keep the local trading date
        h = h.rename(columns=str.lower)
        h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
        h["ticker"] = sym
        frames.append(h[BAR_COLUMNS])
    if not frames:
        raise RuntimeError("yfinance returned no data")
    df = pd.concat(frames, ignore_index=True).dropna(subset=["close"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def synthetic_prices(tickers: list[dict], days: int, seed: int = 7, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Geometric random-walk bars on business days (offline demo only)."""
    rng = np.random.default_rng(seed)
    end = (end or pd.Timestamp.today()).normalize()
    dates = pd.bdate_range(end=end - pd.Timedelta(days=1), periods=int(days * 5 / 7))
    base = {"HK": 100.0, "KR": 80000.0}
    frames = []
    for i, t in enumerate(tickers):
        p0 = base.get(t["exchange"], 100.0) * (1 + 0.3 * i)
        rets = rng.normal(0.0003, 0.018, len(dates))
        close = p0 * np.exp(np.cumsum(rets))
        open_ = close * np.exp(rng.normal(0, 0.006, len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, len(dates))))
        frames.append(pd.DataFrame({"date": dates, "ticker": t["symbol"], "open": open_, "high": high,
                                    "low": low, "close": close, "volume": rng.integers(1e6, 3e7, len(dates))}))
    return pd.concat(frames, ignore_index=True)


def get_prices(tickers: list[dict], days: int, offline: bool = False) -> tuple[pd.DataFrame, str]:
    """Returns (bars, provenance) where provenance is 'yfinance' or 'synthetic'."""
    if not offline:
        try:
            return fetch_yfinance([t["symbol"] for t in tickers], days), "yfinance"
        except Exception as e:  # network / rate-limit
            log.warning("yfinance failed (%s); using synthetic prices", e)
    return synthetic_prices(tickers, days), "synthetic"
