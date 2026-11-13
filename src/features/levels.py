"""Support / resistance: swing pivots (primary) + classic floor pivot points (fallback)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def swing_points(high: np.ndarray, low: np.ndarray, window: int = 3) -> tuple[list[int], list[int]]:
    """Indices of swing highs/lows: extreme within +/- window bars."""
    highs, lows = [], []
    for i in range(window, len(high) - window):
        seg_h = high[i - window:i + window + 1]
        seg_l = low[i - window:i + window + 1]
        if high[i] == seg_h.max():
            highs.append(i)
        if low[i] == seg_l.min():
            lows.append(i)
    return highs, lows


def pivot_points(h: float, l: float, c: float) -> dict[str, float]:
    p = (h + l + c) / 3
    return {"pivot": p, "r1": 2 * p - l, "s1": 2 * p - h, "r2": p + (h - l), "s2": p - (h - l)}


def support_resistance(bars: pd.DataFrame, window: int = 3, lookback: int = 60) -> dict[str, float]:
    """S/R as of the LAST row of `bars` using only that history (no future bars).

    Support = highest swing low below the close; resistance = lowest swing high above it.
    A swing at index i is only confirmed at i+window, so unconfirmed recent pivots are
    excluded automatically by swing_points(). Falls back to S1/R1 floor pivots.
    """
    b = bars.tail(lookback)
    h, l, c = b["high"].to_numpy(float), b["low"].to_numpy(float), b["close"].to_numpy(float)
    last = c[-1]
    sh, sl = swing_points(h, l, window)
    piv = pivot_points(h[-1], l[-1], last)
    sup_c = [l[i] for i in sl if l[i] < last]
    res_c = [h[i] for i in sh if h[i] > last]
    support = max(sup_c) if sup_c else min(piv["s1"], l.min())
    resistance = min(res_c) if res_c else max(piv["r1"], h.max())
    return {"support": float(support), "resistance": float(resistance), "support_method": "swing" if sup_c else "pivot",
            "resistance_method": "swing" if res_c else "pivot", **{k: float(v) for k, v in piv.items()}}


def direction_from_sentiment(score: float, threshold: float = 0.10) -> str:
    if score > threshold:
        return "bullish"
    if score < -threshold:
        return "bearish"
    return "neutral"


def position_rule(price: float, support: float, resistance: float, direction: str) -> tuple[str, str]:
    """Combine sentiment direction with location inside the S/R range.

    long : bullish and price in the lower half of the range (near support) or a breakout above resistance
    short: bearish and price in the upper half (near resistance) or a breakdown below support
    none : neutral sentiment, or sentiment fighting the level (e.g. bullish right under resistance)
    """
    rng = max(resistance - support, 1e-9)
    loc = (price - support) / rng  # 0 = at support, 1 = at resistance
    if direction == "bullish":
        if price > resistance:
            return "long", "bullish sentiment + breakout above resistance"
        if loc <= 0.5:
            return "long", f"bullish sentiment, price near support ({loc:.0%} of S/R range)"
        return "none", f"bullish but price close to resistance ({loc:.0%} of range) - wait for breakout/pullback"
    if direction == "bearish":
        if price < support:
            return "short", "bearish sentiment + breakdown below support"
        if loc >= 0.5:
            return "short", f"bearish sentiment, price near resistance ({loc:.0%} of S/R range)"
        return "none", f"bearish but price close to support ({loc:.0%} of range) - wait for breakdown/bounce"
    return "none", "neutral sentiment"
