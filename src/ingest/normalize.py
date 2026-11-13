"""Unified post schema + defensive field mapping for heterogeneous actor outputs."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

import pandas as pd

POST_COLUMNS = [
    "post_id", "source", "exchange", "ticker", "ts_utc", "text", "likes", "reposts",
    "replies", "views", "author", "author_followers", "url", "is_news", "query",
]


def first(d: dict, *keys: str, default: Any = None) -> Any:
    """Return the first non-empty value among (possibly dotted) keys."""
    for k in keys:
        cur: Any = d
        for part in k.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        if cur not in (None, "", [], {}):
            return cur
    return default


def to_num(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "").lower()
    m = re.match(r"^([\d.]+)\s*([km]?)", s)
    if not m:
        return 0.0
    v = float(m.group(1))
    return v * {"k": 1e3, "m": 1e6}.get(m.group(2), 1)


def to_utc(x: Any) -> pd.Timestamp | None:
    if x in (None, ""):
        return None
    try:
        if isinstance(x, (int, float)):  # epoch seconds or ms
            unit = "ms" if x > 1e11 else "s"
            return pd.Timestamp(x, unit=unit, tz="UTC")
        ts = pd.Timestamp(x)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC")
    except (ValueError, TypeError, OverflowError):
        return None


def make_id(*parts: Any) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _kw_pattern(kw: str) -> re.Pattern:
    # ASCII keywords need word boundaries (avoid "Kakao" matching inside other words
    # and numeric codes matching inside longer numbers); CJK keywords use substring.
    if re.fullmatch(r"[\x00-\x7f]+", kw):
        return re.compile(r"(?<![\w])" + re.escape(kw) + r"(?![\w])", re.IGNORECASE)
    return re.compile(re.escape(kw))


def match_tickers(text: str, tickers: Iterable[dict]) -> list[str]:
    """Return symbols whose keyword list matches the text."""
    hits = []
    for t in tickers:
        pats = t.setdefault("_patterns", [_kw_pattern(k) for k in t["keywords"]])
        if any(p.search(text or "") for p in pats):
            hits.append(t["symbol"])
    return hits


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=POST_COLUMNS)
    if df.empty:
        return df
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    for c in ["likes", "reposts", "replies", "views", "author_followers"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["is_news"] = df["is_news"].astype(bool)
    return df.drop_duplicates(subset=["post_id", "ticker"]).reset_index(drop=True)
