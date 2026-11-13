"""Multilingual sentiment scoring.

Primary: cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual (HF transformers,
handles English / Chinese / Korean). Fallback: a small finance lexicon in EN/ZH/KO
so the pipeline still runs where torch is unavailable. Score is in [-1, 1]
(P(positive) - P(negative) for the model).
"""
from __future__ import annotations

import logging
import math
import re

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

POS = {
    # English
    "bullish", "buy", "beat", "beats", "strong", "surge", "rally", "upgrade", "upside", "growth", "record",
    "outperform", "gain", "gains", "soar", "breakout", "long", "profit", "rebound", "positive", "higher",
    # Chinese (trad + simp)
    "睇好", "看好", "買入", "买入", "強勁", "强劲", "升", "上升", "突破", "增長", "增长", "好正", "利好", "大升",
    "反彈", "反弹", "創新高", "创新高", "超預期", "超预期", "加倉", "加仓",
    # Korean
    "상승", "매수", "호조", "강세", "상향", "급등", "호재", "기대", "신고가", "반등", "흑자", "성장",
}
NEG = {
    "bearish", "sell", "selling", "miss", "weak", "plunge", "drop", "downgrade", "downside", "loss", "lawsuit",
    "underperform", "fall", "falls", "crash", "short", "risk", "negative", "lower", "cut", "warning",
    "看淡", "看空", "沽出", "賣出", "卖出", "跌", "下跌", "差", "利淡", "大跌", "虧損", "亏损", "爆煲", "小心", "減持", "减持",
    "하락", "매도", "부진", "약세", "하향", "급락", "악재", "우려", "손절", "적자", "폭락",
}


def lexicon_score(text: str) -> float:
    """Count lexicon hits (EN by token, ZH/KO by substring); squash to [-1, 1]."""
    t = (text or "").lower()
    words = set(re.findall(r"[a-z]+", t))
    pos = sum(1 for w in POS if (w in words if w.isascii() else w in t))
    neg = sum(1 for w in NEG if (w in words if w.isascii() else w in t))
    if pos == neg == 0:
        return 0.0
    return float(math.tanh((pos - neg) / 2.0))


class SentimentScorer:
    def __init__(self, model_name: str, use_transformers: str | bool = "auto", max_chars: int = 512):
        self.model_name = model_name
        self.max_chars = max_chars
        self.backend = "lexicon"
        self._pipe = None
        if use_transformers in (True, "true", "auto"):
            try:
                from transformers import pipeline  # noqa: WPS433

                self._pipe = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name,
                                      top_k=None, truncation=True)
                self.backend = f"transformers:{model_name}"
            except Exception as e:  # ImportError, no network, no torch...
                if use_transformers in (True, "true"):
                    raise
                log.warning("Transformer model unavailable (%s); using lexicon fallback", type(e).__name__)

    def score(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        texts = [(t or "")[: self.max_chars] for t in texts]
        if self._pipe is None:
            return np.array([lexicon_score(t) for t in texts], dtype=float)
        out = []
        for i in range(0, len(texts), batch_size):
            for res in self._pipe(texts[i:i + batch_size]):
                probs = {r["label"].lower(): r["score"] for r in res}
                out.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
        return np.array(out, dtype=float)


def engagement_weight(df: pd.DataFrame) -> pd.Series:
    """1 + log-engagement + damped reach. Keeps a single viral post from dominating."""
    eng = df["likes"] + 2 * df["reposts"] + df["replies"]
    return 1.0 + np.log1p(eng) + 0.25 * np.log1p(df["views"]) + 0.25 * np.log1p(df["author_followers"])


def score_posts(posts: pd.DataFrame, scorer: SentimentScorer) -> pd.DataFrame:
    posts = posts.copy()
    if posts.empty:
        posts["sentiment"] = []
        posts["weight"] = []
        return posts
    posts["sentiment"] = scorer.score(posts["text"].tolist())
    posts["weight"] = engagement_weight(posts)
    return posts
