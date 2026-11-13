"""Synthetic social/news posts for --demo mode. CLEARLY NOT REAL DATA.

Posts are generated so the pipeline can run end-to-end without API keys. A weak,
noisy link between post tone and the *next* day's return is injected so the model
section has something to find; real-data results will differ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest.normalize import make_id, to_frame

TEMPLATES = {
    "HK": {
        "pos": ["{n} 業績強勁，睇好，準備買入 🚀", "{n} looks bullish, strong buy into results", "{n} 升勢持續，突破阻力位，好正"],
        "neg": ["{n} 跌到冇眼睇，沽出算了", "{n} weak guidance, bearish, selling", "{n} 業績差，看淡，小心再跌"],
        "neu": ["{n} 今日成交一般，觀望中", "Anyone watching {n} today?", "{n} 明天公佈數據"],
    },
    "KR": {
        "pos": ["{n} 실적 호조, 상승 기대 매수", "{n} strong rally, bullish outlook", "{n} 목표가 상향, 강세 지속"],
        "neg": ["{n} 실적 부진, 하락 우려 매도", "{n} downgrade, bearish, weak demand", "{n} 약세 지속, 손절"],
        "neu": ["{n} 오늘 거래량 보통, 관망", "{n} 내일 실적 발표 예정", "What do you think about {n}?"],
    },
}
SOURCES = {"HK": ["lihkg", "telegram_hk"], "KR": ["telegram_kr", "daum_news"]}


def demo_posts(tickers: list[dict], bars: pd.DataFrame, lookback_days: int, now: pd.Timestamp,
               seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = now - pd.Timedelta(days=lookback_days)
    for t in tickers:
        b = bars[bars.ticker == t["symbol"]].sort_values("date")
        fwd = b.set_index("date")["close"].pct_change().shift(-1)  # next-day return
        z = (fwd / (fwd.std() or 1)).fillna(0.0)
        local_name = next((k for k in t["keywords"] if not k.isascii()), t["name"])
        for day in pd.date_range(start.normalize(), now.normalize(), freq="D", tz="UTC"):
            prior = z[z.index <= day.tz_localize(None)]
            tilt = float(prior.iloc[-1]) if len(prior) else 0.0
            p_pos = 1 / (1 + np.exp(-(0.6 * tilt))) * 0.7
            p_neg = 0.7 - p_pos
            for _ in range(rng.poisson(6)):
                ts = day + pd.Timedelta(seconds=int(rng.integers(0, 86400)))
                if ts > now or ts < start:
                    continue
                tone = rng.choice(["pos", "neg", "neu"], p=[p_pos, p_neg, 0.3])
                name = local_name if rng.random() < 0.5 else t["name"]
                text = rng.choice(TEMPLATES[t["exchange"]][tone]).format(n=name)
                is_news = rng.random() < 0.15
                src = "exa_news" if is_news else rng.choice(SOURCES[t["exchange"]])
                rows.append({
                    "post_id": make_id("demo", t["symbol"], ts.value, rng.random()), "source": src,
                    "exchange": t["exchange"], "ticker": t["symbol"], "ts_utc": ts, "text": f"[DEMO] {text}",
                    "likes": float(rng.poisson(15)), "reposts": float(rng.poisson(3)),
                    "replies": float(rng.poisson(5)), "views": float(rng.poisson(800)),
                    "author": f"demo_user_{rng.integers(1, 300)}",
                    "author_followers": float(rng.lognormal(6, 1.5)), "url": "", "is_news": is_news, "query": "demo",
                })
    return to_frame(rows)
