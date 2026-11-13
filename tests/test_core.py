import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, ticker_table  # noqa: E402
from src.features.alignment import assign_bar_dates  # noqa: E402
from src.features.build import build_panel  # noqa: E402
from src.features.levels import position_rule, support_resistance  # noqa: E402
from src.ingest.apify_sources import map_daum, map_lihkg, map_telegram  # noqa: E402
from src.ingest.normalize import match_tickers  # noqa: E402
from src.ingest.prices import synthetic_prices  # noqa: E402
from src.models.walkforward import model_frame, walk_forward  # noqa: E402
from src.nlp.sentiment import lexicon_score  # noqa: E402

TICKERS = ticker_table(load_config())
HK = [t for t in TICKERS if t["exchange"] == "HK"]
KR = [t for t in TICKERS if t["exchange"] == "KR"]


def test_bar_alignment_respects_session_close_and_holidays():
    trading = pd.DatetimeIndex(["2026-11-09", "2026-11-10", "2026-11-11"])  # Mon-Wed
    ts = pd.Series(pd.to_datetime([
        "2026-11-09T07:30:00Z",   # 15:30 HKT Mon, before 16:00 close -> Mon
        "2026-11-09T08:30:00Z",   # 16:30 HKT Mon, after close        -> Tue
        "2026-11-08T03:00:00Z",   # Sunday                            -> Mon
        "2026-11-11T09:00:00Z",   # after last close                  -> NaT (pending)
    ], utc=True))
    out = assign_bar_dates(ts, trading, "Asia/Hong_Kong", "16:00")
    assert list(out.iloc[:3].dt.strftime("%Y-%m-%d")) == ["2026-11-09", "2026-11-10", "2026-11-09"]
    assert pd.isna(out.iloc[3])


def test_lexicon_multilingual_signs():
    assert lexicon_score("Tencent looks bullish, strong buy") > 0
    assert lexicon_score("騰訊業績差，看淡，沽出") < 0
    assert lexicon_score("삼성전자 실적 호조, 상승 기대") > 0
    assert lexicon_score("카카오 하락 우려 매도") < 0
    assert lexicon_score("nothing to see here") == 0


def test_keyword_matching_word_boundaries():
    assert match_tickers("SK하이닉스 실적", KR) == ["000660.KS"]
    assert match_tickers("Buying Tencent today", HK) == ["0700.HK"]
    assert match_tickers("code 107001 is not 0700", HK) == ["0700.HK"]
    assert match_tickers("Kakaotalk", KR) == []  # ASCII keywords need word boundaries


def test_support_below_and_resistance_above_price():
    bars = synthetic_prices(HK[:1], 120)
    lv = support_resistance(bars, 3, 60)
    last = bars["close"].iloc[-1]
    assert lv["support"] <= last <= lv["resistance"] or lv["support_method"] == "pivot"
    assert position_rule(100, 95, 110, "bullish")[0] == "long"
    assert position_rule(108, 95, 110, "bearish")[0] == "short"
    assert position_rule(108, 95, 110, "bullish")[0] == "none"


def test_levels_are_point_in_time():
    bars = synthetic_prices(HK[:1], 120)
    panel = build_panel(bars, pd.DataFrame(columns=["ticker", "date"]))
    i = 50
    expected = support_resistance(bars.iloc[: i + 1], 3, 60)["support"]
    assert np.isclose(panel.loc[i, "support"], expected)


def test_walk_forward_never_trains_on_future():
    bars = synthetic_prices(TICKERS, 60)
    panel = build_panel(bars, pd.DataFrame(columns=["ticker", "date"]))
    df = model_frame(panel, None).dropna(subset=["target_ret"])
    oos = walk_forward(df.tail(150), min_train_dates=3)
    assert not oos.empty
    labelled = df.tail(150)
    for d, g in oos.groupby("date"):
        assert g["n_train"].iloc[0] == int((labelled["next_date"] <= d).sum())
        assert (labelled.loc[labelled["next_date"] <= d, "date"] < d).all()


def test_actor_output_mappers():
    tg = [{"username": "@hanaresearch", "followers": 50000, "message": {
        "fulldate": "2026-11-10T01:00:00.000Z", "views": 1200, "description": "삼성전자 목표가 상향",
        "link": "https://t.me/s/hanaresearch/1"}}]
    rows = map_telegram(tg, KR, "KR", "telegram_kr")
    assert rows[0]["ticker"] == "005930.KS" and rows[0]["views"] == 1200 and rows[0]["author_followers"] == 50000
    lk = [{"thread_id": 1, "title": "騰訊業績", "content": "好正", "like_count": 5, "reply_count": 3,
           "created_at": "2026-11-10T03:00:00Z", "source_query": "騰訊"}]
    assert map_lihkg(lk, HK, "HK", {"騰訊": "0700.HK"})[0]["likes"] == 5
    dn = [{"id": "1", "title": "카카오 급등", "body": "...", "publishedAt": "2026-11-10T03:00:00+09:00"}]
    assert map_daum(dn, KR, "KR")[0]["ticker"] == "035720.KS"
