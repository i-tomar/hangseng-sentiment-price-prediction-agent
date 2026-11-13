# HK / KR Social-Sentiment → Next-Day Price Pipeline

Assignment for **CrowdWisdomTrading (Tzuroni Ltd)** — Ishan Singh Tomar.

A Python pipeline that collects the last 10 days of social posts (≥2 sources per exchange, via **Apify**) and news (**Exa**) for 5 Hong Kong and 5 Korean stocks, scores multilingual sentiment, aligns posts to daily price bars in UTC, computes support/resistance, ranks tickers per exchange, predicts the next bar (n+1) and validates everything strictly out-of-sample with a walk-forward setup. Output is a self-contained HTML report.

| Exchange | Tickers |
|---|---|
| HKEX | 0700.HK Tencent · 9988.HK Alibaba · 3690.HK Meituan · 1810.HK Xiaomi · 0005.HK HSBC |
| KRX  | 005930.KS Samsung Elec. · 000660.KS SK hynix · 035420.KS NAVER · 005380.KS Hyundai Motor · 035720.KS Kakao |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional, better sentiment (~1 GB model download on first run):
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers sentencepiece protobuf

# 1) Demo: no keys needed, synthetic social posts (clearly labelled), real yfinance prices
python scripts/run_pipeline.py --demo                  # add --no-transformers for the fast lexicon path
                                                       # add --offline to use synthetic prices too

# 2) Live: real Apify + Exa data
cp .env.example .env        # then fill APIFY_TOKEN and EXA_API_KEY
python scripts/run_pipeline.py --live                  # cached per day in data/raw/<date>/; --refresh to re-pull

# 3) After the next session closes, score the saved n+1 predictions
python scripts/check_predictions.py                    # --mode demo for demo predictions

# tests
python -m pytest -q
```

Report: `reports/<mode>_report.html` (open in a browser; plotly is inlined so it works offline).
Outputs: `data/processed/<mode>/*.csv` (scored posts, daily sentiment, ticker snapshot, walk-forward predictions, metrics) and `data/predictions/` (n+1 forecasts + running log).

## Frontend

A Streamlit dashboard for demoing the project without walking through the code:

```bash
streamlit run app.py          # opens http://localhost:8501
```

- **Sidebar:** pick demo or live outputs, filter by exchange, and run the pipeline from the UI (it calls `scripts/run_pipeline.py`; options for the fast lexicon scorer, offline prices and re-pulling live data).
- **Overview:** what the pipeline does, an architecture flow diagram, sources/actor IDs with post counts, the scorer used and a DEMO/LIVE badge.
- **Rankings:** HK and KR tables sorted by mentions → sentiment → direction, plus a mentions-by-sentiment bar chart.
- **Ticker deep-dive:** candlesticks with support/resistance, walk-forward predicted vs actual closes and the n+1 star, a sentiment timeline, mentions per bar, and recent posts/news with sentiment labels and links.
- **Model & validation:** walk-forward metrics against the baselines, with an explanation.
- **Predictions n+1:** the forecast table (P(up), predicted return and close) and any past predictions that have been checked.

The app reads `data/processed/<mode>/` (CSV + `meta.json`), which every pipeline run writes. Demo outputs ship in the repo, so the app opens with data right away. The dark theme is set in `.streamlit/config.toml`.

## Data sources (Apify actor IDs)

| Exchange | Source | Actor ID | Key input |
|---|---|---|---|
| HK | LIHKG forum (keyword search) | `claude_code_reviewer/lihkg-scraper-en` | `mode: search`, `keywords`, `dateFrom: "10d"`, `maxItems`, `aiLabels: false` |
| HK | Telegram public channels | `tri_angle/telegram-scraper` | `profiles`, `collectMessages: true`, `scrapeLastNDays: 10` |
| KR | Telegram public channels (broker research / retail) | `tri_angle/telegram-scraper` | as above |
| KR | Daum News (stock + economy sections) | `ardent_fork/daum-news` | `sections: [stock, economy]`, `maxItems`, `fetchArticles: true` |
| all | News search | Exa (`exa-py`, `search` with text contents, `category: news`) | `start_published_date` = now − 10 days, 10 results / ticker |

Input fields were checked against each actor's published input schema (Apify API, Nov 2026). Output mapping in `src/ingest/apify_sources.py` uses field fallbacks (`first(item, "created_at", "createdAt", ...)`) because community actors change shape. Channel lists are in `config.yaml` — swap in any public channels (they must be reachable at `https://t.me/s/<name>`).

Futu/Moomoo, Naver stock discussion boards and KakaoTalk open chats would be better retail sources but have no maintained Apify actor, so they are not used.

## Live run results (13 Nov 2026, data as of the 13 Nov close)

Output: `reports/live_report.html`, `data/processed/live/`, `data/predictions/live_predictions_log.csv` (committed so reviewers can see real results; raw actor dumps in `data/raw/` stay out of git).

| Exchange | Source | Posts tagged to tickers (10d) | Raw items pulled |
|---|---|---|---|
| HK | LIHKG (`claude_code_reviewer/lihkg-scraper-en`) | 14 | 15 threads |
| HK | Telegram (`tri_angle/telegram-scraper`): hongkongstocks, aastocks_news, zaobaosg | 10 | 201 messages |
| HK | Exa news | 50 | 50 |
| KR | Telegram: YeouidoStory2, hanaresearch, shinhanresearch, hedgehara | 68 | 259 messages |
| KR | Daum News (`ardent_fork/daum-news`, stock + economy) | 34 | 84 articles |
| KR | Exa news | 50 | 50 |

226 posts/articles in total (39 published after the latest close, so they're held out as pending). Walk-forward out-of-sample: 50 predictions over 5 dates. Direction accuracy is 60% for sentiment+price, 66% for price-only, 64% for momentum and 52% for majority class. Next-day return MAE is 3.05 pp for the model against 2.16 pp for naive no-change. In other words there's no demonstrated edge from 10 days of data, which is the expected and honest result.

**Actual cost:** Apify $2.19 over 10 actor runs (Telegram $1.97, Daum $0.17, LIHKG $0.06), including one debugging run, plus 10 Exa searches. A fresh full live run costs about $1.6 on Apify plus 10 Exa searches.

**Fixes made during the live run:**
- Telegram: a single multi-channel run hit the spend cap on its first channel and skipped the others, so there is now one run per channel, each with its own `max_usd_per_channel` cap.
- HK channels: the general-news HK channels were swapped for HK stock channels found by checking `t.me/s/` for ticker mentions.
- LIHKG: more than 2 keywords per ticker triggers LIHKG 429 rate limits, so `keywords_per_ticker: 2` is the default.

## Pipeline design

```
src/ingest    apify_sources.py (run + map actors), exa_news.py, prices.py (yfinance), demo.py, cache.py, collect.py
src/nlp       sentiment.py   XLM-R multilingual model, lexicon fallback, engagement weighting
src/features  alignment.py   UTC -> exchange bar mapping, daily aggregates
              levels.py      swing pivots + floor pivots, direction & position rule
              build.py       point-in-time feature panel + per-ticker snapshot
src/models    walkforward.py expanding-window logistic (direction) + ridge (return), baselines, n+1 forecast
              check.py       scores saved predictions vs actual closes
src/report    html.py        self-contained plotly report
scripts/      run_pipeline.py, check_predictions.py
```

1. **Extraction** — every post is normalised to: `ts_utc, text, likes, reposts, replies, views, author, author_followers, url, source, ticker`. Posts are tagged to tickers with English + Chinese/Korean keyword lists (word-boundary matching for ASCII, substring for CJK).
2. **Time alignment** — timestamps converted to UTC, then to exchange-local time. A post at/before the session close (HKEX 16:00 HKT, KRX 15:30 KST) maps to that day's bar; after close / weekend / holiday → next trading bar. Posts after the latest close are "pending" and never used as features for closed bars. Today's bar is dropped if the session hasn't closed (no partial bars).
3. **Sentiment** — `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`, score = P(pos) − P(neg) ∈ [−1, 1]. If transformers/torch are missing it falls back to an EN/ZH/KO finance lexicon automatically (the report header shows which backend ran). Weight = 1 + log(1+likes+2·reposts+replies) + ¼·log(1+views) + ¼·log(1+followers).
4. **Per-ticker output** — ticker, name, current price, sentiment, direction (bullish/bearish/neutral at ±0.10), position (long/short/none with rationale), support, resistance, mention counts.
5. **Support / resistance** — nearest confirmed swing low below / swing high above the close (±3-bar pivots over 60 bars); floor pivots S1/R1 as fallback. Recomputed at every date using only past bars for model features.
6. **Position rule** — long if bullish and price in the lower half of the S/R range (or broke above resistance); short if bearish and in the upper half (or broke below support); otherwise none.
7. **Model** — pooled panel (10 tickers × trading days in the window). Features: weighted sentiment, 3-day EMA, news sentiment, log mentions, bullish share, 1d/5d return, 10d volatility, distance to support/resistance, exchange dummy. Logistic regression → P(up); ridge → next-day return → predicted close.
8. **Validation** — expanding-window walk-forward: for each test date *d* the models are fit only on rows whose target close was known by *d*'s close (`next_date <= d`), then predict all tickers at *d*. Compared against naive no-change, majority-class and momentum baselines, plus a price-only ablation to isolate what sentiment adds. Metrics: direction accuracy, directional hit, MAE and RMSE of next-day return. Unit tests assert no training row ever post-dates the test date.
9. **n+1** — final model fit on all labelled rows predicts the next bar per ticker; saved to `data/predictions/` and checked later with `scripts/check_predictions.py` (or automatically on the next pipeline run).

## Credits / cost control

- Apify free plan: $5/month. Each actor run is capped with `max_total_charge_usd` (`sources.*.max_usd` in `config.yaml`, total ≤ $2.5/run) and small `maxItems`. LIHKG `aiLabels` is off (extra billing).
- Exa: 10 searches per run (1 per ticker).
- Raw pulls are cached in `data/raw/<YYYY-MM-DD>/`, so re-running the same day costs nothing. `--refresh` forces a new pull.
- The report's "Data sources & credit usage" table lists items returned and errors per source.

## Sharing tokens for review (safely)

Never commit `.env` (it is git-ignored). For the reviewer:
1. **Apify** → Console → Settings → API & Integrations → *Create new token*, name it e.g. `crowdwisdom-review`, scope it to the needed actors if offered.
2. **Exa** → Dashboard → API Keys → create a separate key with a low spend limit.
3. Send both only by email, not in the repo, and revoke them after review.

## Limitations (honest read)

- 10 days ≈ 7–8 trading bars, so the out-of-sample set is ~40 predictions; accuracy has wide confidence intervals and is not evidence of an edge. The point here is a clean, leak-free, reproducible pipeline.
- Ticker coverage in Telegram/Daum is uneven; LIHKG and Daum have no follower counts (set to 0); Telegram engagement is mostly views.
- Keyword tagging can mis-attribute (e.g. "Hyundai" group vs Hyundai Motor). The XLM-R model is trained on tweets, not finance text.
- yfinance occasionally lags or returns a NaN close for the latest bar; such bars are dropped.
- Demo mode posts are synthetic, with a weak injected link to next-day returns, purely to exercise the code path.
