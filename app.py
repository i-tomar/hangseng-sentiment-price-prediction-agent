"""Streamlit frontend for the HK/KR social-sentiment -> next-day price pipeline.

    streamlit run app.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import ROOT, load_config
from src.report.html import _mentions_fig, _price_fig, _sent_fig

PROCESSED = ROOT / "data" / "processed"
DIR_ORDER = {"bullish": 0, "neutral": 1, "bearish": 2}
DIR_ICON = {"bullish": "🟢 bullish", "neutral": "⚪ neutral", "bearish": "🔴 bearish"}
POS_ICON = {"long": "⬆️ long", "short": "⬇️ short", "none": "– none"}

st.set_page_config(page_title="HK/KR Sentiment Agent", page_icon="📈", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.6rem}
.badge{display:inline-block;padding:3px 12px;border-radius:999px;font-weight:700;font-size:.85rem;margin-right:6px}
.demo{background:#78350f;color:#fde68a}.live{background:#14532d;color:#bbf7d0}
.chip{background:#1f2937;color:#cbd5e1}
div[data-testid="stMetric"]{background:#161b26;border:1px solid #232a3a;border-radius:12px;padding:12px 16px}
</style>""", unsafe_allow_html=True)


def dark(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=40, r=20, t=50, b=30), legend=dict(orientation="h", y=-0.18))
    return fig


@st.cache_data(show_spinner=False)
def load_outputs(mode: str, stamp: float) -> dict | None:  # stamp busts the cache after a new run
    d = PROCESSED / mode
    if not (d / "meta.json").exists():
        return None

    def csv(name, dates=()):
        p = d / name
        if not p.exists() or p.stat().st_size < 2:
            return pd.DataFrame()
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        for c in dates:
            if c in df:
                df[c] = pd.to_datetime(df[c], utc=(c == "ts_utc"), errors="coerce")
        return df

    return {
        "meta": json.loads((d / "meta.json").read_text(encoding="utf-8")),
        "snapshot": csv("ticker_snapshot.csv"),
        "bars": csv("bars.csv", ["date"]),
        "daily": csv("daily_sentiment.csv", ["date"]),
        "posts": csv("posts_scored.csv", ["ts_utc", "bar_date"]),
        "oos": csv("walkforward_predictions.csv", ["date", "next_date"]),
        "metrics": csv("metrics.csv"),
        "forecast": csv("forecast.csv"),
        "checks": csv("checks.csv"),
    }


def run_pipeline(mode: str, fast: bool, offline: bool, refresh: bool) -> tuple[int, str]:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), f"--{mode}"]
    cmd += ["--no-transformers"] if fast else []
    cmd += ["--offline"] if offline else []
    cmd += ["--refresh"] if refresh else []
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + "\n" + p.stderr)[-6000:]


# ------------------------------------------------------------------ sidebar
cfg = load_config()
with st.sidebar:
    st.title("📈 Sentiment Agent")
    st.caption("CrowdWisdomTrading assignment · Ishan Singh Tomar")
    mode = st.radio("Data", ["demo", "live"], horizontal=True,
                    help="demo = synthetic posts (no keys) · live = Apify + Exa via .env")
    ex_filter = st.radio("Exchange", ["All", "HK", "KR"], horizontal=True)
    st.divider()
    st.subheader("Run pipeline")
    fast = st.checkbox("Fast sentiment (lexicon)", value=False, help="skip the XLM-R model (~1 GB first download)")
    offline = st.checkbox("Offline prices (synthetic)", value=False)
    refresh = st.checkbox("Re-pull live data (spends credits)", value=False, disabled=(mode == "demo"))
    if st.button(f"▶ Run {mode} pipeline", type="primary", width="stretch"):
        with st.spinner(f"Running {mode} pipeline… (first run with the model can take ~2 min)"):
            code, logs = run_pipeline(mode, fast, offline, refresh)
        (st.success if code == 0 else st.error)("Pipeline finished" if code == 0 else f"Pipeline failed (exit {code})")
        with st.expander("Pipeline log", expanded=code != 0):
            st.code(logs[-3000:])
        load_outputs.clear()

meta_path = PROCESSED / mode / "meta.json"
data = load_outputs(mode, meta_path.stat().st_mtime if meta_path.exists() else 0.0)
if data is None:
    st.title("HK / KR Social Sentiment → Next-Day Price")
    st.info(f"No {mode} outputs yet. Click **Run {mode} pipeline** in the sidebar"
            + (" (needs APIFY_TOKEN and EXA_API_KEY in .env)." if mode == "live" else "."))
    st.stop()

meta, snap, bars, daily, posts = data["meta"], data["snapshot"], data["bars"], data["daily"], data["posts"]
oos, met, fc, checks = data["oos"], data["metrics"], data["forecast"], data["checks"]
exchanges = ["HK", "KR"] if ex_filter == "All" else [ex_filter]
snap_f = snap[snap.exchange.isin(exchanges)]

# ------------------------------------------------------------------ header
badge = "<span class='badge demo'>DEMO DATA</span>" if meta["mode"] == "demo" else "<span class='badge live'>LIVE DATA</span>"
st.markdown(f"## HK / KR Social Sentiment → Next-Day Price {badge}"
            f"<span class='badge chip'>run {pd.Timestamp(meta['run_utc']).tz_convert('Asia/Kolkata'):%d %b %Y %H:%M} IST</span>",
            unsafe_allow_html=True)
if meta["mode"] == "demo":
    st.warning("Demo mode: social posts are **synthetic** (prefixed [DEMO]); prices are "
               + ("real yfinance bars." if meta["price_source"] == "yfinance" else "synthetic.")
               + " Metrics only show the pipeline works.", icon="⚠️")

tabs = st.tabs(["🧭 Overview", "🏆 Rankings", "🔎 Ticker deep-dive", "🧪 Model & validation", "🔮 Predictions n+1"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Posts / articles", f"{meta['n_posts']:,}")
    c[1].metric("Tickers", len(snap))
    c[2].metric("Price bars", f"{meta['n_bars']:,}")
    c[3].metric("Window", f"{meta['lookback_days']} days")
    c[4].metric("OOS predictions", int(met["n_oos_predictions"].iloc[0]) if not met.empty else 0)
    st.caption(f"Sentiment scorer: `{meta['sentiment_backend']}` · prices: `{meta['price_source']}`")
    cost_path = PROCESSED / mode / "cost_summary.json"
    if cost_path.exists():
        cost = json.loads(cost_path.read_text(encoding="utf-8"))
        st.caption(f"💳 Credits used: Apify ${cost['apify_total_usd']:.2f} across {cost['apify_runs']} actor runs · "
                   f"Exa {cost['exa_search_requests']} searches · {cost.get('note', '')}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Pipeline")
        st.graphviz_chart("""digraph G { rankdir=LR; bgcolor="transparent"; size="14,5"; ratio=fill; nodesep=0.25; ranksep=0.35;
 node [shape=box style="rounded,filled" fillcolor="#1f2937" fontcolor="#e6e8ee" color="#334155" fontname="Helvetica" fontsize=11];
 edge [color="#64748b"];
 subgraph cluster_src { label="Ingest (Apify + Exa)"; fontcolor="#94a3b8"; color="#334155";
   lihkg [label="LIHKG\\n(HK)"]; tg [label="Telegram\\n(HK + KR)"]; daum [label="Daum News\\n(KR)"]; exa [label="Exa news\\n(all)"]; yf [label="yfinance\\nprices"]; }
 norm [label="Normalise\\nUTC ts · text · engagement · followers"];
 sent [label="Sentiment\\nXLM-R / lexicon\\nengagement-weighted"];
 align [label="Align to bars\\n(after close -> next bar)"];
 lv [label="Support / resistance\\nswing + pivots"];
 feat [label="Point-in-time\\nfeatures"];
 wf [label="Walk-forward\\nlogit + ridge\\nvs baselines"];
 out [label="Rankings · n+1 forecast\\nreport · this app" fillcolor="#14532d"];
 {lihkg tg daum exa} -> norm -> sent -> align -> feat; yf -> align; yf -> lv -> feat; feat -> wf -> out; lv -> out; }""",
                          width="stretch")
    with right:
        st.subheader("Sources")
        rows = []
        for ex_id, srcs in cfg["sources"].items():
            for s in srcs:
                n = int((posts.source == s["id"]).sum()) if not posts.empty else 0
                rows.append({"exchange": ex_id, "source": s["id"], "actor": s["actor"], "posts": n})
        rows.append({"exchange": "ALL", "source": "exa_news", "actor": "Exa search API",
                     "posts": int((posts.source == "exa_news").sum()) if not posts.empty else 0})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if meta.get("source_log"):
            with st.expander("Live source log / credit usage"):
                sl = pd.DataFrame(meta["source_log"])
                for col in sl.columns:  # nested values (e.g. per-channel counts) -> text for Arrow
                    sl[col] = sl[col].map(lambda v: ", ".join(f"{k}: {n}" for k, n in v.items()) if isinstance(v, dict)
                                          else ("" if v is None or (isinstance(v, float) and pd.isna(v)) else v))
                st.dataframe(sl.astype(str), hide_index=True, width="stretch")
    st.markdown("""
**What it does:** collects the last 10 days of social posts and news for 5 HKEX and 5 KRX stocks, scores multilingual
sentiment (EN / 中文 / 한국어), maps every UTC timestamp to the trading bar it can still influence, computes support /
resistance, derives a direction + long/short/none position per ticker, and predicts the next close (n+1) with a strictly
out-of-sample walk-forward evaluation.""")

# ------------------------------------------------------------------ rankings
with tabs[1]:
    for ex_id in exchanges:
        s = snap[snap.exchange == ex_id].copy()
        if s.empty:
            continue
        s["_d"] = s["direction"].map(DIR_ORDER)
        s = s.sort_values(["mentions", "sentiment", "_d"], ascending=[False, False, True])
        st.subheader(cfg["exchanges"][ex_id]["name"])
        k = st.columns(4)
        k[0].metric("Mentions", int(s["mentions"].sum()))
        k[1].metric("Avg sentiment", f"{s['sentiment'].mean():+.3f}")
        k[2].metric("Bullish / Bearish", f"{(s.direction == 'bullish').sum()} / {(s.direction == 'bearish').sum()}")
        vc = s["position"].value_counts()
        k[3].metric("Long / Short / None", f"{vc.get('long', 0)} / {vc.get('short', 0)} / {vc.get('none', 0)}")
        view = s.assign(direction=s["direction"].map(DIR_ICON), position=s["position"].map(POS_ICON))
        a, b = st.columns([3, 2])
        with a:
            st.dataframe(view[["ticker", "name", "current_price", "mentions", "sentiment", "direction", "position",
                               "support", "resistance", "position_rationale"]], hide_index=True, width="stretch",
                         column_config={
                             "current_price": st.column_config.NumberColumn("price", format="%.2f"),
                             "support": st.column_config.NumberColumn(format="%.2f"),
                             "resistance": st.column_config.NumberColumn(format="%.2f"),
                             "sentiment": st.column_config.ProgressColumn(min_value=-1, max_value=1, format="%.3f"),
                             "position_rationale": st.column_config.TextColumn("rationale", width="large")})
        with b:
            st.plotly_chart(dark(_mentions_fig(s, f"{ex_id}: mentions coloured by sentiment"), 360),
                            width="stretch", key=f"m_{ex_id}")

# ------------------------------------------------------------------ deep dive
with tabs[2]:
    opts = [f"{r.ticker} · {r.name}" for r in snap_f.itertuples()]
    pick = st.selectbox("Ticker", opts)
    sym = pick.split(" · ")[0]
    r = snap[snap.ticker == sym].iloc[0]
    k = st.columns(3)
    k[0].metric(f"Price ({r.currency}, {r.price_date})", f"{r.current_price:,.2f}")
    k[1].metric("Sentiment (10d, weighted)", f"{r.sentiment:+.3f}")
    k[2].metric("Mentions (social / news)", f"{r.mentions}  ({r.social_mentions} / {r.news_mentions})")
    k = st.columns(4)
    k[0].metric("Direction", DIR_ICON.get(r.direction, r.direction))
    k[1].metric("Position", POS_ICON.get(r.position, r.position))
    k[2].metric("Support", f"{r.support:,.2f}")
    k[3].metric("Resistance", f"{r.resistance:,.2f}")
    st.caption(f"Rationale: {r.position_rationale} · S/R method: {r.support_method}/{r.resistance_method} · "
               f"pivot {r.pivot:,.2f} (S1 {r.pivot_s1:,.2f}, R1 {r.pivot_r1:,.2f})")
    pf = _price_fig(sym, r["name"], bars, r, oos, fc)
    pf.update_traces(selector=dict(name="Actual close"), marker=dict(color="#f8fafc", size=8))
    st.plotly_chart(dark(pf, 460), width="stretch", key="price")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(dark(_sent_fig(daily, [sym], "Daily engagement-weighted sentiment"), 340), width="stretch", key="sent")
    with b:
        d = daily[daily.ticker == sym]
        fig = go.Figure(go.Bar(x=d["date"], y=d["social_mentions"], name="social"))
        fig.add_bar(x=d["date"], y=d["news_mentions"], name="news")
        fig.update_layout(barmode="stack", title="Mentions per bar")
        st.plotly_chart(dark(fig, 340), width="stretch", key="ment")
    st.subheader("Recent posts & news")
    p = posts[posts.ticker == sym].sort_values("ts_utc", ascending=False).copy()
    if p.empty:
        st.info("No posts matched this ticker in the window.")
    else:
        p["label"] = pd.cut(p["sentiment"], [-1.01, -0.05, 0.05, 1.01], labels=["🔴 negative", "⚪ neutral", "🟢 positive"])
        p["engagement"] = p["likes"] + p["reposts"] + p["replies"]
        src = st.multiselect("Sources", sorted(p.source.unique()), default=sorted(p.source.unique()))
        p = p[p.source.isin(src)]
        st.dataframe(p[["ts_utc", "source", "label", "sentiment", "text", "engagement", "views", "author_followers", "url"]].head(200),
                     hide_index=True, width="stretch", height=420,
                     column_config={"ts_utc": st.column_config.DatetimeColumn("time (UTC)", format="YYYY-MM-DD HH:mm"),
                                    "sentiment": st.column_config.NumberColumn(format="%+.2f"),
                                    "text": st.column_config.TextColumn(width="large"),
                                    "url": st.column_config.LinkColumn("link", display_text="open")})

# ------------------------------------------------------------------ model
with tabs[3]:
    if met.empty:
        st.info("Not enough trading days for walk-forward evaluation.")
    else:
        k = st.columns(4)
        full = met.iloc[0]
        naive = met[met.model.str.startswith("Naive")].iloc[0]
        k[0].metric("Model direction accuracy", f"{full.direction_accuracy:.1%}")
        k[1].metric("Best baseline accuracy", f"{met.iloc[3:]['direction_accuracy'].max():.1%}")
        k[2].metric("Model MAE (ret, pp)", f"{full.MAE_ret_pct:.2f}", delta=f"{full.MAE_ret_pct - naive.MAE_ret_pct:+.2f} vs naive",
                    delta_color="inverse")
        k[3].metric("OOS predictions / dates", f"{int(full.n_oos_predictions)} / {int(full.n_oos_dates)}")
        a, b = st.columns(2)
        with a:
            m = met.dropna(subset=["direction_accuracy"])
            fig = go.Figure(go.Bar(x=m["direction_accuracy"], y=m["model"], orientation="h",
                                   marker_color=["#22c55e", "#3b82f6"] + ["#64748b"] * (len(m) - 2)))
            fig.add_vline(x=0.5, line_dash="dash", line_color="#f59e0b")
            fig.update_layout(title="Direction accuracy (out-of-sample)", xaxis_tickformat=".0%")
            st.plotly_chart(dark(fig, 340), width="stretch", key="acc")
        with b:
            m = met.dropna(subset=["MAE_ret_pct"])
            fig = go.Figure([go.Bar(name="MAE", x=m["model"], y=m["MAE_ret_pct"]),
                             go.Bar(name="RMSE", x=m["model"], y=m["RMSE_ret_pct"])])
            fig.update_layout(title="Next-day return error (pp, lower is better)", barmode="group")
            fig.update_xaxes(tickangle=0, ticktext=[t.split(" (")[0].split(":")[0] for t in m["model"]], tickvals=list(m["model"]))
            dark(fig, 340).update_layout(legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
            st.plotly_chart(fig, width="stretch", key="err")
        st.dataframe(met, hide_index=True, width="stretch")
        if not oos.empty:
            with st.expander("Every walk-forward prediction"):
                st.dataframe(oos[oos.ticker.isin(snap_f.ticker)], hide_index=True, width="stretch")
    st.markdown("""
**How validation works.** Expanding-window walk-forward, pooled across tickers. For each test date *d* the logistic
regression (P(up)) and ridge regression (next-day return) are fit **only** on rows whose next close was already known
at *d*'s close, then predict every ticker at *d*. A price-only ablation shows what sentiment adds, and the naive
no-change, majority-class and momentum baselines are the bar to beat. Unit tests assert no training row post-dates its
test date, and support/resistance features are recomputed point-in-time.

**Caveat.** 10 days ≈ 7–8 trading bars, so the out-of-sample set is small and confidence intervals are wide. The goal
is a clean, leak-free, reproducible pipeline, not a claim of edge.""")

# ------------------------------------------------------------------ predictions
with tabs[4]:
    if fc.empty:
        st.info("No n+1 forecast in this run.")
    else:
        f = fc.merge(snap[["ticker", "name", "exchange", "direction", "position"]], on="ticker", how="left")
        f = f[f.exchange.isin(exchanges)].sort_values(["exchange", "prob_up"], ascending=[True, False])
        f["call"] = f["pred_direction"].map({"up": "⬆️ up", "down": "⬇️ down"})
        f["direction"] = f["direction"].map(DIR_ICON)
        st.dataframe(f[["exchange", "ticker", "name", "asof_date", "asof_close", "prob_up", "call", "pred_ret", "pred_close",
                        "direction", "position"]], hide_index=True, width="stretch",
                     column_config={"prob_up": st.column_config.ProgressColumn("P(up)", min_value=0, max_value=1, format="%.2f"),
                                    "pred_ret": st.column_config.NumberColumn("pred return", format="%+.4f"),
                                    "asof_close": st.column_config.NumberColumn(format="%.2f"),
                                    "pred_close": st.column_config.NumberColumn(format="%.2f")})
        st.caption("P(up) comes from the logistic model; predicted close from the ridge model, so the two can disagree "
                   "on small moves. Forecasts are logged to data/predictions/ and scored with `python scripts/check_predictions.py`.")
    if not checks.empty:
        st.subheader("Past predictions checked against actual closes")
        st.dataframe(checks, hide_index=True, width="stretch")
