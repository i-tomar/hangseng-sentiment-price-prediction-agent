"""Self-contained HTML report (plotly.js inlined once)."""
from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go

DIR_ORDER = {"bullish": 0, "neutral": 1, "bearish": 2}
CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1d2433}
main{max-width:1200px;margin:auto;padding:24px}
h1{margin:0 0 4px} h2{border-bottom:2px solid #dde1e7;padding-bottom:6px;margin-top:40px}
.card{background:#fff;border-radius:10px;padding:16px 20px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.banner{background:#fff3cd;border:1px solid #f0c36d;padding:12px 16px;border-radius:8px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px} th,td{padding:6px 8px;border-bottom:1px solid #e8eaee;text-align:right}
th{background:#f0f2f5;cursor:pointer;position:sticky;top:0} td:first-child,th:first-child,td.l{text-align:left}
.bullish{color:#0a8a3a;font-weight:600}.bearish{color:#c62828;font-weight:600}.neutral{color:#666}
.long{background:#e6f4ea}.short{background:#fdecea}.muted{color:#6b7280;font-size:13px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px} @media(max-width:900px){.grid{grid-template-columns:1fr}}
"""
SORT_JS = """
document.querySelectorAll('table.sortable th').forEach((th,i)=>th.addEventListener('click',()=>{
 const tb=th.closest('table').tBodies[0];const rows=[...tb.rows];const asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
 rows.sort((a,b)=>{const x=a.cells[i].dataset.v??a.cells[i].innerText,y=b.cells[i].dataset.v??b.cells[i].innerText;
 const nx=parseFloat(x),ny=parseFloat(y);const c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc?c:-c});
 rows.forEach(r=>tb.appendChild(r))}));
"""


def _fmt(v, nd=2):
    if isinstance(v, float):
        if pd.isna(v):
            return "–"
        return f"{v:,.{nd}f}"
    return html.escape(str(v))


def _table(df: pd.DataFrame, cols: list[str] | None = None, nd: int = 2, cls: str = "sortable") -> str:
    if df is None or df.empty:
        return "<p class='muted'>No rows.</p>"
    cols = cols or list(df.columns)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            klass = str(v) if c in ("direction", "position") else ""
            dv = f" data-v='{v}'" if isinstance(v, (int, float)) and not pd.isna(v) else ""
            if c == "direction":
                dv = f" data-v='{DIR_ORDER.get(v, 1)}'"
            l = " l" if isinstance(v, str) else ""
            tds.append(f"<td class='{klass}{l}'{dv}>{_fmt(v, nd)}</td>")
        rc = r.get("position", "") if "position" in df.columns else ""
        body.append(f"<tr class='{rc}'>{''.join(tds)}</tr>")
    return f"<table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


class _Figs:
    def __init__(self):
        self.first = True

    def __call__(self, fig: go.Figure, height: int = 380) -> str:
        fig.update_layout(height=height, margin=dict(l=40, r=20, t=40, b=30), template="plotly_white",
                          legend=dict(orientation="h", y=-0.15))
        out = fig.to_html(full_html=False, include_plotlyjs="inline" if self.first else False,
                          config={"displaylogo": False})
        self.first = False
        return out


def _price_fig(sym, name, bars, snap, oos, fc) -> go.Figure:
    b = bars[bars.ticker == sym].sort_values("date").tail(40)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=b["date"], open=b["open"], high=b["high"], low=b["low"], close=b["close"],
                                 name="Price", showlegend=False))
    if snap is not None:
        fig.add_hline(y=snap["support"], line=dict(color="#0a8a3a", dash="dash"),
                      annotation_text=f"Support {snap['support']:,.2f}", annotation_position="bottom left")
        fig.add_hline(y=snap["resistance"], line=dict(color="#c62828", dash="dash"),
                      annotation_text=f"Resistance {snap['resistance']:,.2f}", annotation_position="top left")
    o = oos[oos.ticker == sym] if not oos.empty else oos
    if len(o):
        fig.add_trace(go.Scatter(x=o["next_date"], y=o["pred_close"], mode="markers+lines", name="Walk-forward predicted close",
                                 marker=dict(symbol="x", size=8, color="#7b3fe4")))
        fig.add_trace(go.Scatter(x=o["next_date"], y=o["next_close"], mode="markers", name="Actual close",
                                 marker=dict(size=7, color="#111")))
    f = fc[fc.ticker == sym] if not fc.empty else fc
    if len(f):
        nxt = pd.Timestamp(f["asof_date"].iloc[0]) + pd.offsets.BDay(1)
        fig.add_trace(go.Scatter(x=[nxt], y=f["pred_close"], mode="markers", name="n+1 forecast",
                                 marker=dict(symbol="star", size=14, color="#f59e0b")))
    fig.update_layout(title=f"{sym} {name}", xaxis_rangeslider_visible=False)
    return fig


def _sent_fig(daily: pd.DataFrame, syms: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for s in syms:
        d = daily[daily.ticker == s].sort_values("date")
        if d.empty:
            continue
        fig.add_trace(go.Scatter(x=d["date"], y=d["sent_w"], mode="lines+markers", name=f"{s} sentiment",
                                 text=[f"mentions {m}" for m in d["mentions"]]))
    fig.add_hline(y=0, line=dict(color="#999", width=1))
    fig.update_layout(title=title, yaxis_title="engagement-weighted sentiment (-1..1)")
    return fig


def _mentions_fig(snap: pd.DataFrame, title: str) -> go.Figure:
    s = snap.sort_values("mentions", ascending=False)
    fig = go.Figure(go.Bar(x=s["ticker"], y=s["mentions"], marker=dict(color=s["sentiment"], colorscale="RdYlGn",
                    cmin=-0.5, cmax=0.5, colorbar=dict(title="sent")), text=s["direction"]))
    fig.update_layout(title=title, yaxis_title="mentions (10d)")
    return fig


def build_report(path, *, mode: str, cfg: dict, snapshot: pd.DataFrame, bars: pd.DataFrame, daily: pd.DataFrame,
                 oos: pd.DataFrame, metrics: pd.DataFrame, forecast: pd.DataFrame, checks: pd.DataFrame,
                 source_log: list[dict], meta: dict) -> None:
    fig = _Figs()
    now = datetime.now(timezone.utc)
    parts = [f"<html><head><meta charset='utf-8'><title>HK/KR Sentiment Report</title><style>{CSS}</style></head><body><main>"]
    parts.append("<h1>Social Sentiment → Next-Day Price Report</h1>")
    parts.append(f"<p class='muted'>Generated {now:%Y-%m-%d %H:%M} UTC · mode <b>{mode}</b> · prices: {meta.get('price_source')} · "
                 f"sentiment backend: {html.escape(meta.get('sentiment_backend', ''))} · window: last {cfg['lookback_days']} days · "
                 f"{meta.get('n_posts', 0)} posts/articles</p>")
    if mode == "demo":
        parts.append("<div class='banner'>DEMO MODE — social posts and news below are SYNTHETIC (generated locally, prefixed "
                     "[DEMO]). Prices are " + ("real yfinance bars." if meta.get("price_source") == "yfinance" else "synthetic.")
                     + " Model metrics here only prove the pipeline works; they say nothing about real predictive power. "
                     "Run with <code>--live</code> for real data.</div>")

    cols = ["ticker", "name", "current_price", "mentions", "sentiment", "direction", "position", "support", "resistance",
            "social_mentions", "news_mentions", "position_rationale"]
    for ex_id, ex in cfg["exchanges"].items():
        s = snapshot[snapshot.exchange == ex_id].copy()
        if s.empty:
            continue
        s["_d"] = s["direction"].map(DIR_ORDER)
        s = s.sort_values(["mentions", "sentiment", "_d"], ascending=[False, False, True])
        parts.append(f"<h2>{html.escape(ex['name'])} ({ex_id})</h2>")
        parts.append("<div class='card'><b>Ranking</b> <span class='muted'>(sorted by mentions, then sentiment, then direction; "
                     "click headers to re-sort). Prices in " + ex.get("currency", "") + f", as of {s['price_date'].max()}.</span>"
                     + _table(s, cols) + "</div>")
        parts.append("<div class='grid'><div class='card'>" + fig(_mentions_fig(s, f"{ex_id}: mentions coloured by sentiment"), 340)
                     + "</div><div class='card'>" + fig(_sent_fig(daily, list(s["ticker"]), f"{ex_id}: daily sentiment timeline"), 340)
                     + "</div></div>")
        cards = []
        for _, r in s.iterrows():
            cards.append("<div class='card'>" + fig(_price_fig(r["ticker"], r["name"], bars, r, oos, forecast), 360) + "</div>")
        parts.append("<div class='grid'>" + "".join(cards) + "</div>")

    parts.append("<h2>Next bar (n+1) forecast</h2><div class='card'>")
    if not forecast.empty:
        f = forecast.merge(snapshot[["ticker", "name", "direction", "position"]], on="ticker", how="left")
        parts.append(_table(f, ["ticker", "name", "asof_date", "asof_close", "prob_up", "pred_direction", "pred_ret",
                                "pred_close", "direction", "position"], nd=4))
        parts.append("<p class='muted'>Saved to data/predictions/. Run <code>python scripts/check_predictions.py</code> after the "
                     "next session closes to score these against actual closes.</p>")
    else:
        parts.append("<p class='muted'>Not enough labelled rows to forecast.</p>")
    parts.append("</div>")

    parts.append("<h2>Out-of-sample validation (walk-forward)</h2><div class='card'>")
    parts.append(_table(metrics, nd=3, cls="") if not metrics.empty else "<p class='muted'>Not enough dates for walk-forward.</p>")
    parts.append("<p class='muted'>Expanding window, pooled across tickers: for each test date d the models see only rows whose "
                 "next-day close was known by d's close. Direction = logistic regression on P(up); return = ridge regression; "
                 "price-only ablation isolates what sentiment adds. MAE/RMSE are next-day return errors in percentage points; "
                 "the naive no-change model is the bar to beat.</p></div>")
    if checks is not None and not checks.empty:
        parts.append("<h2>Checked past predictions</h2><div class='card'>" + _table(checks, nd=4) + "</div>")

    parts.append("<h2>Data sources & credit usage</h2><div class='card'>")
    parts.append(_table(pd.DataFrame(source_log), nd=0, cls="") if source_log else "<p class='muted'>Demo mode: no API calls, 0 credits used.</p>")
    parts.append("<p class='muted'>Apify runs are capped with <code>max_total_charge_usd</code> per actor (config.yaml → sources.*.max_usd) "
                 "and raw pulls are cached in data/raw/&lt;date&gt;/ so re-runs on the same day cost nothing. Exa: ~1 search per ticker "
                 "(10 searches per run). Free tiers (Apify $5/month, Exa signup credit) cover several full runs.</p></div>")

    parts.append("""<h2>Methods & limitations</h2><div class='card'><ul>
<li><b>Sources:</b> HK = LIHKG (keyword search) + public Telegram channels; KR = public Telegram channels + Daum News; Exa news for every ticker. Posts are matched to tickers with English + Chinese/Korean keyword lists.</li>
<li><b>Timestamps:</b> all converted to UTC, then mapped to the exchange's trading bar: posts after the local close (HK 16:00, KRX 15:30) or on holidays go to the next bar; posts after the latest close are held out as pending, never used for past bars.</li>
<li><b>Sentiment:</b> multilingual XLM-R model (score = P(pos) − P(neg)), lexicon fallback if unavailable; weighted by 1 + log(engagement) + ¼·log(views) + ¼·log(followers).</li>
<li><b>Support/resistance:</b> nearest confirmed swing low/high (±3 bars, last 60 bars) around the close; floor pivots (S1/R1) when no swing exists. Recomputed point-in-time for features.</li>
<li><b>Position rule:</b> long = bullish and in lower half of S/R range (or breakout); short = bearish and in upper half (or breakdown); otherwise none.</li>
<li><b>Limitations:</b> 10 days ≈ 7–8 trading bars, so the out-of-sample set is tiny and accuracy has very wide confidence intervals; results are not evidence of an edge. Telegram/Daum coverage of specific tickers is sparse and LIHKG has no follower counts; Futu, Naver discussion boards and KakaoTalk have no reliable Apify actors. yfinance data can lag or be revised.</li>
</ul></div>""")
    parts.append(f"<script>{SORT_JS}</script></main></body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
