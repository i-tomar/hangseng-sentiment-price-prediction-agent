"""Walk-forward (expanding window) out-of-sample evaluation + n+1 forecast.

At each test date d the models are fit ONLY on rows whose target (next close) was
already known at d's close (next_date <= d), then predict every ticker's row at d.
Pooled across tickers (10 days x 1 ticker is far too few rows to fit per ticker).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build import PRICE_FEATURES, SENT_FEATURES

log = logging.getLogger(__name__)
FULL = SENT_FEATURES + PRICE_FEATURES + ["is_kr"]
PRICE_ONLY = PRICE_FEATURES + ["is_kr"]


def model_frame(panel: pd.DataFrame, social_start: pd.Timestamp | None) -> pd.DataFrame:
    df = panel.copy()
    df["is_kr"] = df["ticker"].str.endswith(".KS").astype(float)
    if social_start is not None:
        df = df[df["date"] >= social_start]
    return df.dropna(subset=PRICE_FEATURES).reset_index(drop=True)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], C: float, alpha: float):
    X, Xt = train[feats].to_numpy(float), test[feats].to_numpy(float)
    y_up = train["target_up"].to_numpy(int)
    if len(np.unique(y_up)) < 2:  # degenerate window: predict the only class seen
        prob = np.full(len(test), float(y_up[0]))
    else:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
        clf.fit(X, y_up)
        prob = clf.predict_proba(Xt)[:, 1]
    reg = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    reg.fit(X, train["target_ret"].to_numpy(float))
    return prob, reg.predict(Xt)


def walk_forward(df: pd.DataFrame, min_train_dates: int = 3, C: float = 0.5, alpha: float = 1.0) -> pd.DataFrame:
    labelled = df.dropna(subset=["target_ret"])
    dates = sorted(labelled["date"].unique())
    preds = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d in dates[min_train_dates:]:
            train = labelled[labelled["next_date"] <= d]
            test = labelled[labelled["date"] == d]
            if train["date"].nunique() < min_train_dates or test.empty:
                continue
            p_full, r_full = _fit_predict(train, test, FULL, C, alpha)
            p_px, r_px = _fit_predict(train, test, PRICE_ONLY, C, alpha)
            majority_up = float(train["target_up"].mean() >= 0.5)
            t = test[["date", "next_date", "ticker", "close", "next_close", "target_ret", "target_up", "ret_1d"]].copy()
            t["n_train"] = len(train)
            t["prob_up"], t["pred_ret"] = p_full, r_full
            t["prob_up_price_only"], t["pred_ret_price_only"] = p_px, r_px
            t["pred_close"] = t["close"] * (1 + t["pred_ret"])
            t["base_majority_up"] = majority_up
            t["base_momentum_up"] = (t["ret_1d"] > 0).astype(float)
            preds.append(t)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def metrics(oos: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample metrics for models vs naive baselines (returns in %)."""
    if oos.empty:
        return pd.DataFrame()
    y, r = oos["target_up"].to_numpy(), oos["target_ret"].to_numpy() * 100

    def reg_stats(pred_ret):
        e = r - pred_ret * 100
        return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.sign(pred_ret) == np.sign(r)))

    rows = []
    for name, prob, pret in [
        ("Sentiment+price model", oos["prob_up"], oos["pred_ret"]),
        ("Price-only model (ablation)", oos["prob_up_price_only"], oos["pred_ret_price_only"]),
    ]:
        mae, rmse, hit = reg_stats(pret.to_numpy())
        rows.append({"model": name, "direction_accuracy": float(np.mean((prob.to_numpy() >= 0.5) == y)),
                     "ridge_directional_hit": hit, "MAE_ret_pct": mae, "RMSE_ret_pct": rmse})
    mae0, rmse0 = float(np.mean(np.abs(r))), float(np.sqrt(np.mean(r ** 2)))
    rows.append({"model": "Naive: no change (random walk)", "direction_accuracy": np.nan,
                 "ridge_directional_hit": np.nan, "MAE_ret_pct": mae0, "RMSE_ret_pct": rmse0})
    rows.append({"model": "Baseline: majority class", "direction_accuracy": float(np.mean(oos["base_majority_up"] == y)),
                 "ridge_directional_hit": np.nan, "MAE_ret_pct": np.nan, "RMSE_ret_pct": np.nan})
    rows.append({"model": "Baseline: momentum (sign of today's return)",
                 "direction_accuracy": float(np.mean(oos["base_momentum_up"] == y)),
                 "ridge_directional_hit": np.nan, "MAE_ret_pct": np.nan, "RMSE_ret_pct": np.nan})
    m = pd.DataFrame(rows)
    m["n_oos_predictions"] = len(oos)
    m["n_oos_dates"] = oos["date"].nunique()
    return m


def forecast_next(df: pd.DataFrame, C: float = 0.5, alpha: float = 1.0) -> pd.DataFrame:
    """Fit on every labelled row, predict n+1 for each ticker's latest bar."""
    labelled = df.dropna(subset=["target_ret"])
    latest = df.sort_values("date").groupby("ticker").tail(1)
    latest = latest[latest["target_ret"].isna()]
    if labelled.empty or latest.empty:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob, ret = _fit_predict(labelled, latest, FULL, C, alpha)
    out = latest[["ticker", "date", "close"]].rename(columns={"date": "asof_date", "close": "asof_close"}).copy()
    out["prob_up"] = prob
    out["pred_ret"] = ret
    out["pred_close"] = out["asof_close"] * (1 + out["pred_ret"])
    out["pred_direction"] = np.where(out["prob_up"] >= 0.5, "up", "down")
    out["n_train_rows"] = len(labelled)
    out["asof_date"] = pd.to_datetime(out["asof_date"]).dt.date.astype(str)
    return out.reset_index(drop=True)
