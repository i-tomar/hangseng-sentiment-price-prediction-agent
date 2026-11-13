"""Score saved n+1 predictions against actual closes once they exist."""
from __future__ import annotations

import numpy as np
import pandas as pd


def check_predictions(preds: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """For each prediction, the actual is the first bar strictly after asof_date."""
    if preds.empty:
        return pd.DataFrame()
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"])
    rows = []
    for _, p in preds.iterrows():
        b = bars[(bars.ticker == p["ticker"]) & (bars.date > pd.Timestamp(p["asof_date"]))].sort_values("date")
        r = p.to_dict()
        if b.empty:
            r.update(status="pending", actual_date=None, actual_close=np.nan, actual_ret=np.nan, direction_hit=np.nan,
                     abs_err_pct=np.nan, naive_abs_err_pct=np.nan)
        else:
            a = b.iloc[0]
            actual_ret = a["close"] / p["asof_close"] - 1
            r.update(status="checked", actual_date=a["date"].date().isoformat(), actual_close=float(a["close"]),
                     actual_ret=float(actual_ret),
                     direction_hit=float((actual_ret > 0) == (p["pred_direction"] == "up")),
                     abs_err_pct=float(abs(a["close"] - p["pred_close"]) / a["close"] * 100),
                     naive_abs_err_pct=float(abs(a["close"] - p["asof_close"]) / a["close"] * 100))
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(checked: pd.DataFrame) -> dict:
    c = checked[checked.status == "checked"] if not checked.empty else checked
    if c.empty:
        return {"checked": 0}
    return {"checked": int(len(c)), "pending": int((checked.status == "pending").sum()),
            "direction_hit_rate": float(c["direction_hit"].mean()), "model_MAE_pct": float(c["abs_err_pct"].mean()),
            "naive_MAE_pct": float(c["naive_abs_err_pct"].mean())}
