#!/usr/bin/env python
"""Compare saved n+1 predictions with actual closes (run after the next session closes).

  python scripts/check_predictions.py                 # live predictions log
  python scripts/check_predictions.py --mode demo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, load_config, ticker_table  # noqa: E402
from src.ingest.prices import fetch_yfinance  # noqa: E402
from src.models.check import check_predictions, summarise  # noqa: E402


pd.set_option("display.float_format", "{:,.4f}".format)
pd.set_option("display.width", 200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="live", choices=["live", "demo"])
    ap.add_argument("--file", default=None, help="specific predictions CSV (default: <mode>_predictions_log.csv)")
    args = ap.parse_args()
    path = Path(args.file) if args.file else ROOT / "data" / "predictions" / f"{args.mode}_predictions_log.csv"
    if not path.exists():
        raise SystemExit(f"No predictions found at {path}. Run scripts/run_pipeline.py first.")
    preds = pd.read_csv(path)
    if "price_source" in preds:
        preds = preds[preds["price_source"] == "yfinance"]
    cfg = load_config()
    syms = sorted(set(preds["ticker"]) & {t["symbol"] for t in ticker_table(cfg)})
    bars = fetch_yfinance(syms, 30)
    checked = check_predictions(preds, bars)
    out = path.with_name(path.stem.replace("_log", "") + "_checked.csv")
    checked.to_csv(out, index=False)
    cols = ["ticker", "asof_date", "asof_close", "pred_direction", "pred_close", "status", "actual_date", "actual_close",
            "direction_hit", "abs_err_pct", "naive_abs_err_pct"]
    print(checked[cols].to_string(index=False))
    print("\nSummary:", summarise(checked))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
