"""ML evaluation pack for the report (docs/05-ml-plan.md protocol).

1. Forecast: Prophet vs ARIMA vs naive seasonal on each target (90-day holdout).
2. Anomaly detection: precision/recall/F1 vs seeds/injected_anomalies.json (±1 day tolerance).

Writes docs/completions/assets/phase-4/ml-evaluation.json and prints a summary.

Usage: uv run python scripts/evaluate_ml.py
"""

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.database import get_session_factory  # noqa: E402
from app.services.ml import anomaly, forecasting  # noqa: E402
from app.services.ml.features import load_series  # noqa: E402
from app.services.ml.registry import FORECAST_TARGETS  # noqa: E402

ASSETS = ROOT.parent / "docs" / "completions" / "assets" / "phase-4"
LABELS = json.loads((ROOT / "seeds" / "injected_anomalies.json").read_text())


def score_anomalies(flagged_dates: set[str], metric: str) -> dict:
    truth = {
        label["date"]
        for label in LABELS
        if label["metric"] == ("revenue" if metric == "revenue_daily" else "expense_total")
    }

    def match(day: str, pool: set[str]) -> bool:
        ts = pd.Timestamp(day)
        return any(abs((ts - pd.Timestamp(t)).days) <= 1 for t in pool)

    tp = sum(1 for t in truth if match(t, flagged_dates))
    fn = len(truth) - tp
    fp = sum(1 for d in flagged_dates if not match(d, truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "labelled": len(truth),
        "detected": len(flagged_dates),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


async def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    report: dict = {"forecasting": {}, "anomaly_detection": {}}

    async with get_session_factory()() as db:
        for target in FORECAST_TARGETS:
            frame = await load_series(db, target)
            evaluations = forecasting.evaluate_candidates(frame)
            report["forecasting"][target] = {e.model_name: e.metrics for e in evaluations}
            print(f"\n== {target} (holdout {forecasting.HOLDOUT_DAYS}d) ==")
            for e in evaluations:
                print(
                    f"  {e.model_name:>15}: MAPE {e.metrics['mape']:>7.2f}%  "
                    f"RMSE {e.metrics['rmse']:>12,.0f}  MAE {e.metrics['mae']:>12,.0f}"
                )

        for target in ("revenue_daily", "expenses_daily"):
            frame = await load_series(db, target)
            flagged = anomaly.detect_frame(frame, anomaly.TARGET_CONFIG[target]["season"])
            dates = {row["ds"].date().isoformat() for _, row in flagged.iterrows()}
            scores = score_anomalies(dates, target)
            report["anomaly_detection"][target] = scores
            print(f"\n== anomalies: {target} ==")
            print(
                f"  precision {scores['precision']}  recall {scores['recall']}  "
                f"F1 {scores['f1']}  (tp={scores['tp']} fp={scores['fp']} fn={scores['fn']})"
            )

    out = ASSETS / "ml-evaluation.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
