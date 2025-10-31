from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..config import get_settings
from ..utils.logging import logger

class ClaimsInsightTool:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._data = self._load_data(self._settings.claims_data_path)

    def _load_data(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            logger.warning("claims_insights.data_missing", path=str(path))
            return pd.DataFrame(
                columns=[
                    "destination",
                    "activity",
                    "season",
                    "claim_amount",
                    "plan",
                    "age_band",
                ]
            )

        if path.suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported claims data format: {path.suffix}")

        df["claim_amount"] = pd.to_numeric(df["claim_amount"], errors="coerce").fillna(0)
        return df

    def risk_summary(
        self, *, destination: Optional[str] = None, activity: Optional[str] = None
    ) -> Dict[str, Any]:
        df = self._data
        subset = df

        filters = {}
        if destination:
            subset = subset[subset["destination"].str.contains(destination, case=False, na=False)]
            filters["destination"] = destination
        if activity:
            subset = subset[subset["activity"].str.contains(activity, case=False, na=False)]
            filters["activity"] = activity

        if subset.empty:
            return {
                "filters": filters,
                "message": "No claims data available for the specified filters.",
            }

        summary = subset.agg(
            claim_count=("claim_amount", "count"),
            avg_claim=("claim_amount", "mean"),
            p90_claim=("claim_amount", lambda s: s.quantile(0.9)),
            max_claim=("claim_amount", "max"),
        )

        seasonality = (
            subset.groupby("season")["claim_amount"]
            .agg(["count", "mean"])
            .sort_values("count", ascending=False)
            .head(3)
            .reset_index()
            .to_dict(orient="records")
        )

        top_activities = (
            subset.groupby("activity")["claim_amount"].mean().sort_values(ascending=False).head(5)
        )

        return {
            "filters": filters,
            "summary": {
                "claim_count": int(summary["claim_count"]),
                "average_claim": round(summary["avg_claim"], 2),
                "p90_claim": round(summary["p90_claim"], 2),
                "max_claim": round(summary["max_claim"], 2),
            },
            "seasonality": seasonality,
            "top_activities": top_activities.to_dict(),
        }

    def recommend_plan(
        self,
        *,
        destination: Optional[str],
        activity: Optional[str],
        trip_cost: Optional[float],
    ) -> Dict[str, Any]:
        summary = self.risk_summary(destination=destination, activity=activity)
        if "summary" not in summary:
            return {
                "recommendation": "silver",
                "reason": "Default recommendation due to limited data.",
            }

        avg_claim = summary["summary"]["average_claim"]
        p90_claim = summary["summary"]["p90_claim"]

        if p90_claim > 50000:
            tier = "platinum"
            reason = "High 90th percentile claim amount; recommend premium medical coverage"
        elif avg_claim > 20000:
            tier = "gold"
            reason = "Elevated average claim cost; gold tier balances value and protection"
        else:
            tier = "silver"
            reason = "Moderate claim history; silver tier suffices for most travelers"

        if trip_cost and trip_cost > p90_claim:
            reason += " and upgrade trip cancellation coverage to match trip cost."

        return {
            "filters": summary.get("filters"),
            "summary": summary.get("summary"),
            "seasonality": summary.get("seasonality"),
            "recommendation": tier,
            "reason": reason,
        }
