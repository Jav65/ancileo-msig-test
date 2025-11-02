from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:  # Optional dependency guard during import time
    import xgboost as xgb
except ImportError:  # pragma: no cover - dependency validated in runtime guard
    xgb = None  # type: ignore[assignment]

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from ..config import get_settings
from ..utils.logging import logger


MONTH_MAP = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _to_month_label(month: Optional[int | str | date | datetime]) -> str:
    if isinstance(month, str):
        text = month.strip()
        if not text:
            return "Jan"
        if len(text) == 3:
            return text.title()
        try:
            parsed = datetime.strptime(text, "%B")
            return MONTH_MAP[parsed.month]
        except ValueError:
            try:
                parsed = datetime.strptime(text, "%b")
                return MONTH_MAP[parsed.month]
            except ValueError:
                return text[:3].title()

    if isinstance(month, (date, datetime)):
        return MONTH_MAP.get(month.month, "Jan")

    if isinstance(month, int) and month in MONTH_MAP:
        return MONTH_MAP[month]

    return "Jan"


def _clamp_probability(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.01, min(0.95, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


class IncrementalLabelEncoder:
    """Lightweight label encoder that tolerates unseen categories."""

    def __init__(self) -> None:
        self._mapping: Dict[str, int] = {}

    def fit(self, values: pd.Series) -> None:
        self.transform(values)

    def transform(self, values: pd.Series) -> np.ndarray:
        normalized = values.astype(str).fillna("Unknown")
        return np.array([self._ensure(value) for value in normalized], dtype=np.int32)

    def _ensure(self, value: str) -> int:
        if value not in self._mapping:
            self._mapping[value] = len(self._mapping)
        return self._mapping[value]


class TravelInsurancePredictor:
    """Predictive model for travel insurance claims."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.risk_model = None
        self.amount_model: Optional[xgb.XGBRegressor] = None  # type: ignore[attr-defined]
        self.encoders: Dict[str, IncrementalLabelEncoder] = {}
        self.feature_importance: Optional[pd.DataFrame] = None
        self.insights: Dict[str, Any] = {}
        self.df_claims: pd.DataFrame = pd.DataFrame()
        self._prepared = False
        self._trained = False
        self._base_claim_amount: float = 0.0

    def prepare_features(self) -> None:
        if self._prepared:
            return

        if "accident_date" not in self.df.columns:
            raise ValueError("Claims dataset missing 'accident_date' column")

        self.df["accident_date"] = pd.to_datetime(
            self.df["accident_date"], errors="coerce"
        )
        self.df.dropna(subset=["accident_date"], inplace=True)

        self.df["destination"] = (
            self.df.get("destination", "Unknown").astype(str).str.strip().replace("", "Unknown")
        )

        self.df["net_incurred"] = pd.to_numeric(
            self.df.get("net_incurred", 0), errors="coerce"
        ).fillna(0)
        if "gross_incurred" in self.df.columns:
            self.df["gross_incurred"] = pd.to_numeric(
                self.df["gross_incurred"], errors="coerce"
            ).fillna(self.df["net_incurred"])
        else:
            self.df["gross_incurred"] = self.df["net_incurred"]

        self.df["month"] = self.df["accident_date"].dt.month.map(MONTH_MAP)
        self.df["season"] = self.df["month"].apply(self._get_season)
        self.df["has_claim"] = (self.df["net_incurred"] > 0).astype(int)

        self.df_claims = self.df[self.df["has_claim"] == 1].copy()
        self._base_claim_amount = _safe_float(self.df_claims["gross_incurred"].mean(), 0.0)

        self._prepared = True

    def _get_season(self, month_value: Any) -> str:
        month = str(month_value).strip().title()
        winter = {"Dec", "Jan", "Feb"}
        spring = {"Mar", "Apr", "May"}
        summer = {"Jun", "Jul", "Aug"}
        if month in winter:
            return "Winter"
        if month in spring:
            return "Spring"
        if month in summer:
            return "Summer"
        return "Fall"

    def predict_risk(self, age: int, activity: Optional[str], destination: str, month: str) -> float:
        base_rate = 0.18

        if destination in {"China", "Malaysia", "Thailand", "Vietnam"}:
            base_rate *= 1.7
        elif destination in {"Japan", "Indonesia"}:
            base_rate *= 1.3

        if month in {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"}:
            base_rate *= 1.4

        if age < 18 or age > 65:
            base_rate *= 1.5

        risky_activities = {"skiing", "scuba diving", "mountain climbing"}
        if activity and activity.strip().lower() in risky_activities:
            base_rate *= 2.0

        return _clamp_probability(base_rate)

    def train_amount_model(self) -> None:
        if not self._prepared:
            self.prepare_features()

        if self.df_claims.empty:
            raise ValueError("No paid claims available to train amount model")

        if xgb is None:
            raise ImportError("xgboost is not installed")

        features = ["destination", "month", "season"]

        encoded = self.df_claims[features].copy().astype(str)
        for column in features:
            encoder = self.encoders.setdefault(column, IncrementalLabelEncoder())
            encoder.fit(encoded[column])
            encoded[column] = encoder.transform(encoded[column])

        X = encoded
        y = self.df_claims["gross_incurred"]

        if len(X) < 20:
            raise ValueError("Insufficient claim records to train XGBoost model")

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = xgb.XGBRegressor(  # type: ignore[call-arg]
            objective="reg:squarederror",
            max_depth=4,
            learning_rate=0.1,
            n_estimators=200,
            random_state=42,
            eval_metric="rmse",
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        self.amount_model = model
        self.feature_importance = pd.DataFrame(
            {
                "feature": features,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        self.insights = {
            "rmse": float(rmse),
            "mae": float(mae),
            "y_test_mean": float(np.mean(y_test)),
            "y_pred_mean": float(np.mean(y_pred)),
        }
        self._trained = True

    def predict(self, age: int, activity: Optional[str], destination: str, month: str) -> Dict[str, Any]:
        if not self._trained or self.amount_model is None:
            raise RuntimeError("Amount model is not trained")

        season = self._get_season(month)
        encoded = pd.DataFrame(
            {
                "destination": [destination],
                "month": [month],
                "season": [season],
            }
        )

        for column in encoded.columns:
            encoder = self.encoders.setdefault(column, IncrementalLabelEncoder())
            encoder.fit(encoded[column])
            encoded[column] = encoder.transform(encoded[column])

        claim_probability = self.predict_risk(age, activity, destination, month)
        claim_amount = float(self.amount_model.predict(encoded)[0])

        return {
            "destination": destination,
            "month": month,
            "season": season,
            "claim_probability": claim_probability,
            "expected_amount": max(0.0, claim_amount),
        }


@dataclass
class PredictorState:
    refreshed_at: str
    training_rows: int
    claim_rows: int
    baseline_amount: float
    feature_importance: Optional[list[dict[str, Any]]]


class TravelRiskPredictorTool:
    """Wrapper that loads claims data and surfaces risk forecasts."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._lock = threading.Lock()
        self._predictor: Optional[TravelInsurancePredictor] = None
        self._state: Optional[PredictorState] = None
        self._last_inputs: Optional[Dict[str, Any]] = None

    # Public API -----------------------------------------------------
    def predict(
        self,
        *,
        destination: str,
        activity: Optional[str] = None,
        departure_date: Optional[str] = None,
        month: Optional[str] = None,
        age: Optional[int] = None,
        date_of_birth: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_month = self._resolve_month(month, departure_date)
        resolved_age = self._resolve_age(age, date_of_birth, departure_date)

        predictor = self._ensure_predictor()
        input_payload = {
            "destination": destination,
            "activity": activity,
            "month": resolved_month,
            "age": resolved_age,
            "departure_date": departure_date,
            "date_of_birth": date_of_birth,
        }
        self._last_inputs = input_payload

        if predictor is None:
            logger.warning("travel_risk_predictor.fallback", reason="predictor_unavailable")
            fallback = self._fallback_estimate(
                destination=destination,
                month=resolved_month,
                age=resolved_age,
                activity=activity,
            )
            return {
                "status": "fallback",
                "input": input_payload,
                "prediction": fallback,
                "model_state": self._state_as_dict(),
                "notes": [
                    "Model unavailable; using heuristic estimate based on baseline risk multipliers."
                ],
            }

        try:
            prediction = predictor.predict(
                age=resolved_age,
                activity=activity,
                destination=destination,
                month=resolved_month,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("travel_risk_predictor.predict_failed", error=str(exc))
            fallback = self._fallback_estimate(
                destination=destination,
                month=resolved_month,
                age=resolved_age,
                activity=activity,
            )
            return {
                "status": "error",
                "input": input_payload,
                "prediction": fallback,
                "model_state": self._state_as_dict(),
                "notes": [
                    "Model prediction failed; reverted to heuristic estimate.",
                    str(exc),
                ],
            }

        return {
            "status": "ok",
            "input": input_payload,
            "prediction": prediction,
            "model_state": self._state_as_dict(),
            "insights": predictor.insights,
            "notes": [],
        }

    # Internal helpers ----------------------------------------------
    def _ensure_predictor(self) -> Optional[TravelInsurancePredictor]:
        with self._lock:
            if self._predictor is not None:
                return self._predictor

            df = self._load_claims_dataframe()
            if df is None or df.empty:
                logger.warning("travel_risk_predictor.data_missing")
                self._state = None
                return None

            predictor = TravelInsurancePredictor(df)
            try:
                predictor.prepare_features()
                predictor.train_amount_model()
            except Exception as exc:
                logger.exception("travel_risk_predictor.training_failed", error=str(exc))
                self._state = PredictorState(
                    refreshed_at=_now_iso(),
                    training_rows=len(df),
                    claim_rows=len(predictor.df_claims),
                    baseline_amount=_safe_float(predictor._base_claim_amount),
                    feature_importance=None,
                )
                return None

            feature_importance = None
            if predictor.feature_importance is not None:
                feature_importance = predictor.feature_importance.to_dict(orient="records")

            self._state = PredictorState(
                refreshed_at=_now_iso(),
                training_rows=len(df),
                claim_rows=len(predictor.df_claims),
                baseline_amount=_safe_float(predictor._base_claim_amount),
                feature_importance=feature_importance,
            )

            self._predictor = predictor
            return predictor

    def _load_claims_dataframe(self) -> Optional[pd.DataFrame]:
        host = self._settings.claims_db_host
        dbname = self._settings.claims_db_name
        user = self._settings.claims_db_user
        password = self._settings.claims_db_password
        port = self._settings.claims_db_port
        table = self._settings.claims_db_table

        if host and dbname and user and password:
            dsn = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
            try:
                engine: Engine = create_engine(dsn, pool_pre_ping=True)
                with engine.connect() as connection:
                    logger.info(
                        "travel_risk_predictor.loading_from_db",
                        table=table,
                        host=host,
                    )
                    df = pd.read_sql(
                        f"SELECT * FROM {table};",
                        connection,
                        parse_dates=["accident_date", "report_date", "closed_date"],
                    )
                    return df
            except Exception as exc:  # pragma: no cover - network/db dependent
                logger.exception("travel_risk_predictor.db_load_failed", error=str(exc))

        path = self._settings.claims_data_path
        if path.exists():
            try:
                if path.suffix in {".parquet", ".pq"}:
                    return pd.read_parquet(path)
                if path.suffix == ".csv":
                    return pd.read_csv(path, parse_dates=["accident_date"], infer_datetime_format=True)
            except Exception as exc:  # pragma: no cover - file format guard
                logger.exception("travel_risk_predictor.file_load_failed", error=str(exc))

        return None

    def _resolve_month(self, month: Optional[str], departure_date: Optional[str]) -> str:
        if departure_date:
            parsed = self._parse_date(departure_date)
            if parsed:
                return _to_month_label(parsed)
        if month:
            return _to_month_label(month)
        return _to_month_label(datetime.now())

    def _resolve_age(
        self,
        provided_age: Optional[int],
        date_of_birth: Optional[str],
        departure_date: Optional[str],
    ) -> int:
        if isinstance(provided_age, int) and provided_age > 0:
            return provided_age

        dob = self._parse_date(date_of_birth)
        travel_date = self._parse_date(departure_date) or date.today()
        if dob:
            years = travel_date.year - dob.year
            if (travel_date.month, travel_date.day) < (dob.month, dob.day):
                years -= 1
            return max(1, years)

        return 35

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[date]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        return None

    def _fallback_estimate(
        self,
        *,
        destination: str,
        month: str,
        age: int,
        activity: Optional[str],
    ) -> Dict[str, Any]:
        helper = TravelInsurancePredictor(pd.DataFrame())
        probability = helper.predict_risk(
            age=age,
            activity=activity,
            destination=destination,
            month=month,
        )
        baseline_amount = self._state.baseline_amount if self._state else 12000.0
        return {
            "destination": destination,
            "month": month,
            "season": helper._get_season(month),  # type: ignore[arg-type]
            "claim_probability": probability,
            "expected_amount": baseline_amount,
        }

    def _state_as_dict(self) -> Optional[Dict[str, Any]]:
        if not self._state:
            return None
        return {
            "refreshed_at": self._state.refreshed_at,
            "training_rows": self._state.training_rows,
            "claim_rows": self._state.claim_rows,
            "baseline_amount": self._state.baseline_amount,
            "feature_importance": self._state.feature_importance,
        }

