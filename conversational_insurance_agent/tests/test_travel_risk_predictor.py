from __future__ import annotations

from src.config import get_settings
from src.tools.travel_risk_predictor import TravelRiskPredictorTool


def test_travel_risk_predictor_tool_returns_prediction(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CLAIMS_DB_HOST", raising=False)
    monkeypatch.delenv("CLAIMS_DB_NAME", raising=False)
    monkeypatch.delenv("CLAIMS_DB_USER", raising=False)
    monkeypatch.delenv("CLAIMS_DB_PASSWORD", raising=False)
    missing_path = tmp_path / "claims_missing.parquet"
    monkeypatch.setenv("CLAIMS_DATA_PATH", str(missing_path))

    # Ensure settings cache picks up the cleared environment
    get_settings.cache_clear()  # type: ignore[attr-defined]

    tool = TravelRiskPredictorTool()
    result = tool.predict(
        destination="Japan",
        activity="skiing",
        departure_date="2025-12-15",
        date_of_birth="1985-04-10",
    )

    assert "prediction" in result
    prediction = result["prediction"]
    assert prediction["destination"] == "Japan"
    assert "claim_probability" in prediction
    assert "expected_amount" in prediction
    assert isinstance(prediction["claim_probability"], float)
    assert isinstance(prediction["expected_amount"], float)
