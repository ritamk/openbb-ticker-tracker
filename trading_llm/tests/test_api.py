"""API layer tests for the Trading LLM service."""

from __future__ import annotations

import api.app as api_module
import pytest  # type: ignore[import]
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def clear_job_store() -> None:
    """Ensure the in-memory job store is reset between tests."""

    with api_module._job_lock:  # type: ignore[attr-defined]
        api_module._job_store.clear()  # type: ignore[attr-defined]
    yield
    with api_module._job_lock:  # type: ignore[attr-defined]
        api_module._job_store.clear()  # type: ignore[attr-defined]


@pytest.fixture()
def client() -> TestClient:
    return TestClient(api_module.app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_news_endpoint_returns_analysis(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_news(symbol: str, limit: int = 10):  # pragma: no cover - simple stub
        return {
            "symbol_headlines": [
                {"title": "Sample headline", "summary": "Summary", "source": "mock"}
            ]
        }

    def fake_analyze_news(symbol: str, headlines_dict, use_cache: bool = True):  # pragma: no cover
        return {
            "sentiment": "positive",
            "confidence": 0.8,
            "summary": "Positive outlook",
            "global": {"sentiment": "positive", "confidence": 0.7, "counts": {"bullish": 1}},
        }

    monkeypatch.setattr(api_module, "get_news", fake_get_news)
    monkeypatch.setattr(api_module, "analyze_news_agent", fake_analyze_news)

    response = client.post("/analyze/news", json={"tickers": ["AAPL"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["ticker"] == "AAPL"
    assert payload["results"][0]["analysis"]["sentiment"] == "positive"


def test_generate_ta_signals_returns_snapshot(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_technical_data(symbol: str, lookback_days: int = 200):  # pragma: no cover
        return {
            "close": 100.0,
            "rsi": 55.0,
            "macd_signal": "none",
            "_df": None,
        }

    def fake_analyze_technical(symbol: str, ta_data=None, use_csv: bool = False):  # pragma: no cover
        return {"signal": "bullish", "confidence": 0.75, "rationale": "Mock analysis"}

    monkeypatch.setattr(api_module, "get_technical_data", fake_get_technical_data)
    monkeypatch.setattr(api_module, "analyze_technical", fake_analyze_technical)

    response = client.post("/signals/ta", json={"ticker": "AAPL"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["analysis"]["signal"] == "bullish"
    assert payload["snapshot"]["close"] == 100.0


def test_trade_run_sync_returns_result(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    original_mode = api_module.settings.run_trades_in_background
    api_module.settings.run_trades_in_background = False

    def fake_run_cycle(symbol: str):  # pragma: no cover
        return {"symbol": symbol, "meta": {"timings": {"total": 1.23}}}

    monkeypatch.setattr(api_module, "run_trading_cycle", fake_run_cycle)

    try:
        response = client.post(
            "/trade/run",
            json={"ticker": "AAPL", "cash_usd": 1000, "risk_budget": 0.05},
        )
    finally:
        api_module.settings.run_trades_in_background = original_mode

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["result"]["meta"]["request"]["cash_usd"] == 1000

    # Ensure job retrieval works
    job_id = payload["job_id"]
    job_response = client.get(f"/trade/jobs/{job_id}")
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "completed"


def test_trade_job_status_not_found(client: TestClient) -> None:
    response = client.get("/trade/jobs/does-not-exist")
    assert response.status_code == 404

