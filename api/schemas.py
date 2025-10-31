from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class TickerDataRequest(BaseModel):
    """Request body for ticker data lookup."""

    tickers: List[str] = Field(..., min_items=1, description="Symbols to fetch")
    timeframes: Optional[List[str]] = Field(
        default=None,
        description="Optional timeframe overrides; defaults to configuration",
    )

    @validator("tickers", each_item=True)
    def _strip_symbol(cls, value: str) -> str:  # noqa: N805 - pydantic validator signature
        stripped = value.strip()
        if not stripped:
            raise ValueError("Ticker symbols must be non-empty after stripping")
        return stripped

    @validator("timeframes")
    def _normalize_timeframes(
        cls, value: Optional[List[str]]
    ) -> Optional[List[str]]:  # noqa: N805 - pydantic validator signature
        if value is None:
            return None
        cleaned = [tf.strip() for tf in value if tf and tf.strip()]
        unique = list(dict.fromkeys(cleaned))
        return unique or None


class TradingRun(BaseModel):
    symbol: str
    generated_at: str
    results: List[Dict[str, Any]]
    news: Optional[Dict[str, Any]] = None
    news_payload: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class TickerDataResponse(BaseModel):
    requested_at: str
    timeframes: List[str]
    runs: List[TradingRun]

