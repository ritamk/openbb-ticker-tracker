from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from ..schemas import TickerDataRequest, TickerDataResponse
from trading_llm.core import config
from trading_llm.core.orchestrator import run_trading_cycle


router = APIRouter(prefix="/v1/tickers", tags=["tickers"])


@router.post("/data", response_model=TickerDataResponse)
def get_ticker_data(request: TickerDataRequest) -> TickerDataResponse:
    """Return price history + indicators for each requested ticker/timeframe."""

    timeframes = request.timeframes or list(config.TIMEFRAMES)
    runs = []
    for symbol in request.tickers:
        run_result = run_trading_cycle(symbol, timeframes=timeframes, save_log=False)
        runs.append(run_result)

    response = TickerDataResponse(
        requested_at=datetime.now(timezone.utc).isoformat(),
        timeframes=timeframes,
        runs=runs,
    )
    return response

