"""FastAPI application exposing trading LLM functionality."""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.config import get_settings

try:
    import config as project_config
except ImportError:  # pragma: no cover - config should exist, but keep fallback handy
    class _DefaultConfig:
        TA_BARS = 48
        TA_PREC = 3

    project_config = _DefaultConfig()  # type: ignore

from ta_builder import build_ta_csv
from trading_llm.agents.news_analyst import analyze_news as analyze_news_agent
from trading_llm.agents.technical_analyst import analyze_technical
from trading_llm.core.orchestrator import run_trading_cycle
from trading_llm.data.fetchers import get_news, get_technical_data


app = FastAPI(title="Trading LLM API", version="0.1.0")

settings = get_settings()

logger = logging.getLogger("trading_api")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(file_handler)
    logger.propagate = False

_job_store: Dict[str, Dict[str, Any]] = {}
_job_lock = Lock()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _create_job(ticker: str) -> Dict[str, Any]:
    job_id = uuid4().hex
    timestamp = _now_iso()
    record = {
        "job_id": job_id,
        "ticker": ticker,
        "status": "queued",
        "created_at": timestamp,
        "updated_at": timestamp,
        "result": None,
        "error": None,
    }
    with _job_lock:
        _job_store[job_id] = record
    return copy.deepcopy(record)


def _update_job(job_id: str, **fields: Any) -> Dict[str, Any]:
    with _job_lock:
        job = _job_store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.update(fields)
        job["updated_at"] = _now_iso()
        return copy.deepcopy(job)


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _job_lock:
        job = _job_store.get(job_id)
        return copy.deepcopy(job) if job else None


def _public_job_view(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "ticker": job["ticker"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "result": job.get("result"),
        "error": job.get("error"),
    }


def _execute_trade_job(job_id: str, ticker: str, request_meta: Dict[str, Any]) -> None:
    try:
        _update_job(job_id, status="running")
        logger.info("Trade job started", extra={"job_id": job_id, "ticker": ticker})
    except KeyError:  # pragma: no cover - guard against race conditions
        logger.warning("Trade job missing from store", extra={"job_id": job_id})
        return

    try:
        pipeline_result = run_trading_cycle(ticker)
        augmented_result = {
            **pipeline_result,
            "meta": {
                **pipeline_result.get("meta", {}),
                "request": request_meta,
            },
        }
        _update_job(job_id, status="completed", result=augmented_result, error=None)
        logger.info("Trade job completed", extra={"job_id": job_id, "ticker": ticker})
    except Exception as exc:  # pragma: no cover - ensure job failure is recorded
        try:
            _update_job(job_id, status="failed", error=str(exc))
        except KeyError:
            logger.warning("Failed to update missing job", extra={"job_id": job_id})
        logger.exception("Trade job failed", extra={"job_id": job_id, "ticker": ticker})


class HealthResponse(BaseModel):
    status: str


class NewsRequest(BaseModel):
    tickers: List[str]
    limit: Optional[int] = None
    use_cache: Optional[bool] = None


class Headline(BaseModel):
    source: Optional[str] = None
    date: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    url: Optional[str] = None


class NewsAnalysisBreakdown(BaseModel):
    sentiment: str
    confidence: float
    counts: Dict[str, float] = Field(default_factory=dict)


class NewsAnalysis(BaseModel):
    sentiment: str
    confidence: float
    summary: str
    method: Optional[str] = None
    cached: Optional[bool] = None
    global_: Optional[NewsAnalysisBreakdown] = Field(default=None, alias="global")
    india: Optional[NewsAnalysisBreakdown] = None
    symbol: Optional[NewsAnalysisBreakdown] = None

    class Config:
        allow_population_by_field_name = True


class NewsResult(BaseModel):
    ticker: str
    headlines: Dict[str, List[Headline]] = Field(default_factory=dict)
    analysis: Optional[NewsAnalysis] = None
    error: Optional[str] = None


class NewsResponse(BaseModel):
    results: List[NewsResult]


class TARequest(BaseModel):
    ticker: str
    lookback: Optional[int] = None
    keep_rows: Optional[int] = None
    precision: Optional[int] = None


class TechnicalCSV(BaseModel):
    spec: str
    data: str


class TAResponse(BaseModel):
    ticker: str
    snapshot: Dict[str, Any]
    analysis: Dict[str, Any]
    csv: Optional[TechnicalCSV] = None


class TradeRequest(BaseModel):
    ticker: str
    cash_usd: float
    risk_budget: float = 0.01


class TradeResponse(BaseModel):
    job_id: str
    ticker: str
    status: str
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Simple health probe endpoint."""

    return HealthResponse(status="ok")


def _prepare_headlines(raw: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Headline]]:
    allowed_keys = {"source", "date", "title", "summary", "url"}
    prepared: Dict[str, List[Headline]] = {}
    for bucket, entries in raw.items():
        prepared[bucket] = []
        for item in entries:
            filtered = {k: item.get(k) for k in allowed_keys if k in item}
            prepared[bucket].append(Headline(**filtered))
    return prepared


def _build_news_analysis_payload(data: Dict[str, Any]) -> NewsAnalysis:
    def build_bucket(bucket_key: str) -> Optional[NewsAnalysisBreakdown]:
        bucket = data.get(bucket_key)
        if not isinstance(bucket, dict):
            return None
        counts = {
            str(k): float(v)
            for k, v in (bucket.get("counts") or {}).items()
            if isinstance(v, (int, float))
        }
        return NewsAnalysisBreakdown(
            sentiment=str(bucket.get("sentiment", "neutral")),
            confidence=float(bucket.get("confidence", 0.0)),
            counts=counts,
        )

    return NewsAnalysis(
        sentiment=str(data.get("sentiment", "neutral")),
        confidence=float(data.get("confidence", 0.0)),
        summary=str(data.get("summary", "")),
        method=data.get("method"),
        cached=data.get("cached"),
        global_=build_bucket("global"),
        india=build_bucket("india"),
        symbol=build_bucket("symbol"),
    )


@app.post("/analyze/news", response_model=NewsResponse)
def analyze_news_endpoint(request: NewsRequest) -> NewsResponse:
    """Fetch and analyze news sentiment for the requested tickers."""

    limit = request.limit or settings.news_limit
    use_cache = settings.news_use_cache if request.use_cache is None else request.use_cache
    logger.info(
        "News analysis requested",
        extra={"tickers": request.tickers, "limit": limit, "use_cache": use_cache},
    )

    results: List[NewsResult] = []
    for ticker in request.tickers:
        try:
            headlines = get_news(ticker, limit=limit)
            analysis_raw = analyze_news_agent(symbol=ticker, headlines_dict=headlines, use_cache=use_cache)
            news_result = NewsResult(
                ticker=ticker,
                headlines=_prepare_headlines(headlines),
                analysis=_build_news_analysis_payload(analysis_raw) if analysis_raw else None,
            )
        except Exception as exc:  # pragma: no cover - surfacing runtime errors
            logger.exception("News analysis failed", extra={"ticker": ticker})
            news_result = NewsResult(
                ticker=ticker,
                error=str(exc),
            )
        results.append(news_result)

    return NewsResponse(results=results)


@app.post("/signals/ta", response_model=TAResponse)
def generate_ta_signals(request: TARequest) -> TAResponse:
    """Compute technical snapshot and LLM-backed analysis for a ticker."""

    lookback = request.lookback or settings.ta_default_lookback
    logger.info("TA analysis requested", extra={"ticker": request.ticker, "lookback": lookback})

    try:
        ta_snapshot = get_technical_data(request.ticker, lookback_days=lookback)
    except ValueError as exc:
        logger.warning("TA data validation error", extra={"ticker": request.ticker})
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - upstream failures surface as 500
        logger.exception("Failed to fetch technical data", extra={"ticker": request.ticker})
        raise HTTPException(status_code=500, detail=f"Failed to fetch technical data: {exc}") from exc

    df = ta_snapshot.pop("_df", None)
    csv_payload: Optional[TechnicalCSV] = None
    if df is not None:
        rows = df.reset_index().to_dict(orient="records")
        default_keep = getattr(project_config, "TA_BARS", 48)
        default_prec = getattr(project_config, "TA_PREC", 3)
        keep = min(len(rows), request.keep_rows or settings.ta_keep_rows or default_keep)
        prec = request.precision or settings.ta_precision or default_prec
        spec, data = build_ta_csv(rows, keep=max(1, keep), prec=max(0, prec))
        csv_payload = TechnicalCSV(spec=spec, data=data)

    try:
        analysis = analyze_technical(
            request.ticker,
            ta_data=ta_snapshot,
            use_csv=False,
        )
    except Exception as exc:  # pragma: no cover - ensure API stays responsive
        logger.exception("Technical analysis failed", extra={"ticker": request.ticker})
        analysis = {
            "signal": "neutral",
            "confidence": 0.0,
            "rationale": f"Technical analysis failed: {exc}",
        }

    return TAResponse(
        ticker=request.ticker,
        snapshot=ta_snapshot,
        analysis=analysis,
        csv=csv_payload,
    )


@app.post("/trade/run", response_model=TradeResponse)
def run_trade(request: TradeRequest, background: BackgroundTasks) -> TradeResponse:
    """Queue or execute the end-to-end trading pipeline for a ticker."""

    job_record = _create_job(request.ticker)
    request_meta = {"cash_usd": request.cash_usd, "risk_budget": request.risk_budget}

    if settings.run_trades_in_background:
        background.add_task(_execute_trade_job, job_record["job_id"], request.ticker, request_meta)
        logger.info("Trade job queued", extra={"job_id": job_record["job_id"], "ticker": request.ticker})
        return TradeResponse(**_public_job_view(job_record))

    _execute_trade_job(job_record["job_id"], request.ticker, request_meta)
    final_record = _get_job(job_record["job_id"])
    if final_record is None:  # pragma: no cover - should not happen
        raise HTTPException(status_code=500, detail="Trade job result unavailable")
    return TradeResponse(**_public_job_view(final_record))


@app.get("/trade/jobs/{job_id}", response_model=TradeResponse)
def get_trade_job(job_id: str) -> TradeResponse:
    """Return the status of a previously submitted trade job."""

    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return TradeResponse(**_public_job_view(job))

