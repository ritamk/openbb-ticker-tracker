from __future__ import annotations

from datetime import datetime, timezone

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException, Query

from ..schemas import (
    TickerDataRequest,
    TickerDataResponse,
    TickerSearchResponse,
    TickerNewsResponse,
    TickerNewsItem,
)
from trading_llm.core import config
from trading_llm.core.orchestrator import run_trading_cycle
from trading_llm.data.fetchers import prepare_news_payload


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


@router.get("/search", response_model=list[TickerSearchResponse])
def search_tickers(
    q: str = Query(..., min_length=1, description="Company name to search"),
    limit: int = Query(5, ge=1, le=20, description="Max number of tickers to return"),
) -> list[TickerSearchResponse]:
    """Resolve a company name search string to a list of matching tickers.

    Uses Yahoo Finance's public search endpoint. Returns up to ``limit`` matches.
    """

    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query_required")

    quotes_count = max(limit, 6)
    url = "https://query2.finance.yahoo.com/v1/finance/search?" + urlencode(
        {"q": query, "quotesCount": quotes_count, "newsCount": 0, "region": "IN"}
    )

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=6) as resp:
            payload = resp.read().decode("utf-8")
        data = json.loads(payload)
    except Exception:
        raise HTTPException(status_code=502, detail="lookup_failed")

    quotes = data.get("quotes") or []
    results: list[TickerSearchResponse] = []
    for item in quotes:
        symbol = item.get("symbol")
        if not symbol:
            continue
        name = item.get("shortname") or item.get("longname")
        results.append(TickerSearchResponse(ticker=str(symbol), name=str(name) if name else None))
        if len(results) >= limit:
            break

    return results


@router.get("/news", response_model=TickerNewsResponse)
def get_symbol_news(
    symbol: str = Query(..., min_length=1, description="Ticker symbol"),
    limit: int = Query(5, ge=1, le=50, description="Max number of news items to return"),
) -> TickerNewsResponse:
    """Return basic quote details and up to ``limit`` news items for a ticker.

    - News prefers URLs and falls back to headlines.
    - Quote data fetched from Yahoo's public quote API.
    """

    code = symbol.strip()
    if not code:
        raise HTTPException(status_code=400, detail="symbol_required")

    # Only fetch company-specific headlines; cap total to the requested limit
    payload = prepare_news_payload(
        code,
        limit_symbol=limit,
        limit_india=0,
        limit_global=0,
        total_limit=limit,
    )

    headlines = payload.get("symbol_headlines") or []
    items_list: list[TickerNewsItem] = []
    for h in headlines:
        url = h.get("url")
        title = h.get("title")
        if url or title:
            items_list.append(TickerNewsItem(url=url, headline=title))
        if len(items_list) >= limit:
            break

    # Fetch basic quote details
    price = None
    change = None
    change_percent = None
    currency = None
    quote_page_url = f"https://finance.yahoo.com/quote/{code}/"
    try:
        quote_url = "https://query1.finance.yahoo.com/v7/finance/quote?" + urlencode(
            {"symbols": code, "region": "IN"}
        )
        req = Request(quote_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=6) as resp:
            payload = resp.read().decode("utf-8")
        qdata = json.loads(payload)
        results = ((qdata or {}).get("quoteResponse") or {}).get("result") or []
        if results:
            r0 = results[0]
            price = r0.get("regularMarketPrice")
            change = r0.get("regularMarketChange")
            change_percent = r0.get("regularMarketChangePercent")
            currency = r0.get("currency")
            resolved_symbol = r0.get("symbol") or code
            quote_page_url = f"https://finance.yahoo.com/quote/{resolved_symbol}/"
    except Exception:
        pass

    return TickerNewsResponse(
        symbol=code,
        price=price,
        change=change,
        change_percent=change_percent,
        currency=currency,
        quote_url=quote_page_url,
        items=items_list,
    )
