"""Data fetchers for technical indicators and market news."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance.utils import auto_adjust

from ..core import config, utils


# ---------------------------------------------------------------------------
# News helpers (ported from legacy news_handler)
# ---------------------------------------------------------------------------


def weight_for(headline_date: str, source: Optional[str]) -> float:
    """Compute weight for a headline based on recency and source credibility."""
    weight = 1.0
    try:
        dt = datetime.fromisoformat(headline_date.replace("Z", "+00:00"))
        hours_old = max(0.0, (datetime.utcnow() - dt).total_seconds() / 3600.0)
        if hours_old <= config.RECENCY_HOURS_2X:
            weight *= 2.0
        elif hours_old <= config.RECENCY_HOURS_1_5X:
            weight *= 1.5
    except Exception:
        pass
    if source and source in config.SOURCE_WEIGHTS:
        weight *= config.SOURCE_WEIGHTS[source]
    return weight


def aggregate_weighted(per_list: List[Dict[str, Any]], headlines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate sentiment with weights based on recency and source."""
    counts = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
    for lab, headline in zip(per_list, headlines):
        sentiment = str(lab.get("sentiment", "neutral")).lower()
        counts[sentiment if sentiment in counts else "neutral"] += weight_for(
            headline.get("date", ""),
            headline.get("source"),
        )
    total = sum(counts.values()) or 1.0
    majority = max(counts, key=counts.get)
    return {
        "counts": {k: round(v, 2) for k, v in counts.items()},
        "sentiment": majority,
        "confidence": round(counts[majority] / total, 4),
    }


def _fetch_company_headlines(symbol: str, provider: str, per_bucket: int) -> List[Dict[str, Any]]:
    try:
        # Use yfinance Search for news
        search_result = yf.Search(symbol, news_count=per_bucket)
        news_items = search_result.news if hasattr(search_result, 'news') and search_result.news else []

        if not news_items:
            return []

        # Convert yfinance news format to our expected format
        records = []
        for item in news_items:
            record = {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "source": item.get("publisher", "Yahoo Finance"),
                # Convert timestamp if available
                "date": item.get("providerPublishTime", "") if isinstance(item.get("providerPublishTime"), str) else ""
            }
            # Only include if we have at least title or url
            if record["title"] or record["url"]:
                records.append(record)

        return utils.dedupe_and_filter(records)[:per_bucket]
    except Exception:
        return []


def cap_headlines_across_buckets(
    headlines: Dict[str, List[Dict[str, Any]]],
    limit_total: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Cap total headlines across all buckets proportionally."""
    total = sum(len(v) for v in headlines.values())
    if total <= limit_total:
        return headlines

    result: Dict[str, List[Dict[str, Any]]] = {}
    remaining = limit_total
    buckets = list(headlines.keys())
    for idx, key in enumerate(buckets):
        current = headlines[key]
        if idx == len(buckets) - 1:
            result[key] = current[:remaining]
        else:
            allocation = max(1, int(len(current) * limit_total / total))
            allocation = min(allocation, remaining - (len(buckets) - idx - 1))
            result[key] = current[:allocation]
            remaining -= allocation
    return result


def prepare_news_payload(
    symbol: str,
    limit: Optional[int] = None,
    *,
    limit_symbol: Optional[int] = None,
    limit_india: Optional[int] = None,
    limit_global: Optional[int] = None,
    total_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Prepare news payload with provider fallback and per-bucket limits.

    Backwards-compatible: if only ``limit`` is provided, it is used for all buckets.
    Otherwise falls back to per-bucket config values.
    """

    # Resolve per-bucket limits with clear precedence
    per_symbol = (
        limit_symbol
        if limit_symbol is not None
        else (limit if limit is not None else config.NEWS_LIMIT_SYMBOL)
    )
    per_india = (
        limit_india
        if limit_india is not None
        else (limit if limit is not None else config.NEWS_LIMIT_INDIA)
    )
    per_global = (
        limit_global
        if limit_global is not None
        else (limit if limit is not None else config.NEWS_LIMIT_GLOBAL)
    )
    cap_total = total_limit if total_limit is not None else config.NEWS_LIMIT_TOTAL

    def _with_providers(code: str, providers: List[str], per_bucket: int) -> List[Dict[str, Any]]:
        for provider in providers:
            try:
                records = utils.with_retries(
                    lambda: _fetch_company_headlines(code, provider, per_bucket)
                )
                if records:
                    return records
            except Exception:
                continue
        return []

    symbol_headlines = _with_providers(symbol, config.NEWS_PROVIDERS_COMPANY, per_symbol)
    india_headlines = _with_providers("^NSEI", ["yfinance"], per_india)
    global_headlines = _with_providers("SPY", ["yfinance"], per_global)

    capped = cap_headlines_across_buckets(
        {
            "symbol_headlines": symbol_headlines,
            "india_headlines": india_headlines,
            "global_headlines": global_headlines,
        },
        cap_total,
    )

    return {"symbol": symbol, **capped}


# ---------------------------------------------------------------------------
# Company info helpers
# ---------------------------------------------------------------------------


def get_company_long_name(symbol: str) -> Optional[str]:
    """Fetch the long name (company name) for a given ticker symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get("longName") or info.get("shortName")
    except Exception:
        return None


def get_company_info(symbol: str) -> Dict[str, Any]:
    """Fetch company info including long name, current price, and change percentage."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        long_name = info.get("longName") or info.get("shortName")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        currency = info.get("currency")
        
        # Calculate change percentage
        previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        change_percent = None
        if current_price and previous_close and previous_close != 0:
            change_percent = ((current_price - previous_close) / previous_close) * 100
        
        return {
            "long_name": long_name,
            "price": current_price,
            "change_percent": change_percent,
            "currency": currency,
        }
    except Exception:
        return {
            "long_name": None,
            "price": None,
            "change_percent": None,
            "currency": None,
        }


def get_fundamental_data(symbol: str) -> Dict[str, Any]:
    """Fetch comprehensive fundamental metrics for a given ticker symbol.
    
    Extracts key financial metrics including valuation ratios, profitability metrics,
    growth rates, and ownership data from yfinance.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Helper to safely extract numeric values
        def safe_float(key: str) -> Optional[float]:
            val = info.get(key)
            if val is None or val == "N/A":
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        
        # Valuation metrics
        pe_ratio = safe_float("trailingPE") or safe_float("forwardPE")
        pb_ratio = safe_float("priceToBook")
        ev_ebitda = safe_float("enterpriseToEbitda")
        
        # Profitability metrics
        roe = safe_float("returnOnEquity")
        profit_margin = safe_float("profitMargins")
        operating_margin = safe_float("operatingMargins")
        gross_margin = safe_float("grossMargins")
        
        # Growth metrics
        revenue_growth = safe_float("revenueGrowth")
        earnings_growth = safe_float("earningsGrowth") or safe_float("earningsQuarterlyGrowth")
        
        # Financial health
        debt_to_equity = safe_float("debtToEquity")
        
        # Size and ownership
        market_cap = safe_float("marketCap")
        institutional_holdings = safe_float("heldPercentInstitutions")
        insider_holdings = safe_float("heldPercentInsiders")
        
        # Additional context
        sector = info.get("sector")
        industry = info.get("industry")
        
        return {
            "pe_ratio": pe_ratio,
            "pb_ratio": pb_ratio,
            "roe": roe,
            "debt_to_equity": debt_to_equity,
            "profit_margin": profit_margin,
            "operating_margin": operating_margin,
            "gross_margin": gross_margin,
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "ev_ebitda": ev_ebitda,
            "market_cap": market_cap,
            "institutional_holdings": institutional_holdings,
            "insider_holdings": insider_holdings,
            "sector": sector,
            "industry": industry,
        }
    except Exception:
        return {
            "pe_ratio": None,
            "pb_ratio": None,
            "roe": None,
            "debt_to_equity": None,
            "profit_margin": None,
            "operating_margin": None,
            "gross_margin": None,
            "revenue_growth": None,
            "earnings_growth": None,
            "ev_ebitda": None,
            "market_cap": None,
            "institutional_holdings": None,
            "insider_holdings": None,
            "sector": None,
            "industry": None,
        }


def prepare_fundamental_payload(symbol: str) -> Dict[str, Any]:
    """Prepare fundamental data payload for LLM consumption.
    
    Fetches fundamental metrics and formats them in a readable structure
    suitable for the fundamental analyst agent.
    """
    fundamental_data = get_fundamental_data(symbol)
    
    # Format percentages and large numbers for readability
    def format_metric(value: Optional[float], is_percentage: bool = False, is_large_number: bool = False) -> str:
        if value is None:
            return "N/A"
        if is_percentage:
            return f"{value * 100:.2f}%"
        if is_large_number:
            if value >= 1e12:
                return f"${value / 1e12:.2f}T"
            elif value >= 1e9:
                return f"${value / 1e9:.2f}B"
            elif value >= 1e6:
                return f"${value / 1e6:.2f}M"
            return f"${value:,.0f}"
        return f"{value:.2f}"
    
    formatted_metrics = {
        "pe_ratio": format_metric(fundamental_data["pe_ratio"]),
        "pb_ratio": format_metric(fundamental_data["pb_ratio"]),
        "roe": format_metric(fundamental_data["roe"], is_percentage=True),
        "debt_to_equity": format_metric(fundamental_data["debt_to_equity"]),
        "profit_margin": format_metric(fundamental_data["profit_margin"], is_percentage=True),
        "operating_margin": format_metric(fundamental_data["operating_margin"], is_percentage=True),
        "gross_margin": format_metric(fundamental_data["gross_margin"], is_percentage=True),
        "revenue_growth": format_metric(fundamental_data["revenue_growth"], is_percentage=True),
        "earnings_growth": format_metric(fundamental_data["earnings_growth"], is_percentage=True),
        "ev_ebitda": format_metric(fundamental_data["ev_ebitda"]),
        "market_cap": format_metric(fundamental_data["market_cap"], is_large_number=True),
        "institutional_holdings": format_metric(fundamental_data["institutional_holdings"], is_percentage=True),
        "insider_holdings": format_metric(fundamental_data["insider_holdings"], is_percentage=True),
        "sector": fundamental_data["sector"] or "N/A",
        "industry": fundamental_data["industry"] or "N/A",
    }
    
    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": formatted_metrics,
        "raw_metrics": fundamental_data,
    }


# ---------------------------------------------------------------------------
# Price + technical indicator helpers
# ---------------------------------------------------------------------------


def _fetch_price_dataframe(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    start_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()

    def _request() -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            interval=interval,
            auto_adjust=True, # Adjust for splits/dividends
            prepost=False # Exclude pre/post market data
        )
        if df is None or df.empty:
            raise ValueError("empty price dataframe")
        return df

    return utils.with_retries(_request)


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [c.lower() for c in frame.columns]

    # Handle the date/index properly
    if frame.index.name and frame.index.name.lower() == 'date':
        # Index is named 'Date', reset it to make it a column
        frame = frame.reset_index()
    elif not frame.index.name and isinstance(frame.index, pd.DatetimeIndex):
        # Index is DatetimeIndex but not named, reset it
        frame = frame.reset_index()

    # Now find the date column (it might be 'Date', 'date', or from reset_index)
    date_col = None
    if "date" in frame.columns:
        date_col = "date"
    elif "Date" in frame.columns:
        date_col = "Date"

    if date_col:
        frame["timestamp"] = pd.to_datetime(frame[date_col])
    elif "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    else:
        # Fallback - try to convert index if it's still DatetimeIndex
        if isinstance(frame.index, pd.DatetimeIndex):
            frame["timestamp"] = pd.to_datetime(frame.index)
            frame = frame.reset_index(drop=True)
        else:
            raise ValueError("No date/timestamp column found in data")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in frame.columns:
            frame[col] = np.nan

    frame = frame.sort_values("timestamp")
    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return frame[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def get_price_data(symbol: str, timeframe: str) -> List[Dict[str, Any]]:
    """Fetch OHLCV data for the given symbol and timeframe."""
    cfg = config.TIMEFRAME_CONFIG.get(timeframe)
    if cfg is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    interval = cfg.get("interval", "1d")
    lookback = cfg.get("lookback_days", 120)

    try:
        df = _fetch_price_dataframe(symbol, interval=interval, lookback_days=lookback)
    except Exception:
        return []

    ohlcv = _standardize_ohlcv(df)
    max_rows = max(config.TA_BARS + 32, 256)
    return ohlcv.tail(max_rows).to_dict(orient="records")


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(0, 100)


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = _compute_atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    obv = np.zeros(len(close))
    for idx in range(1, len(close)):
        if pd.isna(close.iloc[idx]) or pd.isna(close.iloc[idx - 1]) or pd.isna(volume.iloc[idx]):
            obv[idx] = obv[idx - 1]
        elif close.iloc[idx] > close.iloc[idx - 1]:
            obv[idx] = obv[idx - 1] + volume.iloc[idx]
        elif close.iloc[idx] < close.iloc[idx - 1]:
            obv[idx] = obv[idx - 1] - volume.iloc[idx]
        else:
            obv[idx] = obv[idx - 1]
    return pd.Series(obv, index=close.index)


def compute_indicators(rows: List[Dict[str, Any]], timeframe: str) -> List[Dict[str, Any]]:
    """Enrich OHLCV rows with core technical indicators."""
    if not rows:
        return []

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    close = df["close"]
    df["rsi_14"] = _compute_rsi(close)

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD_12_26_9"] = macd
    df["MACDs_12_26_9"] = macd_signal
    df["MACDh_12_26_9"] = macd - macd_signal

    rolling20 = close.rolling(window=20)
    bb_mid = rolling20.mean()
    bb_std = rolling20.std(ddof=0)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["BBM_20_2.0"] = bb_mid
    df["BBU_20_2.0"] = bb_upper
    df["BBL_20_2.0"] = bb_lower
    band_span = (bb_upper - bb_lower).replace(0, np.nan)
    df["BBP_20_2.0"] = ((close - bb_lower) / band_span).clip(0, 1)

    df["sma_20"] = close.rolling(window=20).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()

    df["atr_14"] = _compute_atr(df)
    df["adx_14"] = _compute_adx(df)

    low = df["low"]
    high = df["high"]
    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()
    stoch_k = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    df["STOCHk_14_3_3"] = stoch_k
    df["STOCHd_14_3_3"] = stoch_k.rolling(window=3).mean()

    df["obv"] = _compute_obv(close, df["volume"].fillna(0))

    df = df.ffill().bfill()
    return df.to_dict(orient="records")


def get_indicator_rows(symbol: str, timeframe: str) -> List[Dict[str, Any]]:
    """Fetch OHLCV data and append indicators for downstream TA prompts."""
    ohlcv = get_price_data(symbol, timeframe)
    return compute_indicators(ohlcv, timeframe)


def _round(value: Any, digits: int = 2) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def summarize_ta(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce a compact TA summary for prompts/logging."""
    if not rows:
        return {
            "rsi": None,
            "macd_signal": "unknown",
            "bollinger": "unknown",
            "sma_50_vs_200": "unknown",
        }

    df = pd.DataFrame(rows)
    for col in ["close", "high", "low", "volume", "rsi_14", "MACDh_12_26_9"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    last = df.iloc[-1]
    close = last.get("close")
    upper = last.get("BBU_20_2.0")
    lower = last.get("BBL_20_2.0")
    middle = last.get("BBM_20_2.0")

    macd_hist = last.get("MACDh_12_26_9")
    prev_hist = df["MACDh_12_26_9"].iloc[-2] if len(df) > 1 else np.nan
    if pd.notna(macd_hist) and pd.notna(prev_hist):
        if macd_hist > 0 and prev_hist <= 0:
            macd_state = "bullish_cross"
        elif macd_hist < 0 and prev_hist >= 0:
            macd_state = "bearish_cross"
        elif macd_hist > 0:
            macd_state = "bullish"
        elif macd_hist < 0:
            macd_state = "bearish"
        else:
            macd_state = "flat"
    elif pd.notna(macd_hist):
        macd_state = "bullish" if macd_hist > 0 else "bearish" if macd_hist < 0 else "flat"
    else:
        macd_state = "unknown"

    bollinger_state = "unknown"
    if pd.notna(close) and pd.notna(upper) and pd.notna(lower):
        if close >= upper:
            bollinger_state = "upper_band_break"
        elif close <= lower:
            bollinger_state = "lower_band_break"
        elif pd.notna(middle):
            bollinger_state = "inside_upper" if close >= middle else "inside_lower"
        else:
            bollinger_state = "inside"

    df["sma_50_calc"] = df["close"].rolling(window=50).mean()
    df["sma_200_calc"] = df["close"].rolling(window=200).mean()
    sma_50 = df["sma_50_calc"].iloc[-1]
    sma_200 = df["sma_200_calc"].iloc[-1]
    if pd.notna(sma_50) and pd.notna(sma_200):
        if sma_50 > sma_200:
            sma_state = "above"
        elif sma_50 < sma_200:
            sma_state = "below"
        else:
            sma_state = "inline"
    else:
        sma_state = "insufficient"

    atr_percent = None
    atr_val = last.get("atr_14")
    if pd.notna(atr_val) and pd.notna(close) and close:
        atr_percent = round(float(atr_val) / float(close), 4)

    stoch_k = last.get("STOCHk_14_3_3")
    if pd.notna(stoch_k):
        if stoch_k >= 80:
            stoch_state = "overbought"
        elif stoch_k <= 20:
            stoch_state = "oversold"
        else:
            stoch_state = "neutral"
    else:
        stoch_state = "unknown"

    adx_val = last.get("adx_14")
    if pd.notna(adx_val):
        if adx_val >= 40:
            adx_state = "strong"
        elif adx_val >= 25:
            adx_state = "rising"
        else:
            adx_state = "weak"
    else:
        adx_state = "unknown"

    return {
        "rsi": _round(last.get("rsi_14"), 1),
        "macd_signal": macd_state,
        "bollinger": bollinger_state,
        "sma_50_vs_200": sma_state,
        "atr_percent": atr_percent,
        "stoch_state": stoch_state,
        "adx_trend": adx_state,
    }
