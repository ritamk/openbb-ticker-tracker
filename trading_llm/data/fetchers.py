"""Data fetchers for technical indicators and market news."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from openbb import obb
import yfinance as yf

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
    ob_result = obb.news.company(symbol=symbol, provider=provider, limit=per_bucket)
    df = ob_result.to_dataframe()
    if df is None or df.empty:
        return []
    
    # Select available columns (removed date)
    cols = [c for c in ["title", "url", "source"] if c in df.columns]
    df = df[cols].dropna(subset=[c for c in ["title", "url"] if c in cols])
    
    records = df.to_dict(orient="records")
    return utils.dedupe_and_filter(records)[:per_bucket]


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
        
        # Calculate change percentage
        previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        change_percent = None
        if current_price and previous_close and previous_close != 0:
            change_percent = ((current_price - previous_close) / previous_close) * 100
        
        return {
            "long_name": long_name,
            "price": current_price,
            "change_percent": change_percent,
        }
    except Exception:
        return {
            "long_name": None,
            "price": None,
            "change_percent": None,
        }


# ---------------------------------------------------------------------------
# Price + technical indicator helpers
# ---------------------------------------------------------------------------


def _fetch_price_dataframe(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    start_date = (datetime.utcnow() - timedelta(days=lookback_days)).date().isoformat()

    def _request() -> pd.DataFrame:
        response = obb.equity.price.historical(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
        )
        df = response.to_dataframe()
        if df is None or df.empty:
            raise ValueError("empty price dataframe")
        return df

    return utils.with_retries(_request)


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [c.lower() for c in frame.columns]
    if "date" not in frame.columns and frame.index.name:
        frame = frame.reset_index()
    if "date" not in frame.columns and "timestamp" in frame.columns:
        frame["date"] = frame["timestamp"]
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    elif "date" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["date"])
    else:
        frame["timestamp"] = pd.to_datetime(frame.index)

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
