"""Data fetching from OpenBB for technical indicators and news."""
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from openbb import obb

# Import config for news weighting and filtering
try:
    import config
    RECENCY_HOURS_2X = config.RECENCY_HOURS_2X
    RECENCY_HOURS_1_5X = config.RECENCY_HOURS_1_5X
    SOURCE_WEIGHTS = config.SOURCE_WEIGHTS
    LOW_SIGNAL_MARKERS = config.LOW_SIGNAL_MARKERS
    NEWS_PROVIDERS_COMPANY = config.NEWS_PROVIDERS_COMPANY
    NEWS_PROVIDERS_WORLD = config.NEWS_PROVIDERS_WORLD
    NEWS_LIMIT_TOTAL = config.NEWS_LIMIT_TOTAL
except ImportError:
    # Fallback defaults if config not available
    RECENCY_HOURS_2X = 24
    RECENCY_HOURS_1_5X = 72
    SOURCE_WEIGHTS = {}
    LOW_SIGNAL_MARKERS = ("press release", "sponsored", "advertorial")
    NEWS_PROVIDERS_COMPANY = ["yfinance", "fmp", "tiingo"]
    NEWS_PROVIDERS_WORLD = ["fmp", "tiingo"]
    NEWS_LIMIT_TOTAL = 24


# Initialize OpenBB SDK once (module-level)
try:
    _obb_initialized = True
except Exception:
    _obb_initialized = False


def get_technical_data(symbol: str, lookback_days: int = 500) -> Dict[str, Any]:
    """
    Fetch OHLCV data and compute technical indicators.
    
    Args:
        symbol: Stock symbol (e.g., "INFY.NS")
        lookback_days: Number of trading days to fetch (default: 200)
    
    Returns:
        Dict with technical indicators, ATR%, and realized volatility
    """
    # Calculate start date (extra buffer to ensure long-window indicators are populated)
    start_date = (datetime.now() - timedelta(days=lookback_days + 100)).strftime("%Y-%m-%d")
    
    try:
        # Fetch OHLCV data
        ohlcv = obb.equity.price.historical(
            symbol, 
            provider="yfinance", 
            start_date=start_date
        )
        df = ohlcv.to_dataframe().copy()
        
        # Ensure proper datetime index
        if "date" in df.columns:
            df = df.sort_values("date").set_index("date")
        else:
            df = df.sort_index()
        
        if df.index.name != "date":
            df.index.name = "date"
        
        # Required columns check
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Available: {df.columns.tolist()}")
        
        # Compute technical indicators
        df["rsi_14"] = ta.rsi(df["close"], length=14)

        # MACD
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df = df.join(macd)  # Adds MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        
        # Bollinger Bands
        bb = ta.bbands(df["close"], length=20, std=2.0)
        df = df.join(bb)  # Adds BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        
        # SMAs/EMA
        df["sma_20"] = ta.sma(df["close"], length=20)
        df["sma_50"] = ta.sma(df["close"], length=50)
        df["sma_200"] = ta.sma(df["close"], length=200)
        df["ema_50"] = ta.ema(df["close"], length=50)
        
        # ATR
        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # Stochastic Oscillator (k/d)
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
        if stoch is not None and not stoch.empty:
            df = df.join(stoch)  # STOCHk_14_3_3, STOCHd_14_3_3

        # ADX
        try:
            adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
            if adx_df is not None and "ADX_14" in adx_df.columns:
                df["adx_14"] = adx_df["ADX_14"]
        except Exception:
            pass

        # OBV
        try:
            df["obv"] = ta.obv(df["close"], df["volume"])
        except Exception:
            pass
        
        # Calculate daily returns for realized volatility
        df["daily_return"] = df["close"].pct_change()
        
        # Clean NaN values conservatively:
        # - Keep rows with valid OHLCV
        # - Drop rows where ALL indicator fields are NaN (allow partial availability)
        indicator_cols = [
            "rsi_14",
            "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
            "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBP_20_2.0",
            "sma_20",
            "sma_50", "sma_200",
            "ema_50",
            "atr_14",
            "STOCHk_14_3_3", "STOCHd_14_3_3",
            "adx_14",
            "obv",
            "daily_return",
        ]
        # Ensure core OHLCV present
        df = df.dropna(subset=required_cols)
        # Drop rows where all indicators are NaN (only use columns that actually exist)
        present_indicator_cols = [c for c in indicator_cols if c in df.columns]
        if present_indicator_cols:
            drop_mask = df[present_indicator_cols].isna().all(axis=1)
            df = df.loc[np.logical_not(drop_mask)]
        
        if df.empty:
            raise ValueError("No data available after computing indicators")
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # ATR percent (active gate)
        atr_14 = float(np.nan_to_num(latest.get("atr_14", 0.0)))
        last_close = float(np.nan_to_num(latest.get("close", 0.0)))
        atr_percent = round((atr_14 / last_close) * 100, 2) if last_close > 0 else 0.0
        
        # Realized volatility (background metric)
        daily_returns = df["daily_return"]
        realized_vol_20 = round(daily_returns.rolling(20).std().iloc[-1] * np.sqrt(252), 4) if len(daily_returns) >= 20 else 0.0
        
        # MACD signal detection
        macd_line = float(np.nan_to_num(latest.get("MACD_12_26_9", 0.0)))
        macd_signal = float(np.nan_to_num(latest.get("MACDs_12_26_9", 0.0)))
        macd_hist = float(np.nan_to_num(latest.get("MACDh_12_26_9", 0.0)))
        
        if prev is not None:
            prev_macd = float(np.nan_to_num(prev.get("MACD_12_26_9", 0.0)))
            prev_signal = float(np.nan_to_num(prev.get("MACDs_12_26_9", 0.0)))
            if prev_macd < prev_signal and macd_line > macd_signal:
                macd_signal_state = "bullish_cross"
            elif prev_macd > prev_signal and macd_line < macd_signal:
                macd_signal_state = "bearish_cross"
            else:
                macd_signal_state = "none"
        else:
            macd_signal_state = "none"
        
        # Bollinger Band position
        bb_lower = float(np.nan_to_num(latest.get("BBL_20_2.0", 0.0)))
        bb_upper = float(np.nan_to_num(latest.get("BBU_20_2.0", 0.0)))
        bb_middle = float(np.nan_to_num(latest.get("BBM_20_2.0", 0.0)))
        bb_pct = float(np.nan_to_num(latest.get("BBP_20_2.0", 0.0)))
        
        if last_close > bb_upper:
            bb_event = "upper_band_break"
        elif last_close < bb_lower:
            bb_event = "lower_band_break"
        elif bb_pct > 0.8:
            bb_event = "near_upper"
        elif bb_pct < 0.2:
            bb_event = "near_lower"
        else:
            bb_event = "within_bands"
        
        # SMA relation
        sma_50_val = float(np.nan_to_num(latest.get("sma_50", 0.0)))
        sma_200_val = float(np.nan_to_num(latest.get("sma_200", 0.0)))
        if sma_50_val > sma_200_val:
            sma_relation = "above"
        elif sma_50_val < sma_200_val:
            sma_relation = "below"
        else:
            sma_relation = "equal"
        
        # Build compact JSON-ready dict
        result = {
            "rsi": round(float(np.nan_to_num(latest.get("rsi_14", 0.0))), 2),
            "macd": {
                "macd": round(macd_line, 2),
                "signal": round(macd_signal, 2),
                "hist": round(macd_hist, 2),
            },
            "macd_signal": macd_signal_state,
            "bollinger": {
                "lower": round(bb_lower, 2),
                "middle": round(bb_middle, 2),
                "upper": round(bb_upper, 2),
                "band_pct": round(bb_pct, 2),
            },
            "bollinger_event": bb_event,
            "sma_50": round(sma_50_val, 2),
            "sma_200": round(sma_200_val, 2),
            "sma_50_vs_200": sma_relation,
            "atr_14": round(float(np.nan_to_num(atr_14)), 2),
            "atr_percent": atr_percent,
            "realized_vol_20": realized_vol_20,
            "close": round(float(np.nan_to_num(last_close)), 2),
            "volume": int(float(np.nan_to_num(latest.get("volume", 0.0)))),
        }
        
        # Also return raw dataframe for CSV encoding
        result["_df"] = df  # Keep for CSV encoding
        
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to fetch technical data for {symbol}: {str(e)}")


def get_technical_data_csv(symbol: str, lookback_days: int = 200, keep: int = 48, prec: int = 3) -> Tuple[str, str]:
    """
    Fetch technical data and return compact CSV representation.
    
    Args:
        symbol: Stock symbol (e.g., "INFY.NS")
        lookback_days: Number of trading days to fetch
        keep: Number of recent bars to include in CSV
        prec: Precision for number formatting
    
    Returns:
        Tuple of (spec_string, csv_data) for compact CSV encoding
    """
    ta_data = get_technical_data(symbol, lookback_days)
    df = ta_data.get("_df")
    
    if df is None:
        raise ValueError("DataFrame not available for CSV encoding")
    
    # Map compact codes to expected keys
    all_fields: List[Tuple[str, str]] = [
        ("o", "open"),
        ("h", "high"),
        ("l", "low"),
        ("c", "close"),
        ("v", "volume"),
        ("rsi", "rsi_14"),
        ("macd", "MACD_12_26_9"),
        ("macds", "MACDs_12_26_9"),
        ("macdh", "MACDh_12_26_9"),
        ("sma", "sma_20"),
        ("sma50", "sma_50"),
        ("sma200", "sma_200"),
        ("ema", "ema_50"),
        ("bbl", "BBL_20_2.0"),
        ("bbm", "BBM_20_2.0"),
        ("bbu", "BBU_20_2.0"),
        ("bbp", "BBP_20_2.0"),
        ("k", "STOCHk_14_3_3"),
        ("d", "STOCHd_14_3_3"),
        ("adx", "adx_14"),
        ("atr", "atr_14"),
        ("obv", "obv"),
    ]
    
    fields = all_fields  # Can filter by TA_COLUMNS config if needed
    
    def format_number_short(x: Any, prec: int) -> str:
        """Format number with adaptive precision to save tokens."""
        try:
            f = float(x)
        except Exception:
            return ""
        # Adaptive precision
        if abs(f) >= 1000:
            s = f"{f:.2f}"
        elif abs(f) >= 1:
            s = f"{f:.{min(max(prec, 1), 5)}f}"
        else:
            s = f"{f:.{min(max(prec + 2, 2), 6)}f}"
        s = s.rstrip("0").rstrip(".")
        return s
    
    lines: List[str] = []
    slice_rows = df.tail(keep).reset_index()
    for _, row in slice_rows.iterrows():
        vals: List[str] = []
        for _, key in fields:
            val = row.get(key, "")
            vals.append(format_number_short(val, prec))
        lines.append("|".join(vals))
    
    csv_data = "\n".join(lines)
    spec = (
        f"headers={'|'.join(code for code, _ in fields)} "
        "(rows are oldest->newest)"
    )
    
    return spec, csv_data


def _weight_for_headline(headline_date: str, source: str | None) -> float:
    """Compute weight for a headline based on recency and source credibility."""
    w = 1.0
    try:
        dt = datetime.fromisoformat(headline_date.replace("Z", "+00:00"))
        hrs = max(0, (datetime.utcnow() - dt).total_seconds() / 3600.0)
        if hrs <= RECENCY_HOURS_2X:
            w *= 2.0
        elif hrs <= RECENCY_HOURS_1_5X:
            w *= 1.5
    except Exception:
        pass
    if source and source in SOURCE_WEIGHTS:
        w *= SOURCE_WEIGHTS[source]
    return w


def aggregate_weighted_sentiment(
    per_list: List[Dict[str, Any]], 
    headlines: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate sentiment with weights based on recency and source."""
    counts = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
    for lab, h in zip(per_list, headlines):
        s = str(lab.get("sentiment", "neutral")).lower()
        # Map positive/negative to bullish/bearish for counting
        if s == "positive":
            s = "bullish"
        elif s == "negative":
            s = "bearish"
        counts[s if s in counts else "neutral"] += _weight_for_headline(
            h.get("date", ""), h.get("source")
        )
    total = sum(counts.values()) or 1.0
    majority = max(counts, key=counts.get)
    # Map back to positive/negative for output
    sentiment_output = {"bullish": "positive", "bearish": "negative", "neutral": "neutral"}[majority]
    return {
        "counts": {k: round(v, 2) for k, v in counts.items()},
        "sentiment": sentiment_output,
        "confidence": round(counts[majority] / total, 4),
    }


def _normalize_title(title: str) -> str:
    """Normalize title for deduplication."""
    return ''.join(ch for ch in title.lower() if ch.isalnum() or ch.isspace()).strip()


def _is_low_signal(title: str) -> bool:
    """Check if title appears to be low-signal content."""
    tl = title.lower()
    return any(m in tl for m in LOW_SIGNAL_MARKERS)


def _dedupe_and_filter(headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates and filter low-signal headlines."""
    seen = set()
    out = []
    for h in headlines:
        t = h.get("title") or ""
        if not t or _is_low_signal(t):
            continue
        key = _normalize_title(t)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _fetch_company_headlines(sym: str, providers: List[str], per_bucket: int) -> List[Dict[str, Any]]:
    """Fetch company headlines with provider fallback."""
    for p in providers:
        try:
            ob_result = obb.news.company(symbol=sym, provider=p, limit=per_bucket)
            df = ob_result.to_dataframe()
            if df is None or df.empty:
                continue
            cols = [c for c in ["date", "title", "url", "source"] if c in df.columns]
            df = df[cols].dropna(subset=[c for c in ["title", "url"] if c in cols])
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
            records = df.to_dict(orient="records")
            filtered = _dedupe_and_filter(records)
            return filtered[:per_bucket]
        except Exception:
            continue
    return []


def _cap_headlines_across_buckets(
    headlines_dict: Dict[str, List[Dict[str, Any]]], 
    limit_total: int
) -> Dict[str, List[Dict[str, Any]]]:
    """Cap total headlines across all buckets."""
    total = sum(len(v) for v in headlines_dict.values())
    if total <= limit_total:
        return headlines_dict
    # Proportionally reduce each bucket
    result = {}
    remaining = limit_total
    buckets = list(headlines_dict.keys())
    for i, key in enumerate(buckets):
        current = headlines_dict[key]
        if i == len(buckets) - 1:
            # Last bucket gets remaining
            result[key] = current[:remaining]
        else:
            # Proportional allocation
            alloc = max(1, int(len(current) * limit_total / total))
            alloc = min(alloc, remaining - (len(buckets) - i - 1))
            result[key] = current[:alloc]
            remaining -= alloc
    return result


def get_news(symbol: str, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch news headlines for symbol, India market, and US market.
    
    Args:
        symbol: Stock symbol (e.g., "INFY.NS")
        limit: Maximum number of headlines per category
    
    Returns:
        Dict with keys: symbol_headlines, india_headlines, global_headlines
        Each value is a list of news items with {source, date, title, summary}
    """
    # Fetch symbol-specific news with provider fallback
    symbol_headlines = _fetch_company_headlines(symbol, NEWS_PROVIDERS_COMPANY, limit)
    
    # Fetch India macro news (NSE index)
    india_providers = NEWS_PROVIDERS_WORLD if NEWS_PROVIDERS_WORLD else ["yfinance"]
    india_headlines = _fetch_company_headlines("^NSEI", india_providers, limit)
    
    # Fetch US macro news (S&P 500)
    global_providers = NEWS_PROVIDERS_WORLD if NEWS_PROVIDERS_WORLD else ["yfinance"]
    global_headlines = _fetch_company_headlines("^GSPC", global_providers, limit)
    
    # Structure headlines by bucket
    headlines_dict = {
        "symbol_headlines": symbol_headlines,
        "india_headlines": india_headlines,
        "global_headlines": global_headlines,
    }
    
    # Cap total headlines across buckets
    capped = _cap_headlines_across_buckets(headlines_dict, NEWS_LIMIT_TOTAL)
    
    # Add summary field and truncate
    for bucket_name, bucket_headlines in capped.items():
        for h in bucket_headlines:
            h["title"] = str(h.get("title", ""))[:200]
            h["summary"] = str(h.get("text", "") or h.get("summary", "") or "")[:500]
            if "text" in h:
                del h["text"]  # Remove raw text field
    
    return capped

