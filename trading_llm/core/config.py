"""Configuration module for LLM trader pipeline."""
import os
import json
from typing import Dict, List

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Model configuration
MODEL = "gpt-4o-mini"
TA_MODEL = os.getenv("TA_MODEL", MODEL)
NEWS_MODEL = os.getenv("NEWS_MODEL", MODEL)
FUNDAMENTAL_MODEL = os.getenv("FUNDAMENTAL_MODEL", MODEL)
TRADER_MODEL = os.getenv("TRADER_MODEL", "gpt-4o")

# Timeframe support
DEFAULT_TIMEFRAMES: List[str] = ["1D", "15m", "5m", "30D"]
_timeframes_env = os.getenv("DEFAULT_TIMEFRAMES") or os.getenv("TIMEFRAMES")
TIMEFRAMES: List[str] = [
    tf.strip() for tf in (_timeframes_env.split(",") if _timeframes_env else DEFAULT_TIMEFRAMES)
    if tf.strip()
]
if not TIMEFRAMES:
    TIMEFRAMES = DEFAULT_TIMEFRAMES

DAILY_LOOKBACK_DAYS = int(os.getenv("DAILY_LOOKBACK_DAYS", "120"))
INTRADAY_LOOKBACK_DAYS = int(os.getenv("INTRADAY_LOOKBACK_DAYS", "5"))
LONG_TERM_LOOKBACK_DAYS = int(os.getenv("LONG_TERM_LOOKBACK_DAYS", "365"))

TIMEFRAME_CONFIG = {
    "1D": {
        "interval": "1d",
        "lookback_days": DAILY_LOOKBACK_DAYS,
    },
    "15m": {
        "interval": "15m",
        "lookback_days": INTRADAY_LOOKBACK_DAYS,
    },
    "5m": {
        "interval": "5m",
        "lookback_days": INTRADAY_LOOKBACK_DAYS,
    },
    "30D": {
        "interval": "30d",
        "lookback_days": LONG_TERM_LOOKBACK_DAYS,
    },
}

# News analysis configuration
NEWS_ENABLED = os.getenv("NEWS_ENABLED", "1") == "1"
NEWS_LIMIT = int(os.getenv("NEWS_LIMIT", "10"))
NEWS_PER_HEADLINE = os.getenv("NEWS_PER_HEADLINE", "0") == "1"

# Fundamental analysis configuration
FUNDAMENTAL_ENABLED = os.getenv("FUNDAMENTAL_ENABLED", "1") == "1"

# Per-bucket limits with sensible fallbacks to NEWS_LIMIT
NEWS_LIMIT_SYMBOL = int(os.getenv("NEWS_LIMIT_SYMBOL", str(NEWS_LIMIT)))
NEWS_LIMIT_INDIA = int(os.getenv("NEWS_LIMIT_INDIA", str(NEWS_LIMIT)))
NEWS_LIMIT_GLOBAL = int(os.getenv("NEWS_LIMIT_GLOBAL", str(NEWS_LIMIT)))

# Overall cap across all buckets
NEWS_LIMIT_TOTAL = int(os.getenv("NEWS_LIMIT_TOTAL", "24"))
NEWS_SUMMARY = os.getenv("NEWS_SUMMARY", "0") == "1"
NEWS_PROVIDERS_COMPANY = [
    p.strip() for p in os.getenv("NEWS_PROVIDERS_COMPANY", "yfinance,fmp,tiingo").split(",") 
    if p.strip()
]
NEWS_PROVIDERS_WORLD = [
    p.strip() for p in os.getenv("NEWS_PROVIDERS_WORLD", "yfinance,fmp,tiingo").split(",") 
    if p.strip()
]

# Recency and source weighting
RECENCY_HOURS_2X = int(os.getenv("RECENCY_HOURS_2X", "24"))
RECENCY_HOURS_1_5X = int(os.getenv("RECENCY_HOURS_1_5X", "72"))
SOURCE_WEIGHTS_RAW = os.getenv("SOURCE_WEIGHTS", "")
SOURCE_WEIGHTS: Dict[str, float] = {}
if SOURCE_WEIGHTS_RAW:
    try:
        SOURCE_WEIGHTS = json.loads(SOURCE_WEIGHTS_RAW)
    except Exception:
        for pair in SOURCE_WEIGHTS_RAW.split(","):
            if ":" in pair:
                k, v = pair.split(":", 1)
                try:
                    SOURCE_WEIGHTS[k.strip()] = float(v.strip())
                except Exception:
                    pass

# TA encoding configuration
TA_FORMAT = os.getenv("TA_FORMAT", "csv")
TA_BARS = int(os.getenv("TA_BARS", "48"))
TA_PREC = int(os.getenv("TA_PREC", "3"))
TA_COLUMNS = os.getenv("TA_COLUMNS", "")

# Robustness settings
SEED = int(os.getenv("SEED", "42")) if os.getenv("SEED") else None
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "120"))
RETRIES = int(os.getenv("RETRIES", "2"))
BACKOFF = float(os.getenv("BACKOFF", "1.5"))
CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
NEWS_CACHE_TTL_MIN = int(os.getenv("NEWS_CACHE_TTL_MIN", "30"))

# Low-signal markers for filtering
LOW_SIGNAL_MARKERS = ("press release", "sponsored", "advertorial")

