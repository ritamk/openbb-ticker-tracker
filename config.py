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

# News analysis configuration
NEWS_ENABLED = os.getenv("NEWS_ENABLED", "1") == "1"
NEWS_LIMIT = int(os.getenv("NEWS_LIMIT", "10"))
NEWS_PER_HEADLINE = os.getenv("NEWS_PER_HEADLINE", "0") == "1"
NEWS_LIMIT_TOTAL = int(os.getenv("NEWS_LIMIT_TOTAL", "24"))
NEWS_SUMMARY = os.getenv("NEWS_SUMMARY", "0") == "1"
NEWS_PROVIDERS_COMPANY = [
    p.strip() for p in os.getenv("NEWS_PROVIDERS_COMPANY", "yfinance,fmp,tiingo").split(",") 
    if p.strip()
]
NEWS_PROVIDERS_WORLD = [
    p.strip() for p in os.getenv("NEWS_PROVIDERS_WORLD", "fmp,tiingo").split(",") 
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
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "90"))
RETRIES = int(os.getenv("RETRIES", "2"))
BACKOFF = float(os.getenv("BACKOFF", "1.5"))
CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
NEWS_CACHE_TTL_MIN = int(os.getenv("NEWS_CACHE_TTL_MIN", "30"))

# Low-signal markers for filtering
LOW_SIGNAL_MARKERS = ("press release", "sponsored", "advertorial")

