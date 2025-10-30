"""News fetching, sentiment analysis, and weighted aggregation."""
import json
import time
from datetime import datetime
from typing import Any, Dict, List

from openbb import obb

import config
import utils


def weight_for(headline_date: str, source: str | None) -> float:
    """Compute weight for a headline based on recency and source credibility."""
    w = 1.0
    try:
        dt = datetime.fromisoformat(headline_date.replace("Z", "+00:00"))
        hrs = max(0, (datetime.utcnow() - dt).total_seconds() / 3600.0)
        if hrs <= config.RECENCY_HOURS_2X:
            w *= 2.0
        elif hrs <= config.RECENCY_HOURS_1_5X:
            w *= 1.5
    except Exception:
        pass
    if source and source in config.SOURCE_WEIGHTS:
        w *= config.SOURCE_WEIGHTS[source]
    return w


def aggregate_weighted(per_list: List[Dict[str, Any]], headlines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate sentiment with weights based on recency and source."""
    counts = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
    for lab, h in zip(per_list, headlines):
        s = str(lab.get("sentiment", "neutral")).lower()
        counts[s if s in counts else "neutral"] += weight_for(h.get("date", ""), h.get("source"))
    total = sum(counts.values()) or 1.0
    majority = max(counts, key=counts.get)
    return {
        "counts": {k: round(v, 2) for k, v in counts.items()},
        "sentiment": majority,
        "confidence": round(counts[majority] / total, 4),
    }


def fetch_company_headlines(sym: str, providers: List[str], per_bucket: int) -> List[Dict[str, Any]]:
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
            return utils.dedupe_and_filter(df.to_dict(orient="records"))[:per_bucket]
        except Exception:
            continue
    return []


def cap_headlines_across_buckets(
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


def prepare_news_payload(sym: str, limit: int = 10) -> Dict[str, Any]:
    """Prepare news payload with provider fallback and capping."""
    def _fetch_with_retry(symbol: str, providers: List[str], per_bucket: int) -> List[Dict[str, Any]]:
        return utils.with_retries(lambda: fetch_company_headlines(symbol, providers, per_bucket))

    # Use provider fallback for company news
    symbol_headlines = _fetch_with_retry(sym, config.NEWS_PROVIDERS_COMPANY, limit)
    
    # For indices, try providers if available, otherwise fall back to yfinance
    india_headlines = _fetch_with_retry(
        "^NSEI", 
        config.NEWS_PROVIDERS_WORLD if config.NEWS_PROVIDERS_WORLD else ["yfinance"], 
        limit
    )
    global_headlines = _fetch_with_retry(
        "^GSPC", 
        config.NEWS_PROVIDERS_WORLD if config.NEWS_PROVIDERS_WORLD else ["yfinance"], 
        limit
    )
    
    headlines_dict = {
        "symbol_headlines": symbol_headlines,
        "india_headlines": india_headlines,
        "global_headlines": global_headlines,
    }
    
    # Cap total headlines across buckets
    capped = cap_headlines_across_buckets(headlines_dict, config.NEWS_LIMIT_TOTAL)
    
    return {
        "symbol": sym,
        **capped
    }

