"""Utility functions for retries, caching, and text normalization."""
import os
import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List

import config


def with_retries(fn: Callable, *, retries: int = None, backoff: float = None) -> Any:
    """Execute function with retries and exponential backoff."""
    retries = retries if retries is not None else config.RETRIES
    backoff = backoff if backoff is not None else config.BACKOFF
    last_exc = None
    for i in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if i == retries:
                break
            time.sleep(backoff ** i)
    raise last_exc


def load_cache(path: str) -> Dict[str, Any] | None:
    """Load cached data if fresh."""
    try:
        with open(path) as f:
            data = json.load(f)
        ts = datetime.fromisoformat(data.get("timestamp"))
        age = (datetime.utcnow() - ts).total_seconds() / 60.0
        if age <= config.NEWS_CACHE_TTL_MIN:
            return data
    except Exception:
        pass
    return None


def save_cache(path: str, data: Dict[str, Any]) -> None:
    """Save data to cache with timestamp."""
    try:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        data = {**data, "timestamp": datetime.utcnow().isoformat()}
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def normalize_title(t: str) -> str:
    """Normalize title for deduplication."""
    return ''.join(ch for ch in t.lower() if ch.isalnum() or ch.isspace()).strip()


def is_low_signal(title: str) -> bool:
    """Check if title appears to be low-signal content."""
    tl = title.lower()
    return any(m in tl for m in config.LOW_SIGNAL_MARKERS)


def dedupe_and_filter(headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates and filter low-signal headlines."""
    seen = set()
    out = []
    for h in headlines:
        t = h.get("title") or ""
        if not t or is_low_signal(t):
            continue
        key = normalize_title(t)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out

