"""Utility functions for retries, caching, logging, and text normalization."""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import config


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


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON content from an LLM response."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to locate JSON substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
    except Exception:
        return None
    return None


_trade_log_path = Path(__file__).resolve().parents[1] / "logs" / "trade_log.jsonl"


def save_trade_log(entry: Dict[str, Any], path: Path | str | None = None) -> None:
    """Append a trade entry to the JSONL log."""
    target = Path(path) if path else _trade_log_path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        json.dump(entry, fh)
        fh.write("\n")


def load_last_trade(symbol: str, timeframe: str, path: Path | str | None = None) -> Optional[Dict[str, Any]]:
    """Return the most recent trade decision for a symbol/timeframe if available."""
    target = Path(path) if path else _trade_log_path
    if not target.exists():
        return None
    try:
        with target.open("r", encoding="utf-8") as fh:
            for line in reversed(fh.readlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("symbol") == symbol and record.get("timeframe") == timeframe:
                    return record
    except Exception:
        return None
    return None


def llm_call(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
    return_response: bool = False,
    temperature: float = 0.0,
    seed: Optional[int] = None,
):
    """Helper to invoke the shared LLM client with retries and JSON response format."""
    from .llm_client import LLMClient  # Lazy import to avoid circular dependency

    client = LLMClient()
    fmt = response_format if response_format is not None else {"type": "json_object"}
    return client.call(
        system,
        user,
        model=model,
        return_response=return_response,
        response_format=fmt,
        temperature=temperature,
        seed=seed,
    )

