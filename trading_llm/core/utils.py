"""Core utility functions for LLM calls, JSON parsing, and logging."""
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
from openai import OpenAI


def llm_call(
    prompt: str, 
    model: str, 
    timeout_s: int = 30, 
    max_retries: int = 2, 
    temperature: float = 0.2
) -> str:
    """
    Make an LLM call with retries and exponential backoff.
    
    Args:
        prompt: The prompt string to send
        model: Model name (e.g., "gpt-4o-mini", "gpt-4o")
        timeout_s: Request timeout in seconds
        max_retries: Maximum number of retries
        temperature: Sampling temperature
    
    Returns:
        Raw response text from the model
    
    Raises:
        Exception: If all retries fail
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key, timeout=timeout_s)
    
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                # Exponential backoff
                wait_time = (1.5 ** attempt)
                time.sleep(wait_time)
            else:
                break
    
    raise Exception(f"LLM call failed after {max_retries + 1} attempts: {str(last_exception)}")


def parse_json_response(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse JSON response with fallback handling.
    
    Args:
        text: Raw text response from LLM
        fallback: Fallback dict to return if parsing fails
    
    Returns:
        Parsed JSON dict or fallback
    """
    if not text:
        return fallback
    
    # Try to extract JSON from markdown code blocks if present
    text_clean = text.strip()
    if "```json" in text_clean:
        start = text_clean.find("```json") + 7
        end = text_clean.find("```", start)
        if end > start:
            text_clean = text_clean[start:end].strip()
    elif "```" in text_clean:
        start = text_clean.find("```") + 3
        end = text_clean.find("```", start)
        if end > start:
            text_clean = text_clean[start:end].strip()
    
    try:
        result = json.loads(text_clean)
        if isinstance(result, dict):
            return result
        else:
            return fallback
    except json.JSONDecodeError:
        return fallback


def load_cache(path: str) -> Dict[str, Any] | None:
    """
    Load cached data if fresh.
    
    Args:
        path: Path to cache file
    
    Returns:
        Cached data dict or None if cache is stale/missing
    """
    try:
        import config
        cache_ttl_min = config.NEWS_CACHE_TTL_MIN
    except ImportError:
        cache_ttl_min = 30  # Default 30 minutes
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = datetime.fromisoformat(data.get("timestamp", ""))
        age = (datetime.utcnow() - ts).total_seconds() / 60.0
        if age <= cache_ttl_min:
            return data
    except Exception:
        pass
    return None


def save_cache(path: str, data: Dict[str, Any]) -> None:
    """
    Save data to cache with timestamp.
    
    Args:
        path: Path to cache file
        data: Data dict to cache
    """
    try:
        import config
        cache_dir = config.CACHE_DIR
    except ImportError:
        cache_dir = ".cache"
    
    try:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else cache_dir, exist_ok=True)
        data = {**data, "timestamp": datetime.utcnow().isoformat()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def save_trade_log(entry: Dict[str, Any], path: str = "trading_llm/logs/trade_log.jsonl") -> None:
    """
    Append a trade log entry to JSONL file.
    
    Args:
        entry: Dictionary to log (will have timestamp added if missing)
        path: Path to JSONL log file
    """
    # Ensure timestamp is present
    if "timestamp" not in entry:
        entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Append to JSONL file
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # Silently fail if logging fails (non-critical)
        print(f"Warning: Failed to write trade log: {str(e)}")

