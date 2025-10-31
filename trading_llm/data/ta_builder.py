"""Technical analysis CSV building and formatting."""
from typing import Any, Dict, List, Tuple

from ..core import config


def format_number_short(x: Any, prec: int) -> str:
    """Format number with adaptive precision to save tokens."""
    try:
        f = float(x)
    except Exception:
        return ""
    # Adaptive precision to save tokens
    if abs(f) >= 1000:
        s = f"{f:.2f}"
    elif abs(f) >= 1:
        s = f"{f:.{min(max(prec, 1), 5)}f}"
    else:
        s = f"{f:.{min(max(prec + 2, 2), 6)}f}"
    s = s.rstrip("0").rstrip(".")
    return s


def build_ta_csv(rows: List[Dict[str, Any]], keep: int, prec: int) -> Tuple[str, str]:
    """Build compact CSV representation of TA data."""
    # Map compact codes to expected keys from test_ta.py payload
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
    
    # Filter fields if TA_COLUMNS is specified
    if config.TA_COLUMNS:
        allowed_codes = {c.strip() for c in config.TA_COLUMNS.split(",") if c.strip()}
        fields = [f for f in all_fields if f[0] in allowed_codes]
    else:
        fields = all_fields
    
    lines: List[str] = []
    slice_rows = rows[-keep:]
    for r in slice_rows:
        vals: List[str] = []
        for _, key in fields:
            vals.append(format_number_short(r.get(key), prec))
        lines.append("|".join(vals))
    csv_data = "\n".join(lines)
    # For readability in prompt, return a brief spec string
    spec = (
        f"headers={'|'.join(code for code, _ in fields)} "
        "(rows are oldest->newest)"
    )
    return spec, csv_data

