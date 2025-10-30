"""Risk manager with volatility gates and duplicate trade checks."""
from typing import Dict, Any, Optional


# In-memory store for last trade decisions per symbol
_last_trades: Dict[str, Dict[str, Any]] = {}


def evaluate_trade(
    symbol: str,
    decision: Dict[str, Any],
    ta_snapshot: Dict[str, Any],
    last_trade: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate trade decision against risk rules.
    
    Args:
        symbol: Stock symbol (e.g., "INFY.NS")
        decision: Trade decision dict with "decision" key
        ta_snapshot: Technical data snapshot with atr_percent and realized_vol_20
        last_trade: Optional last trade dict for this symbol (auto-loaded from memory)
    
    Returns:
        Dict with approved (bool), reason (str), realized_vol_20 (float)
    """
    # Load last trade from memory if not provided
    if last_trade is None:
        last_trade = _last_trades.get(symbol)
    
    # Get ATR percent (active gate)
    atr_percent = ta_snapshot.get("atr_percent", 0.0)
    realized_vol_20 = ta_snapshot.get("realized_vol_20", 0.0)
    
    # Rule 1: Reject if ATR% > 3%
    if atr_percent > 3.0:
        result = {
            "approved": False,
            "reason": f"ATR% ({atr_percent}%) exceeds 3% threshold",
            "realized_vol_20": realized_vol_20
        }
        # Don't update last trade if rejected
        return result
    
    # Rule 2: Reject if same decision repeated consecutively
    current_decision = decision.get("decision", "").upper()
    if last_trade and last_trade.get("decision", "").upper() == current_decision:
        result = {
            "approved": False,
            "reason": f"Duplicate decision: {current_decision} (same as last trade)",
            "realized_vol_20": realized_vol_20
        }
        return result
    
    # All checks passed
    result = {
        "approved": True,
        "reason": "Within limits",
        "realized_vol_20": realized_vol_20
    }
    
    # Update last trade in memory
    _last_trades[symbol] = {
        "decision": current_decision,
        "timestamp": ta_snapshot.get("timestamp", "")
    }
    
    return result


def clear_last_trade(symbol: Optional[str] = None) -> None:
    """
    Clear last trade memory (useful for testing or reset).
    
    Args:
        symbol: Symbol to clear, or None to clear all
    """
    if symbol:
        _last_trades.pop(symbol, None)
    else:
        _last_trades.clear()

