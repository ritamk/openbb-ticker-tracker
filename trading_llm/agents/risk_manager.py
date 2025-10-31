"""Rule-based risk manager for trade approvals."""
from __future__ import annotations

from math import sqrt
from typing import Any, Dict, List, Optional


class RiskManager:
    """Apply lightweight guardrails before executing trades."""

    def __init__(self, max_volatility: float = 0.03, duplicate_penalty: bool = True):
        self.max_volatility = max_volatility
        self.duplicate_penalty = duplicate_penalty

    def evaluate(
        self,
        symbol: str,
        timeframe: str,
        trade: Dict[str, Any],
        *,
        volatility: Optional[float] = None,
        last_trade: Optional[Dict[str, Any]] = None,
        rows: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return approval boolean with reason message and computed metadata."""
        computed_vol = volatility if volatility is not None else self._compute_volatility(rows)
        reasons: List[str] = []
        approved = True

        if computed_vol is not None and computed_vol > self.max_volatility:
            approved = False
            reasons.append(
                f"Volatility {computed_vol:.3f} exceeds limit {self.max_volatility:.3f}"
            )

        if self.duplicate_penalty and last_trade and last_trade.get("decision") == trade.get("decision"):
            approved = False
            reasons.append("Duplicate decision detected")

        if approved:
            reasons.append("Within limits")

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "approved": approved,
            "reason": "; ".join(reasons),
            "volatility": computed_vol,
            "last_decision": last_trade.get("decision") if last_trade else None,
        }

    @staticmethod
    def _compute_volatility(rows: Optional[List[Dict[str, Any]]], window: int = 20) -> Optional[float]:
        if not rows or len(rows) < window + 1:
            return None
        closes = []
        for row in rows[-(window + 1) :]:
            try:
                closes.append(float(row.get("close")))
            except (TypeError, ValueError):
                return None
        if len(closes) < window + 1:
            return None
        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
            if closes[i - 1]
        ]
        if len(returns) < window:
            return None
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return sqrt(variance)
