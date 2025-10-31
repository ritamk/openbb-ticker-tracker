"""LLM trader agent that synthesizes analyst outputs."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..core import config, prompts, utils
from ..core.llm_client import LLMClient

_REQUIRED_KEYS = {"decision", "confidence", "rationale"}


class TraderAgent:
    """Combine TA and news reports into a single trade decision."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or config.TRADER_MODEL
        self._client = LLMClient()

    def decide(
        self,
        symbol: str,
        timeframe: str,
        technical_report: Dict[str, Any],
        news_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Invoke the trader agent and return structured decision + metadata."""
        tech_blob = json.dumps(technical_report, ensure_ascii=False, separators=(",", ":"))
        news_blob = json.dumps(news_report, ensure_ascii=False, separators=(",", ":"))

        system_prompt = prompts.TRADER_PROMPT["system"]
        user_prompt = prompts.TRADER_PROMPT["template"].format(
            symbol=symbol,
            timeframe=timeframe,
            technical_report=tech_blob,
            news_report=news_blob,
        )

        try:
            response = self._client.call(
                system_prompt,
                user_prompt,
                model=self.model,
                return_response=True,
            )
        except Exception as exc:  # pragma: no cover
            return {
                "trade": self._fallback_payload(symbol, timeframe, error=str(exc)),
                "usage": None,
                "raw": None,
            }

        usage = self._client.get_usage(response)
        content = response.choices[0].message.content if response and response.choices else ""
        parsed = utils.parse_json_response(content)

        if isinstance(parsed, dict) and _REQUIRED_KEYS.issubset(parsed.keys()):
            parsed.setdefault("risk_notes", "")
            parsed.setdefault("alignment", {})
            parsed["confidence"] = self._sanitize_confidence(parsed.get("confidence"))
            return {"trade": parsed, "usage": usage, "raw": content}

        return {
            "trade": self._fallback_payload(symbol, timeframe, raw=content),
            "usage": usage,
            "raw": content,
        }

    @staticmethod
    def _sanitize_confidence(confidence: Any) -> float:
        try:
            value = float(confidence)
        except Exception:
            value = 0.4
        return max(0.0, min(1.0, value))

    @staticmethod
    def _fallback_payload(
        symbol: str,
        timeframe: str,
        *,
        error: Optional[str] = None,
        raw: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "decision": "HOLD",
            "confidence": 0.4,
            "rationale": "Defaulting to HOLD after trader agent failure.",
            "risk_notes": "",
            "alignment": {},
        }
        if error:
            payload["error"] = error
        if raw:
            payload["raw"] = raw
        return payload
