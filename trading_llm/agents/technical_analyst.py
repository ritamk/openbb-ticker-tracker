"""LLM-powered technical analyst agent."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..core import config, prompts, utils
from ..core.llm_client import LLMClient
from ..data import ta_builder

_REQUIRED_KEYS = {"signal", "confidence", "timeframe", "indicators", "risk"}


class TechnicalAnalyst:
    """Analyze technical data and produce a structured trade proposal."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or config.TA_MODEL
        self._client = LLMClient()

    def analyze(
        self,
        symbol: str,
        timeframe: str,
        rows: List[Dict[str, Any]],
        summary: Dict[str, Any],
        market_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the technical analysis agent and return response + metadata."""
        ta_spec, ta_csv = ta_builder.build_ta_csv(
            rows,
            keep=config.TA_BARS,
            prec=config.TA_PREC,
        )
        summary_blob = json.dumps(summary, separators=(",", ":")) if summary else "{}"
        market_context = market_context or "(none provided)"

        system_prompt = prompts.TECH_ANALYST_PROMPT["system"]
        user_prompt = prompts.TECH_ANALYST_PROMPT["template"].format(
            symbol=symbol,
            timeframe=timeframe,
            summary=summary_blob,
            market_context=market_context,
            ta_spec=ta_spec,
            ta_csv=ta_csv[:120000],
        )

        try:
            response = self._client.call(
                system_prompt,
                user_prompt,
                model=self.model,
                return_response=True,
            )
        except Exception as exc:  # pragma: no cover - network/LLM failures
            return {
                "ta": self._fallback_payload(timeframe, error=str(exc)),
                "usage": None,
                "raw": None,
            }

        usage = self._client.get_usage(response)
        content = response.choices[0].message.content if response and response.choices else ""
        parsed = utils.parse_json_response(content)

        if isinstance(parsed, dict) and _REQUIRED_KEYS.issubset(parsed.keys()):
            parsed.setdefault("timeframe", timeframe)
            return {"ta": parsed, "usage": usage, "raw": content}

        return {
            "ta": self._fallback_payload(timeframe, raw=content),
            "usage": usage,
            "raw": content,
        }

    @staticmethod
    def _fallback_payload(timeframe: str, *, error: Optional[str] = None, raw: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "signal": "hold",
            "confidence": 0.25,
            "timeframe": timeframe,
            "indicators": {},
            "rules_triggered": [],
            "risk": {"stop_loss": None, "take_profit": None},
            "error": error or "ta_parse_failed",
        }
        if raw:
            payload["raw"] = raw
        return payload
