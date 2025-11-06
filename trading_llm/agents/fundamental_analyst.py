"""LLM-powered fundamental analyst agent."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..core import config, prompts, utils
from ..core.llm_client import LLMClient

_REQUIRED_KEYS = {"signal", "confidence", "summary", "metrics", "drivers"}


class FundamentalAnalyst:
    """Analyze fundamental data and produce a structured valuation assessment."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or config.FUNDAMENTAL_MODEL
        self._client = LLMClient()

    def analyze(
        self,
        symbol: str,
        fundamental_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the fundamental analysis agent and return response + metadata."""
        fundamental_json = json.dumps(
            fundamental_payload.get("metrics", {}),
            ensure_ascii=False,
            separators=(",", ":"),
            indent=2,
        )

        system_prompt = prompts.FUNDAMENTAL_ANALYST_PROMPT["system"]
        user_prompt = prompts.FUNDAMENTAL_ANALYST_PROMPT["template"].format(
            symbol=symbol,
            fundamental_json=fundamental_json,
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
                "fundamental": self._fallback_payload(symbol, error=str(exc)),
                "usage": None,
                "raw": None,
            }

        usage = self._client.get_usage(response)
        content = response.choices[0].message.content if response and response.choices else ""
        parsed = utils.parse_json_response(content)

        if isinstance(parsed, dict) and _REQUIRED_KEYS.issubset(parsed.keys()):
            parsed.setdefault("drivers", [])
            parsed.setdefault("confidence", float(parsed.get("confidence", 0.5)))
            return {"fundamental": parsed, "usage": usage, "raw": content}

        return {
            "fundamental": self._fallback_payload(symbol, raw=content),
            "usage": usage,
            "raw": content,
        }

    @staticmethod
    def _fallback_payload(
        symbol: str,
        *,
        raw: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "signal": "fair",
            "summary": "Unable to analyze fundamental data; defaulting to fair valuation.",
            "confidence": 0.2,
            "metrics": {},
            "drivers": [],
        }
        if raw:
            payload["raw"] = raw
        if error:
            payload["error"] = error
        return payload

