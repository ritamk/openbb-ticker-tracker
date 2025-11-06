"""LLM-powered news and sentiment analyst."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..core import config, prompts, utils
from ..core.llm_client import LLMClient
from ..data import fetchers


class NewsAnalyst:
    """Summarize headlines into structured sentiment."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or config.NEWS_MODEL
        self._client = LLMClient()

    def analyze(
        self,
        symbol: str,
        headlines_payload: Dict[str, List[Dict[str, Any]]],
        prior_sentiment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the news analyst agent and return structured sentiment."""
        compact = {
            bucket: [
                {
                    "title": h.get("title", "")[:180],
                    "date": h.get("date", ""),
                }
                for h in headlines_payload.get(bucket, [])
                if h.get("title")
            ]
            for bucket in ["symbol_headlines", "india_headlines", "global_headlines"]
        }
        headlines_json = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        prior_sentiment = prior_sentiment or "(none)"

        system_prompt = prompts.NEWS_ANALYST_PROMPT["system"]
        user_prompt = prompts.NEWS_ANALYST_PROMPT["template"].format(
            symbol=symbol,
            headlines_json=headlines_json,
            prior_sentiment=prior_sentiment,
        )

        try:
            response = self._client.call(
                system_prompt,
                user_prompt,
                model=self.model,
                return_response=True,
            )
        except Exception as exc:  # pragma: no cover
            return self._fallback_payload(symbol, headlines_payload, error=str(exc))

        usage = self._client.get_usage(response)
        content = response.choices[0].message.content if response and response.choices else ""
        parsed = utils.parse_json_response(content)

        if isinstance(parsed, dict) and {"sentiment", "summary", "confidence"}.issubset(parsed.keys()):
            parsed.setdefault("drivers", [])
            parsed.setdefault("confidence", float(parsed.get("confidence", 0.5)))
            return {"news": parsed, "usage": usage, "raw": content}

        return {
            "news": self._fallback_payload(symbol, headlines_payload, raw=content),
            "usage": usage,
            "raw": content,
        }

    @staticmethod
    def _fallback_payload(
        symbol: str,
        headlines_payload: Dict[str, List[Dict[str, Any]]],
        *,
        raw: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        aggregated = {
            "symbol": fetchers.aggregate_weighted(
                [{"sentiment": "neutral"} for _ in headlines_payload.get("symbol_headlines", [])],
                headlines_payload.get("symbol_headlines", []),
            ),
            "india": fetchers.aggregate_weighted(
                [{"sentiment": "neutral"} for _ in headlines_payload.get("india_headlines", [])],
                headlines_payload.get("india_headlines", []),
            ),
            "global": fetchers.aggregate_weighted(
                [{"sentiment": "neutral"} for _ in headlines_payload.get("global_headlines", [])],
                headlines_payload.get("global_headlines", []),
            ),
        }
        payload: Dict[str, Any] = {
            "sentiment": "neutral",
            "summary": "Unable to parse news sentiment; defaulting to neutral.",
            "confidence": 0.2,
            "drivers": [],
            "fallback": aggregated,
        }
        if raw:
            payload["raw"] = raw
        if error:
            payload["error"] = error
        return payload
