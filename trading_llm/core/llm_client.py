"""OpenAI client wrapper with retries and error handling."""
import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI

from . import config, utils


class LLMClient:
    """Wrapper for OpenAI client with retries and configuration."""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), 
            timeout=config.OPENAI_TIMEOUT
        )
    
    def call(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        *,
        return_response: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        """Make an LLM call with retries and JSON response format.
        
        Args:
            system: System prompt
            user: User prompt
            model: Model name (defaults to config.MODEL)
            return_response: If True, return raw response object; if False, return parsed JSON
        
        Returns:
            Parsed JSON dict or raw response object depending on return_response
        """
        model = model or config.MODEL
        fmt = response_format or {"type": "json_object"}
        seed = seed if seed is not None else config.SEED
        
        def _make_call():
            return self.client.chat.completions.create(
                model=model,
                response_format=fmt,
                temperature=temperature,
                seed=seed,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        
        resp = utils.with_retries(_make_call)
        
        if return_response:
            return resp
        
        try:
            return json.loads(resp.choices[0].message.content)
        except Exception:
            return {"error": "parse_failed", "raw": resp.choices[0].message.content}
    
    def get_usage(self, resp) -> Dict[str, int] | None:
        """Extract usage metrics from response."""
        if hasattr(resp, "usage"):
            return {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return None

