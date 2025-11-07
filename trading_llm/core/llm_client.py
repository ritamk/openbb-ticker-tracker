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
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """Make an LLM call with retries and JSON response format.
        
        Args:
            system: System prompt
            user: User prompt
            model: Model name (defaults to config.MODEL)
            return_response: If True, return raw response object; if False, return parsed JSON
            temperature: Temperature for response generation (defaults to 0.0 if not specified)
            seed: Random seed for reproducibility
        
        Returns:
            Parsed JSON dict or raw response object depending on return_response
        """
        model = model or config.MODEL
        fmt = response_format or {"type": "json_object"}
        temperature = temperature if temperature is not None else 0.0
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
    
    def call_agent(
        self,
        agent_type: str,
        system: str,
        user: str,
        *,
        return_response: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
    ):
        """Call an agent with its specific model and temperature settings.
        
        Args:
            agent_type: One of 'technical', 'news', 'fundamental', 'trader'
            system: System prompt
            user: User prompt
            return_response: If True, return raw response object
            response_format: Custom response format
        
        Returns:
            Parsed JSON dict or raw response object
        """
        agent_configs = {
            "technical": {
                "model": config.TA_MODEL,
                "temperature": config.TA_TEMPERATURE,
            },
            "news": {
                "model": config.NEWS_MODEL,
                "temperature": config.NEWS_TEMPERATURE,
            },
            "fundamental": {
                "model": config.FUNDAMENTAL_MODEL,
                "temperature": config.FUNDAMENTAL_TEMPERATURE,
            },
            "trader": {
                "model": config.TRADER_MODEL,
                "temperature": config.TRADER_TEMPERATURE,
            },
        }
        
        agent_config = agent_configs.get(agent_type, {})
        if not agent_config:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return self.call(
            system=system,
            user=user,
            model=agent_config["model"],
            temperature=agent_config["temperature"],
            return_response=return_response,
            response_format=response_format,
        )

