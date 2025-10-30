"""Technical analyst agent using LLM to interpret technical indicators."""
from typing import Dict, Any
from trading_llm.core.prompts import format_tech_analyst_prompt
from trading_llm.core.utils import llm_call, parse_json_response
from trading_llm.data.fetchers import get_technical_data_csv


def analyze_technical(symbol: str, ta_data: Dict[str, Any] = None, use_csv: bool = True) -> Dict[str, Any]:
    """
    Analyze technical indicators and generate signal.
    
    Args:
        symbol: Stock symbol (e.g., "INFY.NS")
        ta_data: Technical data dict from get_technical_data() (optional if use_csv=True)
        use_csv: If True, use compact CSV format (more token-efficient)
    
    Returns:
        Dict with signal, rationale, confidence
    """
    # Default fallback response
    fallback = {
        "signal": "neutral",
        "rationale": "Failed to parse LLM response - defaulting to neutral",
        "confidence": 0.5
    }
    
    try:
        # Format prompt with CSV or JSON
        if use_csv:
            try:
                ta_csv_spec, ta_csv_data = get_technical_data_csv(symbol, keep=48, prec=3)
                prompt = format_tech_analyst_prompt(
                    symbol=symbol,
                    ta_csv_spec=ta_csv_spec,
                    ta_csv_data=ta_csv_data
                )
            except Exception:
                # Fallback to JSON if CSV fails
                if ta_data:
                    prompt = format_tech_analyst_prompt(symbol=symbol, ta_data=ta_data)
                else:
                    raise ValueError("No technical data available")
        else:
            if not ta_data:
                raise ValueError("ta_data required when use_csv=False")
            prompt = format_tech_analyst_prompt(symbol=symbol, ta_data=ta_data)
        
        # Call LLM
        response_text = llm_call(
            prompt=prompt,
            model="gpt-4o-mini",
            timeout_s=30,
            max_retries=2,
            temperature=0.2
        )
        
        # Parse response
        result = parse_json_response(response_text, fallback)
        
        # Validate schema
        if "signal" not in result:
            result["signal"] = fallback["signal"]
        if "rationale" not in result:
            result["rationale"] = fallback["rationale"]
        if "confidence" not in result:
            result["confidence"] = fallback["confidence"]
        
        # Ensure signal is valid
        valid_signals = ["bullish", "bearish", "neutral"]
        if result["signal"].lower() not in valid_signals:
            result["signal"] = fallback["signal"]
        
        # Ensure confidence is float in [0, 1]
        try:
            conf = float(result["confidence"])
            result["confidence"] = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            result["confidence"] = fallback["confidence"]
        
        return result
        
    except Exception as e:
        # Return fallback on any error
        fallback["rationale"] = f"Error during analysis: {str(e)}"
        return fallback

