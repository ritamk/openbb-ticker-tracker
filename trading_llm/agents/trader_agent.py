"""Trader agent that synthesizes technical and news analysis into a trade decision."""
from typing import Dict, Any
from trading_llm.core.prompts import format_trader_prompt
from trading_llm.core.utils import llm_call, parse_json_response


def make_trade_decision(
    symbol: str, 
    technical_report: Dict[str, Any], 
    news_report: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Make final trade decision by synthesizing technical and news analysis.
    
    Args:
        symbol: Stock symbol (e.g., "INFY.NS")
        technical_report: Output from technical_analyst.analyze_technical()
        news_report: Output from news_analyst.analyze_news()
    
    Returns:
        Dict with decision (BUY/SELL/HOLD), confidence, rationale
    """
    # Default fallback response
    fallback = {
        "decision": "HOLD",
        "confidence": 0.5,
        "rationale": "Failed to parse LLM response - defaulting to HOLD"
    }
    
    try:
        # Format prompt
        prompt = format_trader_prompt(
            symbol=symbol,
            technical_report=technical_report,
            news_report=news_report
        )
        
        # Call LLM with gpt-4o for reasoning
        response_text = llm_call(
            prompt=prompt,
            model="gpt-4o",
            timeout_s=30,
            max_retries=2,
            temperature=0.2
        )
        
        # Parse response
        result = parse_json_response(response_text, fallback)
        
        # Validate schema
        if "decision" not in result:
            result["decision"] = fallback["decision"]
        if "rationale" not in result:
            result["rationale"] = fallback["rationale"]
        if "confidence" not in result:
            result["confidence"] = fallback["confidence"]
        
        # Ensure decision is valid
        valid_decisions = ["BUY", "SELL", "HOLD"]
        decision_upper = result["decision"].upper()
        if decision_upper not in valid_decisions:
            result["decision"] = fallback["decision"]
        else:
            result["decision"] = decision_upper
        
        # Ensure confidence is float in [0, 1]
        try:
            conf = float(result["confidence"])
            result["confidence"] = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            result["confidence"] = fallback["confidence"]
        
        # Truncate rationale if too long
        if len(result["rationale"]) > 500:
            result["rationale"] = result["rationale"][:497] + "..."
        
        return result
        
    except Exception as e:
        # Return fallback on any error
        fallback["rationale"] = f"Error during decision making: {str(e)}"
        return fallback

