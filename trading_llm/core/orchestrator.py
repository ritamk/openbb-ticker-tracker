"""Orchestrator that coordinates the entire trading pipeline."""
import time
from datetime import datetime
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from trading_llm.data.fetchers import get_technical_data, get_news
from trading_llm.agents.technical_analyst import analyze_technical
from trading_llm.agents.news_analyst import analyze_news
from trading_llm.agents.trader_agent import make_trade_decision
from trading_llm.agents.risk_manager import evaluate_trade
from trading_llm.core.utils import save_trade_log


def run_trading_cycle(symbol: str) -> Dict[str, Any]:
    """
    Run a complete trading cycle for a symbol.
    
    Steps:
    1. Fetch technical & news data
    2. Run technical and news analysts (in parallel)
    3. Run trader agent
    4. Run risk manager
    5. Log result
    
    Args:
        symbol: Stock symbol (e.g., "INFY.NS")
    
    Returns:
        Complete result dict with all analyses and decision
    """
    cycle_start = time.time()
    
    try:
        # Step 1: Fetch data
        fetch_start = time.time()
        ta_data = get_technical_data(symbol)
        headlines_dict = get_news(symbol, limit=10)  # Returns dict with buckets
        fetch_time = time.time() - fetch_start
        
        # Step 2: Run analysts in parallel
        analyst_start = time.time()
        
        def run_technical():
            return analyze_technical(symbol, use_csv=True)  # CSV format is more efficient
        
        def run_news():
            return analyze_news(symbol, headlines_dict=headlines_dict, use_cache=True)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            tech_future = executor.submit(run_technical)
            news_future = executor.submit(run_news)
            
            technical_report = tech_future.result()
            news_report = news_future.result()
        
        analyst_time = time.time() - analyst_start
        
        # Step 3: Run trader agent
        trader_start = time.time()
        trade_decision = make_trade_decision(symbol, technical_report, news_report)
        trader_time = time.time() - trader_start
        
        # Step 4: Run risk manager
        risk_start = time.time()
        risk_evaluation = evaluate_trade(
            symbol=symbol,
            decision=trade_decision,
            ta_snapshot=ta_data
        )
        risk_time = time.time() - risk_start
        
        # Step 5: Build final payload
        # Calculate total headlines count
        total_headlines = sum(len(v) for v in headlines_dict.values())
        
        result = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "technical": {
                "data": {k: v for k, v in ta_data.items() if k != "_df"},  # Remove internal _df field
                "analysis": technical_report
            },
            "news": {
                "headlines_count": total_headlines,
                "analysis": news_report
            },
            "trade": {
                "decision": trade_decision.get("decision", "HOLD"),
                "confidence": trade_decision.get("confidence", 0.5),
                "rationale": trade_decision.get("rationale", "")
            },
            "risk": {
                "approved": risk_evaluation.get("approved", False),
                "reason": risk_evaluation.get("reason", ""),
                "realized_vol_20": risk_evaluation.get("realized_vol_20", 0.0),
                "atr_percent": ta_data.get("atr_percent", 0.0)
            },
            "meta": {
                "timings": {
                    "fetch": round(fetch_time, 3),
                    "analyst": round(analyst_time, 3),
                    "trader": round(trader_time, 3),
                    "risk": round(risk_time, 3),
                    "total": round(time.time() - cycle_start, 3)
                }
            }
        }
        
        # Step 6: Log to JSONL
        save_trade_log(result)
        
        return result
        
    except Exception as e:
        # Error handling - return error result
        error_result = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": str(e),
            "trade": {
                "decision": "HOLD",
                "confidence": 0.0,
                "rationale": f"Pipeline error: {str(e)}"
            },
            "risk": {
                "approved": False,
                "reason": "Pipeline error",
                "realized_vol_20": 0.0,
                "atr_percent": 0.0
            }
        }
        save_trade_log(error_result)
        return error_result

