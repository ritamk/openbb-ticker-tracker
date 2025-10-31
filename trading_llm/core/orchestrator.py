"""Pipeline orchestrator for the Lean 4-Agent trading system."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..agents.news_analyst import NewsAnalyst
from ..agents.risk_manager import RiskManager
from ..agents.technical_analyst import TechnicalAnalyst
from ..agents.trader_agent import TraderAgent
from ..core import config, utils
from ..data import fetchers


def _market_context_from_news(news: Optional[Dict[str, Any]]) -> str:
    if not news:
        return "News disabled or unavailable."
    sentiment = news.get("sentiment", "neutral")
    confidence = news.get("confidence", 0.0)
    drivers = news.get("drivers") or []
    drivers_txt = ", ".join(drivers[:3]) if drivers else "n/a"
    return f"Sentiment={sentiment} (confidence={confidence:.2f}), drivers={drivers_txt}"


class TradingOrchestrator:
    """Coordinate data fetchers, LLM agents, and risk checks."""

    def __init__(self) -> None:
        self.tech_agent = TechnicalAnalyst()
        self.news_agent = NewsAnalyst()
        self.trader_agent = TraderAgent()
        self.risk_manager = RiskManager()

    def run(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        *,
        save_log: bool = True,
    ) -> Dict[str, Any]:
        tfs = timeframes or config.TIMEFRAMES
        news_payload: Optional[Dict[str, Any]] = None
        news_result: Optional[Dict[str, Any]] = None
        news_usage: Optional[Dict[str, Any]] = None
        news_raw: Optional[str] = None

        if config.NEWS_ENABLED:
            news_payload = fetchers.prepare_news_payload(symbol, limit=config.NEWS_LIMIT)
            news_analysis = self.news_agent.analyze(symbol, news_payload)
            news_result = news_analysis.get("news")
            news_usage = news_analysis.get("usage")
            news_raw = news_analysis.get("raw")
        else:
            news_payload = {
                "symbol": symbol,
                "symbol_headlines": [],
                "india_headlines": [],
                "global_headlines": [],
            }

        results: List[Dict[str, Any]] = []
        aggregate_usage: Dict[str, Any] = {"news": news_usage}

        for timeframe in tfs:
            rows = fetchers.get_indicator_rows(symbol, timeframe)
            ta_summary = fetchers.summarize_ta(rows)
            market_context = _market_context_from_news(news_result)
            ta_analysis = self.tech_agent.analyze(
                symbol,
                timeframe,
                rows,
                ta_summary,
                market_context=market_context,
            )

            technical = ta_analysis.get("ta")
            ta_usage = ta_analysis.get("usage")
            aggregate_usage.setdefault("technical", {})[timeframe] = ta_usage

            trade_input_news = news_result or {
                "sentiment": "neutral",
                "summary": "News disabled",
                "confidence": 0.0,
            }
            trade_analysis = self.trader_agent.decide(
                symbol,
                timeframe,
                technical,
                trade_input_news,
            )
            trade = trade_analysis.get("trade")
            aggregate_usage.setdefault("trader", {})[timeframe] = trade_analysis.get("usage")

            last_trade = utils.load_last_trade(symbol, timeframe)
            risk = self.risk_manager.evaluate(
                symbol,
                timeframe,
                trade,
                rows=rows,
                last_trade=last_trade,
            )

            record = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.utcnow().isoformat(),
                "technical": technical,
                "news": news_result,
                "trade": trade,
                "risk": risk,
                "meta": {
                    "summary": ta_summary,
                    "usage": {
                        "technical": ta_usage,
                        "trader": trade_analysis.get("usage"),
                        "news": news_usage,
                    },
                    "raw": {
                        "technical": ta_analysis.get("raw"),
                        "trader": trade_analysis.get("raw"),
                        "news": news_raw,
                    },
                },
            }
            results.append(record)

            if save_log:
                utils.save_trade_log(record)

        return {
            "symbol": symbol,
            "generated_at": datetime.utcnow().isoformat(),
            "results": results,
            "news": news_result,
            "news_payload": news_payload,
            "meta": {
                "usage": aggregate_usage,
            },
        }


def run_trading_cycle(
    symbol: str,
    timeframes: Optional[List[str]] = None,
    *,
    save_log: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper for running a trading cycle without instantiating the class."""
    orchestrator = TradingOrchestrator()
    return orchestrator.run(symbol, timeframes=timeframes, save_log=save_log)
