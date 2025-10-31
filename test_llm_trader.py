"""Main orchestration for LLM-based trading analysis."""
import json
import os
import time
from datetime import datetime
from typing import Any, Dict

import config
import llm_client
import news_handler
import ta_builder
import utils


def analyze_news_sentiment(symbol: str, llm: llm_client.LLMClient) -> Dict[str, Any] | None:
    """Analyze news sentiment for symbol, India, and global markets."""
    news_start = time.time()
    cache_path = os.path.join(config.CACHE_DIR, f"news_{symbol.replace('.', '_')}.json")
    cached = utils.load_cache(cache_path)
    
    if cached:
        news_payload = cached.get("payload")
        labels = cached.get("labels")
        drivers = cached.get("drivers")
    else:
        news_payload = news_handler.prepare_news_payload(symbol, limit=config.NEWS_LIMIT)
        symbol_titles = [h.get("title") for h in news_payload.get("symbol_headlines", []) if h.get("title")]
        india_titles = [h.get("title") for h in news_payload.get("india_headlines", []) if h.get("title")]
        global_titles = [h.get("title") for h in news_payload.get("global_headlines", []) if h.get("title")]
        total_headlines = len(symbol_titles) + len(india_titles) + len(global_titles)
        
        labels = None
        drivers = None
        
        if total_headlines > 0:
            news_system = (
                "You are a disciplined market news analyst. "
                "Label each headline as bullish, bearish, or neutral strictly from the headline text. "
                "Do not infer beyond the headline. Return strict JSON."
            )
            labeling_schema = (
                "{\n"
                '  "global": [{"title": "", "sentiment": "bullish|bearish|neutral"}],\n'
                '  "india":  [{"title": "", "sentiment": "bullish|bearish|neutral"}],\n'
                '  "symbol": [{"title": "", "sentiment": "bullish|bearish|neutral"}]\n'
                "}"
            )
            compact_titles = {"global": global_titles, "india": india_titles, "symbol": symbol_titles}
            news_user = (
                f"Symbol: {symbol}\n"
                "Task: Assign one of {bullish,bearish,neutral} to each headline.\n"
                "Rules:\n"
                "- Base labels strictly on headline text.\n"
                "- If unclear, use neutral.\n"
                "- Output strictly as JSON with this schema (order preserved):\n"
                f"{labeling_schema}\n"
                "Here are the headlines (titles only) as JSON:\n"
                f"{json.dumps(compact_titles)[:120000]}"
            )
            try:
                labels = llm.call(news_system, news_user)
                # Coerce labels to match input lengths; fill missing with neutral
                if isinstance(labels, dict):
                    def _coerce(key: str, titles: list[str]) -> list[dict[str, str]]:
                        raw = labels.get(key)
                        safe: list[dict[str, str]] = []
                        for idx, title in enumerate(titles):
                            if isinstance(raw, list) and idx < len(raw) and isinstance(raw[idx], dict):
                                sent = str(raw[idx].get("sentiment", "neutral")).lower()
                            else:
                                sent = "neutral"
                            if sent not in ("bullish", "bearish", "neutral"):
                                sent = "neutral"
                            safe.append({"title": title, "sentiment": sent})
                        return safe
                    labels = {
                        "global": _coerce("global", global_titles),
                        "india": _coerce("india", india_titles),
                        "symbol": _coerce("symbol", symbol_titles),
                    }
                cache_data = {"payload": news_payload, "labels": labels}
            except Exception:
                labels = None
                cache_data = None

            # Optional drivers extraction
            if config.NEWS_SUMMARY and labels and isinstance(labels, dict):
                try:
                    summary_system = "Extract 2-3 terse noun-phrases from titles only. No sentences. Return JSON."
                    summary_user = json.dumps({
                        "global": global_titles[:10],
                        "india": india_titles[:10],
                        "symbol": symbol_titles[:10]
                    })
                    drivers = llm.call(summary_system, summary_user)
                    if cache_data:
                        cache_data["drivers"] = drivers
                except Exception:
                    pass
            
            # Save cache if we fetched new data
            if cache_data:
                utils.save_cache(cache_path, cache_data)

    # Process labels into news_json (works for both cached and fresh)
    if labels and isinstance(labels, dict):
        global_per = labels.get("global") or []
        india_per = labels.get("india") or []
        symbol_per = labels.get("symbol") or []
        
        # Use original headline arrays for weighted aggregation
        g_head = news_payload.get("global_headlines", [])[:len(global_per)]
        i_head = news_payload.get("india_headlines", [])[:len(india_per)]
        s_head = news_payload.get("symbol_headlines", [])[:len(symbol_per)]
        
        news_json = {
            "global": news_handler.aggregate_weighted(global_per, g_head),
            "india": news_handler.aggregate_weighted(india_per, i_head),
            "symbol": news_handler.aggregate_weighted(symbol_per, s_head),
            "method": "llm_headline_majority_weighted",
        }
        if config.NEWS_PER_HEADLINE:
            news_json["details"] = {"global": global_per, "india": india_per, "symbol": symbol_per}
        if drivers:
            news_json["drivers"] = drivers
        
        return {"news_json": news_json, "timing": round(time.time() - news_start, 3)}
    
    return {"news_json": None, "timing": round(time.time() - news_start, 3)}


def analyze_ta(symbol: str, rows: list[Dict[str, Any]], news_json: Dict[str, Any] | None, llm: llm_client.LLMClient) -> Dict[str, Any]:
    """Analyze technical indicators and generate trading signal."""
    ta_start = time.time()
    
    # Build compact TA payload
    ta_spec, ta_csv = ta_builder.build_ta_csv(rows, keep=config.TA_BARS, prec=config.TA_PREC)

    # Prepare prompts with compact data and optional market context
    market_ctx = None
    if news_json:
        g = news_json.get("global", {})
        i = news_json.get("india", {})
        s = news_json.get("symbol", {})
        market_ctx = (
            "Market Context: "
            f"global={g.get('sentiment','neutral')}({g.get('confidence',0)}), "
            f"india={i.get('sentiment','neutral')}({i.get('confidence',0)}), "
            f"symbol={s.get('sentiment','neutral')}({s.get('confidence',0)})"
        )

    # Enhanced system prompt with schema
    ta_schema_text = (
        'Schema: {"signal": "buy|sell|hold", "confidence": 0.0, "timeframe": "1D", '
        '"indicators": {"rsi_14": 0.0, "macd": {"macd": 0.0, "signal": 0.0, "hist": 0.0}, '
        '"bbands": {"lower": 0.0, "middle": 0.0, "upper": 0.0, "band_pct": 0.0}, '
        '"stoch": {"k": 0.0, "d": 0.0}, "adx_14": 0.0, "atr_14": 0.0, "ema_50": 0.0, '
        '"sma_20": 0.0, "obv": 0.0}, "rules_triggered": [], '
        '"risk": {"stop_loss": 0.0, "take_profit": 0.0}}'
    )
    
    system = (
        "You are a disciplined technical analyst. "
        "Only use provided features (already shifted to avoid leakage). "
        "If market context is provided, use it only to adjust confidence/risk, not to override TA. "
        "When stating crossovers/band touches/divergences, include last index offset (e.g., -1 for last bar). "
        "For risk: if signal=buy, set stop_loss = close - 1.5*ATR, take_profit = close + 2.5*ATR "
        "(or adjust based on TA). Use only numbers present in the latest rows; avoid inventing values. "
        f"Return JSON matching this schema: {ta_schema_text}"
    )

    user_parts = [
        f"Symbol: {symbol}",
        "Data: Daily OHLCV with engineered features (compact CSV).",
        ta_spec,
    ]
    if market_ctx:
        user_parts.append(market_ctx)
    user_parts.extend([
        "Task: Produce trade indicators and an actionable signal.",
        "Use RSI, MACD (line/signal/hist), SMA20, EMA50, BBands, ATR, Stoch (k/d), ADX, OBV from the payload.",
        "Summarize the latest state and any recent crossovers/band touches/divergences (cite bar index offsets).",
        "Include risk metrics using ATR-based calculations.",
        "Here is the TA CSV (oldest->newest):",
        ta_csv[:120000],
    ])
    user = "\n".join(user_parts)

    try:
        resp = llm.call(system, user, return_response=True)
        usage = llm.get_usage(resp)
        
        # Parse TA JSON
        try:
            ta_json = json.loads(resp.choices[0].message.content)
            # Schema validation
            required_keys = {"signal", "confidence", "timeframe", "indicators", "risk"}
            if not isinstance(ta_json, dict) or not required_keys.issubset(ta_json.keys()):
                ta_json = {"error": "schema_mismatch", "raw": resp.choices[0].message.content}
        except Exception:
            ta_json = {"error": "parse_failed", "raw": resp.choices[0].message.content if resp else "No response"}
    except Exception as e:
        ta_json = {"error": "api_call_failed", "message": str(e)}
        usage = None
    
    return {
        "ta_json": ta_json,
        "usage": usage,
        "timing": round(time.time() - ta_start, 3)
    }


def main(symbol: str = "RELIANCE.NS"):
    """Main entry point for LLM trading analysis."""
    start_time = time.time()
    
    with open("ta_payload.json") as f:
        rows = json.load(f)

    llm = llm_client.LLMClient()

    # Track metrics
    meta: Dict[str, Any] = {"timings": {}, "usage": {}}

    # Analyze news sentiment
    news_json: Dict[str, Any] | None = None
    if config.NEWS_ENABLED:
        news_result = analyze_news_sentiment(symbol, llm)
        news_json = news_result["news_json"]
        meta["timings"]["news"] = news_result["timing"]

    # Analyze technical indicators
    ta_result = analyze_ta(symbol, rows, news_json, llm)
    ta_json = ta_result["ta_json"]
    if ta_result["usage"]:
        meta["usage"] = ta_result["usage"]
    meta["timings"]["ta"] = ta_result["timing"]
    meta["timings"]["total"] = round(time.time() - start_time, 3)

    final_output: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": "1D",
        "generated_at": datetime.utcnow().isoformat(),
        "ta": ta_json,
        "news": news_json,
        "meta": meta,
    }
    print(json.dumps(final_output))


if __name__ == "__main__":
    main()
