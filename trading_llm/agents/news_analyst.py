"""News analyst agent using LLM to interpret news sentiment."""
import json
import os
from typing import Dict, Any, List
from trading_llm.core.prompts import format_news_analyst_prompt
from trading_llm.core.utils import llm_call, parse_json_response, load_cache, save_cache
from trading_llm.data.fetchers import aggregate_weighted_sentiment


def analyze_news(
    symbol: str, 
    headlines_dict: Dict[str, List[Dict[str, Any]]] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Analyze news headlines and generate sentiment assessment with weighted aggregation.
    
    Args:
        symbol: Stock symbol (e.g., "INFY.NS")
        headlines_dict: Dict with keys symbol_headlines, india_headlines, global_headlines
        use_cache: Whether to use caching for LLM analysis
    
    Returns:
        Dict with sentiment, summary, confidence, and per-bucket aggregations
    """
    # Default fallback response
    fallback = {
        "sentiment": "neutral",
        "summary": "No news available or failed to parse LLM response",
        "confidence": 0.5
    }
    
    # Check cache first
    cache_path = None
    if use_cache:
        try:
            import config
            cache_dir = config.CACHE_DIR
        except ImportError:
            cache_dir = ".cache"
        cache_path = os.path.join(cache_dir, f"news_{symbol.replace('.', '_')}.json")
        cached = load_cache(cache_path)
        
        if cached and cached.get("labels"):
            # Use cached labels and aggregate
            labels = cached.get("labels")
            news_payload = cached.get("payload", headlines_dict or {})
            
            if labels and isinstance(labels, dict) and news_payload:
                # Aggregate using weighted sentiment
                global_per = labels.get("global") or []
                india_per = labels.get("india") or []
                symbol_per = labels.get("symbol") or []
                
                g_head = news_payload.get("global_headlines", [])[:len(global_per)]
                i_head = news_payload.get("india_headlines", [])[:len(india_per)]
                s_head = news_payload.get("symbol_headlines", [])[:len(symbol_per)]
                
                # Weighted aggregation per bucket
                result = {
                    "global": aggregate_weighted_sentiment(global_per, g_head),
                    "india": aggregate_weighted_sentiment(india_per, i_head),
                    "symbol": aggregate_weighted_sentiment(symbol_per, s_head),
                    "method": "llm_headline_majority_weighted",
                    "cached": True,
                }
                
                # Overall sentiment (weighted by bucket confidence)
                sentiments = [result["global"]["sentiment"], result["india"]["sentiment"], result["symbol"]["sentiment"]]
                confidences = [result["global"]["confidence"], result["india"]["confidence"], result["symbol"]["confidence"]]
                overall_sentiment = max(set(sentiments), key=sentiments.count)
                overall_confidence = sum(c for s, c in zip(sentiments, confidences) if s == overall_sentiment) / len(sentiments) if sentiments else 0.5
                
                # Map bullish/bearish to positive/negative for consistency
                sentiment_map = {"bullish": "positive", "bearish": "negative", "neutral": "neutral"}
                overall_sentiment_mapped = sentiment_map.get(overall_sentiment.lower(), overall_sentiment)
                
                result["sentiment"] = overall_sentiment_mapped
                result["confidence"] = round(overall_confidence, 4)
                result["summary"] = cached.get("summary", f"Sentiment: {overall_sentiment_mapped} (cached)")
                
                # Also map per-bucket sentiments
                for bucket in ["global", "india", "symbol"]:
                    if bucket in result:
                        bucket_sentiment = result[bucket].get("sentiment", "neutral")
                        result[bucket]["sentiment"] = sentiment_map.get(bucket_sentiment.lower(), bucket_sentiment)
                
                return result
    
    # If no headlines, return neutral
    if not headlines_dict:
        return fallback
    
    try:
        # Format prompt with headlines dict
        prompt = format_news_analyst_prompt(symbol=symbol, headlines_dict=headlines_dict)
        
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
        
        # LLM returns per-bucket labels
        if isinstance(result, dict) and ("global" in result or "india" in result or "symbol" in result):
            # Extract per-bucket labels
            global_per = result.get("global", [])
            india_per = result.get("india", [])
            symbol_per = result.get("symbol", [])
            
            # Aggregate using weighted sentiment
            g_head = headlines_dict.get("global_headlines", [])[:len(global_per)]
            i_head = headlines_dict.get("india_headlines", [])[:len(india_per)]
            s_head = headlines_dict.get("symbol_headlines", [])[:len(symbol_per)]
            
            aggregated = {
                "global": aggregate_weighted_sentiment(global_per, g_head),
                "india": aggregate_weighted_sentiment(india_per, i_head),
                "symbol": aggregate_weighted_sentiment(symbol_per, s_head),
                "method": "llm_headline_majority_weighted",
            }
            
            # Overall sentiment
            sentiments = [aggregated["global"]["sentiment"], aggregated["india"]["sentiment"], aggregated["symbol"]["sentiment"]]
            confidences = [aggregated["global"]["confidence"], aggregated["india"]["confidence"], aggregated["symbol"]["confidence"]]
            overall_sentiment = max(set(sentiments), key=sentiments.count)
            overall_confidence = sum(c for s, c in zip(sentiments, confidences) if s == overall_sentiment) / len(sentiments) if sentiments else 0.5
            
            # Aggregate already returns positive/negative format
            aggregated["sentiment"] = overall_sentiment
            aggregated["confidence"] = round(overall_confidence, 4)
            aggregated["summary"] = result.get("summary", f"Sentiment: {overall_sentiment}")
            
            # Cache the labels and payload
            if cache_path:
                cache_data = {
                    "payload": headlines_dict,
                    "labels": {"global": global_per, "india": india_per, "symbol": symbol_per},
                    "summary": aggregated.get("summary", "")
                }
                save_cache(cache_path, cache_data)
            
            return aggregated
        else:
            # Simple format (backward compatibility)
            if "sentiment" not in result:
                result["sentiment"] = fallback["sentiment"]
            if "summary" not in result:
                result["summary"] = fallback["summary"]
            if "confidence" not in result:
                result["confidence"] = fallback["confidence"]
            
            # Ensure sentiment is valid
            valid_sentiments = ["positive", "negative", "neutral"]
            if result["sentiment"].lower() not in valid_sentiments:
                result["sentiment"] = fallback["sentiment"]
            
            # Ensure confidence is float in [0, 1]
            try:
                conf = float(result["confidence"])
                result["confidence"] = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                result["confidence"] = fallback["confidence"]
            
            # Truncate summary if too long
            if len(result["summary"]) > 500:
                result["summary"] = result["summary"][:497] + "..."
            
            return result
        
    except Exception as e:
        # Return fallback on any error
        fallback["summary"] = f"Error during analysis: {str(e)}"
        return fallback

