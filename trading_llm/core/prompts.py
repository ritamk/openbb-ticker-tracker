"""Prompt templates for the Lean 4-Agent trading pipeline."""
from __future__ import annotations

from textwrap import dedent


TECH_ANALYST_SCHEMA = dedent(
    """
    {
      "signal": "buy|sell|hold",
      "confidence": 0.0,
      "timeframe": "",
      "indicators": {
        "rsi_14": 0.0,
        "macd": {"macd": 0.0, "signal": 0.0, "hist": 0.0},
        "bbands": {"lower": 0.0, "middle": 0.0, "upper": 0.0, "band_pct": 0.0},
        "stoch": {"k": 0.0, "d": 0.0},
        "adx_14": 0.0,
        "atr_14": 0.0,
        "ema_50": 0.0,
        "sma_20": 0.0,
        "obv": 0.0
      },
      "rules_triggered": [],
      "risk": {"stop_loss": 0.0, "take_profit": 0.0}
    }
    """
).strip()

TECH_ANALYST_PROMPT = {
    "system": dedent(
        f"""
        You are a disciplined technical analyst. Only use provided time-series features (already lagged).
        Cite recent bar offsets (e.g., -1 for latest) for crossovers or band touches. When computing risk,
        use ATR-based logic: if signal == buy, stop_loss = close - 1.5 * ATR, take_profit = close + 2.5 * ATR
        (mirror for sell). Do not hallucinate unseen values. Return strict JSON matching this schema:
        {TECH_ANALYST_SCHEMA}
        """
    ).strip(),
    "template": dedent(
        """
        Symbol: {symbol}
        Timeframe: {timeframe}
        Indicators Summary: {summary}
        Optional Market Context: {market_context}
        TA Spec: {ta_spec}
        Task: Produce the JSON response following the schema.
        TA CSV (oldest -> newest):
        {ta_csv}
        """
    ).strip(),
}

NEWS_ANALYST_SCHEMA = dedent(
    """
    {
      "sentiment": "bullish|bearish|neutral",
      "summary": "",
      "confidence": 0.0,
      "drivers": []
    }
    """
).strip()

NEWS_ANALYST_PROMPT = {
    "system": dedent(
        f"""
        You are a concise market news analyst. Label sentiment strictly from headlines provided. Avoid speculation.
        If unsure, lean neutral. Summaries must be under 60 words and focus on market-moving factors. Return JSON
        matching this schema:
        {NEWS_ANALYST_SCHEMA}
        """
    ).strip(),
    "template": dedent(
        """
        Symbol: {symbol}
        Headlines by bucket (titles only):
        {headlines_json}
        Optional Prior Sentiment Drift: {prior_sentiment}
        Task: Produce the JSON summary.
        """
    ).strip(),
}

FUNDAMENTAL_ANALYST_SCHEMA = dedent(
    """
    {
      "signal": "undervalued|overvalued|fair",
      "confidence": 0.0,
      "summary": "",
      "metrics": {
        "pe_ratio": 0.0,
        "pb_ratio": 0.0,
        "roe": 0.0,
        "debt_to_equity": 0.0,
        "profit_margin": 0.0,
        "operating_margin": 0.0,
        "gross_margin": 0.0,
        "revenue_growth": 0.0,
        "earnings_growth": 0.0,
        "ev_ebitda": 0.0,
        "market_cap": 0.0,
        "institutional_holdings": 0.0,
        "insider_holdings": 0.0
      },
      "drivers": []
    }
    """
).strip()

FUNDAMENTAL_ANALYST_PROMPT = {
    "system": dedent(
        f"""
        You are a disciplined fundamental analyst. Analyze company financial metrics to determine valuation.
        Compare metrics to industry averages and historical norms when possible. Consider profitability, growth,
        debt levels, and valuation ratios. Summaries must be under 60 words and focus on key value drivers.
        Signal should be: undervalued (attractive entry), overvalued (expensive), or fair (reasonably priced).
        Return strict JSON matching this schema:
        {FUNDAMENTAL_ANALYST_SCHEMA}
        """
    ).strip(),
    "template": dedent(
        """
        Symbol: {symbol}
        Fundamental Data:
        {fundamental_json}
        Task: Analyze the fundamental metrics and produce the JSON response following the schema.
        Focus on valuation, profitability, growth trajectory, and financial health.
        """
    ).strip(),
}

TRADER_SCHEMA = dedent(
    """
    {
      "decision": "BUY|SELL|HOLD",
      "confidence": 0.0,
      "rationale": "",
      "risk_notes": "",
      "alignment": {
        "technical": "",
        "news": "",
        "fundamental": ""
      }
    }
    """
).strip()

TRADER_PROMPT = {
    "system": dedent(
        f"""
        You are the trader of record. Synthesize inputs from three analysts: technical, news, and fundamental.
        Start from the technical analyst's proposal (price action is primary). Use news sentiment to adjust
        confidence or highlight risks. Use fundamental analysis to assess long-term value and confirm or
        question the technical signal. If inputs conflict, prioritize price action but note the divergence.
        For example: strong technical buy + undervalued fundamental = high confidence; strong technical buy +
        overvalued fundamental = moderate confidence with caution. Return JSON matching this schema:
        {TRADER_SCHEMA}
        """
    ).strip(),
    "template": dedent(
        """
        Symbol: {symbol}
        Timeframe: {timeframe}
        Technical Report:
        {technical_report}
        News Report:
        {news_report}
        Fundamental Report:
        {fundamental_report}
        Task: Issue a single trade decision JSON following the schema, considering all three inputs.
        """
    ).strip(),
}
