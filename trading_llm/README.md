# Phase 1: Lean 4-Agent LLM Trading System

A modular OpenBB-based LLM trading pipeline that generates structured trade decisions by analyzing technical indicators and news sentiment.

## Architecture

The system uses a **Lean 4-Agent Architecture**:

1. **Technical Analyst** (`gpt-4o-mini`) - Interprets technical indicators
2. **News Analyst** (`gpt-4o-mini`) - Analyzes news sentiment
3. **Trader Agent** (`gpt-4o`) - Synthesizes both into a final decision
4. **Risk Manager** (rule-based) - Applies volatility gates and duplicate checks

## Quick Start

```bash
# Ensure OPENAI_API_KEY is set
export OPENAI_API_KEY="your-key-here"

# Run pipeline
python -m trading_llm.main
```

## Project Structure

```
trading_llm/
├── data/
│   └── fetchers.py          # OpenBB data fetching (TA + news)
├── agents/
│   ├── technical_analyst.py # Technical indicator analysis
│   ├── news_analyst.py      # News sentiment analysis
│   ├── trader_agent.py      # Final trade decision
│   └── risk_manager.py      # Risk gates (ATR%, duplicates)
├── core/
│   ├── orchestrator.py      # Pipeline coordination
│   ├── prompts.py           # LLM prompt templates
│   └── utils.py             # LLM calls, JSON parsing, logging
├── logs/
│   └── trade_log.jsonl      # Structured output logs
├── tests/
│   └── test_risk_and_parser.py  # Unit tests
└── main.py                   # Entry point
```

## Trade Log Schema

Each log entry in `trade_log.jsonl` follows this structure:

### Top-Level Fields

- **`symbol`** (string): Stock symbol (e.g., "INFY.NS")
- **`timestamp`** (string): UTC ISO timestamp (e.g., "2025-01-31T11:00:00Z")

### Technical Analysis Section

**`technical.data`** (object):
- `rsi` (float): RSI(14) value
- `macd` (object): `{macd, signal, hist}` values
- `macd_signal` (string): `"bullish_cross"|"bearish_cross"|"none"`
- `bollinger` (object): `{lower, middle, upper, band_pct}`
- `bollinger_event` (string): `"upper_band_break"|"lower_band_break"|"near_upper"|"near_lower"|"within_bands"`
- `sma_50` (float): SMA(50) value
- `sma_200` (float): SMA(200) value
- `sma_50_vs_200` (string): `"above"|"below"|"equal"`
- `atr_14` (float): ATR(14) value
- `atr_percent` (float): `(ATR_14 / Close) * 100` - **Active risk gate**
- `realized_vol_20` (float): 20-day realized volatility (annualized) - **Background metric**
- `close` (float): Latest closing price
- `volume` (int): Latest volume

**`technical.analysis`** (object):
- `signal` (string): `"bullish"|"bearish"|"neutral"`
- `rationale` (string): Brief explanation
- `confidence` (float): 0.0 to 1.0

### News Analysis Section

**`news.headlines_count`** (int): Number of headlines analyzed

**`news.analysis`** (object):
- `sentiment` (string): `"positive"|"negative"|"neutral"`
- `summary` (string): 2-3 sentence summary
- `confidence` (float): 0.0 to 1.0

### Trade Decision Section

**`trade`** (object):
- `decision` (string): `"BUY"|"SELL"|"HOLD"`
- `confidence` (float): 0.0 to 1.0
- `rationale` (string): Decision explanation

### Risk Evaluation Section

**`risk`** (object):
- `approved` (boolean): Whether trade passed risk checks
- `reason` (string): Approval/rejection reason
- `realized_vol_20` (float): Background volatility metric (always logged)
- `atr_percent` (float): ATR% used for active gate

### Metadata Section

**`meta.timings`** (object):
- `fetch` (float): Data fetch time (seconds)
- `analyst` (float): Analyst LLM calls time (seconds)
- `trader` (float): Trader LLM call time (seconds)
- `risk` (float): Risk evaluation time (seconds)
- `total` (float): Total pipeline time (seconds)

### Error Handling

If the pipeline fails, the log entry includes:
- **`error`** (string): Error message
- **`trade.decision`**: Defaults to `"HOLD"` with `confidence: 0.0`
- **`risk.approved`**: `false` with reason `"Pipeline error"`

## Risk Rules

### Active Gate: ATR%
- **Reject** if `atr_percent > 3.0`
- Reason: `"ATR% ({value}%) exceeds 3% threshold"`

### Duplicate Check
- **Reject** if same decision repeated consecutively for the same symbol
- Reason: `"Duplicate decision: {decision} (same as last trade)"`
- Checks are per-symbol (different symbols can have same decision)

### Background Metric: Realized Volatility
- Always computed and logged: `realized_vol_20`
- Formula: `daily_returns.rolling(20).std() * sqrt(252)`
- Used for observability only (not a gate)

## Cost Optimization

Per symbol cycle:
- **Technical Analyst**: 1 call to `gpt-4o-mini` (~800 tokens)
- **News Analyst**: 1 call to `gpt-4o-mini` (~1200 tokens)
- **Trader Agent**: 1 call to `gpt-4o` (~1800 tokens)
- **Total**: ~3800 tokens per symbol (~₹2-4/day per symbol)

## Guardrails

1. **JSON Parsing**: Fallback to HOLD if LLM response fails to parse
2. **Token Limits**: Headlines truncated to 20, titles to 100 chars, summaries to 500 chars
3. **Schema Validation**: All LLM outputs validated and sanitized
4. **Error Handling**: Pipeline continues with safe defaults on any error

## Running Tests

```bash
python -m unittest trading_llm.tests.test_risk_and_parser
```

## Example Log Entry

```json
{
  "symbol": "INFY.NS",
  "timestamp": "2025-01-31T11:00:00Z",
  "technical": {
    "data": {
      "rsi": 46.3,
      "macd_signal": "bullish_cross",
      "atr_percent": 2.1,
      "realized_vol_20": 0.22
    },
    "analysis": {
      "signal": "bullish",
      "confidence": 0.72,
      "rationale": "RSI recovered from oversold, MACD bullish cross"
    }
  },
  "news": {
    "headlines_count": 15,
    "analysis": {
      "sentiment": "positive",
      "confidence": 0.68,
      "summary": "Strong earnings beat reported"
    }
  },
  "trade": {
    "decision": "BUY",
    "confidence": 0.78,
    "rationale": "Technical recovery confirmed with positive news support"
  },
  "risk": {
    "approved": true,
    "reason": "Within limits",
    "realized_vol_20": 0.22,
    "atr_percent": 2.1
  },
  "meta": {
    "timings": {
      "fetch": 1.2,
      "analyst": 2.5,
      "trader": 3.1,
      "risk": 0.001,
      "total": 6.8
    }
  }
}
```

## Next Steps (Phase 2+)

- `backtest_runner.py` - Backtest using logged decisions
- `dashboard.py` - Streamlit visualization of decisions and PnL

