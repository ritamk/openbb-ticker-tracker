# Lean 4-Agent Trading Pipeline (Phase 1)

This package implements Phase 1 of an LLM-assisted trading workflow tailored for Indian equities. It follows a lean four-agent architecture:

1. **TechnicalAnalyst** – interprets engineered indicators and proposes a structured signal.
2. **NewsAnalyst** – scores short headlines for sentiment and key drivers.
3. **TraderAgent** – synthesises technical + news context into a single trade decision.
4. **RiskManager** – applies rule-based guard rails (volatility, duplicate signals).

All runs are orchestrated by `core/orchestrator.py`, with results saved as JSON lines for audit and future backtesting.

---

## 1. Environment Prerequisites

- Python 3.10+
- [OpenBB Platform](https://docs.openbb.co/platform) (or compatible SDK) installed and authenticated (`openbb` module importable).
- OpenAI Python client (`pip install openai==1.*`).
- API credentials in the environment:
  - `OPENAI_API_KEY`
  - Any provider-specific keys required by OpenBB news sources.
- (Optional) `.env` file – the modules auto-load it if `dotenv` is available.

Install baseline dependencies:

```bash
pip install -r requirements.txt
```

> **Tip:** if you only want the trading package, the minimal extras are `openai`, `openbb`, `numpy`, `pandas`, and `python-dotenv` (optional).

---

## 2. Key Configuration Flags (`trading_llm/core/config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `gpt-4o-mini` | Base model fallback for agents. |
| `TA_MODEL` / `NEWS_MODEL` | inherits `MODEL` | Override per agent if needed. |
| `TRADER_MODEL` | `gpt-4o` | Heavier reasoning model for final decision. |
| `TIMEFRAMES` | `1D,15m` | Comma-separated list of timeframes to process. |
| `DAILY_LOOKBACK_DAYS` | `120` | Days of OHLCV history for daily runs. |
| `INTRADAY_LOOKBACK_DAYS` | `5` | Days of OHLCV history for intraday runs. |
| `NEWS_LIMIT` | `10` | Default per-bucket headline fetch limit. |
| `NEWS_LIMIT_SYMBOL/INDIA/GLOBAL` | — | Override per bucket. |
| `NEWS_LIMIT_TOTAL` | `24` | Cap across all buckets. |
| `NEWS_PER_HEADLINE` | `0` | If set to `1`, retain per-headline sentiment in output. |
| `NEWS_SUMMARY` | `0` | If `1`, attempt driver extraction. |
| `TA_BARS` | `48` | Bars kept in prompts per timeframe. |
| `TA_PREC` | `3` | Numeric precision for compact CSV. |
| `OPENAI_TIMEOUT` | `30` | Seconds for LLM client timeout. |
| `RETRIES` / `BACKOFF` | `2` / `1.5` | Retry policy for LLM + data fetchers. |
| `CACHE_DIR` | `.cache` | Location for news cache (shared helpers). |

Set overrides via environment variables or `.env` file.

---

## 3. Running the Pipeline

### Quick Start

```bash
python -m trading_llm.main --pretty
```

Default behaviour:
- Symbols: `INFY.NS`, `TCS.NS`, `RELIANCE.NS`
- Timeframes: `1D` and `15m`
- Logs appended to `trading_llm/logs/trade_log.jsonl`

### Options

```
usage: python -m trading_llm.main [symbols ...] [--timeframes TF1,TF2] [--no-log] [--pretty]
```

Examples:

- Run only daily timeframe on a single symbol without logging:
  ```bash
  python -m trading_llm.main HDFCBANK.NS --timeframes 1D --no-log --pretty
  ```

- Force intraday-only run with compact JSON output:
  ```bash
  python -m trading_llm.main SBIN.NS --timeframes 15m
  ```

Results stream to stdout and (unless `--no-log`) append to `logs/trade_log.jsonl` for downstream analytics.

---

## 4. Agent Details

| Agent | Location | Model | Notes |
|-------|----------|-------|-------|
| TechnicalAnalyst | `agents/technical_analyst.py` | `TA_MODEL` | Consumes compact TA CSV and summary. Falls back to HOLD on parse failure. |
| NewsAnalyst | `agents/news_analyst.py` | `NEWS_MODEL` | Works off headline titles only; defaults to neutral sentiment if LLM fails. |
| TraderAgent | `agents/trader_agent.py` | `TRADER_MODEL` | Prioritises TA, uses news to nudge confidence / rationale. |
| RiskManager | `agents/risk_manager.py` | rule-based | Rejects if volatility > 3% or consecutive identical decisions. |

Shared helpers:
- `data/fetchers.py` handles OHLCV retrieval, indicator enrichment, news payload preparation, and TA summaries.
- `data/ta_builder.py` compresses indicator tables for prompt inclusion.
- `core/utils.py` centralises retry logic, logging, and JSON parsing.
- `core/prompts.py` contains compact, JSON-first prompt templates.

---

## 5. Logging & Backtesting Hooks

Each run produces a JSON object per timeframe with:
- `technical`, `news`, `trade`, and `risk` sections
- Raw LLM responses for auditing (`meta.raw`)
- Token usage metadata when available (`meta.usage`)

Logs store in `trading_llm/logs/trade_log.jsonl`. You can tail the file:

```bash
tail -f trading_llm/logs/trade_log.jsonl
```

These structured entries are intended for later Phase 2/3 work (backtesting, dashboarding, or post-trade analysis).

---

## 6. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: No module named 'openai'` | OpenAI SDK missing | `pip install openai==1.*` |
| `ModuleNotFoundError: No module named 'openbb'` | OpenBB SDK not installed/authenticated | Install OpenBB Platform, ensure you can `python -c "from openbb import obb"` |
| News outputs are empty | Providers throttled / credentials missing | Reduce `NEWS_LIMIT*`, check provider keys, or disable news (`NEWS_ENABLED=0`). |
| Frequent HOLD fallbacks | Prompt parse errors or missing data | Inspect `meta.raw` in log, ensure data fetchers returning adequate bars/headlines. |
| High volatility rejections | Market is too volatile for rule | Increase `max_volatility` when instantiating `RiskManager` or adjust rule to your tolerance. |

---

## 7. Extending the Pipeline

- Add more indicators: enrich `fetchers.compute_indicators` and update prompt schema.
- Swap LLM vendors: adjust `core/llm_client.py` or last-mile prompts accordingly.
- Plug into a backtester: consume `trade_log.jsonl` and map decisions against historical prices.
- Hook alerts: trigger notifications when `RiskManager` approves a BUY/SELL with high confidence.

Contributions are welcome—keep prompts concise, maintain JSON-first responses, and ensure token budgets remain under ~4k tokens per symbol per run.
