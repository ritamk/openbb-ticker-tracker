# Trading LLM API

This document captures how to run, configure, and validate the FastAPI service that exposes the trading pipeline.

## Prerequisites
- Python environment prepared for the project (`pip install -r requirements.txt`).
- Valid `OPENAI_API_KEY` exported in the environment (required by the LLM agents).
- OpenBB SDK credentials configured as expected by `trading_llm/data/fetchers.py`.

## Running the API locally
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Once running, visit `http://localhost:8000/docs` for interactive Swagger docs.

## Configuration
Runtime settings are defined in `api/config.py` and can be overridden through environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `TRADING_API_NEWS_USE_CACHE` | Toggle news cache usage for the LLM analysis | `True` |
| `TRADING_API_NEWS_LIMIT` | Headline limit per bucket passed to `get_news` | `10` |
| `TRADING_API_TA_LOOKBACK` | Default lookback window (days) for technical data | `200` |
| `TRADING_API_TA_KEEP_ROWS` | Number of rows kept when exporting TA CSV snapshots | `48` |
| `TRADING_API_TA_PRECISION` | Decimal precision used for TA CSV export | `3` |
| `TRADING_API_TRADES_BACKGROUND` | Run `/trade/run` requests asynchronously via background jobs | `True` |
| `TRADING_API_LOG_FILE` | Log file destination for API requests | `trading_llm/logs/api.log` |

## Endpoints
- `GET /health` – service readiness probe.
- `POST /analyze/news` – fetches and summarizes recent headlines. Accepts tickers list plus optional `limit`/`use_cache` overrides.
- `POST /signals/ta` – returns a technical snapshot, optional CSV encoding, and LLM-driven analysis.
- `POST /trade/run` – enqueues (or runs, if background disabled) the end-to-end trading pipeline. Response includes a `job_id`.
- `GET /trade/jobs/{job_id}` – retrieves job status and the final pipeline payload once complete.

## Testing
```
pytest trading_llm/tests/test_api.py
```

The suite patches external dependencies (OpenBB, OpenAI) to keep the tests offline and deterministic.

## Next steps
- Configure deployment credentials (Google Cloud + Firebase) before running the Cloud Run workflow described in `deploy/cloud_run.md`.
- Use `deploy/cloud_run_deploy.sh` for repeatable builds and deployments once credentials are in place.
- Consider adding authentication (Firebase Auth or Cloud IAP) before exposing the API publicly.

