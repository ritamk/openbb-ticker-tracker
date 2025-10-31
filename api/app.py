from fastapi import FastAPI

from .routers.tickers import router as tickers_router


app = FastAPI(title="Trading LLM API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight health probe for Cloud Run load balancer."""
    return {"status": "ok"}


app.include_router(tickers_router)

