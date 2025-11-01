from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.tickers import router as tickers_router


app = FastAPI(title="Trading LLM API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://brok-c367d.web.app",
        "https://brok-c367d.firebaseapp.com",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight health probe for Cloud Run load balancer."""
    return {"status": "ok"}


app.include_router(tickers_router)

