"""Configuration helpers for the Trading LLM API."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    news_use_cache: bool = Field(True, env="TRADING_API_NEWS_USE_CACHE")
    news_limit: int = Field(10, env="TRADING_API_NEWS_LIMIT")
    ta_default_lookback: int = Field(200, env="TRADING_API_TA_LOOKBACK")
    ta_keep_rows: int = Field(48, env="TRADING_API_TA_KEEP_ROWS")
    ta_precision: int = Field(3, env="TRADING_API_TA_PRECISION")
    run_trades_in_background: bool = Field(True, env="TRADING_API_TRADES_BACKGROUND")
    log_file: str = Field("trading_llm/logs/api.log", env="TRADING_API_LOG_FILE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()

