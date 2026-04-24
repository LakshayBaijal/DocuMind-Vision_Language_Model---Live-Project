from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_PATH = Path(__file__).resolve().parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    llm_model: str = "llama-3.1-8b-instant"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.2
    groq_max_image_bytes: int = 3_500_000
    groq_response_json_mode: bool = True
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    review_confidence_threshold: float = 0.85


@lru_cache
def get_settings() -> Settings:
    return Settings()
