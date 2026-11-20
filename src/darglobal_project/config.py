"""Central application configuration managed via Pydantic Settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application runtime settings."""

    # Project directories
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    vector_store_dir: Path = data_dir / "vector_store"

    # API and Server
    host: str = "0.0.0.0"
    port: int = 8000

    # AI & Retrieval
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-3.1-flash-lite-preview"
    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/free"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
