"""Environment-driven settings for ingest and the Streamlit UI."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    embedding_model: str = "hkunlp/instructor-xl"
    embedding_device: str = "cpu"
    chroma_persist_dir: Path = Path("./chroma_db")
    data_dir: Path = Path("./data")
    chunk_size: int = 100
    chunk_overlap: int = 5
    retriever_k: int = 3


settings = Settings()
