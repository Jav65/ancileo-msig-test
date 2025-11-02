from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.1-70b-versatile", env="GROQ_MODEL")
    groq_vision_model: str = Field(
        "meta-llama/llama-4-scout-17b-16e-instruct",
        env="GROQ_VISION_MODEL",
    )
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    vector_db_path: Path = Field(BASE_DIR / "../data/vector_store", env="VECTOR_DB_PATH")
    claims_data_path: Path = Field(BASE_DIR / "../data/claims_stats.parquet", env="CLAIMS_DATA_PATH")
    stripe_api_key: str = Field("", env="STRIPE_API_KEY")
    stripe_secret_key: str = Field("", env="stripe_secret_key")
    stripe_webhook_secret: str = Field("", env="STRIPE_WEBHOOK_SECRET")
    payments_base_url: str = Field("http://localhost:8086", env="PAYMENTS_BASE_URL")
    payment_status_url: str = Field("http://localhost:8086/payments", env="PAYMENT_STATUS_URL")
    claims_db_host: str = Field("", env="CLAIMS_DB_HOST")
    claims_db_port: int = Field(5432, env="CLAIMS_DB_PORT")
    claims_db_name: str = Field("", env="CLAIMS_DB_NAME")
    claims_db_user: str = Field("", env="CLAIMS_DB_USER")
    claims_db_password: str = Field("", env="CLAIMS_DB_PASSWORD")
    claims_db_table: str = Field("hackathon.claims", env="CLAIMS_DB_TABLE")
    taxonomy_path: Path = Field(BASE_DIR / "../Taxonomy/Taxonomy_Hackathon.json", env="TAXONOMY_PATH")
    policy_documents_dir: Path = Field(BASE_DIR / "../Policy_Wordings", env="POLICY_DOCUMENTS_DIR")
    twilio_account_sid: str = Field("", env=["TWILIO_ACCOUNT_SID", "TWILIO_SID"])
    twilio_auth_token: str = Field("", env="TWILIO_AUTH_TOKEN")
    session_secret: str = Field("change-this-session-secret", env="SESSION_SECRET")
    ancileo_api_key: str = Field("", env="ANCILEO_API_KEY")
    ancileo_base_url: str = Field(
        "https://dev.api.ancileo.com/v1/travel/front",
        env="ANCILEO_BASE_URL",
    )
    ancileo_default_market: str = Field("SG", env="ANCILEO_DEFAULT_MARKET")
    ancileo_default_language: str = Field("en", env="ANCILEO_DEFAULT_LANGUAGE")
    ancileo_default_channel: str = Field("white-label", env="ANCILEO_DEFAULT_CHANNEL")
    ancileo_default_device: str = Field("DESKTOP", env="ANCILEO_DEFAULT_DEVICE")
    google_client_id: str = Field("", env="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field("", env="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = Field("", env="GOOGLE_REDIRECT_URI")

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
