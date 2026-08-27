from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: SecretStr
    alpaca_api_key: SecretStr
    alpaca_secret_key: SecretStr
    alpaca_paper: bool = True


@lru_cache
def get_settings() -> Settings:
    # Lazy + cached: importing this module must never crash just because .env is
    # missing/incomplete (pytest collection, Streamlit's degraded first-run state).
    # Only actually instantiating Settings() (i.e. calling get_settings()) requires
    # a valid .env / real env vars.
    #
    # NOTE: this only populates the returned Settings object, NOT os.environ. The
    # Claude Agent SDK spawns the Claude Code CLI as a subprocess and needs
    # ANTHROPIC_API_KEY to already be in os.environ (subprocess inheritance) —
    # scripts/run_pipeline_cli.py calls dotenv.load_dotenv() for that, separately,
    # before any agent code runs.
    return Settings()
