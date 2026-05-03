from pathlib import Path
from functools import lru_cache

import yaml
from dotenv import load_dotenv
from pydantic import Field, PrivateAttr
from pydantic_settings import BaseSettings

load_dotenv()

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config" / "settings.yaml"


def _load_yaml() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class Settings(BaseSettings):
    # Secrets from .env
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    telegram_bot_token: str = Field("", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field("", alias="TELEGRAM_CHAT_ID")
    redis_url: str = Field("redis://127.0.0.1:6379/0", alias="REDIS_URL")
    app_env: str = Field("development", alias="APP_ENV")

    # YAML config stored as a proper Pydantic v2 private attribute
    _yaml: dict = PrivateAttr(default_factory=dict)

    model_config = {"populate_by_name": True}

    def model_post_init(self, __context):
        self._yaml = _load_yaml()

    # --- Convenience accessors ---

    @property
    def claude(self) -> dict:
        return self._yaml["claude"]

    @property
    def server(self) -> dict:
        return self._yaml["server"]

    @property
    def symbols(self) -> dict:
        return self._yaml["symbols"]

    @property
    def risk(self) -> dict:
        return self._yaml["risk"]

    @property
    def news_filter(self) -> dict:
        return self._yaml["news_filter"]

    @property
    def mt5_fetcher(self) -> dict:
        return self._yaml.get("mt5_fetcher", {})

    @property
    def scanner(self) -> dict:
        return self._yaml["scanner"]

    @property
    def sessions(self) -> dict:
        return self._yaml["sessions"]

    @property
    def indicators(self) -> dict:
        return self._yaml["indicators"]

    @property
    def telegram(self) -> dict:
        return self._yaml["telegram"]

    @property
    def logging(self) -> dict:
        return self._yaml["logging"]

    @property
    def database(self) -> dict:
        return self._yaml["database"]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
