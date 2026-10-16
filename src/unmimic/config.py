"""Runtime settings, read from environment variables prefixed ``UNMIMIC_``.

The per-model config file (where a model puts its reasoning, calibrated thresholds)
is added together with the format probe in ``adapt/probe.py``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UNMIMIC_")

    upstream_url: str = "http://localhost:11434/v1"
    # Key the proxy sends to the upstream model. Callers' own keys are never forwarded.
    upstream_api_key: str | None = None
    host: str = "127.0.0.1"
    port: int = 9000
    data_dir: Path = Path("data")
