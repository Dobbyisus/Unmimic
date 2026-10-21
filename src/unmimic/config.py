"""Runtime settings, read from environment variables prefixed ``UNMIMIC_``.

Per-model reasoning formats live separately, in ``data/models/`` (see ``reasoning.py``).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from unmimic.resources import bundled


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UNMIMIC_")

    upstream_url: str = "http://localhost:11434/v1"
    # Key the proxy sends to the upstream model. Callers' own keys are never forwarded.
    upstream_api_key: str | None = None
    # Seconds to wait for the model's reply; reasoning models can take many minutes.
    upstream_timeout: float = 3600.0
    # Guards `unmimic serve` turns on, by registry name, run in this order. The leak
    # detector must come before the strip transform. Env: UNMIMIC_OUTPUT_GUARDS='[...]'
    input_guards: list[str] = ["extraction_guard"]
    output_guards: list[str] = ["leak_detector", "strip_reasoning"]
    # What the input guard learns from: the shipped public attack families plus your own in
    # ./attacks/private, and normal prompts (`unmimic fetch-benign`) plus the shipped hard
    # negatives. Missing folders and files are skipped.
    attack_dirs: list[Path] = [bundled("attacks/public"), Path("attacks/private")]
    benign_files: list[Path] = [
        Path("data/benign/wildchat_train.jsonl"),
        bundled("benign/hard_negatives.yaml"),
    ]
    host: str = "127.0.0.1"
    port: int = 9000
    data_dir: Path = Path("data")
