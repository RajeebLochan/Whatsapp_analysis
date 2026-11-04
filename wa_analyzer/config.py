"""Application limits and secret/setting loading.

Settings come from (in order of precedence) environment variables, then
Streamlit secrets (``.streamlit/secrets.toml`` or the hosting dashboard).
Nothing here ever logs or displays a secret value.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Processing limits (bounded memory / CPU per session)
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_MESSAGES = 300_000  # parsed records kept per upload
MAX_LINE_COUNT = 2_000_000  # physical lines read from an export
MAX_ZIP_MEMBERS = 50
MAX_WARNINGS_KEPT = 50  # individual skipped-record details retained

TABLE_ROW_LIMIT = 500  # rows shown in on-screen tables
CHART_TOP_N = 15  # participants / words / emoji shown in bar charts

SENTIMENT_MAX_MESSAGES = 3_000  # most recent text messages analysed
SENTIMENT_BATCH_SIZE = 32
SENTIMENT_MAX_TOKENS = 128  # tokenizer truncation length

# Together AI defaults. The serverless catalogue changes often; owners should
# verify these IDs against https://docs.together.ai/docs/serverless/models and
# override them with WA_TOGETHER_MODELS.
DEFAULT_TOGETHER_MODELS: tuple[str, ...] = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
AI_MAX_INPUT_TOKENS_PER_CHUNK = 6_000  # estimated (chars / 4)
AI_MAX_CHUNKS = 4
AI_MAX_OUTPUT_TOKENS = 1_200
AI_CONNECT_TIMEOUT_S = 10.0
AI_READ_TIMEOUT_S = 60.0
AI_TOTAL_DEADLINE_S = 180.0
AI_MAX_RETRIES = 2
AI_DEFAULT_MAX_REQUESTS_PER_SESSION = 6  # counts individual HTTP requests


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _read_streamlit_secrets() -> Mapping[str, Any]:
    """Return Streamlit secrets if available, else an empty mapping.

    Accessing ``st.secrets`` without a secrets file raises; that must never
    break ordinary analytics.
    """
    try:
        import streamlit as st

        return dict(st.secrets)
    except Exception:  # noqa: BLE001 - absence of secrets is a normal state
        return {}


def get_setting(name: str, default: Any = None, secrets: Mapping[str, Any] | None = None) -> Any:
    """Read a setting from the environment, then Streamlit secrets."""
    env_value = os.environ.get(name)
    if env_value not in (None, ""):
        return env_value
    secrets = _read_streamlit_secrets() if secrets is None else secrets
    value = secrets.get(name)
    if value in (None, ""):
        return default
    return value


@dataclass(frozen=True)
class AISettings:
    """Deployment-side configuration for the optional AI summary feature."""

    shared_key_available: bool
    shared_key_enabled: bool
    allow_user_key: bool
    models: tuple[str, ...]
    base_url: str
    max_requests_per_session: int
    _shared_key: str | None = field(default=None, repr=False)

    @property
    def shared_key(self) -> str | None:
        return self._shared_key if self.shared_key_enabled else None

    @property
    def usable(self) -> bool:
        return self.allow_user_key or (self.shared_key_available and self.shared_key_enabled)


def load_ai_settings(secrets: Mapping[str, Any] | None = None) -> AISettings:
    key = get_setting("TOGETHER_API_KEY", secrets=secrets)
    models_raw = get_setting("WA_TOGETHER_MODELS", secrets=secrets)
    if isinstance(models_raw, (list, tuple)):
        models = tuple(str(m).strip() for m in models_raw if str(m).strip())
    elif models_raw:
        models = tuple(m.strip() for m in str(models_raw).split(",") if m.strip())
    else:
        models = DEFAULT_TOGETHER_MODELS
    try:
        max_requests = int(
            get_setting("WA_AI_MAX_REQUESTS_PER_SESSION", AI_DEFAULT_MAX_REQUESTS_PER_SESSION, secrets=secrets)
        )
    except (TypeError, ValueError):
        max_requests = AI_DEFAULT_MAX_REQUESTS_PER_SESSION
    return AISettings(
        shared_key_available=bool(key),
        shared_key_enabled=bool(key) and _truthy(get_setting("WA_ENABLE_SHARED_AI_KEY", "false", secrets=secrets)),
        allow_user_key=_truthy(get_setting("WA_ALLOW_USER_AI_KEY", "true", secrets=secrets)),
        models=models or DEFAULT_TOGETHER_MODELS,
        base_url=str(get_setting("WA_TOGETHER_BASE_URL", TOGETHER_BASE_URL, secrets=secrets)).rstrip("/"),
        max_requests_per_session=max(0, max_requests),
        _shared_key=str(key) if key else None,
    )
