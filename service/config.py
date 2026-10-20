"""API settings from environment variables (Phase 8). Safe defaults; see README.md, "API security"."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(env, name, default="0"):
    return env.get(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"           # was 0.0.0.0
    port: int = 5000
    debug: bool = False               # was True: Werkzeug debugger = remote code execution
    api_key: str | None = None        # if set, every request needs header X-API-Key
    cors_origins: tuple = (r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",)
    legacy_code_exec: bool = False    # the old exec() path of /submit-value-function-api
    hmac_key: str | None = None       # signs stored value functions
    max_rows: int = 1_000_000         # rows a synthetic-data request may produce
    max_upload_mb: int = 200


def settings_from_env(env=os.environ) -> Settings:
    origins = env.get("DEBIAS_CORS_ORIGINS")
    return Settings(
        host=env.get("DEBIAS_HOST", "127.0.0.1"),
        port=int(env.get("DEBIAS_PORT", "5000")),
        debug=_flag(env, "DEBIAS_DEBUG"),
        api_key=env.get("DEBIAS_API_KEY") or None,
        cors_origins=tuple(o.strip() for o in origins.split(",")) if origins else Settings.cors_origins,
        legacy_code_exec=_flag(env, "DEBIAS_ENABLE_LEGACY_CODE_EXEC"),
        hmac_key=env.get("DEBIAS_STORE_HMAC_KEY") or None,
        max_rows=int(env.get("DEBIAS_MAX_ROWS", "1000000")),
        max_upload_mb=int(env.get("DEBIAS_MAX_UPLOAD_MB", "200")),
    )
