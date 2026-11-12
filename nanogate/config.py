from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def load_dotenv(path: str | Path = ".env") -> None:
    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8080
    db_path: str = "var/nanogate.db"
    policy_path: str = "config/policy.yaml"
    rates_path: str = "config/rates.yaml"
    router_artifact: str = "var/router.json"
    backend: str = "offline"
    local_base_url: str = "http://127.0.0.1:8000/v1"
    local_model: str = ""
    large_base_url: str = ""
    large_model: str = ""
    local_api_key: str = "EMPTY"
    local_timeout_s: float = 30.0
    remote_mode: str = "disabled"
    receipt_key: str = ""
    key_pepper: str = ""
    cache_similarity: float = 0.82
    cache_ttl_s: int = 86400
    max_prompt_chars: int = 32000

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        base = cls()

        def text(name: str, default: str) -> str:
            value = env.get(name, "")
            return value if value != "" else default

        return cls(
            host=text("NANOGATE_HOST", base.host),
            port=int(text("NANOGATE_PORT", str(base.port))),
            db_path=text("NANOGATE_DB", base.db_path),
            policy_path=text("NANOGATE_POLICY", base.policy_path),
            rates_path=text("NANOGATE_RATES", base.rates_path),
            router_artifact=text("NANOGATE_ROUTER_ARTIFACT", base.router_artifact),
            backend=text("NANOGATE_BACKEND", base.backend),
            local_base_url=text("LOCAL_MODEL_BASE_URL", base.local_base_url),
            local_model=text("LOCAL_MODEL_NAME", base.local_model),
            large_base_url=text("LOCAL_LARGE_MODEL_BASE_URL", base.large_base_url),
            large_model=text("LOCAL_LARGE_MODEL_NAME", base.large_model),
            local_api_key=text("LOCAL_MODEL_API_KEY", base.local_api_key),
            local_timeout_s=float(text("LOCAL_MODEL_TIMEOUT_S", str(base.local_timeout_s))),
            remote_mode=text("NANOGATE_REMOTE_MODE", base.remote_mode),
            receipt_key=text("NANOGATE_RECEIPT_KEY", base.receipt_key),
            key_pepper=text("NANOGATE_KEY_PEPPER", base.key_pepper),
            cache_similarity=float(text("NANOGATE_CACHE_SIMILARITY", str(base.cache_similarity))),
            cache_ttl_s=int(text("NANOGATE_CACHE_TTL_S", str(base.cache_ttl_s))),
            max_prompt_chars=int(text("NANOGATE_MAX_PROMPT_CHARS", str(base.max_prompt_chars))),
        )
