from __future__ import annotations

import time
from typing import Any

import httpx

from nanogate.config import Settings
from nanogate.inference.base import Messages, ModelUnavailable, estimate_tokens, message_text, prompt_tokens
from nanogate.models import Completion, Route

OFFLINE_NOTICE = "[offline backend: no model attached]"


class OpenAICompatBackend:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=settings.local_timeout_s)
        self.has_large = bool(settings.large_model)

    def model_name(self, large: bool) -> str:
        return self.settings.large_model if large else self.settings.local_model

    def _base_url(self, large: bool) -> str:
        url = self.settings.large_base_url if large and self.settings.large_base_url else self.settings.local_base_url
        return url.rstrip("/")

    async def complete(self, messages: Messages, *, large: bool, max_tokens: int, temperature: float) -> Completion:
        model = self.model_name(large)
        if not model:
            raise ModelUnavailable("large tier is not configured" if large else "LOCAL_MODEL_NAME is not set")
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": True,
        }
        started = time.perf_counter()
        try:
            response = await self.client.post(
                f"{self._base_url(large)}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.settings.local_api_key}"},
            )
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            text = choice["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelUnavailable(f"{type(error).__name__}: local model call failed") from error
        usage = body.get("usage") or {}
        tokens = ((choice.get("logprobs") or {}).get("content")) or []
        return Completion(
            text=text,
            model=str(body.get("model") or model),
            route=Route.LOCAL_LARGE if large else Route.LOCAL_SMALL,
            prompt_tokens=int(usage.get("prompt_tokens") or prompt_tokens(messages)),
            completion_tokens=int(usage.get("completion_tokens") or estimate_tokens(text)),
            logprobs=tuple(float(token["logprob"]) for token in tokens if "logprob" in token),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def health(self) -> dict[str, Any]:
        try:
            response = await self.client.get(f"{self._base_url(False)}/models", timeout=3.0)
            response.raise_for_status()
            return {"ok": True, "backend": "openai-compatible", "model": self.settings.local_model}
        except httpx.HTTPError as error:
            return {"ok": False, "backend": "openai-compatible", "error": type(error).__name__}


class OfflineBackend:
    has_large = True

    def model_name(self, large: bool) -> str:
        return "offline-large" if large else "offline-small"

    async def complete(self, messages: Messages, *, large: bool, max_tokens: int, temperature: float) -> Completion:
        last_user = next((message_text(m) for m in reversed(messages) if m.get("role") == "user"), "")
        text = f"{OFFLINE_NOTICE} Received {len(last_user)} characters. Attach a model to get a real answer."
        return Completion(
            text=text,
            model=self.model_name(large),
            route=Route.LOCAL_LARGE if large else Route.LOCAL_SMALL,
            prompt_tokens=prompt_tokens(messages),
            completion_tokens=estimate_tokens(text),
        )

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "backend": "offline", "model": self.model_name(False)}
