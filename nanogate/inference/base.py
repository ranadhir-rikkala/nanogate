from __future__ import annotations

from typing import Any, Protocol

from nanogate.models import Completion

Messages = list[dict[str, Any]]


class ModelUnavailable(Exception):
    pass


class Backend(Protocol):
    has_large: bool

    def model_name(self, large: bool) -> str: ...

    async def complete(self, messages: Messages, *, large: bool, max_tokens: int, temperature: float) -> Completion: ...

    async def health(self) -> dict[str, Any]: ...


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return "" if content is None else str(content)


def prompt_tokens(messages: Messages) -> int:
    return sum(estimate_tokens(message_text(message)) for message in messages)
