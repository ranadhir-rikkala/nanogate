from nanogate.config import Settings
from nanogate.inference.base import Backend, Messages, ModelUnavailable, estimate_tokens, message_text, prompt_tokens
from nanogate.inference.local import OFFLINE_NOTICE, OfflineBackend, OpenAICompatBackend
from nanogate.inference.remote import SIMULATED_NOTICE, EgressDenied, EgressGuard, EgressTicket, RemoteSimulator

BACKENDS = ("offline", "openai")


def build_backend(settings: Settings) -> Backend:
    if settings.backend == "offline":
        return OfflineBackend()
    if settings.backend == "openai":
        return OpenAICompatBackend(settings)
    raise ValueError(f"NANOGATE_BACKEND must be one of {BACKENDS}")


__all__ = [
    "Backend",
    "EgressDenied",
    "EgressGuard",
    "EgressTicket",
    "Messages",
    "ModelUnavailable",
    "OFFLINE_NOTICE",
    "OfflineBackend",
    "OpenAICompatBackend",
    "RemoteSimulator",
    "SIMULATED_NOTICE",
    "build_backend",
    "estimate_tokens",
    "message_text",
    "prompt_tokens",
]
