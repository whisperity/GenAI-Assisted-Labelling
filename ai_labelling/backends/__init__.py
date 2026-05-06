"""Backend registry."""

from .anthropic import AnthropicBackend, AnthropicHTTPError
from .base import AIBackend
from .codex import CodexBackend

CODEX_BACKEND = CodexBackend(name="Codex")
ANTHROPIC_BACKEND = AnthropicBackend(name="Anthropic")
BACKENDS = (ANTHROPIC_BACKEND, CODEX_BACKEND)
BACKENDS_BY_PROVIDER = {
    backend.provider_name(): backend for backend in BACKENDS
}


def get_supported_providers() -> tuple[str, ...]:
    """Return the supported provider strings in help-display order."""

    return tuple(BACKENDS_BY_PROVIDER)


def get_provider_default_model(provider: str) -> str:
    """Return the hard-coded default model for one provider."""

    return BACKENDS_BY_PROVIDER[provider].default_model_name()


def get_provider_reasoning_levels(provider: str) -> tuple[str, ...]:
    """Return the supported reasoning-effort levels for one provider."""

    return tuple(BACKENDS_BY_PROVIDER[provider].reasoning_levels())


def get_provider_help_description(provider: str) -> str:
    """Return the provider-specific ``--help`` description text."""

    return BACKENDS_BY_PROVIDER[provider].help_description()


def get_backend_for_provider(provider: str) -> AIBackend:
    """Return the backend for a specific provider name."""

    try:
        return BACKENDS_BY_PROVIDER[provider]
    except KeyError as exc:
        raise RuntimeError(
            f"unsupported AI provider: {provider}"
        ) from exc


def get_backend_for_model(model: str) -> AIBackend:
    """Return the backend implied by a ``PROVIDER:MODEL:REASONING`` spec."""

    return get_backend_for_provider(
        model.split(":", maxsplit=1)[0].casefold()
    )


__all__ = [
    "AIBackend",
    "AnthropicBackend",
    "AnthropicHTTPError",
    "ANTHROPIC_BACKEND",
    "BACKENDS",
    "BACKENDS_BY_PROVIDER",
    "CodexBackend",
    "CODEX_BACKEND",
    "get_backend_for_model",
    "get_backend_for_provider",
    "get_provider_default_model",
    "get_provider_help_description",
    "get_provider_reasoning_levels",
    "get_supported_providers",
]
