"""Anthropic direct API backend implementation."""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ai_labelling.models import ModelSpec
from ai_labelling.terminal import debug_log, format_prompt_for_debug

from .base import AIBackend


PROVIDER_NAME = "anthropic"
"""Provider string used in ``PROVIDER:MODEL:REASONING`` specs."""

DEFAULT_MODEL_NAME = "claude-haiku-4-5-20251001"
"""Hard-coded Anthropic default model used when the user omits one."""

REASONING_LEVELS = ("low", "medium", "high", "xhigh", "max")
"""Reasoning-effort levels accepted by Anthropic's API."""

API_URL = "https://api.anthropic.com"
"""Base URL for Anthropic's direct API."""

API_VERSION = "2023-06-01"
"""Anthropic API version used for direct HTTP requests."""


class AnthropicHTTPError(RuntimeError):
    """Raised when the Anthropic API returns an HTTP error response."""

    def __init__(self, code: int, body: str) -> None:
        super().__init__(
            f"Anthropic API request failed with HTTP {code}: {body}"
        )
        self.code = code
        self.body = body


def sanitise_headers_for_debug(headers: Dict[str, str]) -> Dict[str, str]:
    """Return headers with secrets redacted for debug output."""

    sanitised = headers.copy()
    if "x-api-key" in sanitised:
        sanitised["x-api-key"] = "***REDACTED***"
    return sanitised


@dataclass
class AnthropicBackend(AIBackend):
    """AI backend implementation that calls Anthropic's direct API."""

    api_url: str = API_URL
    api_version: str = API_VERSION

    @staticmethod
    def provider_name() -> str:
        """Return the stable provider string for this backend."""

        return PROVIDER_NAME

    @staticmethod
    def default_model_name() -> str:
        """Return the provider's hard-coded default model name."""

        return DEFAULT_MODEL_NAME

    @staticmethod
    def reasoning_levels() -> Sequence[str]:
        """Return supported reasoning-effort levels for this backend."""

        return REASONING_LEVELS

    @classmethod
    def help_description(cls) -> str:
        """Return the provider-specific ``--help`` description text."""

        return (
            "direct Claude Platform API "
            "("
            f"default model `{cls.default_model_name()}`, "
            "wildcard `*` selects the newest model from `/v1/models`, "
            "effort: "
            + ", ".join(cls.reasoning_levels())
            + ")"
        )

    def headers(self, *, content_type: bool = False) -> Dict[str, str]:
        """Build common headers for direct Anthropic API requests."""

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; required for anthropic provider"
            )
        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.api_version,
        }
        if content_type:
            headers["content-type"] = "application/json"
        return headers

    def json_request(
        self,
        path: str,
        *,
        method: str,
        body: Optional[Dict[str, object]] = None,
    ) -> object:
        """Send one JSON request to Anthropic's HTTP API."""

        request_data = None
        if body is not None:
            request_data = json.dumps(body).encode("utf-8")

        headers = self.headers(content_type=body is not None)
        request = urllib.request.Request(
            self.api_url + path,
            data=request_data,
            headers=headers,
            method=method,
        )

        debug_log(
            f"+ {method} {path} "
            f"(headers: {sanitise_headers_for_debug(headers)})",
            colour="cyan",
        )

        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise AnthropicHTTPError(exc.code, body_text) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Anthropic API request failed: {exc.reason}"
            ) from exc

    def get_default_model(self) -> str:
        """Return the newest listed Anthropic model ID."""

        payload = self.json_request("/v1/models", method="GET")
        if not isinstance(payload, dict):
            raise RuntimeError("unexpected Anthropic models payload")

        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("Anthropic models list was empty")

        first = data[0]
        if not isinstance(first, dict) or not isinstance(first.get("id"), str):
            raise RuntimeError("unexpected Anthropic model entry")
        return str(first["id"])

    def extract_json(self, text: str) -> Dict[str, object]:
        """Parse one JSON object from an Anthropic text response."""

        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()

        obj_start = stripped.find("{")
        if obj_start == -1:
            raise RuntimeError(
                f"Anthropic returned invalid JSON:\n{stripped}"
            )
        try:
            parsed, _ = json.JSONDecoder().raw_decode(stripped, obj_start)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Anthropic returned invalid JSON:\n{stripped}"
            ) from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Anthropic returned a non-object response")
        return parsed

    def _send_messages_request(
        self, body: Dict[str, object]
    ) -> Dict[str, object]:
        """Send /v1/messages and unwrap the JSON object payload."""

        payload = self.json_request("/v1/messages", method="POST", body=body)
        if not isinstance(payload, dict):
            raise RuntimeError("unexpected Anthropic messages payload")
        return payload

    @staticmethod
    def _is_effort_unsupported(exc: AnthropicHTTPError) -> bool:
        """Return True when the API rejected an effort parameter (HTTP 400)."""

        return (
            exc.code == 400
            and "does not support the effort parameter" in exc.body
        )

    def run_prompt(
        self,
        prompt: str,
        model_spec: ModelSpec,
        *,
        allow_label_removals: bool,  # pylint: disable=unused-argument
    ) -> object:
        """Run a prompt through Anthropic's Messages API."""

        selected_model = model_spec.model or self.get_default_model()
        debug_prompt = format_prompt_for_debug(prompt)
        if debug_prompt is not None:
            debug_log(debug_prompt)

        body: Dict[str, object] = {
            "model": selected_model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if model_spec.reasoning_effort is not None:
            body["output_config"] = {"effort": model_spec.reasoning_effort}

        try:
            payload = self._send_messages_request(body)
        except AnthropicHTTPError as exc:
            if self._is_effort_unsupported(exc) and "output_config" in body:
                debug_log(
                    f"Model {selected_model!r} does not support effort;"
                    " retrying without it.",
                    colour="yellow",
                )
                del body["output_config"]
                payload = self._send_messages_request(body)
            else:
                raise

        content = payload.get("content")
        if not isinstance(content, list):
            raise RuntimeError("unexpected Anthropic content payload")

        text_parts: List[str] = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                text_parts.append(str(block["text"]))
        if not text_parts:
            raise RuntimeError("Anthropic returned no text content")
        return self.extract_json("".join(text_parts))
