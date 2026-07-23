from __future__ import annotations

import http.client
import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from opentelemetry.trace import SpanKind

from .telemetry import set_attributes, span


@dataclass(frozen=True)
class Completion:
    text: str
    response_id: str | None
    provider: str | None
    usage: dict[str, Any]
    raw_model: str | None
    reasoning: str | None
    finish_reason: str | None
    native_finish_reason: str | None
    annotations: list[dict[str, Any]]
    request_attempts: int = 1


class OpenRouterError(RuntimeError):
    pass


class OpenRouterPolicyError(OpenRouterError):
    """A provider-side safety policy outcome, not a transport failure."""


PRIVACY_SETTINGS_URL = "https://openrouter.ai/settings/privacy"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return str(content or "")


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 300,
        attempts: int = 4,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.attempts = attempts

    def list_models(self) -> list[dict[str, Any]]:
        with span(
            "openrouter.models.list",
            {"server.address": "openrouter.ai", "http.request.method": "GET"},
            kind=SpanKind.CLIENT,
        ) as current:
            request = urllib.request.Request(
                f"{self.base_url}/models/user",
                method="GET",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/quinnypig/corey-model-benchmark",
                    "X-Title": "Quinnferno",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = json.loads(response.read())
                    set_attributes(current, {"http.response.status_code": getattr(response, "status", 200)})
            except urllib.error.HTTPError as exc:
                set_attributes(current, {"http.response.status_code": exc.code})
                error_body = exc.read().decode("utf-8", errors="replace")[:2000]
                raise OpenRouterError(f"OpenRouter model catalog failed with HTTP {exc.code}: {error_body}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise OpenRouterError(f"OpenRouter model catalog failed: {exc}") from exc
            data = raw.get("data") if isinstance(raw, dict) else None
            if not isinstance(data, list):
                raise OpenRouterError("OpenRouter model catalog returned an invalid response: expected a data array")
            models = [item for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]
            set_attributes(current, {"openrouter.model_count": len(models)})
            return models

    def require_models_available(self, models: list[str]) -> None:
        """Verify exact model IDs against the API key's routing policy."""
        data = self.list_models()
        available: set[str] = set()
        for index, item in enumerate(data):
            model_id = item.get("id") if isinstance(item, dict) else None
            if not isinstance(model_id, str) or not model_id:
                raise OpenRouterError(
                    f"OpenRouter model preflight returned an invalid model at data[{index}]"
                )
            available.add(model_id)

        missing = list(dict.fromkeys(model for model in models if model not in available))
        if missing:
            raise OpenRouterError(
                "Requested model ID(s) are unavailable for this API key: "
                + ", ".join(missing)
                + ". Check provider preferences, privacy settings, and guardrails at "
                + PRIVACY_SETTINGS_URL
                + ". Model IDs are exact; free and paid variants are not interchangeable."
            )

    def complete_messages(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None,
        seed: int | None,
        reasoning: str,
        condition: str = "weights-only",
        response_format: dict[str, Any] | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed
        if reasoning == "off":
            payload["reasoning"] = {"enabled": False}
        elif reasoning == "on":
            payload["reasoning"] = {"enabled": True}
        elif reasoning != "provider-default":
            payload["reasoning"] = {"effort": reasoning}
        if condition == "search-enabled":
            payload["tools"] = [
                {
                    "type": "openrouter:web_search",
                    "search_context_size": "medium",
                    "max_total_results": 10,
                }
            ]
        if response_format:
            payload["response_format"] = response_format
        return self._send_completion(payload)

    def _send_completion(self, payload: dict[str, Any]) -> Completion:
        with span(
            "gen_ai.chat",
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": "openrouter",
                "gen_ai.request.model": payload.get("model"),
                "gen_ai.request.max_tokens": payload.get("max_tokens"),
                "gen_ai.request.temperature": payload.get("temperature"),
                "gen_ai.request.seed": payload.get("seed"),
                "gen_ai.request.message_count": len(payload.get("messages", [])),
                "gen_ai.request.web_search": bool(payload.get("tools")),
                "server.address": "openrouter.ai",
                "http.request.method": "POST",
            },
            kind=SpanKind.CLIENT,
        ) as current:
            body = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/quinnypig/corey-model-benchmark",
                    "X-Title": "Quinnferno",
                },
            )
            last_retry_after = 0.0
            for attempt in range(1, self.attempts + 1):
                retry_status = 0
                retry_error = ""
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        raw = json.loads(response.read())
                        set_attributes(current, {"http.response.status_code": getattr(response, "status", 200)})
                    choices = raw.get("choices")
                    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                        raise OpenRouterError("OpenRouter returned HTTP 200 without a usable completion choice")
                    choice = choices[0]
                    message = choice.get("message", {})
                    if not isinstance(message, dict):
                        raise OpenRouterError("OpenRouter returned a completion choice without a usable message")
                    usage = raw.get("usage") or {}
                    set_attributes(
                        current,
                        {
                            "gen_ai.response.id": raw.get("id"),
                            "gen_ai.response.model": raw.get("model"),
                            "gen_ai.response.finish_reasons": [choice.get("finish_reason") or "unknown"],
                            "gen_ai.usage.input_tokens": usage.get("prompt_tokens"),
                            "gen_ai.usage.output_tokens": usage.get("completion_tokens"),
                            "gen_ai.usage.total_tokens": usage.get("total_tokens"),
                            "quinnferno.cost_usd": usage.get("cost"),
                            "openrouter.provider": raw.get("provider"),
                            "openrouter.request_attempts": attempt,
                            "openrouter.retry_count": attempt - 1,
                        },
                    )
                    return Completion(
                        text=_content_to_text(message.get("content")),
                        response_id=raw.get("id"),
                        provider=raw.get("provider"),
                        usage=usage,
                        raw_model=raw.get("model"),
                        reasoning=message.get("reasoning"),
                        finish_reason=choice.get("finish_reason"),
                        native_finish_reason=choice.get("native_finish_reason"),
                        annotations=message.get("annotations") if isinstance(message.get("annotations"), list) else [],
                        request_attempts=attempt,
                    )
                except urllib.error.HTTPError as exc:
                    retry_status = exc.code
                    retry_error = "http"
                    error_body = exc.read().decode("utf-8", errors="replace")[:2000]
                    retryable = exc.code == 429 or exc.code >= 500
                    retry_header = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        last_retry_after = min(120.0, max(0.0, float(retry_header))) if retry_header else 0.0
                    except ValueError:
                        last_retry_after = 0.0
                    set_attributes(current, {"http.response.status_code": exc.code})
                    if not retryable or attempt == self.attempts:
                        if exc.code == 400 and (
                            "content_filter" in error_body or "considered high risk" in error_body
                        ):
                            raise OpenRouterPolicyError(
                                f"Provider safety filter blocked the benchmark prompt: {error_body}"
                            ) from exc
                        raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {error_body}") from exc
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, http.client.HTTPException) as exc:
                    retry_error = type(exc).__name__
                    if attempt == self.attempts:
                        raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc
                delay = max(last_retry_after, min(20.0, 2 ** (attempt - 1)) + random.random())
                current.add_event(
                    "openrouter.retry",
                    {
                        "retry.attempt": attempt,
                        "retry.delay_seconds": delay,
                        "retry.http_status_code": retry_status,
                        "retry.error_type": retry_error,
                    },
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def complete(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        seed: int | None,
        reasoning: str,
    ) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.complete_messages(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            reasoning=reasoning,
        )
