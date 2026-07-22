from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


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


class OpenRouterError(RuntimeError):
    pass


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

    def require_models_available(self, models: list[str]) -> None:
        """Verify exact model IDs against the API key's routing policy."""
        request = urllib.request.Request(
            f"{self.base_url}/models/user",
            method="GET",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/QuinnyPig/corey-model-benchmark",
                "X-Title": "Corey Quinn model benchmark",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")[:2000]
            raise OpenRouterError(
                f"OpenRouter model preflight failed with HTTP {exc.code}: {error_body}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OpenRouterError(f"OpenRouter model preflight failed: {exc}") from exc

        data = raw.get("data") if isinstance(raw, dict) else None
        if not isinstance(data, list):
            raise OpenRouterError(
                "OpenRouter model preflight returned an invalid response: expected a data array"
            )
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
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "usage": {"include": True},
        }
        if seed is not None:
            payload["seed"] = seed
        if reasoning == "off":
            payload["reasoning"] = {"enabled": False}
        elif reasoning == "on":
            payload["reasoning"] = {"enabled": True}
        else:
            payload["reasoning"] = {"effort": reasoning}
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/QuinnyPig/corey-model-benchmark",
                "X-Title": "Corey Quinn model benchmark",
            },
        )

        for attempt in range(1, self.attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = json.loads(response.read())
                choices = raw.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise OpenRouterError("OpenRouter returned HTTP 200 without a usable completion choice")
                choice = choices[0]
                message = choice.get("message", {})
                if not isinstance(message, dict):
                    raise OpenRouterError("OpenRouter returned a completion choice without a usable message")
                return Completion(
                    text=_content_to_text(message.get("content")),
                    response_id=raw.get("id"),
                    provider=raw.get("provider"),
                    usage=raw.get("usage") or {},
                    raw_model=raw.get("model"),
                    reasoning=message.get("reasoning"),
                    finish_reason=choice.get("finish_reason"),
                    native_finish_reason=choice.get("native_finish_reason"),
                )
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")[:2000]
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == self.attempts:
                    raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {error_body}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt == self.attempts:
                    raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc
            delay = min(20.0, 2 ** (attempt - 1)) + random.random()
            time.sleep(delay)
        raise AssertionError("unreachable")
