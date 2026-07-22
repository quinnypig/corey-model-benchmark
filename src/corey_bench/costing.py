from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import tiktoken

from .protocol import EvalDefinition, EvalSuite, RenderedEval
if TYPE_CHECKING:
    from .runner import RunConfig


# Expected output sizes are protocol-planning assumptions, not grading limits.
# The upper estimate still respects each eval's max_tokens ceiling.
EXPECTED_OUTPUT_TOKENS = {
    "1.1": 700, "1.2": 650, "1.3": 5000, "1.4": 300,
    "2.1": 6000, "2.2": 6000,
    "3.1": 1200, "3.2": 450, "3.3": 850, "3.4": 700,
    "4.1": 500,
    "5.1": 450, "5.2": 180, "5.3": 250, "5.4": 350,
    "5.5": 250, "5.6": 200, "5.7": 1300, "5.8": 300,
    "6.1": 1100, "6.2": 350, "6.3": 250, "6.4": 200,
    "7.1": 3500, "7.2": 2, "7.3": 100, "7.4": 1600,
    "7.5": 800, "7.6": 1000,
}

# OpenRouter exposes a tokenizer family, not a preflight token-count endpoint.
# GPT uses its native public encoding. Other families use the closest public BPE
# plus a calibrated multiplier and a wider error band.
TOKENIZER_PROFILES: dict[str, tuple[str, float, float]] = {
    "GPT": ("o200k_base", 1.00, 0.08),
    "Claude": ("cl100k_base", 1.08, 0.22),
    "Qwen": ("o200k_base", 1.06, 0.16),
    "Qwen3": ("o200k_base", 1.08, 0.16),
    "Llama2": ("cl100k_base", 1.08, 0.18),
    "Llama3": ("o200k_base", 1.10, 0.17),
    "Llama4": ("o200k_base", 1.08, 0.17),
    "Mistral": ("cl100k_base", 1.07, 0.18),
    "DeepSeek": ("o200k_base", 1.10, 0.20),
    "Gemini": ("o200k_base", 1.04, 0.20),
    "Gemma": ("o200k_base", 1.07, 0.19),
    "Grok": ("o200k_base", 1.06, 0.20),
    "Cohere": ("cl100k_base", 1.10, 0.22),
    "Nova": ("cl100k_base", 1.08, 0.22),
    "Router": ("o200k_base", 1.08, 0.30),
    "Other": ("o200k_base", 1.08, 0.28),
}


@lru_cache(maxsize=4)
def _encoding(name: str) -> Any:
    return tiktoken.get_encoding(name)


def _profile(model: dict[str, Any]) -> tuple[str, str, float, float]:
    family = str((model.get("architecture") or {}).get("tokenizer") or "Other")
    encoding, multiplier, uncertainty = TOKENIZER_PROFILES.get(family, TOKENIZER_PROFILES["Other"])
    model_id = str(model.get("id", ""))
    if family == "GPT" and any(old in model_id for old in ("gpt-3.5", "gpt-4-", "gpt-4:")):
        encoding = "cl100k_base"
    return family, encoding, multiplier, uncertainty


def _count_text(text: str, profile: tuple[str, str, float, float]) -> int:
    _, encoding, multiplier, _ = profile
    return max(1, math.ceil(len(_encoding(encoding).encode(text)) * multiplier))


def _price(pricing: dict[str, Any], key: str, fallback: float = 0.0) -> float:
    try:
        return float(pricing.get(key) or fallback)
    except (TypeError, ValueError):
        return fallback


def _output_sizes(definition: EvalDefinition) -> tuple[int, int, int]:
    expected = min(definition.max_tokens, EXPECTED_OUTPUT_TOKENS.get(definition.id, min(900, definition.max_tokens)))
    low = max(1, math.ceil(expected * 0.55))
    high = min(definition.max_tokens, max(expected, math.ceil(expected * 1.8)))
    return low, expected, high


def _contexts(
    definition: EvalDefinition,
    rendered: RenderedEval,
    profile: tuple[str, str, float, float],
    condition: str,
    output_tokens: int,
    agentic_turns: int,
) -> list[int]:
    overhead = 7
    if condition == "agentic":
        original = _count_text(rendered.messages[0]["content"], profile) + overhead
        feedback = _count_text("The standardized harness found failed gates. Return a complete revised response.", profile) + overhead
        return [original + turn * (output_tokens + feedback) for turn in range(agentic_turns)]
    if definition.execution == "independent":
        return [_count_text(prompt, profile) + overhead for prompt in rendered.prompts]
    if definition.execution == "repeated":
        base = _count_text(rendered.messages[0]["content"], profile) + overhead
        return [base] * definition.sample_count
    if definition.execution == "multi_turn":
        total = _count_text(rendered.messages[0]["content"], profile) + overhead
        contexts = []
        for turn in range(1 + len(rendered.followups)):
            contexts.append(total)
            if turn < len(rendered.followups):
                total += output_tokens + _count_text(rendered.followups[turn], profile) + overhead
        return contexts
    return [_count_text(rendered.messages[0]["content"], profile) + overhead]


def estimate_model_cost(config: "RunConfig", suite: EvalSuite, model: dict[str, Any]) -> dict[str, Any]:
    profile = _profile(model)
    family, encoding, _, uncertainty = profile
    pricing = model.get("pricing") if isinstance(model.get("pricing"), dict) else {}
    prompt_rate = _price(pricing, "prompt")
    completion_rate = _price(pricing, "completion")
    request_rate = _price(pricing, "request")
    reasoning_rate = _price(pricing, "internal_reasoning")
    web_rate = _price(pricing, "web_search", 0.005)
    totals = {"low": 0.0, "expected": 0.0, "high": 0.0}
    token_totals = {
        "low": {"input": 0, "output": 0},
        "expected": {"input": 0, "output": 0},
        "high": {"input": 0, "output": 0},
    }

    for eval_id in config.eval_ids:
        definition = suite.get(eval_id)
        rendered = definition.render(config.seed)
        repetitions = config.repetitions or definition.repetitions
        output_sizes = _output_sizes(definition)
        for condition in config.conditions:
            allowed = condition in definition.conditions or (
                condition == "agentic" and eval_id in {"2.1", "2.2"}
            )
            if not allowed:
                continue
            for band, output_tokens, agentic_turns, search_tokens, search_calls in (
                ("low", output_sizes[0], 1, 0, 0),
                ("expected", output_sizes[1], 2, 6000, 1),
                ("high", output_sizes[2], 10, 12000, 2),
            ):
                contexts = _contexts(definition, rendered, profile, condition, output_tokens, agentic_turns)
                input_tokens = sum(contexts)
                output_total = output_tokens * len(contexts)
                if condition == "search-enabled":
                    input_tokens += search_tokens * len(contexts)
                token_factor = 1 - uncertainty if band == "low" else 1 + uncertainty if band == "high" else 1
                input_tokens = math.ceil(input_tokens * token_factor) * repetitions
                output_total *= repetitions
                calls = len(contexts) * repetitions
                search_cost = web_rate * search_calls * calls if condition == "search-enabled" else 0
                reasoning_cost = 0.0
                if reasoning_rate and config.reasoning != "off":
                    reasoning_factor = 0 if band == "low" else 0.25 if band == "expected" else 1.0
                    reasoning_cost = output_total * reasoning_factor * reasoning_rate
                totals[band] += (
                    input_tokens * prompt_rate
                    + output_total * completion_rate
                    + calls * request_rate
                    + search_cost
                    + reasoning_cost
                )
                token_totals[band]["input"] += input_tokens
                token_totals[band]["output"] += output_total

    priced = any((prompt_rate, completion_rate, request_rate, _price(pricing, "web_search")))
    return {
        "id": model.get("id"),
        "name": model.get("name") or model.get("id"),
        "tokenizer_family": family,
        "encoding": encoding,
        "uncertainty_percent": round(uncertainty * 100),
        "priced": priced,
        "cost": {key: round(value, 4) for key, value in totals.items()},
        "tokens": token_totals,
    }


def estimate_run_cost(config: "RunConfig", suite: EvalSuite, catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    models = [estimate_model_cost(config, suite, catalog[model_id]) for model_id in config.models if model_id in catalog]
    total = {
        band: round(sum(model["cost"][band] for model in models), 4)
        for band in ("low", "expected", "high")
    }
    return {
        "models": models,
        "total": total,
        "missing": [model_id for model_id in config.models if model_id not in catalog],
        "method": "Live OpenRouter prices; tokenizer-family preflight; output/search/agentic range",
    }
