"""Thin wrapper around the Anthropic SDK: structured output, refusal handling, cost.

The API key is read from the ANTHROPIC_API_KEY environment variable (or a local,
git-ignored .env file). It is never stored in code.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TypeVar

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# USD per million tokens: (input, output). Cache reads bill at 0.1x input,
# cache writes at 1.25x input.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

T = TypeVar("T", bound=BaseModel)

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=4)
    return _client


@dataclass
class Call:
    parsed: BaseModel | None  # None when the model declined (stop_reason == "refusal")
    cost_usd: float
    latency_s: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int


def _cost(model: str, usage) -> float:
    price_in, price_out = PRICES.get(model, PRICES["claude-opus-5-5"])
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0
    return (
        usage.input_tokens * price_in
        + cache_read * price_in * 0.1
        + cache_write * price_in * 1.25
        + usage.output_tokens * price_out
    ) / 1_000_000


def ask(
    *,
    model: str,
    system: str | list[dict],
    user: str,
    output_format: type[T],
    effort: str = "medium",
) -> Call:
    start = time.perf_counter()
    response = client().beta.messages.parse(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort},
        output_format=output_format,
        # If a safety classifier declines, retry server-side on Anthropic's
        # recommended fallback model instead of returning the refusal.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    latency = time.perf_counter() - start
    usage = response.usage
    parsed = None if response.stop_reason == "refusal" else response.parsed_output
    return Call(
        parsed=parsed,
        cost_usd=_cost(model, usage),
        latency_s=latency,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
    )


def default_model(env_var: str) -> str:
    return os.environ.get(env_var, "claude-opus-5-5")
