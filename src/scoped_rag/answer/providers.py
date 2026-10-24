"""LLM providers behind one interface, with per-call cost accounted on the way out.

Claude is the only real implementation. The fake one is deterministic and free, so
tests and the leakage eval run in CI with no key and no network.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

CLAUDE_MODEL = "claude-haiku-5-5"
# USD per million tokens, (input, output).
PRICE_PER_MTOK = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (3.00, 15.00)}


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "fake"

    def cost_usd(self) -> float:
        rate = PRICE_PER_MTOK.get(self.model)
        if not rate:
            return 0.0
        return self.input_tokens / 1e6 * rate[0] + self.output_tokens / 1e6 * rate[1]


@runtime_checkable
class LlmProvider(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, system: str, user: str, *, max_tokens: int = 700) -> Completion: ...


_LABEL = re.compile(r"\[([\w:.-]+)\]")


class FakeProvider:
    """Answers by quoting the first context block and citing it.

    Deterministic on purpose: the eval needs to tell a retrieval failure from a
    model failure, which is impossible if the model is free to vary.
    """

    @property
    def name(self) -> str:
        return "fake"

    def complete(self, system: str, user: str, *, max_tokens: int = 700) -> Completion:
        labels = _LABEL.findall(user)
        if not labels:
            return Completion("沒有可用的內容。")
        first = labels[0]
        body = user.split(f"[{first}]", 1)[1]
        body = body.split("\n[", 1)[0]
        lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
        quoted = lines[1] if len(lines) > 1 else (lines[0] if lines else "")
        return Completion(f"{quoted[:120]} [{first}]")


class ClaudeProvider:
    def __init__(self, model: str = CLAUDE_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self._key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        if not self._key:
            raise RuntimeError("set ANTHROPIC_API_KEY, or use the fake provider")

    @property
    def name(self) -> str:
        return self.model

    def complete(self, system: str, user: str, *, max_tokens: int = 700) -> Completion:
        import httpx

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self._key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        last: Exception | None = None
        for attempt in range(3):
            try:
                r = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                r.raise_for_status()
                data = r.json()
                text = "".join(b.get("text", "") for b in data.get("content", []))
                usage = data.get("usage", {})
                return Completion(
                    text,
                    int(usage.get("input_tokens", 0)),
                    int(usage.get("output_tokens", 0)),
                    self.model,
                )
            except Exception as exc:  # noqa: BLE001 - any failure is worth one retry
                last = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Claude call failed after 3 attempts: {last}")
