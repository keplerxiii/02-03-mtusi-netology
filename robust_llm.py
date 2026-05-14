"""
Простой клиент LLM: retry (экспонента + jitter), цепочка провайдеров, usage, circuit breaker, rate limit.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt, wait_exponential, wait_random

logger = logging.getLogger("robust_llm")

UNAVAILABLE_MESSAGE = "Сервис временно недоступен"

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEEPSEEK_BASE = "https://api.deepseek.com"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _now() -> float:
    return time.monotonic()


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APITimeoutError):
        return True
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in (429, 500)
    return False


def _wait_strategy():
    # 1 → 2 → 4 → 8 → 16 с + небольшой jitter
    return wait_exponential(multiplier=1, min=1, max=16) + wait_random(0, 0.5)


def _before_sleep(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    code: Any = type(exc).__name__
    if isinstance(exc, APIStatusError):
        code = exc.status_code
    logger.warning(
        "%s err_code=%s attempt=%s",
        _utc_ts(),
        code,
        retry_state.attempt_number,
    )


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def add_completion(self, usage: Any, cost_usd: float) -> None:
        if usage is None:
            return
        self.prompt_tokens += int(usage.prompt_tokens or 0)
        self.completion_tokens += int(usage.completion_tokens or 0)
        self.estimated_cost_usd += cost_usd


@dataclass
class ChatResult:
    text: str
    provider: str
    model: str | None = None


@dataclass
class _ProviderState:
    name: str
    failures_streak: int = 0
    circuit_open_until: float = 0.0

    def circuit_open(self) -> bool:
        return _now() < self.circuit_open_until

    def on_success(self) -> None:
        self.failures_streak = 0

    def on_hard_fail(self, cooldown_s: float) -> None:
        self.failures_streak += 1
        if self.failures_streak >= 3:
            self.circuit_open_until = _now() + cooldown_s
            logger.warning("%s circuit: 3 ошибки подряд — пауза %.0f с", self.name, cooldown_s)


class _RateLimiter:
    def __init__(self, min_interval_s: float) -> None:
        self._min = max(0.0, min_interval_s)
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        if self._min <= 0:
            return
        with self._lock:
            now = _now()
            wait = self._min - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = _now()


def _pricing_cost_usd(provider: str, model: str | None, prompt: int, completion: int) -> float:
    """Грубая оценка: цены за 1M токенов из env (USD)."""
    p = (prompt / 1_000_000.0) if prompt else 0.0
    c = (completion / 1_000_000.0) if completion else 0.0
    key = f"COST_{provider.upper()}_PROMPT_PER_1M"
    key2 = f"COST_{provider.upper()}_COMPLETION_PER_1M"
    pp = float(os.getenv(key, "0") or 0)
    cc = float(os.getenv(key2, "0") or 0)
    if pp == 0 and cc == 0 and model:
        # необязательные дефолты для демо (можно переопределить .env)
        demo = {
            ("openai", "gpt-4o-mini"): (0.15, 0.6),
            ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"): (0.0, 0.0),
            ("deepseek", "deepseek-v4-flash"): (0.27, 1.1),
            ("deepseek", "deepseek-chat"): (0.14, 0.28),
        }
        for (pr, m), pair in demo.items():
            if pr == provider and m == (model or ""):
                pp, cc = pair
                break
    return p * pp + c * cc


class RobustLLMClient:
    """
    Обёртка над OpenAI SDK: retry на 429/500/таймауты, fallback OpenAI → OpenRouter → DeepSeek → текст.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 5,
        circuit_cooldown_s: float = 60.0,
        rate_limit_min_interval_s: float | None = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.circuit_cooldown_s = circuit_cooldown_s
        rl = rate_limit_min_interval_s
        if rl is None:
            rl = float(os.getenv("CLIENT_RATE_LIMIT_INTERVAL_S", "0.2") or 0)
        self._rate = _RateLimiter(rl)
        self.session_usage = UsageTotals()
        self._states: dict[str, _ProviderState] = {}

    def _state(self, name: str) -> _ProviderState:
        if name not in self._states:
            self._states[name] = _ProviderState(name=name)
        return self._states[name]

    def _make_client(self, name: str) -> tuple[OpenAI, str, str] | None:
        if name == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                return None
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return OpenAI(api_key=key, max_retries=0), model, "openai"
        if name == "openrouter":
            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                return None
            model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
            return OpenAI(base_url=OPENROUTER_BASE, api_key=key, max_retries=0), model, "openrouter"
        if name == "deepseek":
            key = os.getenv("DEEPSEEK_API_KEY")
            if not key:
                return None
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
            return OpenAI(base_url=DEEPSEEK_BASE, api_key=key, max_retries=0), model, "deepseek"
        return None

    def _call_provider(
        self,
        client: OpenAI,
        model: str,
        messages: Sequence[dict[str, str]],
        provider_label: str,
    ) -> ChatResult:
        self._rate.acquire()

        def do_call() -> ChatResult:
            completion = client.chat.completions.create(
                model=model,
                messages=list(messages),
                timeout=float(os.getenv("LLM_TIMEOUT_S", "60")),
            )
            text = (completion.choices[0].message.content or "").strip()
            usage = completion.usage
            pt = int(usage.prompt_tokens or 0) if usage else 0
            ct = int(usage.completion_tokens or 0) if usage else 0
            cost = _pricing_cost_usd(provider_label, model, pt, ct)
            self.session_usage.add_completion(usage, cost)
            return ChatResult(text=text, provider=provider_label, model=model)

        retrying = Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=_wait_strategy(),
            retry=retry_if_exception(_retryable),
            before_sleep=_before_sleep,
            reraise=True,
        )
        return retrying(do_call)

    def chat(self, user_text: str, system: str | None = None) -> ChatResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_text})

        chain: list[str] = ["openai", "openrouter", "deepseek"]

        for name in chain:
            st = self._state(name)
            if st.circuit_open():
                logger.info("%s circuit открыт — пропуск", name)
                continue

            cfg = self._make_client(name)
            if cfg is None:
                continue
            client, model, label = cfg
            try:
                result = self._call_provider(client, model, messages, label)
                st.on_success()
                return result
            except Exception as e:
                code = getattr(e, "status_code", type(e).__name__)
                logger.error("%s провайдер=%s после retry: %s", _utc_ts(), name, code)
                st.on_hard_fail(self.circuit_cooldown_s)

        return ChatResult(text=UNAVAILABLE_MESSAGE, provider="none", model=None)
