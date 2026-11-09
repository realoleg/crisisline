"""One OpenAI-compatible client for every API call, routed through OpenRouter.

The pipeline talks to :class:`ChatClient`; tests replace it with a fake. Each completion carries
the model and provider named in the response, not the requested ones.

Every call ends in one of four classes:

- ``ok``: the model answered.
- ``content_filter``: the provider or OpenRouter declined the request or cut the answer off, or
  the model's refusal came back as ``finish_reason = content_filter``.
- ``empty``: HTTP 200 without any answer text.
- ``transient``: a technical failure that may pass: connection errors, timeouts, HTTP 408, 409,
  429 and 5xx, an error inside an HTTP 200 body, or an HTTP 400 without content-policy metadata
  (a provider rejecting one prompt; one bad prompt must not stop a model's run, and a parameter
  that every request gets wrong stops it as consecutive errors, :func:`crisisline.generate.run`).
  Only these are retried, up to ``max_retries`` times with exponential backoff (longer if the
  response's ``Retry-After`` asks so); then :class:`TransientError` is raised.

The first three are answers and come back as a :class:`Completion`. A request that would fail
every time (bad key, no credits, forbidden, unknown model, unprocessable parameters: HTTP 401,
402, 403 without content-policy metadata, 404, 422) raises :class:`RequestError`.

Every attempt, retries included, first waits for its model's slot in a :class:`RateLimiter`.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import openai

# OpenRouter's error.metadata.error_type for a request the provider declined on content.
_POLICY_ERROR_TYPES = frozenset({"content_policy_violation", "refusal"})
# error.metadata keys of OpenRouter's own moderation (HTTP 403).
_MODERATION_KEYS = frozenset({"reasons", "flagged_input"})
_POLICY_STATUSES = frozenset({400, 403})
_RETRIED_STATUSES = frozenset({400, 408, 409, 429})


class Result(StrEnum):
    OK = "ok"
    CONTENT_FILTER = "content_filter"
    EMPTY = "empty"
    TRANSIENT = "transient"


class MissingKeyError(RuntimeError):
    """The API key environment variable is unset or empty."""


class CompletionError(RuntimeError):
    """An API call produced no answer."""


class TransientError(CompletionError):
    """A technical failure that persisted through the retries; a rerun may succeed."""


class RequestError(CompletionError):
    """The request fails every time (bad key, no credits, unknown model, invalid parameter)."""


@dataclass(frozen=True, slots=True)
class Completion:
    """An answer: ``ok``, ``content_filter`` (``text`` may be partial or empty) or ``empty``.

    ``tokens_out`` includes ``reasoning_tokens``. ``reasoning`` is the model's reasoning text
    when the API returns it; it is stored but never scored. ``detail`` says why a request was
    declined. ``latency_ms`` is how long the HTTP request that returned this answer took, without
    rate-limit waits or earlier attempts.
    """

    result: Result
    text: str
    model: str | None
    provider: str | None
    finish_reason: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    reasoning_tokens: int | None = None
    reasoning: str | None = None
    detail: str | None = None
    latency_ms: int | None = None


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """An OpenRouter model slug and the ``provider`` routing and ``reasoning`` objects sent with
    every request (no ``reasoning`` object if ``None``)."""

    model: str
    provider: Mapping[str, Any]
    reasoning: Mapping[str, Any] | None = None

    @classmethod
    def from_config(cls, settings: Mapping[str, Any]) -> ModelSpec:
        return cls(settings["model"], settings["provider"], settings.get("reasoning"))


class ChatClient(Protocol):
    def complete(
        self, spec: ModelSpec, messages: list[dict[str, str]], params: Mapping[str, Any]
    ) -> Completion: ...


def api_key(env: Mapping[str, str], name: str) -> str:
    key = env.get(name, "").strip()
    if not key:
        raise MissingKeyError(
            f"{name} is not set. Copy .env.example to .env at the repository root and put "
            f"your OpenRouter key there, or export {name}."
        )
    return key


def _extra_body(spec: ModelSpec) -> dict[str, Any]:
    """OpenRouter fields outside the OpenAI schema."""
    body: dict[str, Any] = {"provider": dict(spec.provider)}
    if spec.reasoning is not None:
        body["reasoning"] = dict(spec.reasoning)
    return body


def is_content_policy(error: object) -> bool:
    """Whether an OpenRouter ``error`` object reports a content-policy decline or moderation."""
    if not isinstance(error, Mapping):
        return False
    metadata = error.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    return metadata.get("error_type") in _POLICY_ERROR_TYPES or bool(
        _MODERATION_KEYS & metadata.keys()
    )


def _declined(error: Mapping[str, Any]) -> Completion:
    metadata = error.get("metadata") or {}
    reasons = metadata.get("reasons")
    detail = str(error.get("message", ""))
    if reasons:
        detail += f" ({', '.join(map(str, reasons))})"
    return Completion(
        result=Result.CONTENT_FILTER,
        text="",
        model=metadata.get("model_slug"),
        provider=metadata.get("provider_name"),
        detail=detail,
    )


def classify_error(exc: openai.OpenAIError, model: str) -> Completion:
    """A declined request as a ``content_filter`` completion; raise for any other failure."""
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        if status in _POLICY_STATUSES and is_content_policy(exc.body):
            return _declined(exc.body)  # type: ignore[arg-type]
        if status in _RETRIED_STATUSES or status >= 500:
            raise TransientError(f"{model}: HTTP {status}: {exc.message}") from exc
        raise RequestError(f"{model}: HTTP {status}: {exc.message}") from exc
    raise TransientError(f"{model}: {exc}") from exc


def classify_response(response: Any) -> Completion:  # noqa: ANN401 - an SDK ChatCompletion
    """Classify an HTTP 200 chat completion."""
    extra = response.model_extra or {}
    choice = response.choices[0] if response.choices else None
    choice_extra = (choice.model_extra or {}) if choice else {}
    error = extra.get("error") or choice_extra.get("error")
    if error is not None or (choice and choice.finish_reason == "error"):
        if is_content_policy(error):
            return _declined(error)
        raise TransientError(f"{response.model}: error in response: {error}")

    message = choice.message if choice else None
    text = (message.content or "").strip() if message else ""
    if choice and choice.finish_reason == "content_filter":
        result = Result.CONTENT_FILTER
    else:
        result = Result.OK if text else Result.EMPTY
    usage = response.usage
    details = usage.completion_tokens_details if usage else None
    return Completion(
        result=result,
        text=text,
        model=response.model,
        # OpenRouter adds the serving provider and the reasoning text outside the OpenAI schema.
        provider=extra.get("provider"),
        finish_reason=choice.finish_reason if choice else None,
        tokens_in=usage.prompt_tokens if usage else None,
        tokens_out=usage.completion_tokens if usage else None,
        reasoning_tokens=details.reasoning_tokens if details else None,
        reasoning=(message.model_extra or {}).get("reasoning") if message else None,
        detail=getattr(message, "refusal", None) if message else None,
    )


def retry_after(exc: BaseException | None) -> float | None:
    """Seconds asked for by the ``Retry-After`` header of a failed response, if any."""
    if not isinstance(exc, openai.APIStatusError):
        return None
    try:
        return float(exc.response.headers["retry-after"])
    except (KeyError, ValueError):  # no header, or an HTTP date
        return None


class RateLimiter:
    """Spaces request starts per model evenly, ``60 / per_minute`` seconds apart.

    Thread-safe: a caller reserves its model's next free slot, then sleeps until it.
    """

    def __init__(
        self,
        per_minute: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 60 / per_minute
        self._clock = clock
        self._sleep = sleep
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, model: str) -> None:
        with self._lock:
            now = self._clock()
            start = max(now, self._next.get(model, now))
            self._next[model] = start + self._interval
        if start > now:
            self._sleep(start - now)


class OpenRouterClient:
    """Transient failures are retried ``max_retries`` times, waiting ``backoff_s``, then twice as
    long each time, or as long as ``Retry-After`` asks if that is longer."""

    def __init__(
        self,
        key: str,
        settings: Mapping[str, Any],
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._client = openai.OpenAI(
            api_key=key,
            base_url=settings["base_url"],
            timeout=settings["timeout_s"],
            max_retries=0,
        )
        self.max_retries = settings["max_retries"]
        self.backoff_s = settings["backoff_s"]
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._limiter = RateLimiter(settings["requests_per_minute"], self._clock, self._sleep)

    def complete(
        self, spec: ModelSpec, messages: list[dict[str, str]], params: Mapping[str, Any]
    ) -> Completion:
        for attempt in range(self.max_retries + 1):
            self._limiter.wait(spec.model)
            try:
                return self._attempt(spec, messages, params)
            except TransientError as exc:
                error = exc
            if attempt < self.max_retries:
                backoff = self.backoff_s * 2**attempt
                self._sleep(max(backoff, retry_after(error.__cause__) or 0))
        raise error

    def _attempt(
        self, spec: ModelSpec, messages: list[dict[str, str]], params: Mapping[str, Any]
    ) -> Completion:
        start = self._clock()
        try:
            response = self._client.chat.completions.create(
                model=spec.model,
                messages=messages,  # type: ignore[arg-type]
                extra_body=_extra_body(spec),
                **params,
            )
        except openai.OpenAIError as exc:
            completion = classify_error(exc, spec.model)
        else:
            completion = classify_response(response)
        return dataclasses.replace(completion, latency_ms=round((self._clock() - start) * 1000))
