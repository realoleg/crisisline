"""Generation backends and the resumable generation loop.

Two backends answer a chat: ``ollama`` (local models, native ``/api/chat``) and ``openrouter``
(API models, :mod:`crisisline.llm`). Both return a :class:`~crisisline.llm.Completion`, or raise
:class:`~crisisline.llm.TransientError` after their retries or
:class:`~crisisline.llm.RequestError` for a request that fails every time.

:func:`run` asks the backend for every ``(prompt, sample_idx)`` it is given and hands each
record to a writer as soon as it exists, in task order, so a run can stop anywhere and resume
by skipping what is already written.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from crisisline.llm import ChatClient, Completion, ModelSpec, RequestError, Result, TransientError

BACKEND_OLLAMA = "ollama"
BACKEND_OPENROUTER = "openrouter"
OLLAMA_THINKING = "thinking"  # /api/show capability of models that can think

_THINK_TAG = re.compile(r"</?think>")
_RETRIED_STATUSES = frozenset({408, 429})

Prompt = Mapping[str, Any]
Task = tuple[Prompt, int]  # (prompt, sample_idx)


class Backend(Protocol):
    name: str

    def params(self, sample_idx: int) -> dict[str, Any]:
        """Generation parameters sent for this sample, as recorded."""
        ...

    def complete(self, messages: list[dict[str, str]], sample_idx: int) -> Completion: ...


# --- Ollama ----------------------------------------------------------------------------------


class OllamaHTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


Request = Callable[[str, Mapping[str, Any] | None, float], Any]


def http_json(url: str, payload: Mapping[str, Any] | None, timeout_s: float) -> Any:  # noqa: ANN401 - JSON
    """GET ``url`` (POST ``payload`` as JSON if given) and parse the JSON answer."""
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            message = json.loads(body).get("error", body)
        except (json.JSONDecodeError, AttributeError):
            message = body
        raise OllamaHTTPError(exc.code, str(message)) from exc


class OllamaClient:
    """Ollama's native API. Connection errors, timeouts, HTTP 408, 429 and 5xx are retried with
    exponential backoff; any other HTTP error raises :class:`RequestError` at once."""

    def __init__(
        self,
        settings: Mapping[str, Any],
        request: Request | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.host = settings["host"].rstrip("/")
        self.timeout_s = settings["timeout_s"]
        self.max_retries = settings["max_retries"]
        self.backoff_s = settings["backoff_s"]
        self._request = request or http_json
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic

    def _call(self, path: str, payload: Mapping[str, Any] | None = None) -> Any:  # noqa: ANN401
        return self._timed_call(path, payload)[0]

    def _timed_call(self, path: str, payload: Mapping[str, Any] | None) -> tuple[Any, int]:
        """The answer and how long the request that returned it took, in milliseconds."""
        for attempt in range(self.max_retries + 1):
            start = self._clock()
            try:
                answer = self._request(f"{self.host}{path}", payload, self.timeout_s)
                return answer, round((self._clock() - start) * 1000)
            except OllamaHTTPError as exc:
                if exc.status not in _RETRIED_STATUSES and exc.status < 500:
                    raise RequestError(f"ollama {path}: {exc}") from exc
                error: Exception = exc
            except OSError as exc:  # connection refused, reset, timeout
                error = exc
            if attempt < self.max_retries:
                self._sleep(self.backoff_s * 2**attempt)
        raise TransientError(f"ollama {path}: {error}") from error

    def version(self) -> str:
        return self._call("/api/version")["version"]

    def show(self, tag: str) -> dict[str, Any]:
        return self._call("/api/show", {"model": tag})

    def digest(self, tag: str) -> str | None:
        models = self._call("/api/tags").get("models", [])
        return next((m["digest"] for m in models if m["name"] == tag), None)

    def chat(
        self,
        tag: str,
        messages: list[dict[str, str]],
        options: Mapping[str, Any],
        think: bool | None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": tag,
            "messages": messages,
            "stream": False,
            "options": dict(options),
        }
        if think is not None:
            payload["think"] = think
        answer, latency_ms = self._timed_call("/api/chat", payload)
        message = answer.get("message") or {}
        text = (message.get("content") or "").strip()
        return Completion(
            result=Result.OK if text else Result.EMPTY,
            text=text,
            model=answer.get("model"),
            provider=None,
            finish_reason=answer.get("done_reason"),
            tokens_in=answer.get("prompt_eval_count"),
            tokens_out=answer.get("eval_count"),
            reasoning=message.get("thinking") or None,
            latency_ms=latency_ms,
        )


@dataclass(frozen=True)
class OllamaBackend:
    """A pulled Ollama model. ``think`` is sent only for models that can think (``False``
    then); ``None`` leaves it out. Each sample gets its own seed: ``seed + sample_idx``."""

    client: OllamaClient
    tag: str
    digest: str
    options: Mapping[str, Any]
    think: bool | None
    name: str = BACKEND_OLLAMA

    def params(self, sample_idx: int) -> dict[str, Any]:
        options = {**self.options, "seed": self.options["seed"] + sample_idx}
        return {"options": options, "think": self.think}

    def complete(self, messages: list[dict[str, str]], sample_idx: int) -> Completion:
        params = self.params(sample_idx)
        completion = self.client.chat(self.tag, messages, params["options"], self.think)
        return dataclasses.replace(completion, model=self.digest)


@dataclass(frozen=True)
class OpenRouterBackend:
    client: ChatClient
    spec: ModelSpec
    max_tokens: int
    name: str = BACKEND_OPENROUTER

    def params(self, sample_idx: int) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "reasoning": dict(self.spec.reasoning or {}),
            "provider": dict(self.spec.provider),
        }

    def complete(self, messages: list[dict[str, str]], sample_idx: int) -> Completion:
        return self.client.complete(self.spec, messages, {"max_tokens": self.max_tokens})


# --- records and the run ---------------------------------------------------------------------


class BackendDown(RequestError):
    """Too many records in a row are technical errors: the backend is most likely down (ollama
    serve gone, an API outage, a long 429 storm). Stops the run like any
    :class:`~crisisline.llm.RequestError`; a rerun resumes."""


def generation_record(
    prompt_id: str,
    model_id: str,
    sample_idx: int,
    backend: str,
    params: Mapping[str, Any],
    ts_utc: str,
    outcome: Completion | TransientError,
) -> dict[str, Any]:
    """One generation record.

    ``content_filter`` and ``empty`` completions are answers: ``error`` is null and
    ``response`` holds whatever text came back. ``latency_ms`` is the duration of the request
    that returned the answer (:class:`~crisisline.llm.Completion`). A transient failure that
    persisted through the retries has ``result = transient``, no response, no latency and the
    error message. Reasoning text goes to ``reasoning`` (API) or ``thinking`` (Ollama);
    ``think_tag`` marks a response that contains ``<think>`` or ``</think>``. None of these is
    scored.
    """
    record: dict[str, Any] = {
        "prompt_id": prompt_id,
        "model_id": model_id,
        "model_version": None,
        "backend": backend,
        "provider": None,
        "params": dict(params),
        "sample_idx": sample_idx,
        "ts_utc": ts_utc,
        "result": Result.TRANSIENT.value,
        "response": None,
        "finish_reason": None,
        "tokens_in": None,
        "tokens_out": None,
        "reasoning_tokens": None,
        "reasoning": None,
        "thinking": None,
        "think_tag": None,
        "detail": None,
        "latency_ms": None,
        "error": None,
    }
    if isinstance(outcome, TransientError):
        return record | {"error": str(outcome)}
    reasoning_field = "thinking" if backend == BACKEND_OLLAMA else "reasoning"
    return record | {
        "model_version": outcome.model,
        "provider": outcome.provider,
        "result": outcome.result.value,
        "response": outcome.text,
        "finish_reason": outcome.finish_reason,
        "tokens_in": outcome.tokens_in,
        "tokens_out": outcome.tokens_out,
        "reasoning_tokens": outcome.reasoning_tokens,
        reasoning_field: outcome.reasoning,
        "think_tag": bool(_THINK_TAG.search(outcome.text)),
        "detail": outcome.detail,
        "latency_ms": outcome.latency_ms,
    }


def messages(prompt: Prompt) -> list[dict[str, str]]:
    system = [{"role": "system", "content": prompt["system"]}] if prompt["system"] else []
    return [*system, {"role": "user", "content": prompt["user"]}]


def select_prompts(
    prompts: Sequence[Prompt],
    model_id: str,
    backend: str,
    mode: str,
    experiment: Mapping[str, Any],
) -> list[Prompt]:
    """Prompts of a run: all of them, or the model's robustness blocks."""
    if mode == "main":
        return list(prompts)

    def includes(models: str | Sequence[str]) -> bool:
        # `models` is a list of model_ids or a backend name (every model of that backend).
        return models == backend if isinstance(models, str) else model_id in models

    blocks = [b for b in experiment["robustness"]["blocks"] if includes(b["models"])]
    if not blocks:
        raise ValueError(f"{model_id} is in no robustness block (experiment.yaml)")
    return [
        p
        for p in prompts
        if any(all(p[f] in values for f, values in b["prompts"].items()) for b in blocks)
    ]


def pending_tasks(
    prompts: Iterable[Prompt], samples: int, done: set[tuple[str, int]]
) -> list[Task]:
    return [
        (prompt, sample_idx)
        for prompt in prompts
        for sample_idx in range(samples)
        if (prompt["prompt_id"], sample_idx) not in done
    ]


def run(
    tasks: Sequence[Task],
    model_id: str,
    backend: Backend,
    workers: int,
    write: Callable[[dict[str, Any]], None],
    now: Callable[[], str],
    max_errors: int,
) -> Counter[str]:
    """Generate every task and ``write`` each record in task order; count results.

    A :class:`~crisisline.llm.RequestError` cancels the tasks not yet started and propagates;
    everything written before it stays. So does :class:`BackendDown`, raised once ``max_errors``
    written records in a row are technical errors, instead of an error record for every task
    left.
    """

    def one(task: Task) -> dict[str, Any]:
        prompt, sample_idx = task
        try:
            outcome: Completion | TransientError = backend.complete(messages(prompt), sample_idx)
        except TransientError as exc:
            outcome = exc
        return generation_record(
            prompt["prompt_id"],
            model_id,
            sample_idx,
            backend.name,
            backend.params(sample_idx),
            now(),
            outcome,
        )

    counts: Counter[str] = Counter()
    errors_in_a_row = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, task) for task in tasks]
        try:
            for future in futures:
                record = future.result()
                write(record)
                counts[record["result"]] += 1
                errors_in_a_row = errors_in_a_row + 1 if record["error"] else 0
                if errors_in_a_row == max_errors:
                    raise BackendDown(
                        f"{max_errors} records in a row are technical errors; is the backend"
                        f" down? Last: {record['error']}"
                    )
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return counts
