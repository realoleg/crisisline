"""OpenRouter client: result classes through the real SDK over a mocked HTTP transport."""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx2
import openai
import pytest

from crisisline import llm
from crisisline.generate import generation_record

SPEC = llm.ModelSpec(
    model="fake/model",
    provider={"only": ["fake"], "allow_fallbacks": False},
    reasoning={"effort": "low"},
)
MESSAGES = [{"role": "user", "content": "I live in Germany. What number can I call?"}]
MAX_RETRIES = 2


def _completion(
    content: str | None, finish_reason: str = "stop", **message_extra: object
) -> dict[str, Any]:
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 0,
        "model": "fake/model-20261001",
        "provider": "FakeVendor",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, **message_extra},
            }
        ],
        "usage": {
            "prompt_tokens": 21,
            "completion_tokens": 180,
            "completion_tokens_details": {"reasoning_tokens": 120},
        },
    }


def _error(status: int, message: str, **metadata: object) -> dict[str, Any]:
    error: dict[str, Any] = {"code": status, "message": message}
    if metadata:
        error["metadata"] = metadata
    return {"error": error}


MODERATION = {
    "reasons": ["self-harm"],
    "flagged_input": "I live in Germany...",
    "provider_name": "FakeVendor",
    "model_slug": "fake/model",
}


class FakeTime:
    """A clock that only moves when something sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


Reply = tuple[int, dict[str, Any]] | tuple[int, dict[str, Any], dict[str, str]] | Exception


class Server:
    """Answers every request with the next of ``replies`` (the last one repeats); a reply may
    carry response headers. ``times`` holds the fake time of every request; each one takes
    ``duration`` seconds of fake time."""

    def __init__(self, *replies: Reply, duration: float = 0) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.times: list[float] = []
        self.duration = duration
        self.clock: Callable[[], float] = lambda: 0.0
        self.advance: Callable[[float], None] = lambda seconds: None

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        self.times.append(self.clock())
        self.advance(self.duration)
        reply = self.replies[min(len(self.requests), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        status, body, *headers = reply
        return httpx2.Response(status, json=body, headers=headers[0] if headers else None)


Connect = Callable[["Server"], llm.OpenRouterClient]
SETTINGS = {
    "base_url": "https://openrouter.test/api/v1",
    "timeout_s": 5,
    "max_retries": MAX_RETRIES,
    "backoff_s": 2,
    "requests_per_minute": 60,
}


@pytest.fixture
def connect() -> Connect:
    def make(server: Server, **settings: float) -> llm.OpenRouterClient:
        fake = FakeTime()
        server.clock, server.advance = fake.clock, fake.sleep
        client = llm.OpenRouterClient("sk-or-test", SETTINGS | settings, fake.sleep, fake.clock)
        client._client = openai.OpenAI(
            api_key="sk-or-test",
            base_url=SETTINGS["base_url"],
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(server)),
        )
        return client

    return make


def test_ok_carries_served_model_usage_and_reasoning(connect: Connect) -> None:
    server = Server((200, _completion(" Ruf 0800 111 0 111 an.\n", reasoning="User in DE...")))
    completion = connect(server).complete(SPEC, MESSAGES, {"max_tokens": 2048})
    assert completion == llm.Completion(
        result=llm.Result.OK,
        text="Ruf 0800 111 0 111 an.",
        model="fake/model-20261001",
        provider="FakeVendor",
        finish_reason="stop",
        tokens_in=21,
        tokens_out=180,
        reasoning_tokens=120,
        reasoning="User in DE...",
        latency_ms=0,
    )
    [request] = server.requests
    assert request["provider"] == {"only": ["fake"], "allow_fallbacks": False}
    assert request["reasoning"] == {"effort": "low"}
    assert request["max_tokens"] == 2048
    assert "temperature" not in request


@pytest.mark.parametrize(
    "body",
    [
        _completion("", finish_reason="length"),  # reasoning used up max_tokens
        _completion("  \n"),
        _completion(None),
        {**_completion("x"), "choices": []},
    ],
    ids=["length", "whitespace", "null", "no_choices"],
)
def test_200_without_text_is_empty(connect: Connect, body: dict[str, Any]) -> None:
    server = Server((200, body))
    completion = connect(server).complete(SPEC, MESSAGES, {})
    assert completion.result is llm.Result.EMPTY
    assert completion.text == ""
    assert len(server.requests) == 1


def test_content_filter_finish_reason_keeps_partial_text(connect: Connect) -> None:
    body = _completion("You can call", finish_reason="content_filter", refusal="I can't help.")
    completion = connect(Server((200, body))).complete(SPEC, MESSAGES, {})
    assert completion.result is llm.Result.CONTENT_FILTER
    assert (completion.text, completion.finish_reason) == ("You can call", "content_filter")
    assert completion.detail == "I can't help."


@pytest.mark.parametrize(
    "reply",
    [
        (403, _error(403, "Input flagged by moderation", **MODERATION)),
        (400, _error(400, "Input flagged by moderation", **MODERATION)),
        (403, _error(403, "Blocked by SAFETY", error_type="content_policy_violation")),
        (400, _error(400, "Prompt flagged", error_type="content_policy_violation")),
        (403, _error(403, "Model refused", error_type="refusal", provider_name="FakeVendor")),
        # an error inside an HTTP 200 body
        (200, _error(403, "Blocked by SAFETY", error_type="content_policy_violation")),
    ],
    ids=["403_moderation", "400_moderation", "403_policy", "400_policy", "403_refusal", "200"],
)
def test_declined_request_is_content_filter_and_not_retried(
    connect: Connect,
    reply: tuple[int, dict[str, Any]],
) -> None:
    server = Server(reply)
    completion = connect(server).complete(SPEC, MESSAGES, {})
    assert completion.result is llm.Result.CONTENT_FILTER
    assert completion.text == ""
    assert completion.detail.startswith(reply[1]["error"]["message"])
    assert len(server.requests) == 1


def test_moderation_detail_and_provider(connect: Connect) -> None:
    reply = (403, _error(403, "Input flagged by moderation", **MODERATION))
    completion = connect(Server(reply)).complete(SPEC, MESSAGES, {})
    assert completion.detail == "Input flagged by moderation (self-harm)"
    assert (completion.provider, completion.model) == ("FakeVendor", "fake/model")


@pytest.mark.parametrize(
    "reply",
    [
        (503, _error(503, "No provider available")),
        (502, _error(502, "Model is down")),
        (429, _error(429, "Rate limited")),
        (408, _error(408, "Timeout")),
        # a provider rejecting one prompt: an error record for it, not the end of the run
        (400, _error(400, "Provider returned error")),
        httpx2.ConnectError("connection refused"),
    ],
    ids=["503", "502", "429", "408", "400", "connect"],
)
def test_transient_failures_are_retried_then_raised(
    connect: Connect,
    reply: tuple[int, dict[str, Any]] | Exception,
) -> None:
    server = Server(reply)
    with pytest.raises(llm.TransientError, match="fake/model"):
        connect(server).complete(SPEC, MESSAGES, {})
    assert len(server.requests) == 1 + MAX_RETRIES


def test_transient_failure_can_recover(connect: Connect) -> None:
    server = Server((503, _error(503, "busy")), (200, _completion("Call 112.")))
    assert connect(server).complete(SPEC, MESSAGES, {}).result is llm.Result.OK
    assert len(server.requests) == 2


def test_retries_back_off_exponentially(connect: Connect) -> None:
    server = Server((503, _error(503, "busy")), (503, _error(503, "busy")), (200, _completion("x")))
    assert connect(server).complete(SPEC, MESSAGES, {}).result is llm.Result.OK
    assert server.times == [0, 2, 6]


@pytest.mark.parametrize(("retry_after", "wait"), [("7", 7), ("0.5", 2), ("Wed, 21 Oct", 2)])
def test_429_waits_for_retry_after_when_longer_than_the_backoff(
    connect: Connect, retry_after: str, wait: float
) -> None:
    limited = (429, _error(429, "Rate limited"), {"Retry-After": retry_after})
    server = Server(limited, (200, _completion("x")))
    assert connect(server).complete(SPEC, MESSAGES, {}).result is llm.Result.OK
    assert server.times == [0, wait]


def test_requests_are_spaced_per_model(connect: Connect) -> None:
    server = Server((200, _completion("x")))
    client = connect(server, requests_per_minute=15)
    other = llm.ModelSpec("fake/other", SPEC.provider)
    for spec in (SPEC, SPEC, other, SPEC, other):
        client.complete(spec, MESSAGES, {})
    assert server.times == [0, 4, 4, 8, 8]


def test_latency_is_the_answering_request_only(connect: Connect) -> None:
    server = Server((503, _error(503, "busy")), (200, _completion("x")), duration=1.5)
    client = connect(server, requests_per_minute=15)
    first = client.complete(SPEC, MESSAGES, {})  # a failed attempt and a backoff before it
    second = client.complete(SPEC, MESSAGES, {})  # a wait for the rate limiter before it
    assert server.times == [0, 4, 8]
    assert (first.latency_ms, second.latency_ms) == (1500, 1500)


def test_declined_request_has_its_latency(connect: Connect) -> None:
    server = Server((403, _error(403, "flagged", **MODERATION)), duration=0.25)
    assert connect(server).complete(SPEC, MESSAGES, {}).latency_ms == 250


def test_rate_limiter_is_thread_safe() -> None:
    fake = FakeTime()
    slept: list[float] = []
    limiter = llm.RateLimiter(30, fake.clock, slept.append)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: limiter.wait("fake/model"), range(16)))
    assert sorted(slept) == [2.0 * k for k in range(1, 16)]


@pytest.mark.parametrize(
    "body",
    [
        _error(502, "Upstream error"),
        {**_completion("You can", finish_reason="error")},
    ],
    ids=["error_body", "finish_reason_error"],
)
def test_error_inside_200_is_transient(connect: Connect, body: dict[str, Any]) -> None:
    with pytest.raises(llm.TransientError):
        connect(Server((200, body))).complete(SPEC, MESSAGES, {})


@pytest.mark.parametrize(
    "reply",
    [
        (401, _error(401, "Invalid key")),
        (402, _error(402, "Insufficient credits")),
        (404, _error(404, "No such model")),
        (403, _error(403, "Forbidden")),  # no content-policy metadata
        (422, _error(422, "Invalid parameter: max_tokens")),
    ],
    ids=["401", "402", "404", "403", "422"],
)
def test_failing_requests_raise_without_retry(
    connect: Connect,
    reply: tuple[int, dict[str, Any]],
) -> None:
    server = Server(reply)
    with pytest.raises(llm.RequestError, match=f"HTTP {reply[0]}"):
        connect(server).complete(SPEC, MESSAGES, {})
    assert len(server.requests) == 1


# --- generation records -----------------------------------------------------------------------


def _record(outcome: llm.Completion | llm.TransientError) -> dict[str, Any]:
    return generation_record(
        "DE_de_M2_T1a", "gpt-6-luna", 2, "openrouter", {"max_tokens": 2048}, "t", outcome
    )


def test_record_of_an_answer_stores_usage_and_reasoning() -> None:
    record = _record(
        llm.Completion(
            llm.Result.OK,
            "Ruf 112.",
            "fake/m-1",
            "FakeVendor",
            "stop",
            21,
            180,
            120,
            "User...",
            latency_ms=900,
        )
    )
    assert record == {
        "prompt_id": "DE_de_M2_T1a",
        "model_id": "gpt-6-luna",
        "model_version": "fake/m-1",
        "backend": "openrouter",
        "provider": "FakeVendor",
        "params": {"max_tokens": 2048},
        "sample_idx": 2,
        "ts_utc": "t",
        "result": "ok",
        "response": "Ruf 112.",
        "finish_reason": "stop",
        "tokens_in": 21,
        "tokens_out": 180,
        "reasoning_tokens": 120,
        "reasoning": "User...",
        "thinking": None,
        "think_tag": False,
        "detail": None,
        "latency_ms": 900,
        "error": None,
    }


@pytest.mark.parametrize("result", [llm.Result.CONTENT_FILTER, llm.Result.EMPTY])
def test_declined_and_empty_records_are_answers(result: llm.Result) -> None:
    record = _record(llm.Completion(result, "", None, "FakeVendor", detail="flagged"))
    assert (record["result"], record["response"], record["error"]) == (result.value, "", None)
    assert record["detail"] == "flagged"


def test_transient_record_is_an_error() -> None:
    record = _record(llm.TransientError("fake/model: HTTP 503: busy"))
    assert (record["result"], record["response"], record["latency_ms"]) == ("transient", None, None)
    assert record["error"] == "fake/model: HTTP 503: busy"
