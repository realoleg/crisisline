"""Generation: Ollama client, backends, prompt selection, the resumable run and its manifest."""

import fcntl
import io
import json
import logging
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
import yaml

from crisisline import cli, generate, llm, prompts
from crisisline.cli import load_config, main, read_csv, read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
OLLAMA_SETTINGS = {"host": "http://ollama.test", "timeout_s": 5, "max_retries": 3, "backoff_s": 2}
TAG = "qwen3.5:4b-q8_0"
DIGEST = "sha256:4b4b"
MESSAGES = [{"role": "user", "content": "I live in Germany. What number can I call?"}]


class FakeOllama:
    """Ollama's native API in memory: ``replies`` maps a path to the answers it gives in turn
    (the last repeats); an exception is raised. ``/api/chat`` answers from the request."""

    def __init__(
        self, capabilities: tuple[str, ...] = ("completion", "thinking"), **replies: object
    ) -> None:
        self.calls: list[tuple[str, Mapping[str, Any] | None]] = []
        self.replies: dict[str, list[Any]] = {
            "/api/version": [{"version": "0.30.1"}],
            "/api/show": [
                {"capabilities": list(capabilities), "details": {"quantization_level": "Q8_0"}}
            ],
            "/api/tags": [
                {"models": [{"name": "other:1b", "digest": "x"}, {"name": TAG, "digest": DIGEST}]}
            ],
            "/api/chat": [None],
        }
        self.replies.update({f"/api/{k}": list(v) for k, v in replies.items()})  # type: ignore[call-overload]

    def __call__(self, url: str, payload: Mapping[str, Any] | None, timeout_s: float) -> Any:  # noqa: ANN401
        path = urllib.parse.urlsplit(url).path
        self.calls.append((path, payload))
        answers = self.replies[path]
        answer = answers[min(sum(p == path for p, _ in self.calls), len(answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        if answer is None:  # a chat answer naming the prompt and the seed
            user = payload["messages"][-1]["content"]
            return {
                "model": payload["model"],
                "message": {"role": "assistant", "content": f"Call 112. ({user[:20]})"},
                "done_reason": "stop",
                "prompt_eval_count": 30,
                "eval_count": 12,
            }
        return answer

    def chats(self) -> list[Mapping[str, Any]]:
        return [payload for path, payload in self.calls if path == "/api/chat"]


def _client(server: FakeOllama, sleeps: list[float] | None = None) -> generate.OllamaClient:
    return generate.OllamaClient(
        OLLAMA_SETTINGS,
        request=server,
        sleep=(sleeps if sleeps is not None else []).append,
        clock=lambda: 0.0,
    )


# --- Ollama client ---------------------------------------------------------------------------


def test_chat_sends_think_only_when_given_and_keeps_thinking_apart() -> None:
    answer = {
        "model": TAG,
        "message": {"role": "assistant", "content": " Call 112.\n", "thinking": "User in DE."},
        "done_reason": "length",
        "prompt_eval_count": 30,
        "eval_count": 400,
    }
    server = FakeOllama(chat=[answer])
    completion = _client(server).chat(TAG, MESSAGES, {"temperature": 0, "seed": 2026}, False)
    assert completion == llm.Completion(
        result=llm.Result.OK,
        text="Call 112.",
        model=TAG,
        provider=None,
        finish_reason="length",
        tokens_in=30,
        tokens_out=400,
        reasoning="User in DE.",
        latency_ms=0,
    )
    assert server.chats() == [
        {
            "model": TAG,
            "messages": MESSAGES,
            "stream": False,
            "options": {"temperature": 0, "seed": 2026},
            "think": False,
        }
    ]
    _client(server).chat(TAG, MESSAGES, {}, None)
    assert "think" not in server.chats()[-1]


def test_chat_without_text_is_empty() -> None:
    server = FakeOllama(chat=[{"model": TAG, "message": {"content": "  "}, "done_reason": "stop"}])
    assert _client(server).chat(TAG, MESSAGES, {}, None).result is llm.Result.EMPTY


def test_transient_errors_are_retried_with_backoff() -> None:
    server = FakeOllama(
        version=[ConnectionRefusedError(), generate.OllamaHTTPError(503, "busy"), {"version": "1"}]
    )
    sleeps: list[float] = []
    assert _client(server, sleeps).version() == "1"
    assert sleeps == [2, 4]


def test_chat_latency_is_the_answering_request_only() -> None:
    now = [0.0]
    answer = {"model": TAG, "message": {"content": "Call 112."}, "done_reason": "stop"}
    server = FakeOllama(chat=[generate.OllamaHTTPError(503, "busy"), answer])

    def slow(url: str, payload: Mapping[str, Any] | None, timeout_s: float) -> Any:  # noqa: ANN401
        now[0] += 0.75
        return server(url, payload, timeout_s)

    def sleep(seconds: float) -> None:
        now[0] += seconds

    client = generate.OllamaClient(OLLAMA_SETTINGS, request=slow, sleep=sleep, clock=lambda: now[0])
    assert client.chat(TAG, MESSAGES, {}, None).latency_ms == 750  # not 0.75 + 2 + 0.75 s


@pytest.mark.parametrize(
    "error",
    [TimeoutError(), generate.OllamaHTTPError(500, "crash"), generate.OllamaHTTPError(429, "x")],
)
def test_persistent_transient_error_raises_after_retries(error: Exception) -> None:
    server = FakeOllama(version=[error])
    sleeps: list[float] = []
    with pytest.raises(llm.TransientError, match="ollama /api/version"):
        _client(server, sleeps).version()
    assert len(server.calls) == 4
    assert sleeps == [2, 4, 8]


@pytest.mark.parametrize("status", [400, 404])
def test_failing_request_is_not_retried(status: int) -> None:
    server = FakeOllama(show=[generate.OllamaHTTPError(status, f"model '{TAG}' not found")])
    with pytest.raises(llm.RequestError, match=f"HTTP {status}"):
        _client(server).show(TAG)
    assert len(server.calls) == 1


def test_digest_of_a_tag() -> None:
    assert _client(FakeOllama()).digest(TAG) == DIGEST
    assert _client(FakeOllama()).digest("missing:1b") is None


def test_http_json_posts_the_payload_and_parses_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        sent.append(request)
        return io.BytesIO(b'{"version": "0.30.1"}')

    monkeypatch.setattr(generate.urllib.request, "urlopen", urlopen)
    assert generate.http_json("http://ollama.test/api/show", {"model": TAG}, 5) == {
        "version": "0.30.1"
    }
    assert generate.http_json("http://ollama.test/api/tags", None, 5) == {"version": "0.30.1"}
    assert [(r.get_method(), r.data) for r in sent] == [
        ("POST", json.dumps({"model": TAG}).encode()),
        ("GET", None),
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b'{"error": "model not found"}', "model not found"),
        (b"<html>bad gateway</html>", "<html>bad gateway</html>"),
        (b'["not", "an", "object"]', '["not", "an", "object"]'),
    ],
)
def test_http_json_raises_the_status_and_ollama_error_message(
    monkeypatch: pytest.MonkeyPatch, body: bytes, message: str
) -> None:
    def urlopen(request: urllib.request.Request, timeout: float) -> None:
        raise urllib.error.HTTPError(
            request.full_url, 404, "Not Found", Message(), io.BytesIO(body)
        )

    monkeypatch.setattr(generate.urllib.request, "urlopen", urlopen)
    with pytest.raises(generate.OllamaHTTPError, match=f"HTTP 404: {re.escape(message)}") as exc:
        generate.http_json("http://ollama.test/api/show", {"model": TAG}, 5)
    assert exc.value.status == 404


# --- backends and records --------------------------------------------------------------------


def test_ollama_backend_seeds_each_sample_and_reports_the_digest() -> None:
    server = FakeOllama()
    options = {"temperature": 0.7, "top_p": 1, "seed": 2026, "num_predict": 400}
    backend = generate.OllamaBackend(_client(server), TAG, DIGEST, options, think=False)
    assert backend.params(2) == {"options": {**options, "seed": 2028}, "think": False}
    completion = backend.complete(MESSAGES, 2)
    assert completion.model == DIGEST
    assert server.chats()[0]["options"]["seed"] == 2028


def test_openrouter_backend_sends_the_token_budget() -> None:
    class Client:
        def complete(
            self, spec: llm.ModelSpec, messages: list[dict[str, str]], params: Mapping[str, Any]
        ) -> llm.Completion:
            self.params = params
            return llm.Completion(llm.Result.OK, "Call 112.", "fake/m-1", "Fake")

    client = Client()
    spec = llm.ModelSpec("fake/m", {"only": ["fake"]}, {"effort": "low"})
    backend = generate.OpenRouterBackend(client, spec, 2048)
    assert backend.complete(MESSAGES, 1).text == "Call 112."
    assert client.params == {"max_tokens": 2048}
    assert backend.params(1) == {
        "max_tokens": 2048,
        "reasoning": {"effort": "low"},
        "provider": {"only": ["fake"]},
    }


@pytest.mark.parametrize(
    ("text", "flagged"),
    [("<think>hm</think>Call 112.", True), ("</think>Call 112.", True), ("Call 112.", False)],
)
def test_ollama_record_keeps_thinking_apart_and_flags_think_tags(text: str, flagged: bool) -> None:
    completion = llm.Completion(llm.Result.OK, text, DIGEST, None, "stop", 30, 12, reasoning="hm")
    record = generate.generation_record(
        "DE_de_M1_T1a", "qwen3.5-4b", 0, "ollama", {}, "t", completion
    )
    assert (record["thinking"], record["reasoning"]) == ("hm", None)
    assert record["think_tag"] is flagged
    assert record["model_version"] == DIGEST


# --- prompt selection ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def matrix() -> list[dict[str, Any]]:
    """The full prompt matrix, with the English source standing in for every translation."""
    cfg = load_config(REPO_ROOT / "configs" / "prompts.yaml")
    countries = read_csv(REPO_ROOT / "data" / "countries.csv")
    translations = {
        u.key: {"source_en": u.source_en, "translation": u.source_en, "qa": "auto"}
        for u in prompts.translation_units(countries, cfg)
    }
    m2 = prompts.m2_table(countries, read_csv(REPO_ROOT / "data" / "resources_v1.csv"), cfg)
    return prompts.build_prompts(countries, m2, cfg, translations, "XX")


@pytest.mark.parametrize(
    ("model_id", "count"), [("qwen3.5-4b", 4 + 280), ("gemma4-e4b", 4 + 280), ("llama3.1-8b", 4)]
)
def test_robustness_selects_the_model_blocks(
    matrix: list[dict[str, Any]], config: dict[str, Any], model_id: str, count: int
) -> None:
    selected = generate.select_prompts(matrix, model_id, "ollama", "robustness", config)
    assert len(selected) == count


def test_api_models_are_in_no_robustness_block(
    matrix: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="gpt-6-luna is in no robustness block"):
        generate.select_prompts(matrix, "gpt-6-luna", "openrouter", "robustness", config)


def test_pending_tasks_skip_done_pairs(matrix: list[dict[str, Any]]) -> None:
    first = matrix[:2]
    done = {(first[0]["prompt_id"], 1)}
    tasks = generate.pending_tasks(first, 3, done)
    assert [(p["prompt_id"], i) for p, i in tasks] == [
        (first[0]["prompt_id"], 0),
        (first[0]["prompt_id"], 2),
        *((first[1]["prompt_id"], i) for i in range(3)),
    ]


def test_messages_include_the_system_prompt_only_when_set(matrix: list[dict[str, Any]]) -> None:
    by_id = {p["prompt_id"]: p for p in matrix}
    assert [m["role"] for m in generate.messages(by_id["DE_de_M1_T1a"])] == ["system", "user"]
    assert [m["role"] for m in generate.messages(by_id["DE_de_city_T1a"])] == ["user"]


# --- the generate command --------------------------------------------------------------------


class Disconnect(BaseException):
    """A Colab runtime going away mid-run."""


class ScriptedBackend:
    """Answers every chat; raises ``fail`` on call number ``fail_at`` (1-based)."""

    name = "ollama"

    def __init__(self, fail_at: int | None = None, fail: BaseException | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.fail_at, self.fail = fail_at, fail

    def params(self, sample_idx: int) -> dict[str, Any]:
        return {"options": {"temperature": 0, "seed": 2026 + sample_idx}, "think": False}

    def complete(self, messages: list[dict[str, str]], sample_idx: int) -> llm.Completion:
        self.calls.append((messages[-1]["content"], sample_idx))
        if len(self.calls) == self.fail_at:
            raise self.fail
        if "Tokyo" in messages[-1]["content"]:
            raise llm.TransientError("ollama /api/chat: HTTP 500: crash")
        return llm.Completion(llm.Result.OK, "Call 112.", DIGEST, None, "stop", 30, 12)


ENVIRONMENT = {
    "model_id": "qwen3.5-4b",
    "backend": "ollama",
    "git_commit": "abc123",
    "ollama_version": "0.30.1",
    "tag": TAG,
    "digest": DIGEST,
    "quantization": "Q8_0",
    "capabilities": ["completion", "thinking"],
    "gpu_name": "NVIDIA L4",
}


MAX_ERRORS = 3  # technical errors in a row that stop a run


def _setup(backend: ScriptedBackend, samples: int = 1, **environment: Any) -> cli.GenerationSetup:  # noqa: ANN401
    session = {"driver_version": "550.54", "cuda_version": "12.4"}
    return cli.GenerationSetup(backend, samples, 1, ENVIRONMENT | environment, session, MAX_ERRORS)


@pytest.fixture
def gen_config(
    tmp_path: Path, config: dict[str, Any], matrix: list[dict[str, Any]]
) -> dict[str, Any]:
    """The repository config with prompts and generation outputs under ``tmp_path``."""
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in matrix), encoding="utf-8"
    )
    paths = {name: REPO_ROOT / path for name, path in config["paths"].items()}
    paths |= {
        "prompts": prompts_path,
        "generations": tmp_path / "generations",
        "robustness_generations": tmp_path / "robustness",
    }
    return {**config, "paths": paths}


def _run(
    config: dict[str, Any], mode: str, limit: int, setup: cli.GenerationSetup, now: str
) -> int:
    return cli.run_generate(config, "qwen3.5-4b", mode, limit, setup, lambda: now)


def _out(tmp_path: Path, mode: str = "generations") -> Path:
    return tmp_path / mode / "qwen3.5-4b.jsonl"


def test_generate_writes_records_and_manifest(tmp_path: Path, gen_config: dict[str, Any]) -> None:
    backend = ScriptedBackend()
    assert _run(gen_config, "main", 6, _setup(backend, 2), "t") == 0
    records = read_jsonl(_out(tmp_path))
    assert len(records) == 6 * 2 == len(backend.calls)
    assert [(r["prompt_id"], r["sample_idx"]) for r in records[:3]] == [
        ("GB_en_city_T1a", 0),
        ("GB_en_city_T1a", 1),
        ("GB_en_city_T1b", 0),
    ]
    assert records[1]["params"] == {"options": {"temperature": 0, "seed": 2027}, "think": False}
    assert records[0]["model_version"] == DIGEST and records[0]["result"] == "ok"
    manifest = json.loads((tmp_path / "generations" / "qwen3.5-4b.manifest.json").read_text())
    assert manifest["environment"] == ENVIRONMENT | {
        "mode": "main",
        "prompts_sha256": cli.file_sha256(gen_config["paths"]["prompts"]),
        "params": backend.params(0),
        "samples": 2,
    }
    assert manifest["sessions"] == [
        {
            "started_utc": "t",
            "records_before": 0,
            "driver_version": "550.54",
            "cuda_version": "12.4",
        }
    ]


def test_generate_resumes_after_a_disconnect(
    tmp_path: Path, gen_config: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    interrupted = ScriptedBackend(fail_at=5, fail=Disconnect())
    with pytest.raises(Disconnect):
        _run(gen_config, "main", 5, _setup(interrupted, 2), "t1")
    out = _out(tmp_path)
    assert len(read_jsonl(out)) == 4
    with out.open("a", encoding="utf-8") as f:  # killed in the middle of writing a record
        f.write('{"prompt_id": "DE_de_city_T2b", "model_id": "qwen')

    resumed = ScriptedBackend()
    assert _run(gen_config, "main", 5, _setup(resumed, 2), "t2") == 0
    records = read_jsonl(out)
    keys = [(r["prompt_id"], r["sample_idx"]) for r in records]
    assert len(keys) == len(set(keys)) == 5 * 2
    assert len(resumed.calls) == 6  # only what was missing
    assert "dropping an incomplete last line" in caplog.text
    assert "4 done, 6 to generate" in caplog.text
    manifest = json.loads((tmp_path / "generations" / "qwen3.5-4b.manifest.json").read_text())
    assert [s["records_before"] for s in manifest["sessions"]] == [0, 4]

    again = ScriptedBackend()
    assert _run(gen_config, "main", 5, _setup(again, 2), "t3") == 0
    assert again.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_name", "Tesla T4"),
        ("ollama_version", "0.31.0"),
        ("git_commit", "def456"),
        ("digest", "sha256:9"),
    ],
)
def test_generate_refuses_a_different_environment(
    tmp_path: Path,
    gen_config: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    field: str,
    value: str,
) -> None:
    assert _run(gen_config, "main", 2, _setup(ScriptedBackend()), "t") == 0
    other = ScriptedBackend()
    setup = _setup(other, **{field: value})
    assert _run(gen_config, "main", 4, setup, "t") == 1
    assert other.calls == []
    assert f"Environment differs from the manifest: {field}:" in caplog.text
    assert len(read_jsonl(_out(tmp_path))) == 2


def test_generate_refuses_records_without_a_manifest(
    tmp_path: Path, gen_config: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """Records whose manifest went missing (a sync that lost it) have no environment to check
    against: a fresh manifest would let a different environment continue them."""
    assert _run(gen_config, "main", 2, _setup(ScriptedBackend()), "t") == 0
    manifest = tmp_path / "generations" / "qwen3.5-4b.manifest.json"
    manifest.unlink()
    other = ScriptedBackend()
    assert _run(gen_config, "main", 4, _setup(other, ollama_version="0.31.0"), "t") == 1
    assert other.calls == [] and not manifest.exists()
    assert "no manifest" in caplog.text
    assert len(read_jsonl(_out(tmp_path))) == 2


def test_driver_change_is_logged_not_refused(
    tmp_path: Path, gen_config: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    assert _run(gen_config, "main", 1, _setup(ScriptedBackend()), "t") == 0
    setup = _setup(ScriptedBackend())
    setup.session["driver_version"] = "560.1"
    assert _run(gen_config, "main", 2, setup, "t") == 0
    assert "driver_version changed: '550.54' -> '560.1'" in caplog.text


def test_transient_failures_are_recorded_and_not_retried_on_resume(
    tmp_path: Path, gen_config: dict[str, Any]
) -> None:
    selected = generate.select_prompts(
        read_jsonl(gen_config["paths"]["prompts"]), "qwen3.5-4b", "ollama", "main", gen_config
    )
    limit = next(i for i, p in enumerate(selected) if "Tokyo" in p["user"]) + 1
    assert _run(gen_config, "main", limit, _setup(ScriptedBackend()), "t") == 0
    [failed] = [r for r in read_jsonl(_out(tmp_path)) if r["error"]]
    assert (failed["prompt_id"], failed["result"], failed["response"]) == (
        "JP_ja_city_T1a",
        "transient",
        None,
    )
    rerun = ScriptedBackend()
    assert _run(gen_config, "main", limit, _setup(rerun), "t") == 0
    assert rerun.calls == []


def test_failing_request_stops_and_keeps_what_was_written(
    tmp_path: Path, gen_config: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    backend = ScriptedBackend(fail_at=3, fail=llm.RequestError("ollama /api/chat: HTTP 404"))
    assert _run(gen_config, "main", 5, _setup(backend), "t") == 1
    assert len(read_jsonl(_out(tmp_path))) == 2
    assert "HTTP 404; stopped after 2 records (rerun resumes)" in caplog.text


def test_generate_refuses_foreign_records(
    tmp_path: Path, gen_config: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    out = _out(tmp_path, "generations")
    out.parent.mkdir(parents=True)
    out.write_text('{"prompt_id": "DE_de_city_T1a", "model_id": "llama3.1-8b", "sample_idx": 0}\n')
    assert _run(gen_config, "main", 1, _setup(ScriptedBackend()), "t") == 1
    assert "is for model llama3.1-8b" in caplog.text


def test_generate_needs_the_prompt_matrix(
    gen_config: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    gen_config["paths"]["prompts"].unlink()
    assert _run(gen_config, "main", 1, _setup(ScriptedBackend()), "t") == 1
    assert "run crisisline build first" in caplog.text


# --- setup against a fake Ollama server ------------------------------------------------------


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> FakeOllama:
    server = FakeOllama()
    monkeypatch.setattr(generate, "http_json", server)
    monkeypatch.setattr(generate.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(cli, "git_commit", lambda root: "abc123")
    monkeypatch.setattr(
        cli,
        "gpu_info",
        lambda: {"name": "NVIDIA L4", "driver_version": "550.54", "cuda_version": "12.4"},
    )
    return server


@pytest.mark.parametrize(
    ("mode", "temperature", "samples"), [("main", 0, 1), ("robustness", 0.7, 3)]
)
def test_setup_probes_ollama(
    fake_ollama: FakeOllama, gen_config: dict[str, Any], mode: str, temperature: float, samples: int
) -> None:
    setup = cli.setup_generation(gen_config, REPO_ROOT, "qwen3.5-4b", mode)
    assert setup.environment == ENVIRONMENT
    assert setup.session == {"driver_version": "550.54", "cuda_version": "12.4"}
    assert setup.samples == samples
    assert setup.backend.params(1) == {
        "options": {"temperature": temperature, "top_p": 1, "seed": 2027, "num_predict": 2048},
        "think": False,
    }


def test_setup_leaves_think_out_for_models_that_cannot_think(
    monkeypatch: pytest.MonkeyPatch, fake_ollama: FakeOllama, gen_config: dict[str, Any]
) -> None:
    fake_ollama.replies["/api/show"] = [{"capabilities": ["completion"], "details": {}}]
    setup = cli.setup_generation(gen_config, REPO_ROOT, "qwen3.5-4b", "main")
    setup.backend.complete(MESSAGES, 0)
    assert "think" not in fake_ollama.chats()[0]


def test_setup_names_the_pull_for_a_missing_model(
    fake_ollama: FakeOllama, gen_config: dict[str, Any]
) -> None:
    fake_ollama.replies["/api/show"] = [generate.OllamaHTTPError(404, f"model '{TAG}' not found")]
    with pytest.raises(llm.RequestError, match=f"ollama pull {TAG}"):
        cli.setup_generation(gen_config, REPO_ROOT, "qwen3.5-4b", "main")


def test_setup_rejects_unknown_models(fake_ollama: FakeOllama, gen_config: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="unknown model 'gpt-9'"):
        cli.setup_generation(gen_config, REPO_ROOT, "gpt-9", "main")


def test_setup_of_an_api_model_routes_it_through_openrouter(
    monkeypatch: pytest.MonkeyPatch, gen_config: dict[str, Any]
) -> None:
    monkeypatch.setattr(cli, "git_commit", lambda root: "abc123")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    setup = cli.setup_generation(gen_config, REPO_ROOT, "gpt-6-luna", "main")
    assert isinstance(setup.backend, generate.OpenRouterBackend)
    assert setup.backend.spec == llm.ModelSpec(
        "openai/gpt-6-luna", {"only": ["openai"], "allow_fallbacks": False}, {"effort": "low"}
    )
    assert setup.environment == {
        "model_id": "gpt-6-luna",
        "backend": "openrouter",
        "git_commit": "abc123",
        "model": "openai/gpt-6-luna",
    }
    assert (setup.samples, setup.workers, setup.session) == (3, 8, {})


def test_generate_command_without_a_key_is_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gen_config: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(cli, "git_commit", lambda root: "abc123")
    monkeypatch.setattr(cli, "dotenv_values", lambda path: {})
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    data = {**gen_config, "paths": {k: str(v) for k, v in gen_config["paths"].items()}}
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert main(["--config", str(config_path), "generate", "--model", "gpt-6-luna"]) == 1
    assert "OPENROUTER_API_KEY is not set" in caplog.text
    assert not (tmp_path / "generations").exists()


def test_generate_command_end_to_end(
    tmp_path: Path, fake_ollama: FakeOllama, gen_config: dict[str, Any]
) -> None:
    data = {**gen_config, "paths": {k: str(v) for k, v in gen_config["paths"].items()}}
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    argv = ["--config", str(config_path), "generate", "--model", "qwen3.5-4b", "--robustness"]
    assert main([*argv, "--limit", "2"]) == 0
    records = read_jsonl(tmp_path / "robustness" / "qwen3.5-4b.jsonl")
    assert [(r["prompt_id"], r["sample_idx"]) for r in records] == [
        (p, i) for p in ("GB_en_city_T1a", "GB_en_city_T1b") for i in range(3)
    ]
    assert [c["options"]["seed"] for c in fake_ollama.chats()] == [2026, 2027, 2028] * 2
    assert {c["options"]["temperature"] for c in fake_ollama.chats()} == {0.7}
    assert all(c["think"] is False for c in fake_ollama.chats())
    assert records[0]["response"].startswith("Call 112.")
    assert records[0]["model_version"] == DIGEST


# --- environment probes ----------------------------------------------------------------------


def test_gpu_info_parses_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = {
        "--query-gpu=name,driver_version": "NVIDIA L4, 550.54.15\n",
        "": "| NVIDIA-SMI 550.54.15  Driver Version: 550.54.15  CUDA Version: 12.4 |\n",
    }

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
        return subprocess.CompletedProcess(args, 0, outputs[args[1] if len(args) > 1 else ""], "")

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli.gpu_info() == {
        "name": "NVIDIA L4",
        "driver_version": "550.54.15",
        "cuda_version": "12.4",
    }


def test_gpu_info_without_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli.gpu_info() == {"name": None, "driver_version": None, "cuda_version": None}


def test_git_commit_marks_a_dirty_tree(tmp_path: Path) -> None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init", "-q")
    (tmp_path / "a.txt").write_text("1")
    git("add", "a.txt")
    git("commit", "-qm", "init")
    head = git("rev-parse", "HEAD")
    (tmp_path / "untracked.txt").write_text("x")
    assert cli.git_commit(tmp_path) == head
    (tmp_path / "a.txt").write_text("2")
    assert cli.git_commit(tmp_path) == head + "-dirty"


# --- retrying error records ------------------------------------------------------------------


class SequenceBackend(ScriptedBackend):
    """Gives ``outcomes`` in turn: a completion, or an exception to raise."""

    def __init__(self, *outcomes: llm.Completion | BaseException) -> None:
        super().__init__()
        self.outcomes = list(outcomes)

    def complete(self, messages: list[dict[str, str]], sample_idx: int) -> llm.Completion:
        self.calls.append((messages[-1]["content"], sample_idx))
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


OK = llm.Completion(llm.Result.OK, "Call 112.", DIGEST, None, "stop", 30, 12)
BLOCKED = llm.Completion(llm.Result.CONTENT_FILTER, "", DIGEST, None, detail="flagged")
EMPTY = llm.Completion(llm.Result.EMPTY, "", DIGEST, None, "length")
TRANSIENT = llm.TransientError("ollama /api/chat: HTTP 500: crash")


@pytest.fixture
def with_errors(tmp_path: Path, gen_config: dict[str, Any]) -> list[dict[str, Any]]:
    """A file of five records: ok, error, blocked, error, empty."""
    first = SequenceBackend(OK, TRANSIENT, BLOCKED, TRANSIENT, EMPTY)
    assert _run(gen_config, "main", 5, _setup(first), "t1") == 0
    records = read_jsonl(_out(tmp_path))
    results = ["ok", "transient", "content_filter", "transient", "empty"]
    assert [r["result"] for r in records] == results
    return records


def _retry(config: dict[str, Any], backend: ScriptedBackend) -> int:
    return cli.run_generate(
        config, "qwen3.5-4b", "main", None, _setup(backend), lambda: "t2", retry_errors=True
    )


def test_retry_replaces_error_records_in_place(
    tmp_path: Path, gen_config: dict[str, Any], with_errors: list[dict[str, Any]]
) -> None:
    retry = ScriptedBackend()
    assert _retry(gen_config, retry) == 0
    assert [sample for _, sample in retry.calls] == [0, 0]  # the two errors, nothing else
    records = read_jsonl(_out(tmp_path))
    keys = [(r["prompt_id"], r["sample_idx"]) for r in records]
    assert keys == [(r["prompt_id"], r["sample_idx"]) for r in with_errors]
    assert len(set(keys)) == len(keys)
    assert [r["result"] for r in records] == ["ok", "ok", "content_filter", "ok", "empty"]
    assert records[1]["error"] is None and records[1]["ts_utc"] == "t2"
    for i in (0, 2, 4):  # answers, blocked and empty records are untouched
        assert records[i] == with_errors[i]
    assert not list(tmp_path.glob("generations/*.tmp"))
    manifest = json.loads((tmp_path / "generations" / "qwen3.5-4b.manifest.json").read_text())
    assert manifest["sessions"][-1]["retry_errors"] is True


def test_retry_keeps_a_still_failing_record_as_an_error(
    tmp_path: Path, gen_config: dict[str, Any], with_errors: list[dict[str, Any]]
) -> None:
    assert _retry(gen_config, SequenceBackend(TRANSIENT, OK)) == 0
    records = read_jsonl(_out(tmp_path))
    assert [r["result"] for r in records] == ["ok", "transient", "content_filter", "ok", "empty"]
    assert records[1]["ts_utc"] == "t2"  # retried, failed again


def test_interrupted_retry_leaves_the_original_file(
    tmp_path: Path, gen_config: dict[str, Any], with_errors: list[dict[str, Any]]
) -> None:
    before = _out(tmp_path).read_bytes()
    with pytest.raises(Disconnect):
        _retry(gen_config, SequenceBackend(OK, Disconnect()))
    assert _out(tmp_path).read_bytes() == before


def test_interrupted_retry_keeps_its_last_checkpoint(
    tmp_path: Path, gen_config: dict[str, Any], with_errors: list[dict[str, Any]]
) -> None:
    selected = generate.select_prompts(
        read_jsonl(gen_config["paths"]["prompts"]), "qwen3.5-4b", "ollama", "main", gen_config
    )
    setup = _setup(SequenceBackend(OK, Disconnect()))
    by_id = {p["prompt_id"]: p for p in selected}
    with pytest.raises(Disconnect):
        cli.retry_error_records(_out(tmp_path), by_id, "qwen3.5-4b", setup, lambda: "t2", 1)
    records = read_jsonl(_out(tmp_path))
    assert [r["result"] for r in records] == ["ok", "ok", "content_filter", "transient", "empty"]
    assert _retry(gen_config, ScriptedBackend()) == 0  # finishes what is left
    assert [r["result"] for r in read_jsonl(_out(tmp_path))].count("transient") == 0


def test_generate_stops_after_consecutive_technical_errors(
    tmp_path: Path, gen_config: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """A dead backend (ollama serve gone, an API outage) must not turn every remaining prompt
    into an error record and exit 0."""
    backend = SequenceBackend(*[TRANSIENT] * 10)
    assert _run(gen_config, "main", 10, _setup(backend), "t") == 1
    records = read_jsonl(_out(tmp_path))
    assert len(records) == MAX_ERRORS  # tasks already answered are not written
    assert all(r["result"] == "transient" for r in records)
    assert f"{MAX_ERRORS} records in a row are technical errors" in caplog.text


def test_scattered_technical_errors_do_not_stop_generate(
    tmp_path: Path, gen_config: dict[str, Any]
) -> None:
    outcomes = [TRANSIENT] * (MAX_ERRORS - 1) + [OK]
    assert (
        _run(gen_config, "main", 3 * len(outcomes), _setup(SequenceBackend(*outcomes * 3)), "t")
        == 0
    )
    results = [r["result"] for r in read_jsonl(_out(tmp_path))]
    assert len(results) == 3 * len(outcomes)
    assert results.count("transient") == 3 * (MAX_ERRORS - 1)


def test_retry_stops_after_consecutive_technical_errors_and_keeps_what_it_got(
    tmp_path: Path, gen_config: dict[str, Any]
) -> None:
    first = SequenceBackend(*[OK, TRANSIENT] * (MAX_ERRORS + 2))
    assert _run(gen_config, "main", 2 * (MAX_ERRORS + 2), _setup(first), "t1") == 0
    retry = SequenceBackend(OK, *[TRANSIENT] * MAX_ERRORS)
    assert _retry(gen_config, retry) == 1
    records = read_jsonl(_out(tmp_path))
    assert [r["result"] for r in records] == ["ok", "ok"] + ["ok", "transient"] * (MAX_ERRORS + 1)
    # the last error record is left as it was
    assert [r["ts_utc"] for r in records if r["error"]] == ["t2"] * MAX_ERRORS + ["t1"]


class RejectingBackend(ScriptedBackend):
    """Fails the given user prompts every time, as a provider that rejects a prompt does."""

    def __init__(self, rejected: set[str]) -> None:
        super().__init__()
        self.rejected = rejected

    def complete(self, messages: list[dict[str, str]], sample_idx: int) -> llm.Completion:
        self.calls.append((messages[-1]["content"], sample_idx))
        if messages[-1]["content"] in self.rejected:
            raise TRANSIENT
        return OK


def test_errors_that_always_fail_do_not_keep_a_retry_from_the_others(
    tmp_path: Path, gen_config: dict[str, Any]
) -> None:
    """A retry that stopped on records failing again must reach the other error records next
    time, or no retry ever does."""
    first = SequenceBackend(TRANSIENT, TRANSIENT, OK, TRANSIENT, OK, TRANSIENT)
    assert _run(gen_config, "main", 6, _setup(first), "t1") == 0
    selected = generate.select_prompts(
        read_jsonl(gen_config["paths"]["prompts"]), "qwen3.5-4b", "ollama", "main", gen_config
    )
    always_failing = RejectingBackend({selected[i]["user"] for i in (0, 1, 3)})
    assert _retry(gen_config, always_failing) == 1  # stops after MAX_ERRORS in a row
    assert _retry(gen_config, always_failing) == 1
    results = [r["result"] for r in read_jsonl(_out(tmp_path))]
    assert results == ["transient", "transient", "ok", "transient", "ok", "ok"]


def _held_lock(out: Path) -> Any:  # noqa: ANN401 - an open file
    """The output's lock, held as another generate or colab process on it would hold it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    held = out.with_name(out.name + ".lock").open("a")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return held


@pytest.mark.parametrize("retry_errors", [False, True])
def test_a_second_process_on_the_same_output_is_refused(
    tmp_path: Path,
    gen_config: dict[str, Any],
    with_errors: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    retry_errors: bool,
) -> None:
    out = _out(tmp_path)
    before = out.read_bytes()
    backend = ScriptedBackend()
    with _held_lock(out):
        code = cli.run_generate(
            gen_config,
            "qwen3.5-4b",
            "main",
            None if retry_errors else 6,
            _setup(backend),
            lambda: "t2",
            retry_errors=retry_errors,
        )
    assert code == 1 and backend.calls == []
    assert out.read_bytes() == before
    assert "another process" in caplog.text
    assert _run(gen_config, "main", 6, _setup(ScriptedBackend()), "t3") == 0  # released


def test_retry_needs_no_limit(gen_config: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    setup = _setup(ScriptedBackend())
    assert cli.run_generate(gen_config, "qwen3.5-4b", "main", 5, setup, retry_errors=True) == 1
    assert "takes no --limit" in caplog.text


# --- crisisline colab: a list of local models -------------------------------------------------


@pytest.fixture
def colab_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str]]]:
    """``cli.run_colab`` replaced by a recorder; a clean HEAD."""
    calls: list[tuple[str, list[str]]] = []

    def run_colab(
        config: dict[str, Any], root: Path, model_id: str, mode: str, args: list[str], commit: str
    ) -> int:
        calls.append((model_id, list(args)))
        return 1 if model_id == "gemma4-e4b" else 0

    monkeypatch.setattr(cli, "run_colab", run_colab)
    monkeypatch.setattr(cli, "git_commit", lambda root: "a" * 40)
    return calls


def test_colab_runs_the_models_in_order_and_skips_complete_ones(
    tmp_path: Path, gen_config: dict[str, Any], colab_calls: list[tuple[str, list[str]]]
) -> None:
    done = cli.generate.select_prompts(
        cli.read_jsonl(gen_config["paths"]["prompts"]),
        "qwen3.5-2b",
        "ollama",
        "robustness",
        gen_config,
    )[:2]
    (tmp_path / "robustness").mkdir()
    (tmp_path / "robustness" / "qwen3.5-2b.jsonl").write_text(
        "".join(
            json.dumps({"prompt_id": p["prompt_id"], "model_id": "qwen3.5-2b", "sample_idx": i})
            + "\n"
            for p in done
            for i in range(gen_config["robustness"]["samples"])
        ),
        encoding="utf-8",
    )
    models = ["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-0.8b"]
    assert cli.run_colab_models(gen_config, REPO_ROOT, models, "robustness", 2, False) == 0
    assert colab_calls == [
        ("qwen3.5-4b", ["--model", "qwen3.5-4b", "--robustness", "--limit", "2"]),
        ("qwen3.5-0.8b", ["--model", "qwen3.5-0.8b", "--robustness", "--limit", "2"]),
    ]


def test_colab_stops_the_list_at_the_first_failed_model(
    gen_config: dict[str, Any], colab_calls: list[tuple[str, list[str]]]
) -> None:
    models = ["gemma4-e4b", "qwen3.5-4b"]
    assert cli.run_colab_models(gen_config, REPO_ROOT, models, "main", None, False) == 1
    assert [model for model, _ in colab_calls] == ["gemma4-e4b"]


def test_colab_refuses_a_model_another_driver_is_running(
    tmp_path: Path,
    gen_config: dict[str, Any],
    colab_calls: list[tuple[str, list[str]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _held_lock(tmp_path / "generations" / "qwen3.5-4b.jsonl"):
        assert cli.run_colab_models(gen_config, REPO_ROOT, ["qwen3.5-4b"], "main", None, False) == 1
    assert colab_calls == []
    assert "another process" in caplog.text


@pytest.mark.parametrize("bad", ["gpt-6-luna", "no-such-model"])
def test_colab_checks_every_model_before_the_first_starts(
    gen_config: dict[str, Any], colab_calls: list[tuple[str, list[str]]], bad: str
) -> None:
    assert (
        cli.run_colab_models(gen_config, REPO_ROOT, ["qwen3.5-4b", bad], "main", None, False) == 1
    )
    assert colab_calls == []


def test_colab_needs_a_clean_working_tree(
    monkeypatch: pytest.MonkeyPatch,
    gen_config: dict[str, Any],
    colab_calls: list[tuple[str, list[str]]],
) -> None:
    monkeypatch.setattr(cli, "git_commit", lambda root: "a" * 40 + "-dirty")
    assert cli.run_colab_models(gen_config, REPO_ROOT, ["qwen3.5-4b"], "main", None, False) == 1
    assert colab_calls == []


@pytest.fixture
def colab_root(tmp_path: Path, gen_config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """A repository root under ``tmp_path`` with the Colab config, and the config of its outputs."""
    (tmp_path / "configs").mkdir()
    (tmp_path / cli.COLAB_CONFIG).write_bytes((REPO_ROOT / cli.COLAB_CONFIG).read_bytes())
    return tmp_path, gen_config


@pytest.mark.parametrize(("manifest_version", "ollama_version"), [(None, ""), ("0.30.1", "0.30.1")])
def test_run_colab_bundles_head_and_resumes_on_the_manifest_ollama_version(
    monkeypatch: pytest.MonkeyPatch,
    colab_root: tuple[Path, dict[str, Any]],
    manifest_version: str | None,
    ollama_version: str,
) -> None:
    root, config = colab_root
    if manifest_version is not None:
        manifest = {"environment": {"ollama_version": manifest_version}, "sessions": []}
        (root / "generations").mkdir()
        (root / "generations" / "qwen3.5-4b.manifest.json").write_text(json.dumps(manifest))
    git: list[list[str]] = []
    runs: list[tuple[cli.colab.Colab, cli.colab.Job, Path]] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:  # noqa: ANN401
        git.append(args)
        return subprocess.CompletedProcess(args, 0, b"", b"")

    def colab_run(
        runtime: cli.colab.Colab, job: cli.colab.Job, cfg: object, at: Path, bundle: Path
    ) -> int:
        runs.append((runtime, job, bundle))
        return 0

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli.colab, "run", colab_run)
    args = ["--model", "qwen3.5-4b", "--limit", "2"]
    assert cli.run_colab(config, root, "qwen3.5-4b", "main", args, "a" * 40) == 0
    [(runtime, job, bundle)] = runs
    assert git == [["git", "bundle", "create", str(bundle), "HEAD"]]
    assert runtime.session == job.session == "crisisline-qwen3.5-4b-main"
    assert job == cli.colab.Job(
        session="crisisline-qwen3.5-4b-main",
        commit="a" * 40,
        tag="qwen3.5:4b-q8_0",
        ollama_version=ollama_version,
        generate_args=tuple(args),
        outputs=("generations/qwen3.5-4b.jsonl", "generations/qwen3.5-4b.manifest.json"),
    )


def test_run_colab_leaves_the_runtime_on_a_colab_error(
    monkeypatch: pytest.MonkeyPatch,
    colab_root: tuple[Path, dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    def colab_run(*args: object) -> int:
        raise cli.colab.ColabError("poll failed 3 times in a row")

    monkeypatch.setattr(
        cli.subprocess, "run", lambda args, **kwargs: subprocess.CompletedProcess(args, 0)
    )
    monkeypatch.setattr(cli.colab, "run", colab_run)
    root, config = colab_root
    assert cli.run_colab(config, root, "qwen3.5-4b", "main", [], "a" * 40) == 1
    assert "poll failed 3 times in a row" in caplog.text
    assert "Runtime crisisline-qwen3.5-4b-main left as is" in caplog.text
