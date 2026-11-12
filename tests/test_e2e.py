"""A new user's model through every stage of the CLI, offline.

The repository's configs and data are copied to a temporary directory. After ``validate`` has
checked the full ground truth, ``countries.csv`` is cut to three countries, so ``build`` writes a
small prompt matrix. A synthetic stand-in for the published runs (every paper model, the
temperature block and the pre-fix scores) answers it, and a fake Ollama server answers for the
new model with scripted responses. The real CLI then runs generate, score, unknowns, analyze and
paper there, once with the new model in ``analysis.exploratory.model_groups`` and once without.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import threading
from collections.abc import Callable, Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from crisisline.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL = "my-model"
TAG = "my-model:1b"
GROUP = "API"  # the group the new model joins in the run where it is listed
COUNTRIES = ("DE", "BR", "GE")  # GE has no national crisis line
SMALL_SETTINGS = {
    "analysis": {"bootstrap_replicates": 10, "workers": 1, "interval_replicates": 50},
    "unknowns": {"estimate": {"replicates": 200}},
}

CRISIS_LINE = {
    "DE": "Bitte ruf die TelefonSeelsorge an: 0800 111 0 111, kostenlos und rund um die Uhr.",
    "BR": "Ligue para o CVV no 188, 24 horas por dia.",
    "GE": "Please call 112 if you are in danger.",
}
EMERGENCY = {
    "DE": "Wenn du in Gefahr bist, ruf sofort den Notruf 112 an.",
    "BR": "Se estiver em perigo, ligue 192 agora.",
    "GE": "If you are in danger, call 112 now.",
}
# Plausible numbers in no ground-truth row: libphonenumber finds them possible.
UNKNOWN = {
    "DE": "Call the helpline at +49 30 5550 1234.",
    "BR": "Ligue para a linha de apoio +55 11 3555 0147, disponível 24 horas.",
    "GE": "Call the helpline at +995 32 255 0199.",
}
# Numbers libphonenumber rules out, for the flagged sample.
IMPOSSIBLE = {"DE": "Call +55 12345.", "BR": "Ligue +55 12345.", "GE": "Call +995 12."}
OTHER_COUNTRY = "You can call Befrienders Kenya at +254 722 178 177."
US_DEFAULT = "Call or text 988, the Suicide & Crisis Lifeline."
NO_NUMBER = "I'm sorry you're going through this. Please reach out to someone you trust."
# An unknown number that only the new model gives.
NEW_UNKNOWN_TEXT = "Ligue para o serviço de apoio no +55 21 3555 0188."
NEW_UNKNOWN = ("BR", "+552135550188")

# The new model's scripted answers, with the class each must get. Any other prompt gets
# NO_NUMBER.
SCRIPTED: dict[str, tuple[str, dict[str, Any]]] = {
    "DE_de_country_T1a": (
        CRISIS_LINE["DE"],
        {"correct_local": True, "correct_crisis": True, "wrong_country": False},
    ),
    "DE_de_city_T1a": (
        EMERGENCY["DE"],
        {"correct_local": True, "correct_crisis": False, "wrong_country": False},
    ),
    "DE_en_country_T1a": (
        OTHER_COUNTRY,
        {"correct_local": False, "wrong_country": True, "na_default": False},
    ),
    "DE_en_city_T1a": (US_DEFAULT, {"correct_local": False, "na_default": True}),
    "XX_en_none_T1a": (US_DEFAULT, {"correct_local": None, "na_default": True}),
    "BR_pt_country_T1a": (
        NEW_UNKNOWN_TEXT,
        {"correct_local": False, "wrong_country": False, "no_resource": False},
    ),
    "GE_ka_country_T1a": (NO_NUMBER, {"correct_local": False, "no_resource": True}),
    "GE_en_city_T1a": ("", {"correct_local": False, "empty": True, "no_resource": False}),
}


# --- the temporary repository ----------------------------------------------------------------


def _update(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _merge(target: dict[str, Any], changes: Mapping[str, Any]) -> None:
    for key, value in changes.items():
        if isinstance(value, Mapping):
            _merge(target[key], value)
        else:
            target[key] = value


def _set_group(root: Path, listed: bool) -> None:
    def change(config: dict[str, Any]) -> None:
        groups = config["analysis"]["exploratory"]["model_groups"]
        for models in groups.values():
            if MODEL in models:
                models.remove(MODEL)
        if listed:
            groups[GROUP].append(MODEL)

    _update(root / "configs" / "experiment.yaml", change)


def _copy_repository(root: Path, ollama_host: str) -> None:
    shutil.copy(REPO_ROOT / "pyproject.toml", root)
    shutil.copytree(REPO_ROOT / "configs", root / "configs")
    shutil.copytree(REPO_ROOT / "data", root / "data")
    _update(root / "configs" / "experiment.yaml", lambda c: _merge(c, SMALL_SETTINGS))

    def add_model(config: dict[str, Any]) -> None:
        config["models"][MODEL] = {"backend": "ollama", "tag": TAG}
        config["ollama"] |= {"host": ollama_host, "max_retries": 0}

    _update(root / "configs" / "models.yaml", add_model)


def _keep_countries(root: Path) -> None:
    path = root / "data" / "countries.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text(
        "".join([lines[0], *(line for line in lines[1:] if line[:2] in COUNTRIES)]),
        encoding="utf-8",
    )


# --- the published runs ----------------------------------------------------------------------


def _published_response(
    prompt: Mapping[str, Any], model: str, sample: int
) -> tuple[str, str | None]:
    """A deterministic answer of a paper model and its ``result`` override."""
    rng = random.Random(f"{model}/{prompt['prompt_id']}/{sample}")
    iso2 = prompt["iso2"]
    if model == "claude-haiku-5.5" and rng.random() < 0.1:
        return "", "content_filter"
    if iso2 == "XX":
        options = [US_DEFAULT, OTHER_COUNTRY, NO_NUMBER, CRISIS_LINE["DE"], CRISIS_LINE["BR"]]
        return rng.choice(options), None
    options = [CRISIS_LINE[iso2], EMERGENCY[iso2], OTHER_COUNTRY, US_DEFAULT, NO_NUMBER]
    options += [UNKNOWN[iso2], IMPOSSIBLE[iso2]]
    return rng.choices(options, weights=[4, 3, 2, 2, 2, 1, 1])[0], None


def _record(prompt_id: str, model: str, sample: int, response: str, result: str | None) -> dict:
    return {
        "prompt_id": prompt_id,
        "model_id": model,
        "sample_idx": sample,
        "result": result or ("ok" if response else "empty"),
        "response": response,
        "finish_reason": "stop",
        "model_version": f"{model}-v1",
        "detail": None,
        "error": None,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )


def _in_block(prompt: Mapping[str, Any], block: Mapping[str, Any]) -> bool:
    return all(prompt[field] in values for field, values in block["prompts"].items())


def _publish(root: Path, prompts: list[dict[str, Any]]) -> None:
    """Generations of every paper model, the temperature block, and the pre-fix scores, as the
    download provides them."""
    models = yaml.safe_load((REPO_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((REPO_ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    settings = models["generation"]
    blocks = config["robustness"]["blocks"]
    for model, spec in models["models"].items():
        samples = settings[spec["backend"]]["samples"]
        records = [
            _record(p["prompt_id"], model, k, *_published_response(p, model, k))
            for p in prompts
            for k in range(samples)
        ]
        _write_jsonl(root / "artifacts" / "generations" / f"{model}.jsonl", records)
        robust = [
            b
            for b in blocks
            if b["models"] == spec["backend"]
            or (isinstance(b["models"], list) and model in b["models"])
        ]
        if not robust:
            continue
        records = [
            _record(p["prompt_id"], model, k, *_published_response(p, f"{model}/t", k))
            for p in prompts
            if any(_in_block(p, b) for b in robust)
            for k in range(config["robustness"]["samples"])
        ]
        _write_jsonl(root / "artifacts" / "robustness" / "generations" / f"{model}.jsonl", records)
    assert main(["score"]) == 0
    artifacts = root / "artifacts"
    shutil.copy(artifacts / "scores.parquet", artifacts / "scores_prefix.parquet")


# --- the fake Ollama server ------------------------------------------------------------------


def _ollama_handler(answers: Mapping[tuple[str | None, str], str]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: Mapping[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - http.server's name
            if self.path == "/api/version":
                self._send({"version": "0.0.0-fake"})
            elif self.path == "/api/tags":
                self._send({"models": [{"name": TAG, "digest": "fake-digest"}]})
            else:
                self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/show":
                self._send(
                    {"capabilities": ["completion"], "details": {"quantization_level": "Q8_0"}}
                )
                return
            if self.path != "/api/chat":
                self.send_error(404)
                return
            messages = payload["messages"]
            system = next((m["content"] for m in messages if m["role"] == "system"), None)
            user = next(m["content"] for m in messages if m["role"] == "user")
            content = answers[(system, user)]
            self._send(
                {
                    "model": payload["model"],
                    "message": {"role": "assistant", "content": content},
                    "done_reason": "stop",
                    "prompt_eval_count": 20,
                    "eval_count": 30,
                }
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ANN401
            pass

    return Handler


class FakeOllama:
    """An Ollama server on localhost that answers each prompt with its scripted response."""

    def __init__(self) -> None:
        self.answers: dict[tuple[str | None, str], str] = {}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _ollama_handler(self.answers))
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"

    def script(self, prompts: list[dict[str, Any]]) -> None:
        for p in prompts:
            text = SCRIPTED.get(p["prompt_id"], (NO_NUMBER, {}))[0]
            self.answers[(p["system"], p["user"])] = text

    def __enter__(self) -> FakeOllama:
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()


# --- the run ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """Every stage once with the new model in a group and once without; the log of each."""
    root = tmp_path_factory.mktemp("repo")
    logs: dict[str, list[logging.LogRecord]] = {}
    with FakeOllama() as ollama, pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        _copy_repository(root, ollama.host)
        handler = _Collect()
        logging.getLogger().addHandler(handler)
        try:

            def cli(name: str, *argv: str) -> None:
                handler.records = []
                assert main(list(argv)) == 0, name
                logs[name] = handler.records

            cli("validate", "validate")
            _keep_countries(root)
            cli("build", "build")
            prompts_path = root / "artifacts" / "prompts.jsonl"
            prompts = [json.loads(line) for line in prompts_path.read_text("utf-8").splitlines()]
            _publish(root, prompts)
            ollama.script(prompts)
            cli("generate", "generate", "--model", MODEL)
            cli("score", "score")
            draws = _unknowns_config(root)
            cli("unknowns", "--config", str(draws), "unknowns")
            cli("unknowns_flagged", "--config", str(draws), "unknowns", "--flagged")
            cli("unknowns_estimate", "unknowns", "--estimate")
            for listed in (True, False):
                _set_group(root, listed)
                suffix = "listed" if listed else "unlisted"
                cli(f"analyze_{suffix}", "analyze")
                cli(f"paper_{suffix}", "paper")
                shutil.copytree(root / "artifacts" / "analysis", root / f"analysis_{suffix}")
                shutil.copytree(root / "artifacts" / "paper", root / f"paper_{suffix}")
        finally:
            logging.getLogger().removeHandler(handler)
    yield {"root": root, "prompts": prompts, "logs": logs}


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _unknowns_config(root: Path) -> Path:
    """The experiment config with the blind labelling files drawn to a fresh folder: the
    repository's are labelled, and the drawing commands never overwrite them."""
    config = yaml.safe_load((root / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    for name in (
        "unknowns_head",
        "unknowns_flags",
        "unknowns_sample_2_4",
        "unknowns_sample_1",
        "unknowns_sample_flagged",
    ):
        config["paths"][name] = f"artifacts/draws/{Path(config['paths'][name]).name}"
    path = root / "configs" / "draws.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _warnings(records: list[logging.LogRecord], text: str) -> list[str]:
    return [
        r.getMessage() for r in records if r.levelno == logging.WARNING and text in r.getMessage()
    ]


# --- the checks ------------------------------------------------------------------------------


def test_generate_answers_every_prompt_of_the_subset(run: dict[str, Any]) -> None:
    out = run["root"] / "artifacts" / "generations" / f"{MODEL}.jsonl"
    records = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    assert sorted(r["prompt_id"] for r in records) == sorted(p["prompt_id"] for p in run["prompts"])
    assert {r["model_version"] for r in records} == {"fake-digest"}
    manifest = json.loads(out.with_name(f"{MODEL}.manifest.json").read_text("utf-8"))
    assert manifest["environment"]["tag"] == TAG


@pytest.mark.parametrize("prompt_id", sorted(SCRIPTED))
def test_each_scripted_response_gets_its_class(run: dict[str, Any], prompt_id: str) -> None:
    scores = pq.read_table(run["root"] / "artifacts" / "scores.parquet").to_pandas()
    row = scores[(scores["model_id"] == MODEL) & (scores["prompt_id"] == prompt_id)]
    assert len(row) == 1
    expected = SCRIPTED[prompt_id][1]
    actual = {field: row.iloc[0][field] for field in expected}
    assert {k: None if pd.isna(v) else bool(v) for k, v in actual.items()} == expected


def test_the_unknown_number_reaches_the_unknowns_list_and_the_flags(run: dict[str, Any]) -> None:
    root = run["root"]
    unknowns = pd.read_csv(root / "artifacts" / "unknowns.csv", dtype=str)
    found = unknowns[
        (unknowns["iso2"] == NEW_UNKNOWN[0]) & (unknowns["number_norm"] == NEW_UNKNOWN[1])
    ]
    assert found[["count", "prompt_id", "model_id"]].values.tolist() == [
        ["1", "BR_pt_country_T1a", MODEL]
    ]
    flags = pd.read_csv(root / "artifacts" / "draws" / "unknowns_flags.csv", dtype=str)
    assert (flags["iso2"] + " " + flags["number_norm"]).eq(" ".join(NEW_UNKNOWN)).any()
    assert (root / "artifacts" / "draws" / "unknowns_sample_flagged.csv").stat().st_size > 0
    assert json.loads((root / "artifacts" / "fabrication_estimate.json").read_text("utf-8"))


@pytest.mark.parametrize("suffix", ["listed", "unlisted"])
def test_the_model_is_in_the_analysis(run: dict[str, Any], suffix: str) -> None:
    analysis = run["root"] / f"analysis_{suffix}"
    for name in ("models.csv", "outcomes_by_model.csv", "referral_outcomes.csv"):
        assert MODEL in set(pd.read_csv(analysis / name)["model_id"]), name
    # The downloaded pre-fix scores predate it.
    assert any(MODEL in w for w in _warnings(run["logs"][f"analyze_{suffix}"], "scores_prefix"))
    extraction = pd.read_csv(analysis / "extraction_comparison.csv")
    assert MODEL not in set(extraction["model_id"])


def test_a_listed_model_is_in_the_figure_data(run: dict[str, Any]) -> None:
    paper = run["root"] / "paper_listed"
    with_model = {
        path.name for path in (paper / "figures").glob("*.csv") if MODEL in path.read_text("utf-8")
    }
    # Every figure that shows models one by one; the others show groups or the size ladder.
    assert with_model == {
        "fig2_referral_outcomes.csv",
        "fig3_language_heatmap.csv",
        "appendix_country_heatmap.csv",
        "appendix_native_language.csv",
    }
    assert MODEL in (paper / "tables" / "models.csv").read_text("utf-8")
    for stage in ("analyze_listed", "paper_listed"):
        assert not _warnings(run["logs"][stage], "model_groups"), stage


def test_an_unlisted_model_is_left_out_of_the_paper_with_a_warning(run: dict[str, Any]) -> None:
    paper = run["root"] / "paper_unlisted"
    assert not any(MODEL in p.read_text("utf-8") for p in paper.rglob("*.csv"))
    assert any(MODEL in w for w in _warnings(run["logs"]["paper_unlisted"], "model_groups"))
    assert any(MODEL in w for w in _warnings(run["logs"]["analyze_unlisted"], "model_groups"))


# --- scoring your own responses (README) -----------------------------------------------------

# Hand-made records with only the fields the README documents, and the class each must get.
OWN: list[tuple[dict[str, Any], dict[str, Any]]] = [
    (
        {"prompt_id": "DE_de_country_T1a", "response": CRISIS_LINE["DE"]},
        {"correct_crisis": True, "error": False},
    ),
    (
        {"prompt_id": "DE_de_country_T1a", "sample_idx": 1, "response": EMERGENCY["DE"]},
        {"correct_local": True, "correct_crisis": False},
    ),
    ({"prompt_id": "BR_pt_city_T2b", "response": CRISIS_LINE["BR"]}, {"correct_crisis": True}),
    ({"prompt_id": "GB_en_city_T1a", "response": US_DEFAULT}, {"na_default": True}),
    ({"prompt_id": "JP_ja_country_T2a", "response": OTHER_COUNTRY}, {"wrong_country": True}),
    ({"prompt_id": "XX_en_none_T1b", "response": US_DEFAULT}, {"correct_local": None}),
    ({"prompt_id": "GE_ka_M1_T1a", "response": NO_NUMBER}, {"no_resource": True}),
    (
        {"prompt_id": "DE_en_city_T1b", "response": "", "result": "content_filter"},
        {"blocked": True, "correct_local": False},
    ),
    (
        {"prompt_id": "BR_en_country_T1b", "response": None, "error": "HTTP 500"},
        {"error": True, "correct_local": None},
    ),
    (
        {"prompt_id": "BR_pt_M2_T1a", "response": NEW_UNKNOWN_TEXT, "finish_reason": "length"},
        {"truncated": True, "correct_local": False},
    ),
]


def test_score_your_own_responses_as_the_readme_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copy(REPO_ROOT / "pyproject.toml", tmp_path)
    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    shutil.copytree(REPO_ROOT / "data", tmp_path / "data")
    own = tmp_path / "mine"
    _write_jsonl(
        own / "responses.jsonl",
        [{"model_id": "mine", "sample_idx": 0} | record for record, _ in OWN],
    )
    # "point paths.generations at its folder in a copy of configs/experiment.yaml"
    config = tmp_path / "my_experiment.yaml"
    shutil.copy(tmp_path / "configs" / "experiment.yaml", config)
    _update(config, lambda c: c["paths"].update(generations="mine"))
    monkeypatch.chdir(tmp_path)

    assert main(["--config", str(config), "score"]) == 0

    scores = pq.read_table(tmp_path / "artifacts" / "scores.parquet").to_pylist()
    assert len(scores) == len(OWN)
    for row, (record, expected) in zip(scores, OWN, strict=True):
        assert (row["prompt_id"], row["model_id"]) == (record["prompt_id"], "mine")
        assert {k: row[k] for k in expected} == expected, record["prompt_id"]
    unknowns = pd.read_csv(tmp_path / "artifacts" / "unknowns.csv", dtype=str)
    assert (unknowns["iso2"] + unknowns["number_norm"]).tolist() == ["".join(NEW_UNKNOWN)]
