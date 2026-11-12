"""Extraction audit: samples, labelling rows, precision and recall."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from crisisline import audit, cli
from crisisline.cli import load_config, main, read_csv
from crisisline.extract import ExtractConfig, NumberIndex

REPO_ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ["en", "de", "es", "ar", "ja", "ka"]
PROMPT_OF = {
    "en": "DE_en_city_T1a",
    "de": "DE_de_city_T1a",
    "es": "MX_es_city_T1a",
    "ar": "EG_ar_city_T1a",
    "ja": "JP_ja_city_T1a",
    "ka": "GE_ka_city_T1a",
}


def _record(language: str, model: str, n: int, **fields: Any) -> dict[str, Any]:  # noqa: ANN401
    """A generation record; ``n`` makes the (prompt_id, sample_idx) key unique."""
    return {
        "prompt_id": PROMPT_OF[language],
        "model_id": model,
        "sample_idx": n,
        "result": "ok",
        "response": f"Call 112. ({language} {model} {n})",
        "error": None,
        **fields,
    }


# --- labelling rows ---------------------------------------------------------------------------


def test_only_responses_with_text_are_auditable() -> None:
    records = [
        _record("de", "m1", 0),
        _record("de", "m1", 1, error="HTTP 500", result="transient", response=None),
        _record("de", "m1", 2, result="content_filter", response="You can"),
        _record("de", "m1", 3, result="empty", response=""),
        _record("de", "m1", 4, response="  "),  # a local model's empty answer
    ]
    assert [audit.auditable(r) for r in records] == [True, False, False, False, False]


def test_audit_rows_carry_the_extracted_numbers(
    index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    record = _record("de", "m1", 0, response="Ruf die 0800 111 0 111 an oder im Notfall 112.")
    [row] = audit.audit_rows([record], index, extract_cfg)
    assert row == {
        "response_id": "m1/DE_de_city_T1a/0",
        "language": "de",
        "model_id": "m1",
        "prompt_id": "DE_de_city_T1a",
        "sample_idx": "0",
        "response": record["response"],
        "extracted": "+498001110111; 112",
        "extracted_raw": "0800 111 0 111; 112",
        "labelled": "",
        "missed": "",
        "false": "",
        "note": "",
    }


# --- score -----------------------------------------------------------------------------------


def _row(rid: str, language: str, extracted: str, **labels: str) -> dict[str, str]:
    return {
        "response_id": rid,
        "language": language,
        "extracted": extracted,
        "labelled": "yes",
        "missed": "",
        "false": "",
        "note": "",
        **labels,
    }


def test_score_precision_and_recall() -> None:
    rows = [
        _row("a", "de", "+498001110111; 112"),  # 2 correct
        _row("b", "de", "112; 2024", false="2024"),  # 1 correct, 1 false
        _row(
            "c", "ja", "110", missed="0120-061-338", note="full-width digits"
        ),  # 1 correct, 1 missed
        _row("d", "ja", ""),  # nothing to find
        _row("e", "ka", "112", labelled=""),  # not labelled yet
    ]
    report = audit.score(rows)
    assert (report.overall.extracted, report.overall.false, report.overall.missed) == (5, 1, 1)
    assert report.overall.precision == pytest.approx(4 / 5)
    assert report.overall.recall == pytest.approx(4 / 5)
    assert report.by_language["de"].precision == pytest.approx(3 / 4)
    assert report.by_language["de"].recall == 1
    assert report.by_language["ja"].recall == pytest.approx(1 / 2)
    assert "ka" not in report.by_language
    assert report.unlabelled == ["e"]
    assert report.findings == [
        audit.Finding("b", "de", "false", "2024", ""),
        audit.Finding("c", "ja", "missed", "0120-061-338", "full-width digits"),
    ]
    assert report.problems == []


def test_false_label_must_name_an_extracted_number() -> None:
    report = audit.score([_row("a", "de", "112", false="110")])
    assert report.problems == ["a: false ['110'] not among extracted ['112']"]
    assert report.overall.extracted == 0


def test_rates_are_undefined_without_numbers() -> None:
    report = audit.score([_row("a", "de", "")])
    assert report.overall.precision is None and report.overall.recall is None


# --- samples and commands --------------------------------------------------------------------

MODELS = tuple(f"m{i}" for i in range(11))


def _main(per_cell: int, models: tuple[str, ...] = MODELS) -> list[dict[str, Any]]:
    return [_record(lang, m, n) for lang in LANGUAGES for m in models for n in range(per_cell)]


def test_balanced_sample_gives_each_language_its_share_over_rotating_models() -> None:
    chosen = audit.balanced_sample(_main(3), per_language=5, seed=2026)
    assert Counter(audit.prompt_lang(r["prompt_id"]) for r in chosen) == dict.fromkeys(LANGUAGES, 5)
    for language in LANGUAGES:
        models = [r["model_id"] for r in chosen if audit.prompt_lang(r["prompt_id"]) == language]
        assert len(set(models)) == 5
    assert set(Counter(r["model_id"] for r in chosen).values()) == {2, 3}  # 30 over 11 models


def test_balanced_sample_is_seeded_and_skips_unauditable_records() -> None:
    errors = [_record(lang, m, 9, error="HTTP 500", response=None) for lang, m in [("de", "m0")]]
    records = _main(3) + errors
    first = audit.balanced_sample(records, per_language=5, seed=2026)
    assert first == audit.balanced_sample(list(reversed(records)), per_language=5, seed=2026)
    assert first != audit.balanced_sample(records, per_language=5, seed=7)
    assert all(not r["error"] for r in first)


def test_targeted_rows_follow_the_two_known_patterns(
    config: dict[str, Any], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    targets = cli.audit_targets(config)
    records = [
        _record("de", "m1", 0, response="Die Beratungsstelle (116 117) hilft."),
        _record("de", "m1", 1, response="Du kannst den Notruf unter 112 anrufen."),
        _record("de", "m1", 2, response="Seit 2015 gibt es das Angebot."),
        _record("de", "m1", 3, response="Du kannst den Notruf unter 110 anrufen."),  # excluded
        _record("en", "m1", 0, response="Call 365 for help."),
        _record("en", "m1", 1, response="We are open 365 days a year."),
        _record("ja", "m1", 0, response="窓口は24時間365日対応しています。"),
        _record("ja", "m1", 1, result="content_filter", response="24時間365日"),
    ]
    rows = audit.targeted(records, {"m1/DE_de_city_T1a/3"}, targets, index, extract_cfg)
    assert [(stratum, audit.response_id(r)) for stratum, r in rows] == [
        ("false_number", "m1/JP_ja_city_T1a/0"),
        ("false_number_extracted", "m1/DE_en_city_T1a/0"),
        ("short_number_call_verb", "m1/DE_de_city_T1a/1"),
        ("short_number_unextracted", "m1/DE_de_city_T1a/0"),
    ]


def test_score_counts_responses_and_groups_languages() -> None:
    rows = [
        _row("a", "de", "112; 2024", false="2024"),
        _row("b", "en", "112", missed="999"),
        _row("c", "ja", "110"),
    ]
    report = audit.score(rows)
    assert report.by_language["de"].responses == 1
    assert report.overall.responses == 3
    groups = audit.by_group(report.by_language, {"latin": ["de", "en"], "cjk": ["ja", "zh"]})
    assert list(groups) == ["latin", "cjk"]
    assert (groups["latin"].responses, groups["latin"].extracted) == (2, 3)
    assert (groups["latin"].false, groups["latin"].missed) == (1, 1)
    assert (groups["cjk"].responses, groups["cjk"].extracted) == (1, 1)


def test_pattern_counts_count_responses_per_finding() -> None:
    rows = [
        _row("a", "ja", "365; +81120061338", false="365"),
        _row("b", "ja", "365", false="365"),
        _row("c", "ja", "110"),
        _row("d", "de", "112", missed="116 117"),
        _row("e", "de", ""),
    ]
    assert audit.pattern_counts(audit.score(rows)) == {
        "de": Counter({"missed 116 117": 1}),
        "ja": Counter({"false 365": 2}),
    }


@pytest.fixture
def audit_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    generations = tmp_path / "generations"
    generations.mkdir()
    for model in MODELS:
        records = [
            _record(lang, model, n, response=f"Im Notfall 112. Antwort {n}.")
            for lang in LANGUAGES
            for n in range(3)
        ]
        if model == "m0":
            records.append(_record("ja", model, 9, response="電話 365 または 110。"))
        (generations / f"{model}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    paths = {name: str(REPO_ROOT / path) for name, path in config["paths"].items()}
    paths |= {
        "generations": str(generations),
        "main_audit": str(tmp_path / "main_audit.csv"),
        "main_audit_targeted": str(tmp_path / "main_audit_targeted.csv"),
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(json.dumps({**config, "paths": paths}), encoding="utf-8")  # JSON is YAML
    return path


def test_audit_commands_end_to_end(
    tmp_path: Path, audit_config: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    args = ["--config", str(audit_config), "audit"]
    assert main([*args, "sample"]) == 0
    out, targeted = tmp_path / "main_audit.csv", tmp_path / "main_audit_targeted.csv"
    rows, target_rows = read_csv(out), read_csv(targeted)
    assert list(rows[0]) == list(target_rows[0]) == list(audit.COLUMNS)
    assert Counter(r["language"] for r in rows) == dict.fromkeys(LANGUAGES, 5)
    assert [r["response_id"] for r in target_rows] == ["m0/JP_ja_city_T1a/9"]

    assert main([*args, "sample"]) == 1
    assert "may hold labels" in caplog.text

    for row in rows:
        row["labelled"] = "yes"
    rows[0]["missed"] = "0800 111"
    cli.write_csv(out, rows, audit.COLUMNS)
    target_rows[0] |= {"labelled": "yes", "false": "365"}
    cli.write_csv(targeted, target_rows, audit.COLUMNS)
    caplog.clear()
    assert main([*args, "score"]) == 0
    assert "all      precision 1.000, recall 0.968 (30 extracted, 0 false, 1 missed)" in caplog.text
    assert "latin    precision" in caplog.text
    assert "cyrillic precision n/a" in caplog.text
    assert "Criterion precision and recall >= 0.95: met" in caplog.text
    assert "Targeted ja   1 responses: false 365 1" in caplog.text


def test_audit_score_fails_on_unmatched_labels(
    tmp_path: Path, audit_config: Path, caplog: pytest.LogCaptureFixture
) -> None:
    args = ["--config", str(audit_config), "audit"]
    assert main([*args, "sample"]) == 0
    out = tmp_path / "main_audit.csv"
    rows = read_csv(out)
    rows[0] |= {"labelled": "yes", "false": "999"}
    cli.write_csv(out, rows, audit.COLUMNS)
    assert main([*args, "score"]) == 1
    assert "false ['999'] not among extracted" in caplog.text


# --- changed responses -------------------------------------------------------------------------


def _changed_records() -> list[dict[str, Any]]:
    return [_record(lang, "m1", n) for lang in LANGUAGES for n in range(10)]


def test_changed_responses_are_auditable_changed_and_not_excluded() -> None:
    records = _changed_records()
    ids = [audit.response_id(r) for r in records]
    before = dict.fromkeys(ids, frozenset({"112"}))
    after = before | dict.fromkeys(ids[:5], frozenset({"911"}))
    records[1] |= {"result": "content_filter"}
    found = audit.changed_responses(records, before, after, {ids[2]})
    assert [audit.response_id(r) for r in found] == sorted([ids[0], ids[3], ids[4]])


def test_changed_sample_is_seeded_and_in_row_order() -> None:
    population = _changed_records()
    first = audit.changed_sample(population, 10, 2026)
    assert len(first) == 10
    assert first == audit.changed_sample(population, 10, 2026)
    assert first != audit.changed_sample(population, 10, 2027)
    assert first == sorted(first, key=audit.row_order)


def test_changed_sample_takes_what_there_is() -> None:
    population = _changed_records()[:3]
    assert len(audit.changed_sample(population, 10, 2026)) == 3


def test_sample_changed_command(
    tmp_path: Path, audit_config: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    settings = load_config(audit_config)
    records = cli.read_records(Path(settings["paths"]["generations"]))
    prefix, scores = tmp_path / "scores_prefix.parquet", tmp_path / "scores.parquet"
    out = tmp_path / "main_audit_changed.csv"
    labelled = tmp_path / "main_audit.csv"
    targeted = tmp_path / "main_audit_targeted.csv"
    changed = records[:3]
    cli.write_csv(
        labelled,
        [{"response_id": audit.response_id(changed[0])}],
        ["response_id"],
    )
    cli.write_csv(targeted, [{"response_id": "none"}], ["response_id"])

    def table(numbers: list[list[str]]) -> pa.Table:
        return pa.Table.from_pylist(
            [
                {
                    "prompt_id": r["prompt_id"],
                    "model_id": r["model_id"],
                    "sample_idx": r["sample_idx"],
                    "numbers": [{"number": n} for n in found],
                }
                for r, found in zip(records, numbers, strict=True)
            ]
        )

    pq.write_table(table([["112"] for _ in records]), prefix)
    pq.write_table(
        table([["911"] if r in changed else ["112"] for r in records]),
        scores,
    )
    settings["paths"] |= {
        "scores": str(scores),
        "scores_prefix": str(prefix),
        "main_audit_changed": str(out),
    }
    settings["audit"]["changed"] = {"size": 80, "seed": 2026}
    audit_config.write_text(json.dumps(settings), encoding="utf-8")
    args = ["--config", str(audit_config), "audit", "sample", "--changed"]

    assert main(args) == 0
    rows = read_csv(out)
    assert list(rows[0]) == list(audit.COLUMNS)
    assert {r["response_id"] for r in rows} == {audit.response_id(r) for r in changed[1:]}
    assert "Only 2 responses for 80" in caplog.text

    assert main(args) == 1
    assert "may hold labels" in caplog.text


def test_score_changed_command(
    tmp_path: Path, audit_config: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    out = tmp_path / "main_audit_changed.csv"
    settings = load_config(audit_config)
    settings["paths"]["main_audit_changed"] = str(out)
    audit_config.write_text(json.dumps(settings), encoding="utf-8")
    args = ["--config", str(audit_config), "audit", "score", "--changed"]
    rows = [
        _row("a", "de", "112; 2024", false="2024"),
        _row("b", "ja", "110", missed="119"),
    ]
    cli.write_csv(out, rows, audit.COLUMNS)

    assert main(args) == 1
    assert "false 2024 [de] a" in caplog.text
    assert "missed 119 [ja] b" in caplog.text
    assert "latin    precision 0.500, recall 1.000 (2 extracted, 1 false, 0 missed)" in caplog.text
    assert "cjk      precision 1.000, recall 0.500 (1 extracted, 0 false, 1 missed)" in caplog.text
    assert "all      precision 0.667, recall 0.667 (3 extracted, 1 false, 1 missed)" in caplog.text
    assert (
        "Extraction criterion not met: precision 0.667 and recall 0.667 must both reach 0.95"
        in caplog.text
    )

    cli.write_csv(out, [_row("a", "de", "112", false="911")], audit.COLUMNS)
    assert main(args) == 1
