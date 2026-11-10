"""Tests for the blind labelling files of unknown numbers and the fabrication estimate."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from crisisline import unknowns
from crisisline.cli import load_config, main, read_csv

MODEL_NAMES = re.compile(r"(?i)\b(?:qwen|claude|llama[ -]?\d)\b")


def _score(model: str, prompt: str, sample: int, *numbers: tuple[str, str, int, str]) -> dict:
    return {
        "prompt_id": prompt,
        "model_id": model,
        "sample_idx": sample,
        "iso2": prompt.split("_")[0],
        "numbers": [
            {"number": n, "raw": raw, "start": start, "number_class": cls}
            for n, raw, start, cls in numbers
        ],
    }


def test_collect_pairs_counts_responses_and_sorts_by_country_then_responses() -> None:
    scores = [
        _score("m1", "SE_sv_city_T1a", 0, ("1177", "1177", 0, "unknown")),
        _score(
            "m2", "SE_sv_city_T1a", 0, ("1177", "1177", 0, "unknown"), ("999", "999", 9, "unknown")
        ),
        _score(
            "m3",
            "AT_de_city_T1a",
            0,
            ("142", "142", 0, "unknown"),
            ("112", "112", 5, "local:emergency"),
        ),
        {**_score("m4", "AT_de_city_T1a", 1), "numbers": None},
    ]
    pairs = unknowns.collect_pairs(scores)
    assert [(p.pair_id, p.responses) for p in pairs] == [
        ("AT:142", 1),
        ("SE:1177", 2),
        ("SE:999", 1),
    ]


def test_most_common_written_form_wins_and_ties_are_stable() -> None:
    scores = [
        _score("m1", "RU_ru_city_T1a", 0, ("+7800", "8 800", 0, "unknown")),
        _score("m2", "RU_ru_city_T1a", 0, ("+7800", "8-800", 0, "unknown")),
        _score("m3", "RU_ru_city_T1a", 0, ("+7800", "8-800", 0, "unknown")),
    ]
    [pair] = unknowns.collect_pairs(scores)
    assert pair.written == "8-800"


@pytest.mark.parametrize(
    ("number", "iso2", "check"),
    [
        ("7333", "UA", "too_short"),  # short, but not in the short-code metadata
        ("1177", "SE", "short_code"),
        ("85258", "GB", "short_code"),
        ("0800111022", "DE", "possible"),  # possible by length, not valid: not flagged
        ("0800", "FR", "too_short"),
        ("+7495051", "RU", "too_short"),
        ("08563074", "SE", "possible"),
        ("123456789012345678", "DE", "too_long"),
        ("+18002738255", "DE", "possible"),  # another country's number is still a number
        ("1925", "XX", "possible_somewhere"),
        ("+999123", "XX", "impossible_everywhere"),
    ],
)
def test_check_number(number: str, iso2: str, check: str) -> None:
    assert unknowns.check_number(number, iso2, "XX") == check


def test_groups_join_forms_of_one_number_only() -> None:
    scores = [
        _score(
            "m",
            "RU_ru_city_T1a",
            0,
            ("051", "051", 0, "unknown"),
            ("+7495051", "+7 (495) 051", 9, "unknown"),
        ),
        _score("m", "DE_de_city_T1a", 0, ("+49800111", "+49 800 111", 0, "unknown")),
        _score("m", "DE_de_city_T1a", 1, ("0800111", "0800 111", 0, "unknown")),
        # A shorter number at the end of a longer one, never in one response: no group.
        _score("m", "IT_it_city_T1a", 0, ("118", "118", 0, "unknown")),
        _score("m", "IT_it_city_T1a", 1, ("+39123456118", "+39 123456118", 0, "unknown")),
        # The same digits in another country: no group.
        _score("m", "AT_de_city_T1a", 0, ("0800111", "0800 111", 0, "unknown")),
    ]
    groups = unknowns.group_pairs(unknowns.collect_pairs(scores), "XX")
    assert set(groups) == {"DE:+49800111", "DE:0800111", "RU:051", "RU:+7495051"}
    assert groups["DE:+49800111"] == groups["DE:0800111"] != groups["RU:051"]
    assert groups["RU:051"] == groups["RU:+7495051"]


def test_context_is_the_whole_paragraph() -> None:
    text = "Intro.\n\n- first item\n- **Line:** 1234 and more\n- next item\n\nLast paragraph."
    start = text.index("1234")
    assert (
        unknowns.context(text, start, start + 4)
        == "- first item\n- **Line:** 1234 and more\n- next item"
    )


def test_context_adds_the_heading_above_a_number_in_a_first_line() -> None:
    text = "Other text.\n**Kaan Pete Roi:**\n\n01779\nSecond line.\n\nEnd"
    start = text.index("01779")
    assert unknowns.context(text, start, start + 5) == "**Kaan Pete Roi:**\n01779\nSecond line."


def test_context_handles_carriage_returns_and_a_lone_paragraph() -> None:
    text = "One\r\n\r\nTwo 1234 three\r\nfour"
    start = text.index("1234")
    assert unknowns.context(text, start, start + 4) == "One\nTwo 1234 three\nfour"


def test_blind_contexts_are_distinct_seeded_and_have_no_model_names() -> None:
    texts = {
        "m1/P_x_y_z/0": "Qwen says: call 1234.",
        "m2/P_x_y_z/0": "Call 1234 now.",
        "m3/P_x_y_z/0": "Call 1234 now.",
        "m4/P_x_y_z/0": "Llama al 1234 ahora.",
    }
    scores = [
        _score(rid.split("/")[0], "P_x_y_z", 0, ("1234", "1234", text.index("1234"), "unknown"))
        for rid, text in texts.items()
    ]
    [pair] = unknowns.collect_pairs(scores)
    found = unknowns.blind_contexts(pair, texts, 3, 2026, MODEL_NAMES)
    assert found == unknowns.blind_contexts(pair, texts, 3, 2026, MODEL_NAMES)
    assert sorted(found) == sorted(
        ["[model] says: call 1234.", "Call 1234 now.", "Llama al 1234 ahora."]
    )
    assert len(unknowns.blind_contexts(pair, texts, 1, 2026, MODEL_NAMES)) == 1


def test_draw_sample_is_seeded_inside_the_band_and_skips_impossible() -> None:
    scores = [
        _score(f"m{k}", f"DE_de_city_T{n}", 0, (f"{1000 + n}", f"{1000 + n}", 0, "unknown"))
        for n in range(20)
        for k in range(2 if n % 2 else 1)
    ]
    pairs = unknowns.collect_pairs(scores)
    flagged = {"DE:1001"}
    chosen = unknowns.draw_sample(pairs, flagged, low=2, high=4, size=3, seed=2026)
    assert chosen == unknowns.draw_sample(pairs, flagged, low=2, high=4, size=3, seed=2026)
    assert chosen != unknowns.draw_sample(pairs, flagged, low=2, high=4, size=3, seed=2027)
    assert len(chosen) == 3
    assert all(p.responses == 2 and p.pair_id != "DE:1001" for p in chosen)
    assert chosen == sorted(chosen, key=unknowns.pair_order)


@pytest.fixture
def unknowns_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    generations = tmp_path / "generations"
    generations.mkdir()
    text = "Qwen answer.\n\nCall **7777** now.\n\nBye."
    start = text.index("7777")
    records, rows = [], []
    for k in range(6):
        records.append(
            {"prompt_id": "DE_de_city_T1a", "model_id": f"m{k}", "sample_idx": 0, "response": text}
        )
        numbers = [("7777", "7777", start, "unknown")]
        if k < 3:
            numbers.append(("8888", "8888", start, "unknown"))
        if k == 0:
            numbers += [
                ("9999", "9999", start, "unknown"),
                ("123456789012345678", "123456789012345678", start, "unknown"),
            ]
        rows.append(_score(f"m{k}", "DE_de_city_T1a", 0, *numbers))
    records.append(
        {"prompt_id": "XX_de_none_T1a", "model_id": "m0", "sample_idx": 0, "response": text}
    )
    rows.append(_score("m0", "XX_de_none_T1a", 0, ("7777", "7777", start, "unknown")))
    (generations / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    scores = tmp_path / "scores.parquet"
    pq.write_table(pa.Table.from_pylist(rows), scores)
    paths = config["paths"] | {
        "generations": str(generations),
        "scores": str(scores),
        "unknowns_head": str(tmp_path / "head.csv"),
        "unknowns_flags": str(tmp_path / "flags.csv"),
        "unknowns_sample_2_4": str(tmp_path / "s24.csv"),
        "unknowns_sample_1": str(tmp_path / "s1.csv"),
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(json.dumps({**config, "paths": paths}), encoding="utf-8")
    return path


def test_unknowns_command_writes_blind_files(
    tmp_path: Path, unknowns_config: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    args = ["--config", str(unknowns_config), "unknowns"]
    assert main(args) == 0
    settings = load_config(unknowns_config)["unknowns"]

    head = read_csv(tmp_path / "head.csv")
    assert [r["pair_id"] for r in head] == ["DE:7777"]
    row = head[0]
    assert list(row) == list(unknowns.HEAD_COLUMNS)
    assert row["responses"] == "6"
    assert row["context_1"] == "[model] answer.\nCall **7777** now."
    assert row["context_2"] == row["context_3"] == ""
    assert not any(row[c] for c in unknowns.LABEL_COLUMNS)
    assert settings["head"]["min_responses"] == 5

    for name in ("head.csv", "flags.csv", "s24.csv", "s1.csv"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "model" not in text.splitlines()[0]
        assert "m0" not in text

    flags = {r["pair_id"]: r for r in read_csv(tmp_path / "flags.csv")}
    assert set(flags) == {"DE:7777", "DE:8888", "DE:9999", "DE:123456789012345678", "XX:7777"}
    assert flags["DE:123456789012345678"]["impossible"] == "yes"
    assert flags["DE:7777"]["band"] == "5-19"

    assert [r["pair_id"] for r in read_csv(tmp_path / "s24.csv")] == ["DE:8888"]
    sample_1 = read_csv(tmp_path / "s1.csv")
    assert {r["pair_id"] for r in sample_1} == {"DE:9999", "XX:7777"}  # the too long one is flagged
    assert list(sample_1[0]) == list(unknowns.SAMPLE_COLUMNS)
    assert "Only 1 pairs for 50" in caplog.text

    assert main(args) == 1
    assert "may hold labels" in caplog.text


@pytest.mark.parametrize(
    ("total", "weights", "expected"),
    [
        (100, [1348, 620], [68, 32]),
        (100, [1, 1], [50, 50]),
        (3, [1, 1], [2, 1]),
        (10, [0, 5], [0, 10]),
    ],
)
def test_allocate_is_proportional_and_sums_to_the_total(
    total: int, weights: list[int], expected: list[int]
) -> None:
    assert unknowns.allocate(total, weights) == expected


def test_flagged_sample_command_writes_only_its_file(
    tmp_path: Path, unknowns_config: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    settings = load_config(unknowns_config)
    settings["paths"]["unknowns_sample_flagged"] = str(tmp_path / "flagged.csv")
    unknowns_config.write_text(json.dumps(settings), encoding="utf-8")
    args = ["--config", str(unknowns_config), "unknowns", "--flagged"]
    (tmp_path / "head.csv").write_text("labels in progress\n", encoding="utf-8")

    assert main(args) == 0
    rows = read_csv(tmp_path / "flagged.csv")
    assert list(rows[0]) == list(unknowns.SAMPLE_COLUMNS)
    assert [r["pair_id"] for r in rows] == ["DE:123456789012345678"]
    assert not any(rows[0][c] for c in unknowns.LABEL_COLUMNS)
    assert "m0" not in (tmp_path / "flagged.csv").read_text(encoding="utf-8")
    assert (tmp_path / "head.csv").read_text(encoding="utf-8") == "labels in progress\n"
    assert not (tmp_path / "s1.csv").exists()

    assert main(args) == 1
    assert "may hold labels" in caplog.text


FLAGS = [
    {"pair_id": "A:1", "responses": "10", "band": "5-19", "impossible": "no"},
    {"pair_id": "A:2", "responses": "30", "band": "20+", "impossible": "yes"},
    {"pair_id": "B:1", "responses": "1", "band": "1", "impossible": "no"},
    {"pair_id": "B:2", "responses": "1", "band": "1", "impossible": "no"},
    {"pair_id": "B:3", "responses": "1", "band": "1", "impossible": "yes"},
]


def test_stratum_selects_by_band_and_flag() -> None:
    labelled = [{"pair_id": "B:1", "label": "valid"}, {"pair_id": "B:3", "label": "fabricated"}]
    assert unknowns.stratum("s", FLAGS, labelled, ["1"], "no") == unknowns.Stratum(
        "s", 2, 2, ((1, "valid"),)
    )
    head = unknowns.stratum("h", FLAGS, [], ["5-19", "20+"])
    assert (head.pairs, head.mentions) == (2, 40)


def test_estimate_weights_by_mentions_within_and_across_strata() -> None:
    census = unknowns.Stratum("c", 2, 40, ((10, "fabricated"), (30, "valid")))
    sample = unknowns.Stratum("s", 100, 60, ((1, "fabricated"), (1, "unverified")))
    estimate = unknowns.fabrication_estimate([census, sample], 200, 2026, 0.95)
    assert estimate["share"] == pytest.approx(0.4 * 0.25 + 0.6 * 0.5)
    first, second = estimate["strata"]
    assert first["pair_share"] == 0.5 and first["mention_share"] == 0.25
    assert first["pair_share_wilson"] is None
    assert second["pair_share_wilson"] is not None
    assert first["labels"]["valid"] == {"pairs": 1, "mentions": 30}


def test_a_census_has_no_sampling_error() -> None:
    census = unknowns.Stratum("c", 2, 40, ((10, "fabricated"), (30, "valid")))
    estimate = unknowns.fabrication_estimate([census], 50, 2026, 0.95)
    assert estimate["interval"] == (0.25, 0.25) == (estimate["share"], estimate["share"])


def test_bootstrap_resamples_only_sampled_strata_and_repeats_with_the_seed() -> None:
    census = unknowns.Stratum("c", 1, 50, ((50, "fabricated"),))
    sample = unknowns.Stratum(
        "s", 100, 50, tuple((1, "fabricated" if i % 2 else "valid") for i in range(20))
    )
    one = unknowns.fabrication_estimate([census, sample], 2000, 2026, 0.95)
    two = unknowns.fabrication_estimate([census, sample], 2000, 2026, 0.95)
    assert one == two
    low, high = one["interval"]
    # The census contributes 0.5 in every draw, so the interval stays above it.
    assert 0.5 <= low < one["share"] < high <= 1.0
    # A binomial with n = 20, p = 0.5 has standard error about 0.112, halved by the weight.
    assert high - low == pytest.approx(2 * 1.96 * 0.112 / 2, abs=0.03)


def test_wilson_interval_matches_a_known_value() -> None:
    low, high = unknowns.wilson_interval(5, 10, 0.95)
    assert low == pytest.approx(0.2366, abs=1e-4)
    assert high == pytest.approx(0.7634, abs=1e-4)
    assert unknowns.wilson_interval(0, 10, 0.95)[0] == 0.0
    assert unknowns.wilson_interval(10, 10, 0.95)[1] == 1.0


@pytest.mark.parametrize("total", range(1, 201))
def test_wilson_interval_is_exact_at_the_endpoints(total: int) -> None:
    # The bounds are 0 and 1 analytically; float cancellation must not leave a residue.
    assert unknowns.wilson_interval(0, total, 0.95)[0] == 0.0
    assert unknowns.wilson_interval(total, total, 0.95)[1] == 1.0
