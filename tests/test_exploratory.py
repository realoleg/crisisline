"""Tests of the exploratory analyses of ``crisisline analyze`` on synthetic data."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from crisisline.analysis import (
    breakdowns,
    exploratory,
    fabrication,
    frames,
    intervals,
    language_gap,
    near_miss,
)
from crisisline.score import FABRICATED, NOT_A_NUMBER

TEMPLATES = ("T1a", "T1b", "T2a", "T2b")


def _response(iso2: str, model: str, prompt: str, **values: object) -> dict[str, Any]:
    template = values.pop("template", "T1a")
    return {
        "prompt_id": prompt,
        "model_id": model,
        "sample_idx": values.pop("sample_idx", 0),
        "iso2": iso2,
        "template": template,
        "outcome": values.pop("outcome", 0.0),
        **values,
    }


# Intervals.


def test_share_averages_responses_within_a_prompt_and_resamples_countries() -> None:
    rows = [
        _response("A", "api", "A_1", sample_idx=0, outcome=1.0),
        _response("A", "api", "A_1", sample_idx=1, outcome=0.0),
        _response("A", "api", "A_1", sample_idx=2, outcome=0.0),
        _response("B", "api", "B_1", outcome=1.0),
        _response("B", "local", "B_1", outcome=np.nan),
    ]
    result = intervals.share(pd.DataFrame(rows), "outcome", 500, 1, ("k",))
    assert result is not None
    assert result["estimate"] == pytest.approx((1 / 3 + 1) / 2)
    assert result["cluster"] == "iso2"
    assert result["units"] == 2
    assert result["responses"] == 4
    again = intervals.share(pd.DataFrame(rows), "outcome", 500, 1, ("k",))
    assert again == result


def test_share_inside_one_country_resamples_prompts() -> None:
    rows = [_response("A", "m", f"A_{k}", outcome=float(k % 2)) for k in range(10)]
    result = intervals.share(pd.DataFrame(rows), "outcome", 500, 1, ("k",))
    assert result is not None
    assert result["cluster"] == "prompt_id"
    assert result["ci_low"] < 0.5 < result["ci_high"]


def test_share_of_an_outcome_never_defined_is_none() -> None:
    rows = [_response("A", "m", "A_1", outcome=np.nan)]
    assert intervals.share(pd.DataFrame(rows), "outcome", 10, 1, ("k",)) is None


def test_difference_draws_the_same_countries_for_both_arms() -> None:
    rng = np.random.default_rng(0)
    rows = [
        _response(f"C{c}", "m", f"C{c}_{k}", outcome=float(rng.random() < c / 10))
        for c in range(10)
        for k in range(4)
    ]
    arm = pd.DataFrame(rows)
    result = intervals.difference(arm, arm.copy(), "outcome", 300, 1, ("k",))
    assert result is not None
    assert (result["estimate"], result["ci_low"], result["ci_high"]) == (0.0, 0.0, 0.0)


def test_difference_inside_one_country_pairs_on_templates() -> None:
    a = pd.DataFrame([_response("A", "m", f"A_en_{t}", template=t, outcome=1.0) for t in TEMPLATES])
    b = pd.DataFrame([_response("A", "m", f"A_lo_{t}", template=t, outcome=0.0) for t in TEMPLATES])
    result = intervals.difference(a, b, "outcome", 100, 1, ("k",))
    assert result is not None
    assert result["cluster"] == "template"
    assert result["estimate"] == pytest.approx(1.0)


def test_scopes_give_each_model_then_groups_then_all() -> None:
    frame = pd.DataFrame({"model_id": ["a", "b", "c"]})
    groups = {"G1": ["a", "b"], "G2": ["c"]}
    found = [(g, m, len(part)) for g, m, part in intervals.scopes(frame, groups)]
    assert found == [
        ("G1", "a", 1),
        ("G1", "b", 1),
        ("G2", "c", 1),
        ("G1", "all", 2),
        ("G2", "all", 1),
        ("all", "all", 3),
    ]


def test_a_model_outside_the_groups_gets_its_own_group_and_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    listed = {"G1": ["a", "b"]}
    with caplog.at_level("WARNING", logger=exploratory.__name__):
        groups = exploratory.model_groups(listed, ["new-2", "a", "new-1", "b", "new-1"])
    assert groups == {"G1": ["a", "b"], "new-1": ["new-1"], "new-2": ["new-2"]}
    assert listed == {"G1": ["a", "b"]}
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert "new-1, new-2" in caplog.records[0].getMessage()
    frame = pd.DataFrame({"model_id": ["a", "new-1"]})
    found = [(g, m, len(part)) for g, m, part in intervals.scopes(frame, groups)]
    assert ("new-1", "new-1", 1) in found


def test_listed_models_keep_the_groups_as_they_are(caplog: pytest.LogCaptureFixture) -> None:
    listed = {"G1": ["a", "b"], "G2": ["c"]}
    with caplog.at_level("WARNING", logger=exploratory.__name__):
        assert exploratory.model_groups(listed, ["c", "a"]) is listed
    assert not caplog.records


# Breakdowns.


def _analysis_row(
    iso2: str, model: str, condition: str, cue: str, mode: str, **values: object
) -> dict[str, Any]:
    template = values.pop("template", "T1a")
    return {
        "prompt_id": f"{iso2}_{mode}_{condition}_{cue}_{template}",
        "model_id": model,
        "sample_idx": values.pop("sample_idx", 0),
        "iso2": iso2,
        "lang": "en" if mode == "en" else "xx",
        "cue": cue,
        "template": template,
        "type": "disclosure" if template.startswith("T2") else "direct",
        "condition": condition,
        "lang_mode": mode,
        "english_country": False,
        "has_crisis_line": True,
        "has_numbers": True,
        "correct_local": 0.0,
        "correct_crisis": 0.0,
        "wrong_country": 0.0,
        "fabricated": 0.0,
        "mentions_directory": 0.0,
        "no_resource": 0.0,
        "blocked": 0.0,
        "empty": 0.0,
        "loop": 0.0,
        **values,
    }


def test_setting_names_s0_by_cue() -> None:
    frame = pd.DataFrame(
        {
            "condition": ["S0", "S0", "S0", "M1", "M2"],
            "cue": ["city", "country", "none"] + ["none"] * 2,
        }
    )
    assert breakdowns.setting(frame).tolist() == ["S0-city", "S0-country", "S0-none", "M1", "M2"]


def test_grounding_steps_are_differences_of_the_settings() -> None:
    rows = []
    for c, (condition, cue, hit) in itertools.product(
        range(6), (("S0", "country", 0.2), ("M1", "none", 0.4), ("M2", "none", 0.9))
    ):
        for k, t in enumerate(TEMPLATES):
            value = float(k < round(hit * 4))
            rows.append(
                _analysis_row(
                    f"C{c}", "m", condition, cue, "local", template=t, correct_local=value
                )
            )
    table = breakdowns.grounding(
        frames.add_derived(pd.DataFrame(rows)), {"G": ["m"]}, 200, 1
    ).set_index(["model_id", "measure", "outcome"])
    level = table["estimate"]
    for step, a, b in breakdowns.GROUNDING_STEPS:
        expected = level[("m", a, "correct_local")] - level[("m", b, "correct_local")]
        assert level[("m", step, "correct_local")] == pytest.approx(expected)
    assert level[("m", "M2 - M1", "correct_local")] == pytest.approx(1.0 - 0.5)


def _contrast(template: str, estimate: float, low: float, high: float) -> dict[str, Any]:
    return {
        "template": template,
        "model_group": "all",
        "contrast": "x",
        "outcome": "o",
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
    }


def test_a_result_that_holds_in_one_template_is_flagged() -> None:
    table = pd.DataFrame(
        [
            _contrast("T1a", 0.1, 0.05, 0.2),
            *(_contrast(t, 0.02, -0.01, 0.05) for t in TEMPLATES[1:]),
            _contrast("all", 0.04, 0.01, 0.07),
        ]
    )
    flagged = breakdowns.flag_single_template(table).iloc[0]
    assert flagged["flag"] == "holds only in T1a"
    assert flagged["templates_excluding_zero"] == 1
    assert bool(flagged["pooled_excludes_zero"])


def test_templates_that_disagree_in_sign_are_flagged_and_consistent_ones_are_not() -> None:
    opposite = pd.DataFrame(
        [
            _contrast("T1a", 0.1, 0.05, 0.2),
            _contrast("T1b", -0.1, -0.2, -0.05),
            *(_contrast(t, 0.0, -0.1, 0.1) for t in TEMPLATES[2:]),
            _contrast("all", 0.0, -0.1, 0.1),
        ]
    )
    assert breakdowns.flag_single_template(opposite)["flag"].iloc[0] == "templates disagree in sign"
    consistent = pd.DataFrame(
        [_contrast(t, 0.1, 0.05, 0.2) for t in TEMPLATES] + [_contrast("all", 0.1, 0.05, 0.2)]
    )
    assert breakdowns.flag_single_template(consistent)["flag"].iloc[0] == ""


# Language gap.


def _gap_frame(seed: int, strengths: dict[str, float], slope: float) -> pd.DataFrame:
    """Country cue, local and English prompts; the log odds ratio of English over local is
    ``1 + slope * (strength - 0.5)``."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(40):
        shift = rng.normal(0, 0.2)
        for model, strength in strengths.items():
            for t, mode, sample in itertools.product(TEMPLATES, ("local", "en"), range(6)):
                gap = 1.0 + slope * (strength - 0.5)
                eta = -0.5 + shift + 2 * strength + gap * (mode == "en")
                hit = float(rng.random() < 1 / (1 + math.exp(-eta)))
                rows.append(
                    _analysis_row(
                        f"C{c:02d}",
                        model,
                        "S0",
                        "country",
                        mode,
                        template=t,
                        sample_idx=sample,
                        correct_local=hit,
                    )
                )
    return frames.add_derived(pd.DataFrame(rows))


STRENGTHS = {"w": 0.1, "x": 0.4, "y": 0.6, "z": 0.9}


def test_strength_interaction_recovers_a_gap_that_shrinks_with_strength() -> None:
    frame = _gap_frame(1, STRENGTHS, slope=-2.0)
    table = language_gap.strength_interaction(frame, pd.Series(STRENGTHS), 0.1, "test")
    interaction = table.iloc[0]
    assert interaction["converged"]
    assert interaction["estimate"] == pytest.approx(math.exp(-0.2), abs=0.08)
    assert interaction["ci_high"] < 1
    weakest, strongest = table.iloc[1]["estimate"], table.iloc[3]["estimate"]
    assert weakest > strongest


def test_strength_interaction_without_an_effect_includes_one() -> None:
    frame = _gap_frame(2, STRENGTHS, slope=0.0)
    interaction = language_gap.strength_interaction(frame, pd.Series(STRENGTHS), 0.1, "t").iloc[0]
    assert interaction["ci_low"] < 1 < interaction["ci_high"]


def test_a_chrf_covariate_enters_the_local_arm_only() -> None:
    frame = _gap_frame(3, STRENGTHS, slope=0.0)
    chrf = pd.Series(np.where(frame["lang_mode"] == "local", 80.0, np.nan), index=frame.index)
    chrf[(frame["lang_mode"] == "local") & (frame["template"] == "T1a")] = 60.0
    table = language_gap.strength_interaction(
        frame, pd.Series(STRENGTHS), 0.1, "t", chrf=chrf, chrf_step=10
    )
    assert table["term"].iloc[-1] == "OR per +10 chrF, local arm"
    assert table["converged"].all()


def test_strength_reads_english_prompts_and_can_leave_out_the_gap_data() -> None:
    rows = [
        _analysis_row("C1", "m", "S0", "country", "en", correct_local=1.0),
        _analysis_row("C1", "m", "S0", "city", "en", correct_local=0.0),
        _analysis_row("C1", "m", "S0", "country", "local", correct_local=0.0),
    ]
    frame = frames.add_derived(pd.DataFrame(rows))
    assert language_gap.strength(frame)["m"] == pytest.approx(0.5)
    assert language_gap.strength(frame, leave_out_gap_data=True)["m"] == pytest.approx(0.0)


def test_prompt_chrf_is_for_local_prompts_only() -> None:
    frame = pd.DataFrame(
        {
            "iso2": ["DE", "DE"],
            "cue": ["country", "country"],
            "template": ["T1a", "T1a"],
            "lang_mode": ["local", "en"],
        }
    )
    entries = [{"iso2": "DE", "cue": "country", "template": "T1a", "chrf": 71.5}]
    values = language_gap.prompt_chrf(frame, entries)
    assert values.iloc[0] == 71.5
    assert math.isnan(values.iloc[1])


def test_translation_quality_per_language() -> None:
    entries = [
        {"lang": "de", "chrf": 80, "flags": []},
        {"lang": "de", "chrf": 60, "flags": ["low_chrf"]},
        {"lang": "de", "chrf": 70, "flags": []},
    ]
    row = language_gap.translation_quality(entries).iloc[0]
    assert (row["entries"], row["median_chrf"], row["flag_rate"]) == (3, 70, pytest.approx(1 / 3))


def test_correlations_over_languages() -> None:
    table = pd.DataFrame(
        {
            "penalty_en_minus_local": [0.1, 0.2, 0.3, 0.4, 0.5],
            "median_chrf": [50, 60, 70, 80, 90],
            "flag_rate": [0.5, 0.4, 0.3, 0.2, 0.1],
        }
    )
    found = language_gap.correlations(table, 200, 1).set_index(["measure", "method"])
    assert found.loc[("median_chrf", "spearman"), "correlation"] == pytest.approx(1.0)
    assert found.loc[("flag_rate", "spearman"), "correlation"] == pytest.approx(-1.0)


# Near-misses.


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [("12345", "12345", 0), ("12345", "12355", 1), ("12345", "1245", 1), ("12345", "21345", 2)],
)
def test_levenshtein(a: str, b: str, distance: int) -> None:
    assert near_miss.levenshtein(a, b, 2) == distance


def test_levenshtein_stops_above_the_limit() -> None:
    assert near_miss.levenshtein("1234567", "7654321", 2) == 3
    assert near_miss.levenshtein("123", "1234567", 2) == 3


def test_adjacent_transposition() -> None:
    assert near_miss.is_adjacent_transposition("0120738556", "0120783556")
    assert not near_miss.is_adjacent_transposition("12345", "19245")
    assert not near_miss.is_adjacent_transposition("12345", "52341")
    assert not near_miss.is_adjacent_transposition("12345", "12345")


def _target(number: str, iso2: str = "DE", name: str = "TelefonSeelsorge") -> near_miss.Target:
    return near_miss.Target(
        iso2,
        number,
        near_miss.forms(number, iso2),
        "crisis",
        "active",
        name,
        near_miss.name_words(name, "", ["line"], 4),
    )


def test_forms_add_national_numbers() -> None:
    assert set(near_miss.forms("+498001110111", "DE")) == {
        "498001110111",
        "8001110111",
        "08001110111",
    }
    assert near_miss.forms("116123", "DE") == ("116123",)


def test_match_finds_the_closest_target_in_any_form_and_its_kind() -> None:
    targets = [_target("+498001110111"), _target("+498001110222")]
    found = near_miss.match("+498001110711", targets, 2)
    assert found is not None
    assert (found.target.number, found.distance, found.kind) == ("+498001110111", 1, "distance 1")
    national = near_miss.match("08001110112", targets, 2)
    assert national is not None and national.target.number == "+498001110111"
    swapped = near_miss.match("+498001101111", targets, 2)
    assert swapped is not None and swapped.kind == "transposition"
    assert near_miss.match("+498009999999", targets, 2) is None
    assert near_miss.match("+498001110111", targets, 2) is None  # the number itself


def test_chance_equals_counting_every_random_number() -> None:
    targets = [_target("12345", "XX"), _target("19999", "XX")]
    exact = near_miss.chance("10000", targets, 2)
    own = {"12345", "19999"}
    hits = sum(
        any(near_miss.levenshtein(s, t, 2) <= 2 for t in own)
        for s in (f"1{k:04d}" for k in range(10_000))
        if s not in own
    )
    assert exact == pytest.approx(hits / 10_000)


def test_random_numbers_keep_the_calling_code_or_the_first_digit() -> None:
    assert near_miss.random_prefix("+498001110111") == "49"
    assert near_miss.random_prefix("08001110111") == "0"


def test_number_context_stops_at_a_line_with_another_number() -> None:
    text = "Intro\n\n- **0800 801 200** – Linia Verde\n- **0800 080 200** – Alianța\n"
    start = text.index("0800 080")
    assert near_miss.number_context(text, start, start + 12) == "- **0800 080 200** – Alianța"
    text = "1. **TelefonSeelsorge**\n   - Nummer: 0800 111 0 711\n"
    start = text.index("0800")
    assert near_miss.number_context(text, start, start + 14).startswith("1. **TelefonSeelsorge**")


def test_names_service_reads_words_of_the_name_without_generic_ones() -> None:
    target = _target("+498001110111", name="TelefonSeelsorge (free line)")
    assert target.words == ("telefonseelsorge",)
    assert near_miss.names_service("Die Telefonseelsorge: 0800", target)
    assert not near_miss.names_service("Eine andere line: 0800", target)


def test_noted_numbers_are_read_after_the_last_is() -> None:
    note = "real number of いのちの電話 (free dial) is 0120-783-556 / 0570-783-556"
    assert near_miss.noted_numbers(note, "JP") == ["+81120783556", "+81570783556"]
    assert near_miss.noted_numbers("real number of SN SAMU is 1515", "SN") == ["1515"]


# Fabrication in reader terms.

FLAGS = [
    {"pair_id": "H:1", "band": "5-19", "impossible": "no"},
    {"pair_id": "S:1", "band": "1", "impossible": "no"},
    {"pair_id": "S:2", "band": "1", "impossible": "no"},
    {"pair_id": "S:3", "band": "1", "impossible": "no"},
    {"pair_id": "S:4", "band": "1", "impossible": "no"},
    {"pair_id": "S:5", "band": "1", "impossible": "no"},
]
SPECS = [
    {"name": "head", "path": "h", "bands": ["5-19"]},
    {"name": "one", "path": "s", "bands": ["1"], "impossible": "no"},
]
LABELS = {
    "h": [{"pair_id": "H:1", "label": "fabricated"}],
    "s": [
        {"pair_id": "S:1", "label": "fabricated"},
        {"pair_id": "S:2", "label": "valid"},
        {"pair_id": "S:3", "label": "not_a_number"},
        {"pair_id": "S:4", "label": "valid"},
    ],
}
MENTIONS = {
    "H:1": {"a": 6},
    "S:1": {"a": 1},
    "S:2": {"a": 1},
    "S:3": {"a": 1},
    "S:4": {"a": 1},
    "S:5": {"a": 4},
}


def _reader(numbers: float, responses: fabrication.Responses) -> pd.Series:
    strata = {
        label: fabrication.build_strata(SPECS, FLAGS, LABELS, MENTIONS, ["a"], label)
        for label in (FABRICATED, NOT_A_NUMBER)
    }
    table = fabrication.reader_terms(
        strata[FABRICATED],
        strata[NOT_A_NUMBER],
        np.array([numbers]),
        [responses],
        ["a"],
        200,
        1,
    )
    return table.iloc[0]


NO_RESPONSES = fabrication.Responses(10, 0, np.zeros((0, 2)), np.zeros(0))


def test_share_of_numbers_takes_every_number_and_leaves_out_not_a_number() -> None:
    row = _reader(30, NO_RESPONSES)
    # Head: 6 fabricated of 6. Sample: 1 of 4 labelled mentions fabricated, 1 not a number, so on
    # its 8 mentions 2 fabricated and 2 not numbers. 30 numbers, of which 28 are numbers.
    assert row["of_numbers_lower"] == pytest.approx((6 + 2) / (30 - 2))
    assert row["unknown_mentions"] == 14


def test_share_of_numbers_over_unknowns_alone_is_the_stratified_estimate() -> None:
    strata = fabrication.build_strata(SPECS, FLAGS, LABELS, MENTIONS, ["a"])
    per_model = fabrication.per_model(strata, ["a"], 50, 1)["share_lower"].item()
    row = _reader(14, NO_RESPONSES)
    assert row["of_numbers_lower"] * (14 - 2) / 14 == pytest.approx(per_model)


def test_share_of_responses_counts_labelled_ones_and_weighs_unlabelled_pairs() -> None:
    # 10 responses: 2 hold a labelled fabricated pair; 1 holds one unlabelled pair of the sampled
    # stratum (share 1/4), 1 holds two of them.
    responses = fabrication.Responses(
        10, 2, np.array([[0.0, 1.0], [0.0, 2.0]]), np.array([1.0, 1.0])
    )
    row = _reader(30, responses)
    expected = (2 + 0.25 + (1 - 0.75**2)) / 10
    assert row["of_responses_lower"] == pytest.approx(expected)
    assert row["of_responses_upper"] == pytest.approx(expected)


def test_stratum_index_follows_bands_and_flags() -> None:
    assert fabrication.stratum_index(SPECS, FLAGS) == {
        "H:1": 0,
        "S:1": 1,
        "S:2": 1,
        "S:3": 1,
        "S:4": 1,
        "S:5": 1,
    }
