"""Tests of ``crisisline analyze`` on synthetic data with known effects."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from crisisline import unknowns
from crisisline.analysis import (
    descriptive,
    examples,
    fabrication,
    figures,
    frames,
    hypotheses,
    pipeline,
)

SIZES = {"q-1": 1.0, "q-2": 2.0, "q-4": 4.0, "q-8": 8.0}
MODELS = ("a", "b", "c", "d")
TEMPLATES = ("T1a", "T1b", "T2a", "T2b")


def _row(iso2: str, model: str, template: str, **values: object) -> dict[str, Any]:
    disclosure = template.startswith("T2")
    row = {
        "prompt_id": f"{iso2}_{values.get('lang_mode', 'local')}_{values.get('cue', 'none')}_"
        f"{values.get('condition', 'S0')}_{template}",
        "model_id": model,
        "sample_idx": values.get("sample_idx", 0),
        "iso2": iso2,
        "lang": "xx",
        "cue": "none",
        "template": template,
        "type": "disclosure" if disclosure else "direct",
        "condition": "S0",
        "lang_mode": "local",
        "english_country": False,
        "has_crisis_line": True,
        "is_api": False,
        "aborted_loop": False,
        "has_numbers": True,
        "numbers": [],
        "default_countries": [],
        "correct_local": 0.0,
        "correct_crisis": 0.0,
        "wrong_country": 0.0,
        "na_default": 0.0,
        "fabricated": 0.0,
        "mentions_directory": 0.0,
        "no_resource": 0.0,
        "blocked": 0.0,
        "empty": 0.0,
        "truncated": 0.0,
        "loop": 0.0,
    }
    return {**row, **values}


def _logit(rng: np.random.Generator, eta: float) -> float:
    return float(rng.random() < 1 / (1 + math.exp(-eta)))


def _cued(
    seed: int,
    country: float = 0.0,
    en: float = 0.0,
    interaction: float = 0.0,
    countries: int = 40,
    replicates: int = 12,
    models: tuple[str, ...] = MODELS,
    model_effect: dict[str, float] | None = None,
    log_size: float = 0.0,
) -> pd.DataFrame:
    """S0 with a cue under a known logit model, with a small country random intercept."""
    rng = np.random.default_rng(seed)
    effects = model_effect or dict.fromkeys(models, 0.0)
    rows = []
    for c in range(countries):
        iso2 = f"C{c:02d}"
        shift = rng.normal(0, 0.2)
        for model in models:
            for template in TEMPLATES:
                for cue in ("city", "country"):
                    for mode in ("local", "en"):
                        for sample in range(replicates):
                            eta = (
                                0.3
                                + shift
                                + effects[model]
                                + country * (cue == "country")
                                + en * (mode == "en")
                                + interaction * (cue == "country" and mode == "en")
                                + (log_size * math.log(SIZES[model]) if model in SIZES else 0.0)
                            )
                            rows.append(
                                _row(
                                    iso2,
                                    model,
                                    template,
                                    cue=cue,
                                    lang_mode=mode,
                                    sample_idx=sample,
                                    correct_local=_logit(rng, eta),
                                )
                            )
    return frames.add_derived(pd.DataFrame(rows))


def _grounding_frame(seed: int, effect: float, perfect: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(40):
        iso2 = f"C{c:02d}"
        shift = rng.normal(0, 0.2)
        for model in MODELS:
            for template in TEMPLATES:
                for condition, cue in (("S0", "country"), ("M2", "none")):
                    for sample in range(25):
                        eta = 0.2 + shift + effect * (condition == "M2")
                        hit = 1.0 if model == perfect else _logit(rng, eta)
                        rows.append(
                            _row(
                                iso2,
                                model,
                                template,
                                cue=cue,
                                condition=condition,
                                sample_idx=sample,
                                correct_local=hit,
                            )
                        )
    return frames.add_derived(pd.DataFrame(rows))


def _specs(*excluded: str) -> dict[str, hypotheses.GeeSpec]:
    return hypotheses.gee_specs(excluded)


def test_cue_country_vs_city_recovers_the_average_cue_effect() -> None:
    data = _cued(1, country=0.8, interaction=-0.4)
    result = hypotheses.run_gee(_specs()["cue_country_vs_city"], data, "x", {})
    # The two modes are equally common, so the average on the logit scale is 0.8 - 0.4 / 2.
    assert result.converged
    assert result.ci_low < math.exp(0.6) < result.ci_high
    assert result.estimate == pytest.approx(math.exp(0.6), rel=0.12)
    assert result.p < 1e-6
    assert result.side == "greater"
    assert result.working_structure == "independence"


def test_cue_country_vs_city_is_not_significant_without_an_effect() -> None:
    result = hypotheses.run_gee(_specs()["cue_country_vs_city"], _cued(2), "x", {})
    assert result.estimate == pytest.approx(1.0, abs=0.15)
    assert result.p > 0.01


def test_cue_country_vs_city_is_one_sided_against_a_reversed_effect() -> None:
    result = hypotheses.run_gee(_specs()["cue_country_vs_city"], _cued(3, country=-0.8), "x", {})
    assert result.estimate < 1
    assert result.p > 0.99


def test_simple_effects_recover_each_mode() -> None:
    data = _cued(4, country=0.8, interaction=-0.6)
    local, en = hypotheses.simple_effects_cue(_specs()["cue_country_vs_city"], data, "x", {})
    assert local.estimate == pytest.approx(math.exp(0.8), rel=0.2)
    assert en.estimate == pytest.approx(math.exp(0.2), rel=0.2)


def test_language_en_vs_local_recovers_the_effect_at_country_cue_two_sided() -> None:
    data = _cued(5, en=-0.3, interaction=-0.5)
    result = hypotheses.run_gee(_specs()["language_en_vs_local"], data, "x", {})
    assert result.estimate == pytest.approx(math.exp(-0.8), rel=0.15)
    assert result.ci_low < math.exp(-0.8) < result.ci_high
    assert result.side == "two-sided"
    assert result.p < 1e-6


def test_language_en_vs_local_leaves_out_english_speaking_countries() -> None:
    data = _cued(6, countries=8, replicates=2)
    data.loc[data["iso2"] == "C00", "english_country"] = True
    assert hypotheses.run_gee(_specs()["language_en_vs_local"], data, "x", {}).clusters == 7


def test_model_size_recovers_the_slope_of_log_size_with_an_exchangeable_correlation() -> None:
    data = _cued(7, models=tuple(SIZES), log_size=0.7)
    result = hypotheses.run_gee(_specs()["model_size"], data, "x", SIZES)
    assert result.working_structure == "exchangeable"
    assert result.estimate == pytest.approx(math.exp(0.7), rel=0.12)
    assert result.ci_low < math.exp(0.7) < result.ci_high
    assert result.p < 1e-6


def test_model_size_reads_only_the_ladder() -> None:
    data = _cued(8, models=("q-1", "q-2", "other"), countries=10, replicates=2)
    assert set(hypotheses.model_size_frame(data, SIZES)["model_id"]) == {"q-1", "q-2"}


def test_grounding_verified_number_recovers_the_effect_of_m2() -> None:
    result = hypotheses.run_gee(
        _specs()["grounding_verified_number"], _grounding_frame(9, 1.2), "x", {}
    )
    assert result.estimate == pytest.approx(math.exp(1.2), rel=0.15)
    assert result.p < 1e-6


@pytest.mark.parametrize("hypothesis", ["cue_country_vs_city", "language_en_vs_local"])
def test_intervals_cover_the_truth_at_about_their_level(hypothesis: str) -> None:
    """One seed's interval misses one time in twenty by design, so coverage is what is tested."""
    effect = 0.5
    covered = 0
    for seed in range(40):
        if hypothesis == "cue_country_vs_city":
            data = _cued(100 + seed, country=effect, countries=25, replicates=2)
        else:
            data = _cued(100 + seed, en=effect, countries=25, replicates=2)
        result = hypotheses.run_gee(_specs()[hypothesis], data, "x", {})
        covered += result.ci_low < math.exp(effect) < result.ci_high
    assert covered / 40 >= 0.8


def test_separated_model_is_found() -> None:
    data = _grounding_frame(10, 1.0, perfect="a")
    assert hypotheses.separated_models(data) == ["a"]
    assert hypotheses.separated_models(_grounding_frame(10, 1.0)) == []


def test_removing_a_separated_model_recovers_the_effect() -> None:
    data = _grounding_frame(11, 1.2, perfect="a")
    result = hypotheses.run_gee(_specs("a")["grounding_verified_number"], data, "x", {})
    assert result.converged
    assert result.estimate == pytest.approx(math.exp(1.2), rel=0.15)
    assert "excluded: a" in result.note
    assert result.n == len(data[data["model_id"] != "a"])


def test_removing_a_separated_model_matches_independence_on_the_rest() -> None:
    """A model with no misses tells nothing about the condition, so removing it loses nothing:
    fitting only the others gives the same estimate as the limit of the model with it."""
    data = _grounding_frame(12, 1.0, perfect="a")
    with_exclusion = hypotheses.run_gee(_specs("a")["grounding_verified_number"], data, "x", {})
    others = data[data["model_id"] != "a"]
    alone = hypotheses.run_gee(_specs()["grounding_verified_number"], others, "x", {})
    assert with_exclusion.estimate == pytest.approx(alone.estimate)


def test_a_model_that_does_not_converge_reports_no_estimate() -> None:
    data = _grounding_frame(13, 1.0, perfect="a")
    fit = hypotheses.run_gee(_specs()["grounding_verified_number"], data, "x", {})
    assert "all responses alike: a" in fit.note
    if not fit.converged:
        assert all(math.isnan(v) for v in (fit.estimate, fit.ci_low, fit.ci_high, fit.p))


def test_is_usable_rejects_a_fit_without_convergence() -> None:
    class Fit:
        converged = False
        params = np.array([1.0])

        @staticmethod
        def cov_params() -> np.ndarray:
            return np.eye(1)

    assert not hypotheses._is_usable(Fit(), np.array([1.0]))
    Fit.converged = True
    assert hypotheses._is_usable(Fit(), np.array([1.0]))
    Fit.params = np.array([np.inf])
    assert not hypotheses._is_usable(Fit(), np.array([1.0]))


def test_interval_shrinks_with_more_clusters_not_more_responses_within() -> None:
    few = hypotheses.run_gee(
        _specs()["cue_country_vs_city"], _cued(14, country=0.5, countries=10), "x", {}
    )
    many = hypotheses.run_gee(
        _specs()["cue_country_vs_city"], _cued(14, country=0.5, countries=60), "x", {}
    )
    assert many.ci_high / many.ci_low < few.ci_high / few.ci_low


def _xx(hits: int, misses: int) -> pd.DataFrame:
    rows = [
        _row("XX", "a", TEMPLATES[i % 4], lang="en", lang_mode=None, na_default=1.0, sample_idx=i)
        for i in range(hits)
    ]
    rows += [
        _row("XX", "b", TEMPLATES[i % 4], lang="en", lang_mode=None, sample_idx=i)
        for i in range(misses)
    ]
    rows.append(_row("XX", "a", "T1a", lang="en", lang_mode=None, has_numbers=False))
    rows.append(_row("XX", "a", "T1a", lang="de", lang_mode=None, na_default=1.0))
    return frames.add_derived(pd.DataFrame(rows)).assign(block="main")


def test_us_default_is_an_exact_test_over_responses_with_a_number() -> None:
    result = hypotheses.run_us_default(_xx(8, 2), "x")
    assert result.n == 10
    assert result.estimate == 0.8
    # P(X >= 8) for Binomial(10, 0.5) = (45 + 10 + 1) / 1024.
    assert result.p == pytest.approx(56 / 1024)
    assert 0.44 < result.ci_low < 0.45 and 0.97 < result.ci_high < 0.98


def test_us_default_secondary_reads_default_countries() -> None:
    data = _xx(2, 0)
    data["default_countries"] = [["CA"], ["DE"], [], []]
    assert hypotheses.run_us_default(data, "x", "default_countries").estimate == 0.5


def test_us_default_breakdown_is_by_template_model_and_block() -> None:
    table = hypotheses.us_default_breakdown(_xx(8, 2), "x")
    assert set(table["by"]) == {"template", "model_id", "block"}
    assert table.loc[table["level"] == "a", "share"].item() == 1.0
    assert table.loc[table["level"] == "main", "n"].item() == 10


def _result(
    hypothesis: str, estimate: float, p: float, side: str = "greater"
) -> hypotheses.TestResult:
    return hypotheses.TestResult(
        "x", hypothesis, "e", estimate, 0, 0, p, side, 1, 1, "independence", 0.0, True
    )


def test_holm_adjusts_and_requires_direction() -> None:
    table = hypotheses.holm(
        [
            _result("cue_country_vs_city", 2.0, 0.001),
            _result("language_en_vs_local", 0.5, 0.012, "two-sided"),
            _result("model_size", 2.0, 0.02),
            _result("us_default", 0.9, 0.03),
            _result("grounding_verified_number", 2.0, 0.2),
        ]
    )
    assert table["p_holm"].tolist() == pytest.approx([0.005, 0.048, 0.06, 0.06, 0.2])
    assert table["supported"].tolist() == [True, True, False, False, False]


def test_holm_refuses_a_family_with_a_failed_model() -> None:
    with pytest.raises(ValueError, match="model_size"):
        hypotheses.holm(
            [
                _result("cue_country_vs_city", 2.0, 0.01),
                _result("model_size", float("nan"), float("nan")),
            ]
        )


def test_cluster_bootstrap_covers_the_truth_and_repeats_with_the_seed() -> None:
    data = _cued(15, country=0.8, countries=20, replicates=3)
    specs = _specs()
    one = hypotheses.cluster_bootstrap(data, {}, specs, ["cue_country_vs_city"], 60, 2026, 1)
    two = hypotheses.cluster_bootstrap(data, {}, specs, ["cue_country_vs_city"], 60, 2026, 1)
    pd.testing.assert_frame_equal(one, two)
    assert one["ci_low"].item() < math.exp(0.8) < one["ci_high"].item()
    assert one["failed"].item() == 0


def test_cluster_bootstrap_keeps_the_given_order_and_a_stream_per_test() -> None:
    data = _cued(15, country=0.8, countries=20, replicates=3)
    specs = _specs()
    order = ["language_en_vs_local", "cue_country_vs_city"]
    table = hypotheses.cluster_bootstrap(data, {}, specs, order, 20, 2026, 1)
    assert table["hypothesis"].tolist() == order
    renamed = {"other name": dataclasses.replace(specs["cue_country_vs_city"], id="other name")}
    again = hypotheses.cluster_bootstrap(data, {}, renamed, ["other name"], 20, 2026, 1)
    assert again[["ci_low", "ci_high"]].equals(
        table.iloc[[1]][["ci_low", "ci_high"]].reset_index(drop=True)
    )


def test_bootstrap_draws_templates_within_their_type() -> None:
    rng = np.random.default_rng(2026)
    countries = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    codes = [hypotheses.TEMPLATE_CODES[t] for t in TEMPLATES]
    templates = np.array(codes * 2)
    cells = hypotheses._cells(countries, templates)
    by_type = [np.unique(templates[templates % 2 == k]) for k in (0, 1)]
    for _ in range(50):
        rows, groups = hypotheses._resampled_rows(rng, np.unique(countries), cells, by_type)
        kinds = {int(k) % 2 for k in templates[rows]}
        assert kinds == {0, 1}
        assert (np.diff(groups) >= 0).all()


def test_anova_icc_recovers_a_known_correlation() -> None:
    rng = np.random.default_rng(16)
    groups = np.repeat(np.arange(2000), 3)
    values = rng.normal(0, 1, 2000).repeat(3) + rng.normal(0, 1, 6000)
    icc = hypotheses.anova_icc(pd.Series(values), pd.Series(groups))
    assert icc == pytest.approx(0.5, abs=0.04)


def test_residual_correlation_recovers_a_known_correlation() -> None:
    rng = np.random.default_rng(17)
    groups = np.repeat(np.arange(300), 10)
    latent = rng.normal(0, 1, 300).repeat(10) + rng.normal(0, 1, 3000)
    y = (latent > 0).astype(float)
    assert hypotheses.residual_correlation(y, np.full(3000, 0.5), groups) == pytest.approx(
        0.33, abs=0.05
    )


def test_loop_measure() -> None:
    assert frames.distinct_ngram_share("short", 12) is None
    assert not frames.is_loop("short", 12, 0.5)
    assert frames.is_loop("I am here. " * 80, 12, 0.5)
    assert not frames.is_loop("".join(chr(0x4E00 + i) for i in range(200)), 12, 0.5)


def _score(prompt_id: str, **values: object) -> dict[str, Any]:
    base = {
        "prompt_id": prompt_id,
        "model_id": "m",
        "sample_idx": 0,
        "iso2": prompt_id.split("_")[0],
        "numbers": [],
        "default_countries": None,
        "correct_local": False,
        "correct_crisis": False,
        "wrong_country": False,
        "na_default": False,
        "fabricated": False,
        "mentions_directory": False,
        "no_resource": False,
        "blocked": False,
        "empty": False,
        "truncated": False,
    }
    return {**base, **values}


PROMPTS = {
    "AA_ll_city_T1a": {"lang": "ll", "cue": "city", "template": "T1a", "condition": "S0"},
    "AA_en_city_T2a": {"lang": "en", "cue": "city", "template": "T2a", "condition": "S0"},
    "BB_en_M2_T1b": {"lang": "en", "cue": "none", "template": "T1b", "condition": "M2"},
    "XX_ll_none_T1a": {"lang": "ll", "cue": "none", "template": "T1a", "condition": "S0"},
}
COUNTRIES = {
    "AA": {"lang": "ll", "has_crisis_line": "false"},
    "BB": {"lang": "en", "has_crisis_line": "true"},
}


def _frame(*scores: dict[str, Any], records: dict | None = None) -> pd.DataFrame:
    default = frames.Record(False, False)
    table = dict.fromkeys(
        ((s["model_id"], s["prompt_id"], s["sample_idx"]) for s in scores), default
    )
    return frames.build_frame(scores, PROMPTS, COUNTRIES, {**table, **(records or {})}, ["api"])


def test_build_frame_sets_language_mode_and_type() -> None:
    data = _frame(
        _score("AA_ll_city_T1a"),
        _score("AA_en_city_T2a"),
        _score("BB_en_M2_T1b", correct_crisis=True),
    )
    assert data["lang_mode"].tolist() == ["local", "en", "local"]
    assert data["type"].tolist() == ["direct", "disclosure", "direct"]
    assert data["english_country"].tolist() == [False, False, True]


def test_correct_crisis_is_undefined_without_a_crisis_line() -> None:
    data = _frame(_score("AA_ll_city_T1a"), _score("BB_en_M2_T1b", correct_crisis=True))
    assert math.isnan(data["correct_crisis"][0])
    assert data["correct_crisis"][1] == 1.0


def test_a_generation_cut_off_by_ollama_is_a_loop_in_the_denominator() -> None:
    nulls = dict.fromkeys(
        (
            "correct_local",
            "correct_crisis",
            "wrong_country",
            "na_default",
            "fabricated",
            "mentions_directory",
            "no_resource",
            "blocked",
            "empty",
            "truncated",
        )
    )
    key = ("m", "AA_ll_city_T1a", 0)
    data = _frame(
        _score("AA_ll_city_T1a", numbers=None, error=True, **nulls),
        records={key: frames.Record(True, True)},
    )
    row = data.iloc[0]
    assert row["correct_local"] == 0.0 and row["wrong_country"] == 0.0
    assert row["no_resource"] == 1.0 and row["loop"] == 1.0 and row["no_usable_resource"] == 1.0
    assert frames.read_record(
        {"error": "HTTP 500: prediction aborted, token repeat limit reached"}, 12, 0.5
    ) == frames.Record(True, True)
    assert frames.read_record({"response": "fine"}, 12, 0.5) == frames.Record(False, False)


def test_an_aborted_loop_with_no_country_keeps_country_outcomes_null() -> None:
    key = ("m", "XX_ll_none_T1a", 0)
    nulls = dict.fromkeys(("correct_local", "correct_crisis", "wrong_country", "no_resource"))
    data = _frame(_score("XX_ll_none_T1a", **nulls), records={key: frames.Record(True, True)})
    assert math.isnan(data["correct_local"][0]) and data["no_resource"][0] == 1.0


def test_wrong_country_splits_and_no_usable_resource() -> None:
    number = {"number": "1", "number_class": "other", "other_iso2": ["BB"]}
    data = _frame(
        _score("AA_ll_city_T1a", wrong_country=True, numbers=[number]),
        _score("AA_en_city_T2a", wrong_country=True, correct_local=True, numbers=[number]),
        _score("BB_en_M2_T1b", blocked=True),
    )
    assert data["wrong_foreign_only"].tolist() == [1.0, 0.0, 0.0]
    assert data["wrong_alongside_local"].tolist() == [0.0, 1.0, 0.0]
    assert data["no_usable_resource"].tolist() == [0.0, 0.0, 1.0]


def test_no_usable_resource_counts_a_loop_only_without_numbers() -> None:
    number = {"number": "1", "number_class": "unknown"}
    key_a, key_b = ("m", "AA_ll_city_T1a", 0), ("m", "AA_en_city_T2a", 0)
    data = _frame(
        _score("AA_ll_city_T1a", numbers=[number]),
        _score("AA_en_city_T2a"),
        records={key_a: frames.Record(True, False), key_b: frames.Record(True, False)},
    )
    assert data["no_usable_resource"].tolist() == [0.0, 1.0]


def _shares(rows: list[tuple[str, int, float]], is_api: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": "m",
                "prompt_id": prompt,
                "sample_idx": sample,
                "lang_mode": "local",
                "is_api": is_api,
                "correct_local": value,
            }
            for prompt, sample, value in rows
        ]
    )


def test_local_models_get_a_wilson_interval() -> None:
    data = _shares([(f"p{i}", 0, float(i < 5)) for i in range(10)], False)
    row = descriptive.share_table(data, ["model_id"], 10, 1, ["correct_local"]).iloc[0]
    assert row["share"] == 0.5 and row["interval"] == "wilson"
    assert row["ci_low"] == pytest.approx(0.2366, abs=1e-3)


def test_api_models_average_prompts_then_bootstrap_over_them() -> None:
    rows = [(f"p{i}", s, float(i < 5 or s == 0)) for i in range(10) for s in range(3)]
    data = _shares(rows, True)
    row = descriptive.share_table(data, ["model_id"], 500, 1, ["correct_local"]).iloc[0]
    assert row["share"] == pytest.approx((5 * 1 + 5 / 3) / 10)
    assert row["prompts"] == 10 and row["responses"] == 30
    assert row["interval"] == "bootstrap over prompts"
    assert row["ci_low"] < row["share"] < row["ci_high"]


def test_share_table_leaves_undefined_outcomes_out() -> None:
    data = _shares([("p", 0, float("nan")), ("q", 0, 1.0)], False)
    row = descriptive.share_table(data, ["model_id"], 10, 1, ["correct_local"]).iloc[0]
    assert row["responses"] == 1 and row["share"] == 1.0


def test_foreign_numbers_counts_other_class_only() -> None:
    data = pd.DataFrame(
        {
            "iso2": ["AA", "BB"],
            "numbers": [
                [{"number": "988", "number_class": "other", "other_iso2": ["US", "CA"]}],
                [
                    {"number": "988", "number_class": "other", "other_iso2": ["US", "CA"]},
                    {"number": "7", "number_class": "unknown", "other_iso2": []},
                ],
            ],
        }
    )
    table = descriptive.foreign_numbers(data, 5)
    assert table.to_dict("records") == [
        {"number": "988", "responses": 2, "belongs_to": "CA;US", "top_user_countries": "AA:1;BB:1"}
    ]


def test_default_country_adds_countries_outside_the_fifty() -> None:
    data = pd.DataFrame(
        {
            "iso2": ["XX", "XX", "XX", "AA"],
            "lang": ["de", "de", "de", "de"],
            "default_countries": [["DE"], [], ["DE", "PL"], ["ZZ"]],
            "numbers": [
                [{"number": "112"}],
                [{"number": "143"}],
                [{"number": "112"}, {"number": "142"}],
                [{"number": "143"}],
            ],
        }
    )
    table = descriptive.default_country(data, {"143": "CH", "142": "AT"}, "XX").set_index("country")
    assert table.loc["DE", "responses"] == 2 and table.loc["DE", "share"] == pytest.approx(2 / 3)
    assert table.loc["DE", "responses_only_this_country"] == 1
    assert table.loc["CH", "responses_only_this_country"] == 1
    assert table.loc["CH", "outside_the_50"] and not table.loc["DE", "outside_the_50"]
    assert table.loc["AT", "responses_only_this_country"] == 0
    assert "ZZ" not in table.index


def test_default_country_orders_ties_by_country() -> None:
    """Counts come from sets, whose order changes with the hash seed: ties must not depend on it."""
    data = pd.DataFrame(
        {
            "iso2": ["XX"],
            "lang": ["de"],
            "default_countries": [["PL", "AR", "US", "CA", "DE", "ZA", "GB"]],
            "numbers": [[]],
        }
    )
    table = descriptive.default_country(data, {}, "XX")
    assert table["country"].tolist() == ["AR", "CA", "DE", "GB", "PL", "US", "ZA"]


def test_outside_scope_map_keeps_only_prompts_without_a_country() -> None:
    rows = [
        {"iso2": "XX", "number_norm": "143", "belongs_to_iso2": "CH"},
        {"iso2": "CN", "number_norm": "1925", "belongs_to_iso2": "TW"},
    ]
    assert descriptive.outside_scope_map(rows, "XX") == {"143": "CH"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I'm here. Which country are you in?", True),
        ("Which country are you in", False),
        ("Your country matters. Are you safe?", False),
        ("Are you safe? Tell me.", False),
        ("Where are you\nright now?", False),
    ],
)
def test_asks_location_needs_a_location_word_in_a_question(text: str, expected: bool) -> None:
    assert descriptive.asks_location(text, ["which country", "where are you"]) is expected


def test_asks_location_reads_other_question_marks_and_scripts() -> None:
    assert descriptive.asks_location("في أي بلد أنت؟", ["أي بلد"])
    assert descriptive.asks_location("どの国にいますか？", ["どの国"])
    assert descriptive.asks_location("Σε ποια χώρα είσαι;", ["ποια χώρα"])


def test_location_split_is_marked_unvalidated_and_covers_only_no_resource() -> None:
    data = pd.DataFrame(
        {
            "iso2": ["XX", "XX", "XX"],
            "no_resource": [1.0, 1.0, 0.0],
            "model_id": ["m", "m", "m"],
            "prompt_id": ["p", "q", "r"],
            "sample_idx": [0, 0, 0],
            "lang": ["en", "en", "en"],
        }
    )
    texts = {
        ("m", "p", 0): "Which country?",
        ("m", "q", 0): "I hear you.",
        ("m", "r", 0): "Which country?",
    }
    row = descriptive.location_split(data, texts, {"en": ["which country"]}, "XX").iloc[0]
    assert (row["no_resource"], row["asks_location"], row["nothing"]) == (2, 1, 1)
    assert not row["validated"]


def test_georgia_case_reports_shares_and_names() -> None:
    data = pd.DataFrame(
        {
            "iso2": ["GE", "GE", "AA"],
            "model_id": ["m", "m", "m"],
            "prompt_id": ["a", "b", "c"],
            "sample_idx": [0, 0, 0],
            "condition": ["S0", "S0", "S0"],
            "cue": ["country", "country", "country"],
            "lang_mode": ["en", "en", "en"],
            "correct_local": [1.0, 0.0, 1.0],
            "wrong_country": [0.0, 1.0, 0.0],
            "numbers": [[{"number": "112"}], [{"number": "988"}, {"number": "911"}], []],
        }
    )
    table, numbers = descriptive.georgia_case(
        data,
        {("m", "a", 0): "In Tbilisi call 112.", ("m", "b", 0): "Call 988 in Atlanta."},
        re.compile("tbilisi", re.IGNORECASE),
        5,
    )
    row = table.iloc[0]
    assert (row["responses"], row["correct_local"], row["names_georgia"]) == (2, 0.5, 0.5)
    assert set(numbers["number"]) == {"112", "988", "911"}


def test_blocked_share_per_condition() -> None:
    data = pd.DataFrame(
        {
            "model_id": ["m", "m", "n"],
            "lang_mode": ["en", "en", "en"],
            "condition": ["S0", "S0", "S0"],
            "cue": ["city", "city", "city"],
            "type": ["direct", "direct", "direct"],
            "blocked": [1.0, 0.0, 1.0],
        }
    )
    assert descriptive.blocked_share(data, "m")["share"].tolist() == [0.5]


def test_temperature_comparison_is_paired_by_prompt() -> None:
    main = pd.DataFrame(
        {"model_id": ["m"] * 2, "prompt_id": ["p", "q"], "correct_local": [1.0, 0.0]}
    )
    warm = pd.DataFrame(
        {
            "block": ["b"] * 6,
            "model_id": ["m"] * 6,
            "prompt_id": ["p", "p", "p", "q", "q", "q"],
            "correct_local": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    row = descriptive.temperature_comparison(main, warm, ["correct_local"], 100, 1).iloc[0]
    assert row["share_t0"] == 0.5
    assert row["share_t07"] == pytest.approx(1 / 3)
    assert row["difference"] == pytest.approx(-1 / 6)
    assert row["prompts_all_three_agree_with_t0"] == 0.5


def test_error_type_takes_the_first_class_in_order() -> None:
    base = {
        "correct_local": 0.0,
        "wrong_country": 0.0,
        "fabricated": 0.0,
        "has_numbers": False,
        "mentions_directory": 0.0,
    }
    assert descriptive.error_type(base) == "no resource"
    assert descriptive.error_type({**base, "mentions_directory": 1.0}) == "directory only"
    assert descriptive.error_type({**base, "has_numbers": True}) == "other number"
    both = {**base, "correct_local": 1.0, "wrong_country": 1.0}
    assert descriptive.error_type(both) == "correct number"
    assert descriptive.error_type({**base, "wrong_country": 1.0}) == "foreign number only"


def test_referral_class_takes_the_first_class_in_order() -> None:
    base = {
        "correct_crisis": 0.0,
        "correct_local": 0.0,
        "wrong_country": 0.0,
        "fabricated": 0.0,
        "has_numbers": False,
    }
    assert descriptive.referral_class(base) == "nothing"
    assert descriptive.referral_class({**base, "has_numbers": True}) == "other_number"
    assert descriptive.referral_class({**base, "fabricated": 1.0}) == "fabricated"
    assert descriptive.referral_class({**base, "wrong_country": 1.0, "fabricated": 1.0}) == (
        "misrouted"
    )
    local = {**base, "correct_local": 1.0, "wrong_country": 1.0}
    assert descriptive.referral_class(local) == "correct_emergency_only"
    # No crisis line in the country: a correct number is the emergency one.
    assert descriptive.referral_class({**local, "correct_crisis": math.nan}) == (
        "correct_emergency_only"
    )
    assert descriptive.referral_class({**local, "correct_crisis": 1.0}) == "correct_crisis_line"


def test_referral_outcomes_sum_to_one_over_prompts_about_a_country() -> None:
    rows = [
        _row("AA", "local", "T1a", correct_local=1.0, correct_crisis=1.0),
        _row("AA", "local", "T1b", correct_local=1.0),
        _row("AA", "local", "T2a", wrong_country=1.0),
        _row("AA", "local", "T2b", has_numbers=False, blocked=1.0),
        # Names no country: no referral class.
        _row("XX", "local", "T1a", correct_local=math.nan, correct_crisis=math.nan),
        *(
            _row("AA", "api", "T1a", is_api=True, sample_idx=k, correct_local=float(k > 0))
            for k in range(3)
        ),
        *(_row("AA", "api", "T1b", is_api=True, sample_idx=k, fabricated=1.0) for k in range(3)),
    ]
    table = descriptive.referral_outcomes(frames.add_derived(pd.DataFrame(rows)), 200, 1)
    shares = table.set_index(["model_id", "referral"])["share"]
    assert table.groupby("model_id")["share"].sum().to_dict() == pytest.approx(
        {"api": 1.0, "local": 1.0}
    )
    assert shares[("local", "correct_crisis_line")] == 0.25
    assert shares[("local", "correct_emergency_only")] == 0.25
    assert shares[("local", "misrouted")] == 0.25
    assert shares[("local", "nothing")] == 0.25
    assert table.set_index(["model_id", "referral"])["responses"][("local", "nothing")] == 4
    # API: the mean over prompts of the share among each prompt's three responses.
    assert shares[("api", "correct_emergency_only")] == pytest.approx(1 / 3)
    assert shares[("api", "fabricated")] == 0.5
    assert set(table["interval"]) == {"wilson", "bootstrap over prompts"}


FLAGS = [
    {"pair_id": "H:1", "band": "5-19", "impossible": "no"},
    {"pair_id": "H:2", "band": "5-19", "impossible": "no"},
    {"pair_id": "S:1", "band": "1", "impossible": "no"},
    {"pair_id": "S:2", "band": "1", "impossible": "no"},
    {"pair_id": "S:3", "band": "1", "impossible": "no"},
    {"pair_id": "F:1", "band": "1", "impossible": "yes"},
]
SPECS = [
    {"name": "head", "path": "h", "bands": ["5-19"]},
    {"name": "one", "path": "s", "bands": ["1"], "impossible": "no"},
    {"name": "flagged", "path": "f", "bands": ["1"], "impossible": "yes"},
]
LABELS = {
    "h": [{"pair_id": "H:1", "label": "fabricated"}, {"pair_id": "H:2", "label": "valid"}],
    "s": [{"pair_id": "S:1", "label": "fabricated"}, {"pair_id": "S:2", "label": "valid"}],
    "f": [{"pair_id": "F:1", "label": "fabricated"}],
}
MENTIONS = {
    "H:1": {"a": 2, "b": 8},
    "H:2": {"a": 8, "b": 2},
    "S:1": {"a": 1},
    "S:2": {"a": 3},
    "S:3": {"a": 6, "b": 10},
    "F:1": {"a": 5},
}


def test_per_model_share_is_stratified_with_each_model_s_weights() -> None:
    strata = fabrication.build_strata(SPECS, FLAGS, LABELS, MENTIONS, ["a", "b"])
    table = fabrication.per_model(strata, ["a", "b"], 200, 1).set_index("model_id")
    # a: head 2/10 on 10 mentions, sample 1/4 on 10, flagged 5/5 on 5.
    expected_a = (10 * 0.2 + 10 * 0.25 + 5 * 1.0) / 25
    assert table.loc["a", "share_lower"] == pytest.approx(expected_a)
    assert table.loc["a", "share_upper"] == pytest.approx(expected_a)
    assert table.loc["a", "empty_strata"] == ""


def test_a_cell_with_no_labelled_mention_gives_bounds_at_zero_and_one() -> None:
    strata = fabrication.build_strata(SPECS, FLAGS, LABELS, MENTIONS, ["a", "b"])
    row = fabrication.per_model(strata, ["a", "b"], 200, 1).set_index("model_id").loc["b"]
    # b: head 8/10 on 10 mentions; 10 mentions of the sampled stratum are all on an unlabelled pair.
    assert row["share_lower"] == pytest.approx(10 * 0.8 / 20)
    assert row["share_upper"] == pytest.approx((10 * 0.8 + 10 * 1.0) / 20)
    assert row["empty_strata"] == "one"
    assert row["mentions_in_empty_strata"] == 10


def test_a_census_has_no_interval_of_its_own() -> None:
    specs = [SPECS[0]]
    strata = fabrication.build_strata(specs, FLAGS, LABELS, MENTIONS, ["a", "b"])
    row = fabrication.per_model(strata, ["a", "b"], 50, 1).iloc[0]
    assert row["ci_lower_low"] == pytest.approx(row["share_lower"])
    assert row["ci_lower_high"] == pytest.approx(row["share_lower"])


def test_per_model_over_all_models_equals_the_estimate_of_section_7_2() -> None:
    pooled = {pair: {"all": sum(m.values())} for pair, m in MENTIONS.items()}
    strata = fabrication.build_strata(SPECS, FLAGS, LABELS, pooled, ["all"])
    share = fabrication.per_model(strata, ["all"], 100, 1)["share_lower"].item()
    flags = [{**row, "responses": str(sum(MENTIONS[row["pair_id"]].values()))} for row in FLAGS]
    labelled = [
        unknowns.stratum("head", flags, LABELS["h"], ["5-19"]),
        unknowns.stratum("one", flags, LABELS["s"], ["1"], "no"),
        unknowns.stratum("flagged", flags, LABELS["f"], ["1"], "yes"),
    ]
    assert share == pytest.approx(unknowns.fabrication_estimate(labelled, 100, 1, 0.95)["share"])


def _fabrication_row(model: str, low: tuple[float, float], high: tuple[float, float]) -> dict:
    return {
        "model_id": model,
        "share_lower": 0.05,
        "ci_lower_low": low[0],
        "ci_lower_high": low[1],
        "share_upper": 0.09,
        "ci_upper_low": high[0],
        "ci_upper_high": high[1],
    }


def test_collapsed_intervals_finds_a_point_in_either_bound() -> None:
    table = pd.DataFrame(
        [
            _fabrication_row("both", (0.05, 0.05), (0.09, 0.09)),
            _fabrication_row("upper", (0.01, 0.05), (0.09, 0.09)),
            _fabrication_row("neither", (0.01, 0.05), (0.05, 0.09)),
        ]
    )
    assert fabrication.collapsed_intervals(table)["model_id"].tolist() == ["both", "upper"]


def test_notes_report_failed_replicates_and_ranges_only() -> None:
    boot = pd.DataFrame(
        {
            "hypothesis": ["cue_country_vs_city", "model_size"],
            "replicates": [2000, 2000],
            "failed": [0, 132],
        }
    )
    table = pd.DataFrame(
        [
            _fabrication_row("m", (0.05, 0.05), (0.09, 0.09)),
            _fabrication_row("n", (0.01, 0.05), (0.05, 0.09)),
        ]
    )
    notes = pipeline.results_notes(boot, table)
    assert "model_size: 132 of 2000 replicates did not converge and were excluded" in notes
    assert "cue_country_vs_city" not in notes
    assert "of m: report only the range 5.0-9.0%" in notes
    assert "of n" not in notes
    assert "bootstrap" not in pipeline.results_notes(None, table)


def test_manifest_outside_a_git_checkout_has_no_commit(tmp_path: Path) -> None:
    # An unpacked archive of the repository: analyze still writes its manifest.
    assert pipeline.git_state(tmp_path) == ("", False)


def test_manifest_names_inputs_relative_to_the_repository(tmp_path: Path) -> None:
    root, outside = tmp_path / "repo", tmp_path / "elsewhere"
    (root / "artifacts").mkdir(parents=True)
    outside.mkdir()
    (root / "artifacts" / "scores.parquet").write_bytes(b"a")
    (outside / "labels.csv").write_bytes(b"b")
    checksums = pipeline.input_checksums(
        [root / "artifacts" / "scores.parquet", outside / "labels.csv"], root
    )
    assert list(checksums) == ["artifacts/scores.parquet", "../elsewhere/labels.csv"]
    assert checksums["artifacts/scores.parquet"] == hashlib.sha256(b"a").hexdigest()


def _scored(model: str, error: bool = False) -> dict[str, Any]:
    return {
        "model_id": model,
        "prompt_id": "DE_de_city_T1a",
        "sample_idx": 0,
        "error": error,
        "blocked": None if error else 0,
        "empty": None if error else 0,
        "truncated": None if error else 0,
        "correct_local": None if error else 1,
    }


def test_scores_of_other_generations_are_out_of_date() -> None:
    current = [_scored("a"), _scored("b")]
    scores = {
        "same records, other ground truth": [{**_scored("b"), "correct_local": 0}, _scored("a")],
        "a model short (score alone after a new model)": [_scored("a")],
        "an error since retried": [_scored("a"), _scored("b", error=True)],
    }
    assert pipeline.out_of_date(scores, current) == [
        "a model short (score alone after a new model)",
        "an error since retried",
    ]


def test_analyze_stops_on_scores_of_other_generations(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def scores(name: str, *models: str) -> Path:
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist([_scored(m) for m in models]), path)
        return path

    paths = {
        "analysis": tmp_path / "analysis",
        "scores": scores("v2", "a", "new"),
        "scores_v1": scores("v1", "a"),
        "scores_prelabels": scores("prelabels", "a"),
    }
    config = {
        "paths": paths,
        "analysis": {"seed": 0, "interval_replicates": 0, "model_sizes": {}},
        "no_country_iso2": "XX",
    }
    with caplog.at_level("ERROR"):
        code = pipeline.analyze(config, {"official": scores("official", "a", "new")}, False)
    assert code == 1
    assert caplog.messages == [
        f"scores are out of date; run `crisisline reproduce` ({paths['scores_v1']}, "
        f"{paths['scores_prelabels']})"
    ]
    assert not paths["analysis"].exists()


def test_cells_report_each_model_s_counts() -> None:
    strata = fabrication.build_strata(SPECS, FLAGS, LABELS, MENTIONS, ["a", "b"])
    table = fabrication.cells(strata, ["a", "b"]).set_index(["model_id", "stratum"])
    assert table.loc[("a", "one"), "labelled_mentions"] == 4
    assert table.loc[("a", "one"), "fabricated_mentions"] == 1
    assert table.loc[("b", "one"), "mentions"] == 10
    assert math.isnan(table.loc[("b", "one"), "share"])


def test_snippet_marks_the_number_within_the_limit() -> None:
    text = "x" * 400 + " call 112 now " + "y" * 400
    made = examples.snippet(text, text.index("112"), "112", 300)
    assert "**112**" in made and len(made) <= 300
    assert len(examples.snippet("a\nb " * 200, None, "", 300)) <= 300


def test_choose_spreads_over_models_and_languages() -> None:
    rows = [{"model_id": m, "lang": lang} for m in "ab" for lang in "xyz" for _ in range(5)]
    chosen = examples.choose(rows, 6, 1)
    assert {r["model_id"]: 0 for r in chosen}.keys() == {"a", "b"}
    assert sorted(sum(r["model_id"] == m for r in chosen) for m in "ab") == [3, 3]
    assert sorted(sum(r["lang"] == lang for r in chosen) for lang in "xyz") == [2, 2, 2]
    assert examples.choose(rows, 6, 1) == chosen
    assert len(examples.choose(rows[:2], 10, 1)) == 2


def test_candidates_wait_for_a_human() -> None:
    number = {"number": "112", "raw": "112", "start": 5, "number_class": "fabricated"}
    data = pd.DataFrame(
        [
            {
                "prompt_id": "AA_ll_city_T1a",
                "model_id": "m",
                "sample_idx": 0,
                "lang": "ll",
                "numbers": [number],
            }
        ]
    )
    tests = {"fabricated": lambda r: (True, r["numbers"][0])}
    row = examples.candidates(
        data, {("m", "AA_ll_city_T1a", 0): "Call 112."}, {}, tests, 5, 300, 1
    )[0]
    assert row["snippet"] == "Call **112**."
    assert row["snippet_en"] is None and row["snippet_en_status"].startswith("pending")
    assert row["safety_review"] == "pending"


def test_example_classes_follow_the_plan() -> None:
    groups = {"AA": "es", "BB": "es", "CC": ""}
    tests = examples.classes(groups, ["988"], lambda _: False)
    base = {
        "iso2": "AA",
        "numbers": [],
        "na_default": 0.0,
        "wrong_country": 0.0,
        "directory_only": 0.0,
        "no_resource": 0.0,
        "blocked": 0.0,
        "aborted_loop": False,
        "cue": "city",
        "correct_local": 0.0,
    }
    other = {"number": "1", "number_class": "other", "other_iso2": ["BB"]}
    far = {"number": "1", "number_class": "other", "other_iso2": ["CC"]}
    same = tests["wrong_country_same_language_group"]
    assert same({**base, "wrong_country": 1.0, "numbers": [other]})[0]
    assert not same({**base, "wrong_country": 1.0, "numbers": [far]})[0]
    assert tests["refusal"]({**base, "blocked": 1.0})[0]
    correct = {"number": "1", "number_class": "local:crisis"}
    city = tests["positive_correct_from_city_cue"]
    assert city({**base, "correct_local": 1.0, "numbers": [correct]})[0]
    assert not city({**base, "cue": "country", "correct_local": 1.0, "numbers": [correct]})[0]


def _pdf_size_cm(path: Path) -> tuple[float, float]:
    box = re.search(rb"/MediaBox \[ ?0 0 ([\d.]+) ([\d.]+)", path.read_bytes())
    assert box is not None
    return float(box[1]) / 72 * 2.54, float(box[2]) / 72 * 2.54


def test_figures_are_vector_pdf_at_acl_widths(tmp_path: Path) -> None:
    ladder = pd.DataFrame(
        {
            "model_id": ["q-1", "q-2"] * 2,
            "cue": ["city", "city", "country", "country"],
            "share": [0.2, 0.3, 0.4, 0.5],
            "ci_low": [0.1, 0.2, 0.3, 0.4],
            "ci_high": [0.3, 0.4, 0.5, 0.6],
        }
    )
    figures.qwen_ladder(ladder, {"q-1": 1.0, "q-2": 2.0}, tmp_path / "l.pdf")
    assert _pdf_size_cm(tmp_path / "l.pdf")[0] == pytest.approx(7.7, abs=0.01)
    types = pd.DataFrame(
        {
            "model_id": ["m", "m", "n"],
            "error_type": ["correct number", "no resource", "correct number"],
            "share": [0.5, 0.5, 1.0],
        }
    )
    figures.error_types_by_model(types, tmp_path / "e.pdf")
    assert _pdf_size_cm(tmp_path / "e.pdf")[0] == pytest.approx(16.0, abs=0.01)
    assert figures.mpl.rcParams["pdf.fonttype"] == 42
    assert min(figures.mpl.rcParams[k] for k in ("font.size", "legend.fontsize")) >= 8
