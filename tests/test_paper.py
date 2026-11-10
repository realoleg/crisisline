"""Tests of ``crisisline paper`` on small analysis tables with known content."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd
import pytest
from matplotlib.text import Text

from crisisline.analysis.descriptive import REFERRAL_CLASSES
from crisisline.paper import figures, style, subsets, tables

GROUPS = {"API": ["api-b", "api-a"], "local": ["q-2", "other", "q-1"]}
SIZES = {"q-1": 1.0, "q-2": 2.0}
STYLES = style.model_styles(GROUPS, SIZES)
GROUP_STYLES = style.group_styles(list(GROUPS))
MODELS = list(STYLES)
SCOPE = subsets.Scope.from_rows(
    [
        {
            "iso2": "AA",
            "name_en": "Aland",
            "who_region": "AFR",
            "lang": "aa",
            "has_crisis_line": "true",
        },
        {
            "iso2": "BB",
            "name_en": "Bland",
            "who_region": "AFR",
            "lang": "en",
            "has_crisis_line": "true",
        },
        {
            "iso2": "CC",
            "name_en": "Cland",
            "who_region": "EUR",
            "lang": "cc",
            "has_crisis_line": "false",
        },
    ],
    temperature_block=["AA", "CC"],
    grounding_excluded=["api-a"],
)


def _ci(estimate: float) -> dict[str, float]:
    return {
        "estimate": estimate,
        "ci_low": max(0.0, estimate - 0.05),
        "ci_high": min(1.0, estimate + 0.05),
    }


def test_models_are_ordered_by_group_with_the_size_ladder_by_size() -> None:
    assert MODELS == ["api-b", "api-a", "q-1", "q-2", "other"]
    assert {STYLES[m].colour for m in GROUPS["API"]} == {style.GROUP_COLOURS[0]}
    assert {STYLES[m].colour for m in GROUPS["local"]} == {style.GROUP_COLOURS[1]}
    assert [STYLES[m].marker for m in ("q-1", "q-2", "other")] == ["o", "s", "^"]
    assert GROUP_STYLES["local"] == (style.GROUP_COLOURS[1], style.GROUP_MARKERS[1])


def test_referral_colours_follow_the_outcome_semantics() -> None:
    colours = {name: colour for name, (_, colour) in style.REFERRAL.items()}
    assert list(colours) == list(REFERRAL_CLASSES)
    assert colours == {
        "correct_crisis_line": "#02692b",
        "correct_local_not_crisis_line": "#4e88cb",
        "misrouted": "#dd8e00",
        "fabricated": "#b4534b",
        "unconfirmed_number": "#D9D9D9",
        "nothing": "#7A7A7A",
    }


# Figure data.


def _referral() -> pd.DataFrame:
    correct = {"api-b": 0.9, "api-a": 0.95, "q-1": 0.3, "q-2": 0.6, "other": 0.5}
    rows = []
    for model, share in correct.items():
        parts = [share * 0.6, share * 0.4, 0.1, 0.05, 0.05, 0.8 - share, share]
        rows += [
            {"model_id": model, "referral": name, "share": p, "ci_low": p, "ci_high": p}
            for name, p in zip([*REFERRAL_CLASSES, "correct_local"], parts, strict=True)
        ]
    return pd.DataFrame(rows)


def test_referral_data_sorts_models_by_correct_share_within_their_group() -> None:
    data = figures.referral_data(_referral(), STYLES, SCOPE)
    assert list(data["model_id"].unique()) == ["api-a", "api-b", "q-2", "other", "q-1"]
    model = data[data["model_id"] == "q-2"].set_index("referral")
    assert list(model.index) == [*REFERRAL_CLASSES, "correct_local"]
    bars = model.loc[list(REFERRAL_CLASSES)]
    assert bars["left"].tolist() == pytest.approx([0, *bars["share"].cumsum().iloc[:-1].tolist()])
    assert model.loc["correct_local", "share"] == pytest.approx(
        bars.loc[["correct_crisis_line", "correct_local_not_crisis_line"], "share"].sum()
    )
    assert set(data["subset"]) == {
        "prompts about a country (S0 with a city or country cue, M1, M2), 2 countries with a "
        "crisis line, all prompt languages"
    }


def _grounding() -> pd.DataFrame:
    rows = []
    for group in [*GROUPS, figures.ALL]:
        for measure in [*figures.CONDITIONS, "M2 - M1"]:
            for outcome in ("correct_local", "correct_crisis", "fabricated"):
                rows.append(
                    {
                        "model_group": group,
                        "model_id": figures.ALL,
                        "measure": measure,
                        "outcome": outcome,
                        **_ci(0.5),
                    }
                )
        rows.append(
            {
                "model_group": group,
                "model_id": GROUPS.get(group, ["x"])[0],
                "measure": "M1",
                "outcome": "fabricated",
                **_ci(0.1),
            }
        )
    return pd.DataFrame(rows)


def test_grounding_data_keeps_the_groups_conditions_and_two_outcomes() -> None:
    data = figures.grounding_data(_grounding(), list(GROUPS), SCOPE)
    assert len(data) == 2 * len(GROUPS) * len(figures.CONDITIONS)
    assert set(data["model_group"]) == set(GROUPS)
    assert data["condition"].tolist()[:3] == list(figures.CONDITIONS)
    assert data["outcome"].iloc[0] == "correct_crisis"


def _language_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    by_model = pd.DataFrame(
        [
            {
                "model_id": m,
                **{
                    f"share_{lang}{suffix}": value + shift
                    for lang, value in (("en", 0.7), ("local", 0.5))
                    for suffix, shift in (("", 0.0), ("_ci_low", -0.1), ("_ci_high", 0.1))
                },
                "difference_en_minus_local": 0.2,
                # The gap of q-1 excludes zero; that of every other model does not.
                "difference_ci_low": 0.05 if m == "q-1" else -0.1,
                "difference_ci_high": 0.3,
            }
            for m in MODELS
        ]
    )
    pooled = {"am": 0.2, "de": 0.9, "si": 0.4}
    by_language = pd.DataFrame(
        [
            {"model_group": group, "lang": lang, "countries": 1, **_ci(value)}
            for lang, value in pooled.items()
            for group in [figures.ALL, *GROUPS]
        ]
    )
    return by_model, by_language


def test_language_data_orders_languages_from_the_lowest_pooled_share() -> None:
    data = figures.language_data(*_language_inputs(), STYLES, list(GROUPS), SCOPE)
    models = data[data["panel"] == "by model"]
    assert list(models["model_id"].unique()) == MODELS
    assert set(models["prompt_language"]) == {"en", "local"}
    langs = data[data["panel"] == "by language"]
    assert list(langs["lang"].unique()) == ["am", "si", "de"]
    assert figures.ALL not in set(langs["model_group"])


def _geography_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = {"AFR": 0.5, "EUR": 0.9, "SEAR": 0.6}
    regions = pd.DataFrame(
        [
            {"model_group": group, "who_region": region, "outcome": outcome, **_ci(value)}
            for region, value in pooled.items()
            for group in [figures.ALL, *GROUPS]
            for outcome in ("correct_local", "correct_crisis")
        ]
    )
    worst = pd.DataFrame(
        [
            {
                "iso2": iso2,
                "name_en": name,
                "who_region": "AFR",
                "correct_local estimate (all)": local,
                "correct_local ci_low (all)": local - 0.05,
                "correct_local ci_high (all)": local + 0.05,
                "correct_crisis estimate (all)": crisis,
                "correct_crisis ci_low (all)": crisis - 0.05,
                "correct_crisis ci_high (all)": crisis + 0.05,
            }
            for iso2, name, local, crisis in (
                ("ET", "Ethiopia", 0.2, math.nan),
                ("LK", "Sri Lanka", 0.4, 0.3),
            )
        ]
    )
    return regions, worst


def test_geography_data_orders_regions_and_skips_countries_without_a_crisis_line() -> None:
    data = figures.geography_data(*_geography_inputs(), list(GROUPS), SCOPE)
    regions = data[data["panel"] == "region"]
    assert list(regions["who_region"].unique()) == ["AFR", "SEAR", "EUR"]
    assert set(regions["model_group"]) == set(GROUPS)
    countries = data[data["panel"] == "country"]
    assert countries[countries["outcome"] == "correct_crisis"]["iso2"].tolist() == ["LK"]
    assert countries[countries["outcome"] == "correct_local"]["iso2"].tolist() == ["ET", "LK"]


def _ladder() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model_id": m, "cue": cue, "outcome": "correct_local", **_ci(SIZES[m] / 4)}
            for m in ("q-2", "q-1")
            for cue in ("city", "country")
        ]
    ).rename(columns={"estimate": "share"})


def test_ladder_data_is_by_cue_then_size() -> None:
    data = figures.ladder_data(_ladder(), SIZES, SCOPE)
    assert data[["cue", "size"]].values.tolist() == [
        ["country", 1.0],
        ["country", 2.0],
        ["city", 1.0],
        ["city", 2.0],
    ]


def _countries() -> pd.DataFrame:
    pooled = {("AFR", "ET"): 0.2, ("AFR", "NG"): 0.6, ("EUR", "DE"): 0.9, ("EUR", "FR"): 0.95}
    return pd.DataFrame(
        [
            {
                "model_group": group,
                "who_region": region,
                "iso2": iso2,
                "name_en": iso2,
                "outcome": outcome,
                **_ci(value),
            }
            for (region, iso2), value in pooled.items()
            for group in [figures.ALL, *GROUPS]
            for outcome in ("correct_local", "correct_crisis")
        ]
    )


def test_countries_data_sorts_within_region_from_the_highest_pooled_share() -> None:
    data = figures.countries_data(_countries(), list(GROUPS))
    assert list(data["iso2"].unique()) == ["NG", "ET", "FR", "DE"]
    assert set(data["model_group"]) == set(GROUPS)


# Drawing.


def _who_inputs(
    lowest: int = 1,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, style.ModelStyle], subsets.Scope, int
]:
    shares = {"api-b": 0.9, "api-a": 0.95, "q-1": 0.1, "q-2": 0.3, "other": 0.2}
    cells = pd.DataFrame(
        [
            {"model_id": m, "iso2": iso2, "responses": 24, **_ci(v)}
            for m, v in shares.items()
            for iso2 in ("AA", "BB")
        ]
    ).rename(columns={"estimate": "share"})
    margins = cells.groupby("model_id", as_index=False).agg(
        share=("share", "mean"),
        ci_low=("ci_low", "mean"),
        ci_high=("ci_high", "mean"),
        responses=("responses", "sum"),
    )
    pooled = {
        (figures.ALL, "AA"): 0.6,
        (figures.ALL, "BB"): 0.5,
        ("API", "AA"): 0.9,
        ("API", "BB"): 0.9,
        ("local", "AA"): 0.0,
        ("local", "BB"): 0.0,
    }
    countries = pd.DataFrame(
        [
            {
                "model_group": group,
                "iso2": iso2,
                "outcome": outcome,
                "responses": 48,
                **_ci(value if outcome == "correct_crisis" else 1 - value),
            }
            for (group, iso2), value in pooled.items()
            for outcome in ("correct_crisis", "correct_local")
        ]
    )
    return cells, margins, countries, STYLES, SCOPE, lowest


def _fails() -> pd.DataFrame:
    referral = figures.referral_data(_referral(), STYLES, SCOPE)
    fabrication = pd.DataFrame(
        [
            {
                "kind": "model",
                "domain": m,
                "measure": "numbers_fabricated",
                "bound": bound,
                "value": value,
                "ci_low": value - 0.01,
                "ci_high": value + 0.01,
                "range_only": m == "api-a",
            }
            for m, (lower, upper) in {
                "api-a": (0.01, 0.02),
                "api-b": (0.05, 0.05),
                "q-1": (0.5, 0.5),
                "q-2": (0.3, 0.32),
                "other": (0.2, 0.2),
            }.items()
            for bound, value in (("lower", lower), ("upper", upper))
        ]
    )
    return figures.how_it_fails_data(referral, fabrication, "reader subset")


def _grounding_with_steps() -> pd.DataFrame:
    """The grounding table with the step from S0-country to M1 (it has M2 - M1 already)."""
    steps = pd.DataFrame(
        [
            {
                "model_group": figures.ALL,
                "model_id": figures.ALL,
                "measure": measure,
                "outcome": "correct_crisis",
                "estimate": value,
                "ci_low": value - 0.05,
                "ci_high": value + 0.05,
            }
            for measure, value in (("M1 - S0-country", 0.0),)
        ]
    )
    return pd.concat([_grounding(), steps], ignore_index=True)


def _drawings() -> dict[str, tuple[Callable[[Path], None], float]]:
    referral = figures.referral_data(_referral(), STYLES, SCOPE)
    grounding = figures.grounding_data(_grounding(), list(GROUPS), SCOPE)
    language = figures.language_data(*_language_inputs(), STYLES, list(GROUPS), SCOPE)
    geography = figures.geography_data(*_geography_inputs(), list(GROUPS), SCOPE)
    ladder = figures.ladder_data(_ladder(), SIZES, SCOPE)
    countries = figures.countries_data(_countries(), list(GROUPS))
    return {
        "referral": (lambda p: figures.draw_referral(referral, p), style.FULL),
        "grounding": (lambda p: figures.draw_grounding(grounding, GROUP_STYLES, p), style.FULL),
        "language": (
            lambda p: figures.draw_language(language, STYLES, GROUP_STYLES, p),
            style.FULL,
        ),
        "geography": (lambda p: figures.draw_geography(geography, GROUP_STYLES, p), style.FULL),
        "ladder": (lambda p: figures.draw_ladder(ladder, STYLES["q-1"].colour, p), style.COLUMN),
        "countries": (
            lambda p: figures.draw_countries(countries, GROUP_STYLES, p),
            style.FULL,
        ),
        "who_gets_help": (
            lambda p: figures.draw_who_gets_help(figures.who_gets_help_data(*_who_inputs()), p),
            style.FULL,
        ),
        "how_it_fails": (
            lambda p: figures.draw_how_it_fails(_fails(), p),
            style.FULL,
        ),
        "native_language": (
            lambda p: figures.draw_native_language(
                figures.native_language_data(_language_inputs()[0], STYLES, SCOPE),
                STYLES,
                GROUP_STYLES,
                p,
            ),
            style.COLUMN,
        ),
        "what_fixes": (
            lambda p: figures.draw_what_fixes(
                figures.what_fixes_data(_grounding_with_steps(), list(GROUPS), SCOPE),
                GROUP_STYLES,
                p,
            ),
            style.COLUMN,
        ),
    }


@pytest.mark.parametrize("name", list(_drawings()))
def test_figures_are_pdf_and_png_at_the_printed_width_and_rebuild_byte_identical(
    name: str, tmp_path: Path
) -> None:
    draw, width = _drawings()[name]
    draw(tmp_path / "a" / name)
    draw(tmp_path / "b" / name)
    pdf = (tmp_path / "a" / f"{name}.pdf").read_bytes()
    png = tmp_path / "a" / f"{name}.png"
    assert pdf == (tmp_path / "b" / f"{name}.pdf").read_bytes()
    assert png.read_bytes() == (tmp_path / "b" / f"{name}.png").read_bytes()
    box = re.search(rb"/MediaBox \[ ?0 0 ([\d.]+) ([\d.]+)", pdf)
    assert box is not None
    assert float(box[1]) / 72 == pytest.approx(width, abs=1e-3)
    assert b"CreationDate" not in pdf
    assert b"STIXGeneral" in pdf
    assert mpimg.imread(png).shape[1] == round(width * style.RC["savefig.dpi"])


def test_style_sizes_meet_the_template() -> None:
    sizes = [style.RC[k] for k in ("font.size", "xtick.labelsize", "legend.fontsize")]
    assert sizes == [9, 8.5, 8.5]
    assert style.RC["font.serif"] == ["STIXGeneral"]
    assert style.RC["mathtext.fontset"] == "stix"


# Tables.


def test_models_table_reads_family_size_access_and_sampling() -> None:
    analysis_models = pd.DataFrame(
        [
            {
                "model_id": "q-1",
                "backend": "ollama",
                "identifier": "qwen3.5:0.8b-q8_0",
                "responses_per_prompt": 1,
                "records": 10,
                "versions_seen": 1,
            },
            {
                "model_id": "g",
                "backend": "ollama",
                "identifier": "gemma4:e2b-it-q8_0",
                "responses_per_prompt": 1,
                "records": 10,
                "versions_seen": 1,
            },
            {
                "model_id": "api-a",
                "backend": "openrouter",
                "identifier": "vendor/model-x",
                "responses_per_prompt": 3,
                "records": 30,
                "versions_seen": 2,
            },
        ]
    )
    config = {
        "models": {
            "q-1": {"backend": "ollama"},
            "g": {"backend": "ollama"},
            "api-a": {"backend": "openrouter"},
        },
        "generation": {
            "ollama": {"params": {"temperature": 0, "top_p": 1, "seed": 7}, "max_tokens": 9},
            "openrouter": {"max_tokens": 9, "reasoning": {"effort": "low"}},
        },
    }
    styles = style.model_styles({"API": ["api-a"], "local": ["q-1", "g"]}, {"q-1": 0.8})
    table = tables.models_table(analysis_models, config, styles)
    value = table.set_index(["model_id", "field"])["value"]
    assert value[("q-1", "family")] == "qwen3.5"
    assert value[("q-1", "size")] == "0.8B"
    assert value[("g", "size")] == "E2B"
    assert value[("q-1", "quantization")] == "q8_0"
    assert value[("q-1", "seed")] == 7
    assert value[("api-a", "family")] == "vendor"
    assert value[("api-a", "size")] == "undisclosed"
    assert value[("api-a", "access")] == "API"
    assert value[("api-a", "temperature")] == "provider default"
    assert value[("api-a", "reasoning_effort")] == "low"
    assert ("api-a", "quantization") not in value
    assert ("q-1", "reasoning_effort") not in value


def test_fabrication_table_flags_collapsed_intervals_as_range_only() -> None:
    bounds = {
        f"{prefix}_{bound}{suffix}": 0.1
        for prefix in ("of_numbers", "of_responses")
        for bound in ("lower", "upper")
        for suffix in ("", "_ci_low", "_ci_high")
    }
    reader = pd.DataFrame(
        [
            {"kind": "model", "domain": m, "numbers": 10, "responses": 5, **bounds}
            for m in ("tight", "wide")
        ]
    )
    by_model = pd.DataFrame(
        [
            {
                "model_id": "tight",
                "share_lower": 0.1,
                "ci_lower_low": 0.1,
                "ci_lower_high": 0.1,
                "share_upper": 0.2,
                "ci_upper_low": 0.2,
                "ci_upper_high": 0.2,
                "mentions": 4,
            },
            {
                "model_id": "wide",
                "share_lower": 0.1,
                "ci_lower_low": 0.0,
                "ci_lower_high": 0.3,
                "share_upper": 0.1,
                "ci_upper_low": 0.0,
                "ci_upper_high": 0.3,
                "mentions": 40,
            },
        ]
    )
    near_miss = pd.DataFrame(
        [
            {
                "model_group": "API",
                "model_id": "tight",
                "mentions": 4,
                "near_miss": 0.05,
                "near_miss_ci_low": 0.01,
                "near_miss_ci_high": 0.1,
                "chance": 1e-6,
                "chance_ci_low": 1e-7,
                "chance_ci_high": 1e-5,
            }
        ]
    )
    table = tables.fabrication_table(
        reader, {"share": 0.5, "interval": [0.4, 0.6], "mentions": 44}, by_model, near_miss, 7
    )
    flagged = table[table["range_only"]]
    assert set(flagged["domain"]) == {"tight"}
    assert set(flagged["measure"]) == {
        "numbers_fabricated",
        "responses_with_a_fabricated_number",
        "unknown_number_mentions_fabricated",
    }
    pooled = table[(table["kind"] == "all") & (table["measure"] == tables.UNKNOWN_MENTIONS)]
    assert pooled[["value", "ci_low", "ci_high", "n"]].drop_duplicates().values.tolist() == [
        [0.5, 0.4, 0.6, 44]
    ]
    assert table["subset"].ne("").all()
    rates = table[table["measure"].str.startswith("near_miss")].set_index("measure")
    assert rates.loc["near_miss_by_chance", "value"] == 1e-6
    assert not rates["range_only"].any()


def test_no_location_table_counts_questions_and_the_commonest_country() -> None:
    us_default = pd.Series({"estimate": 0.9, "ci_low": 0.8, "ci_high": 0.95, "n": 50})
    breakdown = pd.DataFrame(
        [
            {
                "by": "template",
                "level": "T1a",
                "hits": 9,
                "n": 10,
                "share": 0.9,
                "ci_low": 0.6,
                "ci_high": 1.0,
            }
        ]
    )
    asks = pd.DataFrame(
        [
            {"model_id": "a", "no_resource": 4, "asks_location": 3, "nothing": 1},
            {"model_id": "b", "no_resource": 0, "asks_location": 0, "nothing": 0},
        ]
    )
    cells = pd.DataFrame(
        [
            {
                "model_id": m,
                "condition": condition,
                "lang_mode": mode,
                "outcome": outcome,
                "responses": 5,
            }
            for m in ("a", "b")
            for condition, mode in (("S0", "none"), ("S0", "local"), ("M1", "local"))
            for outcome in ("blocked", "no_resource")
            for _ in range(2)  # two templates
        ]
    )
    defaults = pd.DataFrame(
        [
            {"lang": "de", "country": "AT", "responses": 3, "language_responses": 10},
            {"lang": "de", "country": "DE", "responses": 6, "language_responses": 10},
            {"lang": "en", "country": "US", "responses": 4, "language_responses": 10},
            {"lang": "en", "country": "CA", "responses": 4, "language_responses": 10},
        ]
    )
    table = tables.no_location_table(
        us_default, breakdown, asks, cells, defaults, ["a", "b"], SCOPE
    )
    asked = table[table["measure"] == "asks_where_the_user_is"].set_index("level")
    assert asked.loc["a", "value"] == pytest.approx(3 / 10)
    assert asked.loc[figures.ALL, ["hits", "n"]].tolist() == [3, 20]
    assert asked.loc["b", "ci_low"] == 0.0
    top = table[table["part"] == "default_country"].set_index("level")
    assert top.loc["de", "country"] == "DE"
    assert top.loc["de", "value"] == pytest.approx(0.6)
    assert top.loc["en", "country"].tolist() == ["CA", "US"]
    us = table[table["part"] == "north_american_number"]
    assert us["level"].tolist() == [figures.ALL, "T1a"]


def test_sensitivity_table_keeps_the_comparisons_without_the_extraction_rows() -> None:
    hypotheses = pd.DataFrame(
        [
            {"analysis": analysis, "hypothesis": h}
            for analysis, h in (
                ("v2", "cue_country_vs_city"),
                ("v1, Exchangeable", "language_en_vs_local"),
                ("without GE", "cue_country_vs_city"),
                ("v2", "cue_country_vs_city at local"),
                ("v2", "us_default secondary"),
                ("main block, prefix", "us_default"),
            )
        ]
    )
    for column in (
        "effect",
        "estimate",
        "ci_low",
        "ci_high",
        "p",
        "p_holm",
        "side",
        "n",
        "clusters",
        "working_structure",
        "converged",
        "note",
    ):
        hypotheses[column] = None
    table = tables.sensitivity_table(hypotheses, SCOPE)
    assert table["analysis"].tolist() == ["v2", "v1, Exchangeable", "without GE"]
    assert table["comparison"].iloc[0] == tables.COMPARISONS["cue_country_vs_city"]
    assert not table.columns.str.contains("hypothes").any()


def test_numbers_are_written_to_six_significant_digits() -> None:
    assert tables._number("x", 1 / 3, "proportion", "s", "")["value"] == 0.333333
    assert tables._number("x", "ET", "country", "s", "")["value"] == "ET"


def test_names_of_numbers_have_no_spaces_or_slashes() -> None:
    assert tables._name("georgia", "S0/country/en", "correct_local") == (
        "georgia_S0_country_en_correct_local"
    )
    assert tables._name("fabricated", "M2 - S0-country", "other local") == (
        "fabricated_M2_minus_S0-country_other_local"
    )


# Subsets.


def test_counts_leave_out_countries_without_a_crisis_line_for_crisis_outcomes() -> None:
    assert SCOPE.count() == "3 countries"
    assert SCOPE.count("correct_local") == "3 countries"
    assert SCOPE.count("correct_crisis") == "2 countries with a crisis line"
    assert SCOPE.count(non_english=True) == "2 countries"
    assert SCOPE.count("correct_crisis", within=SCOPE.temperature_block) == (
        "1 country with a crisis line"
    )
    assert SCOPE.count(region="AFR", without=["AA"]) == "1 country"
    assert SCOPE.languages == 3


def test_comparison_subsets_follow_the_sensitivity_analysis() -> None:
    assert subsets.comparison(SCOPE, "cue_country_vs_city", "v2").endswith("3 countries")
    assert subsets.comparison(SCOPE, "cue_country_vs_city", "without CC").endswith("2 countries")
    crisis = subsets.comparison(SCOPE, "model_size", "only countries with a crisis line")
    assert crisis.endswith("2 countries with a crisis line")
    assert subsets.comparison(SCOPE, "grounding_verified_number", "v2").endswith("without api-a")
    assert "api-a" not in subsets.comparison(
        SCOPE, "grounding_verified_number", "v2, Exchangeable, all models"
    )
    assert subsets.comparison(SCOPE, "us_default", "without truncated").endswith(
        "without truncated responses"
    )
    with pytest.raises(ValueError, match="no subset"):
        subsets.comparison(SCOPE, "unknown", "v2")


def test_every_figure_row_names_its_subset_and_crisis_rows_count_crisis_line_countries() -> None:
    frames = [
        figures.referral_data(_referral(), STYLES, SCOPE),
        figures.grounding_data(_grounding(), list(GROUPS), SCOPE),
        figures.language_data(*_language_inputs(), STYLES, list(GROUPS), SCOPE),
        figures.geography_data(*_geography_inputs(), list(GROUPS), SCOPE),
        figures.ladder_data(_ladder(), SIZES, SCOPE),
        figures.countries_data(_countries(), list(GROUPS)),
    ]
    for data in frames:
        assert data["subset"].str.len().gt(0).all()
    grounding, geography = frames[1], frames[3]
    for data in (grounding, geography[geography["panel"] == "region"]):
        crisis = data["outcome"] == "correct_crisis"
        assert data.loc[crisis, "subset"].str.contains("with a crisis line").all()
        assert not data.loc[~crisis, "subset"].str.contains("crisis line").any()


def test_template_contrasts_have_subsets_and_unknown_ones_fail() -> None:
    assert "minus" in subsets.contrast(SCOPE, "M2 - S0-country, local", "T1a", "correct_crisis")
    assert "2 countries with a crisis line" in subsets.contrast(
        SCOPE, "API - Gemma, S0", "all", "correct_crisis"
    )
    with pytest.raises(ValueError, match="no subset"):
        subsets.contrast(SCOPE, "something else", "T1a", "correct_local")


def _template_inputs(step: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    levels = pd.DataFrame(
        [
            {
                "template": "all",
                "model_group": "API",
                "outcome": "fabricated",
                "responses": 9,
                **_ci(0.1),
            }
        ]
    )
    contrasts = pd.DataFrame(
        [
            {
                "template": template,
                "model_group": "API",
                "contrast": "M2 - S0-country, local",
                "outcome": "fabricated",
                "responses": 9,
                "flag": None,
                "estimate": -0.05,
                "ci_low": -0.09,
                "ci_high": -0.01,
            }
            for template in ("T1a", "all")
        ]
    )
    grounding = pd.DataFrame(
        [
            {
                "model_group": "API",
                "model_id": figures.ALL,
                "measure": "M2 - S0-country",
                "outcome": "fabricated",
                "estimate": step,
                "ci_low": -0.08,
                "ci_high": -0.02,
            }
        ]
    )
    return levels, contrasts, grounding


def test_pooled_template_contrasts_take_the_interval_of_the_grounding_step() -> None:
    table = tables.templates_table(*_template_inputs(-0.05), SCOPE)
    pooled = table[(table["template"] == "all") & (table["contrast"] != "")]
    assert pooled[["value", "ci_low", "ci_high"]].values.tolist() == [[-0.05, -0.08, -0.02]]
    within = table[table["template"] == "T1a"]
    assert within[["ci_low", "ci_high"]].values.tolist() == [[-0.09, -0.01]]
    assert pooled["subset"].iloc[0] == subsets.condition(SCOPE, "M2 - S0-country", "fabricated")
    assert within["subset"].iloc[0].endswith(", template T1a")
    with pytest.raises(ValueError, match="differ"):
        tables.templates_table(*_template_inputs(-0.04), SCOPE)


# The redesigned figures.


def test_display_names_cover_unknown_models_and_label_columns() -> None:
    assert style.model_name("qwen3.5-0.8b") == "Qwen3.5 0.8B"
    assert style.model_name("local-8") == "Ministral 3 8B"
    assert style.model_name("new-model") == "new-model"
    data = figures.referral_data(_referral(), STYLES, SCOPE)
    assert (data["model"] == data["model_id"]).all()  # the test models have no display name


def test_spread_keeps_order_and_minimum_gap() -> None:
    placed = style.spread([10.0, 11.0, 50.0], 4.0)
    assert placed[1] - placed[0] == pytest.approx(4.0)
    assert placed[2] > placed[1]
    assert style.spread([1.0, 20.0], 4.0) == [1.0, 20.0]


def test_who_gets_help_orders_models_by_strength_and_marks_tied_lowest_countries() -> None:
    data = figures.who_gets_help_data(*_who_inputs())
    cells = data[data["panel"] == "cell"]
    assert list(dict.fromkeys(cells["model_id"])) == ["api-a", "api-b", "q-2", "other", "q-1"]
    # AFR before EUR; within AFR from the highest all-model share; no line last.
    assert list(dict.fromkeys(cells["iso2"])) == ["AA", "BB", "CC"]
    no_line = cells[cells["iso2"] == "CC"]
    assert no_line["share"].isna().all() and not no_line["has_crisis_line"].any()
    # One lowest country asked for, two tied at zero: both are marked.
    lowest = data[data["panel"] == "lowest"]
    assert set(lowest["iso2"]) == {"AA", "BB"} and set(lowest["model_group"]) == {"local"}
    margin = data[data["panel"] == "country share"].set_index("iso2")
    assert margin.loc["AA", "share"] == 0.6 and pd.isna(margin.loc["CC", "share"])
    assert data["n"].notna().sum() == len(cells) - len(no_line) + 5 + 2 + 2


def test_how_it_fails_stacks_the_failures_from_zero_and_totals_them() -> None:
    data = _fails()
    failures = data[(data["panel"] == "failures") & (data["model_id"] == "q-2")].set_index(
        "measure"
    )
    assert list(failures.index) == [*figures.FAILURES, "correct_local", "failure_total"]
    stacked = failures.loc[list(figures.FAILURES)]
    assert stacked["left"].tolist() == pytest.approx(
        [0, *stacked["value"].cumsum().iloc[:-1].tolist()]
    )
    assert failures.loc["failure_total", "value"] == pytest.approx(
        1 - failures.loc["correct_local", "value"]
    )
    numbers = data[data["panel"] == "numbers fabricated"].set_index("model_id")
    assert numbers.loc["q-2", ["value", "upper"]].tolist() == [0.3, 0.32]
    assert numbers.loc["api-a", "range_only"]
    assert list(dict.fromkeys(data["model_id"])) == list(
        dict.fromkeys(figures.referral_data(_referral(), STYLES, SCOPE)["model_id"])
    )


def test_native_language_flags_only_gaps_whose_interval_excludes_zero() -> None:
    data = figures.native_language_data(_language_inputs()[0], STYLES, SCOPE)
    flagged = set(data.loc[data["significant_gap"], "model_id"])
    assert flagged == {"q-1"}


def test_what_fixes_has_the_lines_and_the_two_steps_over_all_models() -> None:
    data = figures.what_fixes_data(_grounding_with_steps(), list(GROUPS), SCOPE)
    assert set(data.loc[data["panel"] == "line", "outcome"]) == {"correct_crisis"}
    steps = data[data["panel"] == "step"]
    assert steps["condition"].tolist() == list(figures.STEPS)
    assert steps["subset"].str.contains("with a crisis line").all()


def test_figure_numbers_hold_what_the_new_figures_write() -> None:
    who = figures.who_gets_help_data(*_who_inputs())
    native = figures.native_language_data(_language_inputs()[0], STYLES, SCOPE)
    fixes = figures.what_fixes_data(_grounding_with_steps(), list(GROUPS), SCOPE)
    numbers = tables.figure_numbers(who, _fails(), native, fixes).set_index("name")
    assert numbers.loc["numbers_invented_one_in", "value"] == 2
    assert numbers.loc["lowest_countries_crisis_line_local", "value"] == 2
    assert {"en_prompt_q-1", "local_prompt_q-1", "en_minus_local_q-1"} <= set(numbers.index)
    assert "en_minus_local_q-2" not in numbers.index
    assert numbers.loc["correct_crisis_M2_minus_M1_all", "value"] == 0.5
    assert numbers.index.is_unique
    assert numbers["subset"].str.len().gt(0).all()


def test_saved_text_is_single_spaced(tmp_path: Path) -> None:
    with style.paper_style():
        fig = style.new_figure(style.COLUMN, 1.5)
        ax = fig.add_subplot()
        ax.set_xticks([0, 1], ["two\nlines", "one"])
        style.save(fig, tmp_path / "f")
        multi_line = [t for t in fig.findobj(Text) if "\n" in t.get_text()]
        assert multi_line and {t.get_linespacing() for t in multi_line} == {1.0}
