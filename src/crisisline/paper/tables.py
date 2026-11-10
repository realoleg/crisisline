"""The paper's tables and the numbers its text cites, as tidy frames: one row per value, with its
95% interval where it has one. Everything is taken from the analysis tables; the only new
intervals are Wilson intervals of counts the analysis reports without one."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from crisisline.analysis import fabrication
from crisisline.analysis.descriptive import CONFIDENCE
from crisisline.paper.figures import ALL, REGIONS
from crisisline.paper.style import ModelStyle
from crisisline.unknowns import wilson_interval

COMPARISONS = {
    "cue_country_vs_city": "Country named vs city named",
    "language_en_vs_local": "English vs local-language prompt, country named",
    "model_size": "Qwen model size, per unit of log parameters",
    "us_default": "North American number, English prompt that names no country",
    "grounding_verified_number": "Location and numbers in the system prompt vs country named",
}
NAN = math.nan
_SIZE = re.compile(r":(e?\d+(?:\.\d+)?b)")
_QUANTIZATION = re.compile(r"(q\d+_\w+)$")


def _wilson(hits: int, n: int) -> tuple[float, float]:
    return wilson_interval(hits, n, CONFIDENCE)


def _name(*parts: str) -> str:
    return "_".join(p.replace(" - ", " minus ").replace(" ", "_").replace("/", "_") for p in parts)


def _long(
    table: pd.DataFrame,
    ids: Sequence[str],
    measures: Mapping[str, tuple[str, str | None, str | None]],
    n: str | None = None,
) -> pd.DataFrame:
    """One row per row of ``table`` and measure: ``value`` and its interval from the columns
    named in ``measures``."""
    rows = []
    for _, row in table.iterrows():
        for measure, (value, low, high) in measures.items():
            rows.append(
                {
                    **{c: row[c] for c in ids},
                    "measure": measure,
                    "value": row[value],
                    "ci_low": row[low] if low else NAN,
                    "ci_high": row[high] if high else NAN,
                    **({"n": row[n]} if n else {}),
                }
            )
    return pd.DataFrame(rows)


def models_table(
    analysis_models: pd.DataFrame,
    models_config: Mapping[str, Any],
    styles: Mapping[str, ModelStyle],
) -> pd.DataFrame:
    """Family, size, access and sampling of each model, from its identifier and the generation
    settings; records and versions from the analysis."""
    info = analysis_models.set_index("model_id")
    rows = []
    for model, model_style in styles.items():
        backend = models_config["models"][model]["backend"]
        settings = models_config["generation"][backend]
        params = settings.get("params", {})
        row = info.loc[model]
        identifier = row["identifier"]
        open_weights = backend == "ollama"
        size = _SIZE.search(identifier)
        quantization = _QUANTIZATION.search(identifier)
        fields = {
            "family": identifier.split(":" if open_weights else "/")[0],
            "size": size[1].upper() if open_weights and size else "undisclosed",
            "access": "open weights" if open_weights else "API",
            "backend": backend,
            "identifier": identifier,
            "quantization": quantization[1] if quantization else "",
            "responses_per_prompt": row["responses_per_prompt"],
            "temperature": params.get("temperature", "provider default"),
            "top_p": params.get("top_p", "provider default"),
            "seed": params.get("seed", "none"),
            "max_tokens": settings["max_tokens"],
            "reasoning_effort": settings.get("reasoning", {}).get("effort", ""),
            "records": row["records"],
            "versions_seen": row["versions_seen"],
        }
        rows.extend(
            {"model_id": model, "model_group": model_style.group, "field": k, "value": v}
            for k, v in fields.items()
            if v != ""
        )
    return pd.DataFrame(rows)


def _domain(group: str, model: str) -> tuple[str, str]:
    if model != ALL:
        return "model", model
    return ("all", ALL) if group == ALL else ("model group", group)


def fabrication_table(
    reader: pd.DataFrame, by_model: pd.DataFrame, near_miss: pd.DataFrame
) -> pd.DataFrame:
    """Fabricated numbers in reader terms (share of all numbers; share of responses with one),
    the share of unknown-number mentions, each between its lower and upper bound; then the rate
    of near-misses of a real number against chance. ``range_only``: the interval collapses, so
    only the range of the two bounds is reported."""
    rows = []
    for _, r in reader.iterrows():
        for measure, prefix, n in (
            ("numbers_fabricated", "of_numbers", r["numbers"]),
            ("responses_with_a_fabricated_number", "of_responses", r["responses"]),
        ):
            rows.extend(
                {
                    "kind": r["kind"],
                    "domain": r["domain"],
                    "measure": measure,
                    "bound": bound,
                    "value": r[f"{prefix}_{bound}"],
                    "ci_low": r[f"{prefix}_{bound}_ci_low"],
                    "ci_high": r[f"{prefix}_{bound}_ci_high"],
                    "n": n,
                }
                for bound in ("lower", "upper")
            )
    for _, r in by_model.iterrows():
        kind, domain = ("all", ALL) if r["model_id"] == "all models" else ("model", r["model_id"])
        rows.extend(
            {
                "kind": kind,
                "domain": domain,
                "measure": "unknown_number_mentions_fabricated",
                "bound": bound,
                "value": r[f"share_{bound}"],
                "ci_low": r[f"ci_{bound}_low"],
                "ci_high": r[f"ci_{bound}_high"],
                "n": r["mentions"],
            }
            for bound in ("lower", "upper")
        )
    for _, r in near_miss.iterrows():
        kind, domain = _domain(r["model_group"], r["model_id"])
        rows.extend(
            {
                "kind": kind,
                "domain": domain,
                "measure": measure,
                "bound": "",
                "value": r[column],
                "ci_low": r[f"{column}_ci_low"],
                "ci_high": r[f"{column}_ci_high"],
                "n": r["mentions"],
            }
            for measure, column in (
                ("near_miss_rate", "near_miss"),
                ("near_miss_by_chance", "chance"),
            )
        )
    table = pd.DataFrame(rows)
    collapsed = set(fabrication.collapsed_intervals(by_model)["model_id"])
    table["range_only"] = (
        (table["kind"] == "model") & table["domain"].isin(collapsed) & (table["bound"] != "")
    )
    return table


def _no_location_responses(cells: pd.DataFrame) -> pd.Series:
    """Responses to prompts that name no country, per model."""
    rows = cells[
        (cells["condition"] == "S0")
        & (cells["lang_mode"] == "none")
        & (cells["outcome"] == "blocked")
    ]
    return rows.groupby("model_id")["responses"].sum()


def _share_row(part: str, by: str, level: str, measure: str, hits: int, n: int) -> dict[str, Any]:
    low, high = _wilson(hits, n)
    return {
        "part": part,
        "by": by,
        "level": level,
        "country": "",
        "measure": measure,
        "value": hits / n,
        "ci_low": low,
        "ci_high": high,
        "hits": hits,
        "n": n,
    }


def no_location_table(
    us_default: pd.Series,
    breakdown: pd.DataFrame,
    asks: pd.DataFrame,
    cells: pd.DataFrame,
    defaults: pd.DataFrame,
    models: Sequence[str],
) -> pd.DataFrame:
    """Prompts that name no country: the share of North American numbers in English (overall,
    by template, model and block); how often a model asks where the user is, or gives nothing
    (unvalidated rule); per prompt language, the country whose number is named most often (each
    of them on a tie)."""
    part = "north_american_number"
    rows = [
        {
            "part": part,
            "by": "all",
            "level": ALL,
            "country": "",
            "measure": "share_of_responses_with_a_number",
            "value": us_default["estimate"],
            "ci_low": us_default["ci_low"],
            "ci_high": us_default["ci_high"],
            "hits": NAN,
            "n": us_default["n"],
        }
    ]
    rows.extend(
        {
            "part": part,
            "by": r["by"],
            "level": r["level"],
            "country": "",
            "measure": "share_of_responses_with_a_number",
            "value": r["share"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "hits": r["hits"],
            "n": r["n"],
        }
        for _, r in breakdown.iterrows()
    )
    responses = _no_location_responses(cells)
    counts = asks.set_index("model_id")
    for model in [*models, ALL]:
        asked = int(
            counts["asks_location"].sum() if model == ALL else counts.loc[model, "asks_location"]
        )
        nothing = int(counts["nothing"].sum() if model == ALL else counts.loc[model, "nothing"])
        n = int(responses.sum() if model == ALL else responses[model])
        by = "all" if model == ALL else "model_id"
        rows.append(_share_row("asks_location", by, model, "asks_where_the_user_is", asked, n))
        rows.append(_share_row("asks_location", by, model, "no_number_and_no_question", nothing, n))
    for lang, part_rows in defaults.groupby("lang", sort=True):
        # Every country tied for the most responses.
        top = part_rows[part_rows["responses"] == part_rows["responses"].max()]
        for _, r in top.sort_values("country").iterrows():
            row = _share_row(
                "default_country",
                "prompt_language",
                lang,
                "names_a_number_of_this_country",
                int(r["responses"]),
                int(r["language_responses"]),
            )
            rows.append({**row, "country": r["country"]})
    return pd.DataFrame(rows)


def languages_table(by_language: pd.DataFrame, gap: pd.DataFrame) -> pd.DataFrame:
    """Per prompt language: each group's share of a correct number, and the English-minus-local
    gap over all models with the quality of the prompt's translation."""
    shares = _long(
        by_language,
        ["lang", "model_group"],
        {"correct_local": ("estimate", "ci_low", "ci_high")},
        "responses",
    )
    gaps = _long(
        gap.assign(model_group=ALL),
        ["lang", "model_group"],
        {
            "correct_local_english_prompt": ("share_en", None, None),
            "correct_local_local_prompt": ("share_local", None, None),
            "english_minus_local": ("penalty_en_minus_local", "ci_low", "ci_high"),
            "median_chrf_of_translation": ("median_chrf", None, None),
            "translation_flag_rate": ("flag_rate", None, None),
            "countries": ("countries", None, None),
        },
    )
    return pd.concat([shares, gaps], ignore_index=True).sort_values("lang", kind="stable")


def templates_table(levels: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    """Shares per prompt template and model group, then each contrast within each template;
    ``flag`` marks contrasts whose templates disagree."""
    columns = ["template", "model_group", "contrast", "outcome", "value", "ci_low", "ci_high"]
    level_rows = levels.assign(contrast="", flag="").rename(columns={"estimate": "value"})
    contrast_rows = contrasts.rename(columns={"estimate": "value"}).fillna({"flag": ""})
    return pd.concat(
        [
            level_rows[[*columns, "responses", "flag"]],
            contrast_rows[[*columns, "responses", "flag"]],
        ],
        ignore_index=True,
    ).rename(columns={"responses": "n"})


def sensitivity_table(hypotheses: pd.DataFrame) -> pd.DataFrame:
    """Every model-based comparison under both ground-truth versions, both working correlations,
    and each sensitivity analysis; no estimate where the model did not converge."""
    rows = hypotheses[
        hypotheses["hypothesis"].isin(COMPARISONS)
        & ~hypotheses["analysis"].str.startswith("main block")
    ]
    rows = rows.assign(comparison=rows["hypothesis"].map(COMPARISONS)).rename(
        columns={"hypothesis": "comparison_id", "working_structure": "working_correlation"}
    )
    return rows[
        [
            "analysis",
            "comparison_id",
            "comparison",
            "effect",
            "estimate",
            "ci_low",
            "ci_high",
            "p",
            "p_holm",
            "side",
            "n",
            "clusters",
            "working_correlation",
            "converged",
            "note",
        ]
    ].fillna({"note": ""})


def robustness_table(temperature: pd.DataFrame) -> pd.DataFrame:
    """Temperature 0.7 (three responses) against 0 on the same prompts, local models."""
    return _long(
        temperature,
        ["block", "model_id", "outcome"],
        {
            "share_at_temperature_0": ("share_t0", None, None),
            "share_at_temperature_0.7": ("share_t07", None, None),
            "difference": ("difference", "ci_low", "ci_high"),
            "prompts_all_three_agree_with_temperature_0": (
                "prompts_all_three_agree_with_t0",
                None,
                None,
            ),
        },
        "prompts",
    )


def near_miss_table(near_miss: pd.DataFrame) -> pd.DataFrame:
    """Numbers within two digit edits of a real number of the country, per model and group."""
    counts = (
        "mentions",
        "near_miss_mentions",
        "near_miss_unknown",
        "near_miss_fabricated",
        "near_miss_other_service",
        "near_miss_unverified",
        "names_service",
        "names_service_share",
        "countries",
        "mentions_without_country",
    )
    return _long(
        near_miss,
        ["model_group", "model_id"],
        {
            "near_miss_rate": ("near_miss", "near_miss_ci_low", "near_miss_ci_high"),
            "near_miss_by_chance": ("chance", "chance_ci_low", "chance_ci_high"),
            **{c: (c, None, None) for c in counts},
        },
    )


# The numbers the text cites.


def _number(
    name: str,
    value: float | str,
    unit: str,
    description: str,
    low: float = NAN,
    high: float = NAN,
) -> dict[str, Any]:
    if isinstance(value, float):
        value = float(f"{value:.6g}")  # as the numeric columns are written
    return {
        "name": name,
        "value": value,
        "ci_low": low,
        "ci_high": high,
        "unit": unit,
        "description": description,
    }


def _comparison_numbers(hypotheses: pd.DataFrame) -> Iterable[dict[str, Any]]:
    for _, r in hypotheses[hypotheses["hypothesis"].isin(COMPARISONS)].iterrows():
        unit = "proportion" if r["hypothesis"] == "us_default" else "odds ratio"
        yield _number(
            r["hypothesis"],
            r["estimate"],
            unit,
            f"{COMPARISONS[r['hypothesis']]}: {r['effect']} (n = {r['n']})",
            r["ci_low"],
            r["ci_high"],
        )


def _georgia_numbers(georgia: pd.DataFrame) -> Iterable[dict[str, Any]]:
    data = georgia.assign(
        **{
            f"{c}_hits": (georgia[c] * georgia["responses"]).round().astype(int)
            for c in ("correct_local", "wrong_country", "names_georgia")
        }
    )
    for setting, part in data.groupby("condition_cue", sort=True):
        n = int(part["responses"].sum())
        for outcome, what in (
            ("correct_local", "a correct number"),
            ("wrong_country", "another country's number"),
        ):
            hits = int(part[f"{outcome}_hits"].sum())
            yield _number(
                _name("georgia", setting, outcome),
                hits / n,
                "proportion",
                f"Georgia, {setting} (condition/cue/language), all models: responses with {what} "
                f"({hits} of {n})",
                *_wilson(hits, n),
            )
    hits, n = int(data["names_georgia_hits"].sum()), int(data["responses"].sum())
    yield _number(
        "georgia_names_the_country",
        hits / n,
        "proportion",
        f"Georgia, all prompts and models: responses that name the country (Tbilisi, country of "
        f"Georgia; unvalidated pattern) ({hits} of {n})",
        *_wilson(hits, n),
    )


def numbers_table(
    *,
    hypotheses: pd.DataFrame,
    no_location: pd.DataFrame,
    who_regions: pd.DataFrame,
    worst: pd.DataFrame,
    georgia: pd.DataFrame,
    template_levels: pd.DataFrame,
    by_model: pd.DataFrame,
    blocked_model: str,
    ladder: pd.DataFrame,
    temperature: pd.DataFrame,
    grounding: pd.DataFrame,
    groups: Sequence[str],
    fabrication_rows: pd.DataFrame,
    referral: pd.DataFrame,
) -> pd.DataFrame:
    """Every number the text cites, one row each, ground truth v2."""
    rows = list(_comparison_numbers(hypotheses))

    asked = no_location[(no_location["part"] == "asks_location") & (no_location["level"] == ALL)]
    for _, r in asked.iterrows():
        rows.append(
            _number(
                _name("no_country", r["measure"]),
                r["value"],
                "proportion",
                f"Prompts that name no country, all models: {r['measure'].replace('_', ' ')} "
                f"(unvalidated rule; {r['hits']} of {r['n']})",
                r["ci_low"],
                r["ci_high"],
            )
        )

    pooled = who_regions[who_regions["model_group"] == ALL]
    local = pooled[pooled["outcome"] == "correct_local"].sort_values(["estimate", "who_region"])
    region = local.iloc[0]["who_region"]
    name = REGIONS[region]
    rows.append(
        _number(
            "worst_region",
            region,
            "WHO region",
            f"WHO region with the lowest share of a correct number over all models: {name}",
        )
    )
    for outcome in ("correct_local", "correct_crisis"):
        r = pooled[(pooled["who_region"] == region) & (pooled["outcome"] == outcome)].iloc[0]
        rows.append(
            _number(
                _name("worst_region", outcome),
                r["estimate"],
                "proportion",
                f"{name}, all models, prompts that name a city or the country: {outcome}",
                r["ci_low"],
                r["ci_high"],
            )
        )
    for rank, (_, r) in enumerate(worst.iterrows(), start=1):
        rows.append(_number(f"worst_country_{rank}", r["iso2"], "country", r["name_en"]))
        for outcome in ("correct_local", "correct_crisis"):
            if pd.notna(r[f"{outcome} estimate ({ALL})"]):
                rows.append(
                    _number(
                        f"worst_country_{rank}_{outcome}",
                        r[f"{outcome} estimate ({ALL})"],
                        "proportion",
                        f"{r['name_en']}, all models, prompts that name a city or the country: "
                        f"{outcome}",
                        r[f"{outcome} ci_low ({ALL})"],
                        r[f"{outcome} ci_high ({ALL})"],
                    )
                )

    rows.extend(_georgia_numbers(georgia))

    nothing = template_levels[
        (template_levels["model_group"] == ALL)
        & (template_levels["outcome"] == "no_usable_resource")
    ]
    for _, r in nothing.iterrows():
        rows.append(
            _number(
                _name("no_usable_resource", r["template"]),
                r["estimate"],
                "proportion",
                f"Template {r['template']}, all models, prompts that name a city or the country: "
                "responses with no usable resource (no number, a loop, blocked or empty)",
                r["ci_low"],
                r["ci_high"],
            )
        )

    blocked = by_model[(by_model["model_id"] == blocked_model) & (by_model["outcome"] == "blocked")]
    r = blocked.iloc[0]
    rows.append(
        _number(
            _name("blocked", blocked_model),
            r["share"],
            "proportion",
            f"{blocked_model}: responses blocked by the provider's moderation, all prompts",
            r["ci_low"],
            r["ci_high"],
        )
    )

    sizes = sorted(ladder["size"].unique())
    for _, r in ladder[ladder["size"].isin((sizes[0], sizes[-1]))].iterrows():
        rows.append(
            _number(
                _name("ladder", r["model_id"], r["cue"]),
                r["share"],
                "proportion",
                f"{r['model_id']}, prompts that name the {r['cue']}: correct number",
                r["ci_low"],
                r["ci_high"],
            )
        )

    cited = (("cue_20", "correct_local"), ("none_en", "na_default"))
    for _, r in temperature.iterrows():
        if (r["block"], r["outcome"]) in cited:
            rows.append(
                _number(
                    _name("temperature", r["block"], r["model_id"], r["outcome"]),
                    r["difference"],
                    "difference in proportion",
                    f"{r['model_id']}, block {r['block']}: {r['outcome']} at temperature 0.7 minus "
                    f"at 0 ({r['share_t07']:.3f} vs {r['share_t0']:.3f}, {r['prompts']} prompts)",
                    r["ci_low"],
                    r["ci_high"],
                )
            )

    conditions = ("S0-country", "M1", "M2", "M2 - S0-country")
    for _, r in grounding[
        (grounding["model_id"] == ALL)
        & grounding["model_group"].isin([*groups, ALL])
        & grounding["measure"].isin(conditions)
        & grounding["outcome"].isin(("correct_crisis", "fabricated"))
    ].iterrows():
        difference = " - " in r["measure"]
        rows.append(
            _number(
                _name(r["outcome"], r["measure"], r["model_group"]),
                r["estimate"],
                "difference in proportion" if difference else "proportion",
                f"{r['outcome']}, {r['measure']}, local-language prompts, models: "
                f"{r['model_group']}",
                r["ci_low"],
                r["ci_high"],
            )
        )

    overall = fabrication_rows[
        (fabrication_rows["kind"] == "all") & (fabrication_rows["bound"] == "lower")
    ]
    for _, r in overall.iterrows():
        rows.append(
            _number(
                _name("all_models", r["measure"]),
                r["value"],
                "proportion",
                f"All models: {r['measure'].replace('_', ' ')} (lower bound; n = {r['n']})",
                r["ci_low"],
                r["ci_high"],
            )
        )

    for _, r in referral[
        referral["referral"].isin(("correct_crisis_line", "correct_any"))
    ].iterrows():
        rows.append(
            _number(
                _name(r["referral"], r["model_id"]),
                r["share"],
                "proportion",
                f"{r['model_id']}: responses to prompts about a country with "
                + (
                    "the correct crisis line"
                    if r["referral"] == "correct_crisis_line"
                    else "any correct number"
                ),
                r["ci_low"],
                r["ci_high"],
            )
        )
    return pd.DataFrame(rows)
