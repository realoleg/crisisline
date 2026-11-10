"""The paper's tables and the numbers its text cites, as tidy frames: one row per value, with its
95% interval where it has one and the prompt subset it covers. Everything is taken from the
analysis outputs; the only new intervals are Wilson intervals of counts the analysis reports
without one."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from crisisline.analysis import fabrication
from crisisline.analysis.descriptive import CONFIDENCE
from crisisline.paper import subsets
from crisisline.paper.figures import ALL
from crisisline.paper.style import REFERRAL, ModelStyle, group_label, model_name
from crisisline.paper.subsets import Scope
from crisisline.unknowns import wilson_interval

COMPARISONS = {
    "cue_country_vs_city": "Country named vs city named",
    "language_en_vs_local": "English vs local-language prompt, country named",
    "model_size": "Qwen model size, per unit of log parameters",
    "us_default": "North American number, English prompt that names no country",
    "grounding_verified_number": "Location and numbers in the system prompt vs country named",
}
NAN = math.nan
UNKNOWN_MENTIONS = "unknown_number_mentions_fabricated"
_SIZE = re.compile(r":(e?\d+(?:\.\d+)?b)")
_QUANTIZATION = re.compile(r"(q\d+_\w+)$")


def _wilson(hits: int, n: int) -> tuple[float, float]:
    return wilson_interval(hits, n, CONFIDENCE)


def _name(*parts: str) -> str:
    return "_".join(p.replace(" - ", " minus ").replace(" ", "_").replace("/", "_") for p in parts)


def _with_names(
    table: pd.DataFrame, ids: str = "model_id", where: pd.Series | None = None
) -> pd.DataFrame:
    """``table`` with a ``model`` column right after ``ids``: the display name of each model, empty
    where ``where`` is false or the id is not a model."""
    names = table[ids].map(lambda m: model_name(m) if m != ALL else ALL)
    if where is not None:
        names = names.where(where, "")
    columns = list(table.columns)
    position = columns.index(ids) + 1
    return table.assign(model=names)[[*columns[:position], "model", *columns[position:]]]


def _long(
    table: pd.DataFrame,
    ids: Sequence[str],
    measures: Mapping[str, tuple[str, str | None, str | None]],
    subset: Callable[[pd.Series, str], str],
    n: str | None = None,
) -> pd.DataFrame:
    """One row per row of ``table`` and measure: ``value`` and its interval from the columns
    named in ``measures``, and the subset of that row and measure."""
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
                    "subset": subset(row, measure),
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
            {
                "model_id": model,
                "model": model_name(model),
                "model_group": model_style.group,
                "field": k,
                "value": v,
                "subset": subsets.MODELS,
            }
            for k, v in fields.items()
            if v != ""
        )
    return pd.DataFrame(rows)


def _domain(group: str, model: str) -> tuple[str, str]:
    if model != ALL:
        return "model", model
    return ("all", ALL) if group == ALL else ("model group", group)


def reader_subset(kind: str, domain: str) -> str:
    if kind == "language group":
        return f"{subsets.FABRICATION_READER}, prompts in the {domain} language group"
    return subsets.FABRICATION_READER


def fabrication_table(
    reader: pd.DataFrame,
    estimate: Mapping[str, Any],
    by_model: pd.DataFrame,
    near_miss: pd.DataFrame,
    min_digits: int,
) -> pd.DataFrame:
    """Fabricated numbers in reader terms (share of all numbers; share of responses with one),
    the share of unknown-number mentions (per model; over all models the stratified estimate),
    each between its lower and upper bound; then the rate of near-misses of a real number against
    chance. ``range_only``: the interval collapses, so only the range of the two bounds is
    reported."""
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
                    "subset": reader_subset(r["kind"], r["domain"]),
                }
                for bound in ("lower", "upper")
            )
    # Over all models, the estimate itself rather than this table's check of it.
    rows.extend(
        {
            "kind": "all",
            "domain": ALL,
            "measure": UNKNOWN_MENTIONS,
            "bound": bound,
            "value": estimate["share"],
            "ci_low": estimate["interval"][0],
            "ci_high": estimate["interval"][1],
            "n": estimate["mentions"],
            "subset": subsets.UNKNOWN_MENTIONS,
        }
        for bound in ("lower", "upper")
    )
    models = by_model[by_model["model_id"] != "all models"]
    for _, r in models.iterrows():
        rows.extend(
            {
                "kind": "model",
                "domain": r["model_id"],
                "measure": UNKNOWN_MENTIONS,
                "bound": bound,
                "value": r[f"share_{bound}"],
                "ci_low": r[f"ci_{bound}_low"],
                "ci_high": r[f"ci_{bound}_high"],
                "n": r["mentions"],
                "subset": subsets.UNKNOWN_MENTIONS,
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
                "subset": subsets.near_miss(min_digits),
            }
            for measure, column in (
                ("near_miss_rate", "near_miss"),
                ("near_miss_by_chance", "chance"),
            )
        )
    table = pd.DataFrame(rows)
    collapsed = set(fabrication.collapsed_intervals(models)["model_id"])
    table["range_only"] = (
        (table["kind"] == "model") & table["domain"].isin(collapsed) & (table["bound"] != "")
    )
    return _with_names(table, "domain", table["kind"] == "model")


def _no_location_responses(cells: pd.DataFrame) -> pd.Series:
    """Responses to prompts that name no country, per model."""
    rows = cells[
        (cells["condition"] == "S0")
        & (cells["lang_mode"] == "none")
        & (cells["outcome"] == "blocked")
    ]
    return rows.groupby("model_id")["responses"].sum()


def _share_row(
    part: str, by: str, level: str, measure: str, hits: int, n: int, subset: str
) -> dict[str, Any]:
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
        "subset": subset,
    }


def no_location_table(
    us_default: pd.Series,
    breakdown: pd.DataFrame,
    asks: pd.DataFrame,
    cells: pd.DataFrame,
    defaults: pd.DataFrame,
    models: Sequence[str],
    scope: Scope,
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
            "subset": subsets.NORTH_AMERICAN,
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
            "subset": subsets.NORTH_AMERICAN_BLOCKS[r["level"]]
            if r["by"] == "block"
            else subsets.NORTH_AMERICAN,
        }
        for _, r in breakdown.iterrows()
    )
    responses = _no_location_responses(cells)
    counts = asks.set_index("model_id")
    asked_subset = subsets.asks_location(scope)
    for model in [*models, ALL]:
        asked = int(
            counts["asks_location"].sum() if model == ALL else counts.loc[model, "asks_location"]
        )
        nothing = int(counts["nothing"].sum() if model == ALL else counts.loc[model, "nothing"])
        n = int(responses.sum() if model == ALL else responses[model])
        by = "all" if model == ALL else "model_id"
        for measure, hits in (
            ("asks_where_the_user_is", asked),
            ("no_number_and_no_question", nothing),
        ):
            rows.append(_share_row("asks_location", by, model, measure, hits, n, asked_subset))
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
                subsets.default_country(lang),
            )
            rows.append({**row, "country": r["country"]})
    table = pd.DataFrame(rows)
    return _with_names(table, "level", table["by"] == "model_id")


TRANSLATION_MEASURES = ("median_chrf_of_translation", "translation_flag_rate")


def languages_table(by_language: pd.DataFrame, gap: pd.DataFrame) -> pd.DataFrame:
    """Per prompt language: each group's share of a correct number, and the English-minus-local
    gap over all models with the quality of the prompt's translation."""
    shares = _long(
        by_language,
        ["lang", "model_group"],
        {"correct_local": ("estimate", "ci_low", "ci_high")},
        lambda row, _: subsets.language(row["lang"], row["countries"]),
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
        lambda row, measure: (
            subsets.translations(row["lang"])
            if measure in TRANSLATION_MEASURES
            else subsets.language_gap(row["lang"], row["countries"])
        ),
    )
    return pd.concat([shares, gaps], ignore_index=True).sort_values("lang", kind="stable")


# Pooled over templates, these contrasts are steps of the grounding analysis, which reports them
# with its own bootstrap.
GROUNDING_STEPS = {
    "M1 - S0-country, local": "M1 - S0-country",
    "M2 - S0-country, local": "M2 - S0-country",
}


def templates_table(
    levels: pd.DataFrame, contrasts: pd.DataFrame, grounding: pd.DataFrame, scope: Scope
) -> pd.DataFrame:
    """Shares per prompt template and model group, then each contrast within each template;
    ``flag`` marks contrasts whose templates disagree. A contrast pooled over templates that the
    grounding analysis also reports takes its interval from there, so that each number has one."""
    columns = ["template", "model_group", "contrast", "outcome", "value", "ci_low", "ci_high"]
    level_rows = levels.assign(
        contrast="",
        flag="",
        subset=[
            subsets.template(scope, t, o)
            for t, o in zip(levels["template"], levels["outcome"], strict=True)
        ],
    ).rename(columns={"estimate": "value"})
    contrast_rows = (
        contrasts.assign(
            subset=[
                subsets.contrast(scope, c, t, o)
                for c, t, o in zip(
                    contrasts["contrast"], contrasts["template"], contrasts["outcome"], strict=True
                )
            ]
        )
        .rename(columns={"estimate": "value"})
        .fillna({"flag": ""})
    )
    steps = grounding[grounding["model_id"] == ALL].set_index(["model_group", "measure", "outcome"])
    pooled = (contrast_rows["template"] == "all") & contrast_rows["contrast"].isin(GROUNDING_STEPS)
    for index in contrast_rows.index[pooled]:
        row = contrast_rows.loc[index]
        step = steps.loc[(row["model_group"], GROUNDING_STEPS[row["contrast"]], row["outcome"])]
        if not math.isclose(step["estimate"], row["value"], rel_tol=1e-5, abs_tol=1e-9):
            raise ValueError(f"the grounding step and the pooled template contrast differ: {row}")
        contrast_rows.loc[index, ["value", "ci_low", "ci_high"]] = [
            step["estimate"],
            step["ci_low"],
            step["ci_high"],
        ]
    return pd.concat(
        [
            level_rows[[*columns, "responses", "flag", "subset"]],
            contrast_rows[[*columns, "responses", "flag", "subset"]],
        ],
        ignore_index=True,
    ).rename(columns={"responses": "n"})


def sensitivity_table(hypotheses: pd.DataFrame, scope: Scope) -> pd.DataFrame:
    """Every model-based comparison under both ground-truth versions, both working correlations,
    and each sensitivity analysis; no estimate where the model did not converge."""
    rows = hypotheses[
        hypotheses["hypothesis"].isin(COMPARISONS)
        & ~hypotheses["analysis"].str.startswith("main block")
    ]
    rows = rows.assign(
        comparison=rows["hypothesis"].map(COMPARISONS),
        subset=[
            subsets.comparison(scope, h, a)
            for h, a in zip(rows["hypothesis"], rows["analysis"], strict=True)
        ],
    ).rename(columns={"hypothesis": "comparison_id", "working_structure": "working_correlation"})
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
            "subset",
        ]
    ].fillna({"note": ""})


def robustness_table(temperature: pd.DataFrame, scope: Scope) -> pd.DataFrame:
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
        lambda row, _: subsets.temperature(scope, row["block"], row["outcome"]),
        "prompts",
    ).pipe(_with_names)


def near_miss_table(near_miss: pd.DataFrame, min_digits: int) -> pd.DataFrame:
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
        lambda _, __: subsets.near_miss(min_digits),
    ).pipe(_with_names)


# The numbers the text cites.


def _number(
    name: str,
    value: float | str,
    unit: str,
    subset: str,
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
        "subset": subset,
        "description": description,
    }


def _comparison_numbers(hypotheses: pd.DataFrame, scope: Scope) -> Iterable[dict[str, Any]]:
    for _, r in hypotheses[hypotheses["hypothesis"].isin(COMPARISONS)].iterrows():
        unit = "proportion" if r["hypothesis"] == "us_default" else "odds ratio"
        yield _number(
            r["hypothesis"],
            r["estimate"],
            unit,
            subsets.comparison(scope, r["hypothesis"], r["analysis"]),
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
                f"{subsets.georgia(setting)}, all models",
                f"Responses with {what} ({hits} of {n})",
                *_wilson(hits, n),
            )
    hits, n = int(data["names_georgia_hits"].sum()), int(data["responses"].sum())
    yield _number(
        "georgia_names_the_country",
        hits / n,
        "proportion",
        "Georgia, all prompts, all models",
        "Responses that name the country (Tbilisi, country of Georgia): an unvalidated "
        f"automatic pattern match ({hits} of {n})",
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
    scope: Scope,
) -> pd.DataFrame:
    """Every number the text cites, one row each, ground truth v2."""
    rows = list(_comparison_numbers(hypotheses, scope))

    asked = no_location[(no_location["part"] == "asks_location") & (no_location["level"] == ALL)]
    for _, r in asked.iterrows():
        rows.append(
            _number(
                _name("no_country", r["measure"]),
                r["value"],
                "proportion",
                r["subset"],
                f"All models: {r['measure'].replace('_', ' ')} "
                f"({int(r['hits'])} of {int(r['n'])}). "
                "Unvalidated automatic measure: a rule (a question with a location word, per "
                "language) that was not checked by hand; precision and recall are unknown",
                r["ci_low"],
                r["ci_high"],
            )
        )

    pooled = who_regions[who_regions["model_group"] == ALL]
    local = pooled[pooled["outcome"] == "correct_local"].sort_values(["estimate", "who_region"])
    region = local.iloc[0]["who_region"]
    rows.append(
        _number(
            "worst_region",
            region,
            "WHO region",
            subsets.all_regions(scope, "correct_local"),
            f"WHO region with the lowest share of a correct number over all models: {region}",
        )
    )
    for outcome in ("correct_local", "correct_crisis"):
        r = pooled[(pooled["who_region"] == region) & (pooled["outcome"] == outcome)].iloc[0]
        rows.append(
            _number(
                _name("worst_region", outcome),
                r["estimate"],
                "proportion",
                subsets.region(scope, region, outcome),
                f"{region}, all models: {outcome}",
                r["ci_low"],
                r["ci_high"],
            )
        )
    for rank, (_, r) in enumerate(worst.iterrows(), start=1):
        rows.append(
            _number(
                f"worst_country_{rank}",
                r["iso2"],
                "country",
                subsets.all_regions(scope, "correct_local"),
                f"Country with the {rank}. lowest share of a correct number over all models: "
                f"{r['name_en']}",
            )
        )
        for outcome in ("correct_local", "correct_crisis"):
            if pd.notna(r[f"{outcome} estimate ({ALL})"]):
                rows.append(
                    _number(
                        f"worst_country_{rank}_{outcome}",
                        r[f"{outcome} estimate ({ALL})"],
                        "proportion",
                        subsets.country(r["name_en"]),
                        f"{r['name_en']}, all models: {outcome}",
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
                subsets.template(scope, r["template"], r["outcome"]),
                f"Template {r['template']}, all models: responses with no usable resource (no "
                "number, a loop, blocked or empty)",
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
            f"all prompts, {subsets.MAIN_RUN}",
            f"{model_name(blocked_model)}: responses blocked by the provider's moderation",
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
                r["subset"],
                f"{model_name(r['model_id'])}: correct number",
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
                    subsets.temperature(scope, r["block"], r["outcome"]),
                    f"{model_name(r['model_id'])}: {r['outcome']} at temperature 0.7 minus at 0 "
                    f"({r['share_t07']:.3f} vs {r['share_t0']:.3f}, {r['prompts']} prompts)",
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
                subsets.condition(scope, r["measure"], r["outcome"]),
                f"{r['outcome']}, {r['measure']}, models: {r['model_group']}",
                r["ci_low"],
                r["ci_high"],
            )
        )

    overall = fabrication_rows[
        (fabrication_rows["kind"] == "all") & (fabrication_rows["bound"] == "lower")
    ]
    for _, r in overall[overall["measure"] != UNKNOWN_MENTIONS].iterrows():
        rows.append(
            _number(
                _name("all_models", r["measure"]),
                r["value"],
                "proportion",
                r["subset"],
                f"All models: {r['measure'].replace('_', ' ')} (lower bound; n = {r['n']})",
                r["ci_low"],
                r["ci_high"],
            )
        )
    r = overall[overall["measure"] == UNKNOWN_MENTIONS].iloc[0]
    rows.append(
        _number(
            "unconfirmed_numbers_estimated_fabricated",
            r["value"],
            "proportion",
            r["subset"],
            f"Estimated share fabricated among the {r['n']:,} mentions of unknown numbers "
            "(stratified estimate from the blind labelling, stratified bootstrap interval). "
            "The 'Unconfirmed number' class of the referral figure holds the unknown numbers "
            "left unlabelled and those labelled unverified or another service; this estimate "
            "covers all unknown numbers before labelling, not that class alone",
            r["ci_low"],
            r["ci_high"],
        )
    )

    shown = referral[referral["referral"].isin(("correct_crisis_line", "correct_local"))]
    for _, r in shown.iterrows():
        what = (
            "the correct crisis line"
            if r["referral"] == "correct_crisis_line"
            else "a correct local number (the crisis line or another)"
        )
        rows.append(
            _number(
                _name(r["referral"], r["model_id"]),
                r["share"],
                "proportion",
                r["subset"],
                f"{model_name(r['model_id'])}: responses with {what}",
                r["ci_low"],
                r["ci_high"],
            )
        )
    return pd.DataFrame(rows)


def figure_numbers(
    who: pd.DataFrame,
    fails: pd.DataFrame,
    native: pd.DataFrame,
    fixes: pd.DataFrame,
    ladder: pd.DataFrame,
) -> pd.DataFrame:
    """The numbers of the figures' captions that no other row of ``numbers.csv`` holds."""
    rows = []
    for _, r in who[who["panel"] == "model share"].iterrows():
        rows.append(
            _number(
                _name("crisis_line_S0", r["model_id"]),
                r["share"],
                "proportion",
                r["subset"],
                f"{model_name(r['model_id'])}: responses with the correct crisis line (n = "
                f"{int(r['n'])})",
                r["ci_low"],
                r["ci_high"],
            )
        )
    lowest = who[who["panel"] == "lowest"]
    group = lowest["model_group"].iloc[0]
    names = ", ".join(lowest["name_en"])
    rows.append(
        _number(
            _name("lowest_countries_crisis_line", group),
            len(lowest),
            "count",
            f"{subsets.CUED}, {subsets.ALL_LANGUAGES}",
            f"{group_label(group)} models: countries with the lowest crisis-line share, the "
            f"three lowest with every tie: {names}",
        )
    )
    for _, r in lowest.iterrows():
        rows.append(
            _number(
                _name("lowest_crisis_line", group, r["iso2"]),
                r["share"],
                "proportion",
                r["subset"],
                f"{group_label(group)} models in {r['name_en']}: responses with the correct "
                f"crisis line (n = {int(r['n'])})",
                r["ci_low"],
                r["ci_high"],
            )
        )
    no_line = who[who["panel"] == "no crisis line"]
    rows.append(
        _number(
            "countries_without_crisis_line",
            len(no_line),
            "count",
            no_line["subset"].iloc[0],
            "countries without a national crisis line, left out of the crisis-line grid: "
            + ", ".join(no_line["name_en"]),
        )
    )
    for _, r in fails[fails["measure"] == "failure_total"].iterrows():
        rows.append(
            _number(
                _name("failure", r["model_id"]),
                r["value"],
                "proportion",
                r["subset"],
                f"{model_name(r['model_id'])}: responses with no correct local number "
                "(misrouted, fabricated, unconfirmed or no number)",
                r["ci_low"],
                r["ci_high"],
            )
        )
    invented = fails[fails["panel"] == "numbers fabricated"]
    top = invented.loc[invented["upper"].idxmax()]
    rows.append(
        _number(
            "numbers_invented_largest",
            top["upper"],
            "proportion",
            top["subset"],
            f"{model_name(top['model_id'])}: the largest share of the numbers a model gives that "
            "are fabricated (upper bound)",
            top["ci_low"],
            top["ci_high"],
        )
    )
    rows.append(
        _number(
            "numbers_invented_one_in",
            round(1 / top["upper"]),
            "count",
            top["subset"],
            f"{model_name(top['model_id'])}: 1 in this many numbers given is fabricated",
        )
    )
    for model, part in native[native["significant_gap"]].groupby("model_id", sort=False):
        shares = part.set_index("prompt_language")
        for language, label in (("en", "English"), ("local", "local-language")):
            r = shares.loc[language]
            rows.append(
                _number(
                    _name(f"{language}_prompt", model),
                    r["estimate"],
                    "proportion",
                    r["subset"],
                    f"{model_name(model)}: correct number, {label} prompts",
                    r["ci_low"],
                    r["ci_high"],
                )
            )
        r = shares.iloc[0]
        rows.append(
            _number(
                _name("en_minus_local", model),
                r["difference_en_minus_local"],
                "difference in proportion",
                subsets.language_by_model_pair(r["subset"]),
                f"{model_name(model)}: correct number, English minus local-language prompts",
                r["difference_ci_low"],
                r["difference_ci_high"],
            )
        )
    for _, r in fixes[fixes["panel"] == "step"].iterrows():
        rows.append(
            _number(
                _name("correct_crisis", r["condition"], ALL),
                r["estimate"],
                "difference in proportion",
                r["subset"],
                f"correct_crisis, {r['condition']}, models: all",
                r["ci_low"],
                r["ci_high"],
            )
        )
    sizes = sorted(ladder["size"].unique())
    for _, r in ladder[ladder["size"].isin((sizes[0], sizes[-1]))].iterrows():
        rows.append(
            _number(
                _name("ladder", r["referral"], r["model_id"]),
                r["share"],
                "proportion",
                r["subset"],
                f"{model_name(r['model_id'])}: {REFERRAL[r['referral']][0].lower()} (n = "
                f"{int(r['responses'])})",
                r["ci_low"],
                r["ci_high"],
            )
        )
    return pd.DataFrame(rows)
