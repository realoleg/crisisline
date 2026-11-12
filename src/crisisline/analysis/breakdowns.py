"""Exploratory breakdowns of the outcomes.

Grounding by condition, ``correct_local`` by prompt language, template consistency, countries and
WHO regions, and directories. Every share comes with a cluster bootstrap interval
(:mod:`crisisline.analysis.intervals`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from crisisline.analysis import intervals
from crisisline.analysis.frames import s0_cued
from crisisline.analysis.intervals import ALL

GROUNDING_OUTCOMES = (
    "correct_local",
    "correct_crisis",
    "fabricated",
    "wrong_foreign_only",
    "no_usable_resource",
)
TEMPLATE_OUTCOMES = ("correct_local", "correct_crisis", "fabricated", "no_usable_resource")
COUNTRY_OUTCOMES = ("correct_local", "correct_crisis")
DIRECTORY_OUTCOMES = ("mentions_directory", "directory_only")
GROUNDING_STEPS = (
    ("M1 - S0-country", "M1", "S0-country"),
    ("M2 - M1", "M2", "M1"),
    ("M2 - S0-country", "M2", "S0-country"),
)

Select = Callable[[pd.DataFrame], pd.DataFrame]


def setting(frame: pd.DataFrame) -> pd.Series:
    """``S0-city``, ``S0-country``, ``S0-none``, ``M1`` or ``M2``."""
    s0 = frame["condition"] == "S0"
    return frame["condition"].where(~s0, "S0-" + frame["cue"])


def _row(keys: Mapping[str, Any], outcome: str, result: Mapping[str, Any] | None) -> list[dict]:
    return [] if result is None else [{**keys, "outcome": outcome, **result}]


def grounding_settings(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Local-language prompts: S0 with the country cue, M1 and M2."""
    local = frame[frame["lang_mode"] == "local"]
    named = setting(local)
    return {name: local[named == name] for name in ("S0-country", "M1", "M2")}


def grounding(
    frame: pd.DataFrame, groups: Mapping[str, Sequence[str]], replicates: int, seed: int
) -> pd.DataFrame:
    """Each outcome under S0-country, M1 and M2, and the steps between them: what naming the
    country (M1) changes, and what the verified numbers change on top of it (M2 - M1)."""
    rows = []
    for group, model, part in intervals.scopes(frame, groups):
        arms = grounding_settings(part)
        for outcome in GROUNDING_OUTCOMES:
            for name, data in arms.items():
                keys = {"model_group": group, "model_id": model, "measure": name}
                result = intervals.share(data, outcome, replicates, seed, tuple(keys.values()))
                rows += _row(keys, outcome, result)
            for name, a, b in GROUNDING_STEPS:
                keys = {"model_group": group, "model_id": model, "measure": name}
                result = intervals.difference(
                    arms[a], arms[b], outcome, replicates, seed, tuple(keys.values())
                )
                rows += _row(keys, outcome, result)
    return pd.DataFrame(rows)


def by_language(
    frame: pd.DataFrame, groups: Mapping[str, Sequence[str]], replicates: int, seed: int
) -> pd.DataFrame:
    """``correct_local`` per prompt language, local-language S0 prompts with a cue."""
    data = s0_cued(frame)
    data = data[data["lang_mode"] == "local"]
    rows = []
    for group, _, part in intervals.scopes(data, groups, per_model=False):
        for lang, cell in part.groupby("lang", sort=True):
            keys = {"model_group": group, "lang": lang, "countries": cell["iso2"].nunique()}
            result = intervals.share(cell, "correct_local", replicates, seed, (group, lang))
            rows += _row(keys, "correct_local", result)
    return pd.DataFrame(rows)


def _local(frame: pd.DataFrame, named: str) -> pd.DataFrame:
    local = frame[frame["lang_mode"] == "local"]
    return local[setting(local) == named]


def _cued_non_english(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    data = s0_cued(frame)
    return data[(data["cue"] == "country") & ~data["english_country"] & (data["lang_mode"] == mode)]


def contrasts(non_english_countries: int) -> tuple[tuple[str, Select, Select], ...]:
    """Contrasts within one scope: (name, arm a, arm b), each a difference a - b. The first names
    how many countries it covers: those whose language is not English."""
    return (
        (
            f"en - local, S0-country, {non_english_countries} countries",
            lambda f: _cued_non_english(f, "en"),
            lambda f: _cued_non_english(f, "local"),
        ),
        (
            "country - city, S0",
            lambda f: s0_cued(f)[lambda d: d["cue"] == "country"],
            lambda f: s0_cued(f)[lambda d: d["cue"] == "city"],
        ),
        ("M1 - S0-country, local", lambda f: _local(f, "M1"), lambda f: _local(f, "S0-country")),
        ("M2 - S0-country, local", lambda f: _local(f, "M2"), lambda f: _local(f, "S0-country")),
    )


def _group_contrasts(groups: Sequence[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(groups) for b in groups[i + 1 :]]


def templates(
    frame: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    template_names: Sequence[str],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outcomes of S0 with a cue per template (in the order of ``template_names``) and model
    group, and the main contrasts per template, pooled (``template = all``) and in each
    template."""
    levels, differences = [], []
    named = frame[frame["lang_mode"].notna() & ~frame["english_country"]]
    steps = contrasts(named["iso2"].nunique())
    subsets = [(t, frame[frame["template"] == t]) for t in template_names] + [(ALL, frame)]
    for template, part in subsets:
        for group, _, scoped in intervals.scopes(part, groups, per_model=False):
            for outcome in TEMPLATE_OUTCOMES:
                keys = {"template": template, "model_group": group}
                result = intervals.share(
                    s0_cued(scoped), outcome, replicates, seed, (*keys.values(), "level")
                )
                levels += _row(keys, outcome, result)
                for name, arm_a, arm_b in steps:
                    keys = {"template": template, "model_group": group, "contrast": name}
                    result = intervals.difference(
                        arm_a(scoped),
                        arm_b(scoped),
                        outcome,
                        replicates,
                        seed,
                        tuple(keys.values()),
                    )
                    differences += _row(keys, outcome, result)
        cued = s0_cued(part)
        for a, b in _group_contrasts(list(groups)):
            for outcome in TEMPLATE_OUTCOMES:
                keys = {"template": template, "model_group": ALL, "contrast": f"{a} - {b}, S0"}
                result = intervals.difference(
                    cued[cued["model_id"].isin(groups[a])],
                    cued[cued["model_id"].isin(groups[b])],
                    outcome,
                    replicates,
                    seed,
                    tuple(keys.values()),
                )
                differences += _row(keys, outcome, result)
    return pd.DataFrame(levels), flag_single_template(pd.DataFrame(differences))


def flag_single_template(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Per contrast, outcome and group: in how many templates the interval excludes zero, whether
    the pooled one does, and a flag where exactly one template does or templates disagree in
    sign."""
    table = contrasts.copy()
    table["excludes_zero"] = [
        intervals.excludes_zero(low, high)
        for low, high in zip(table["ci_low"], table["ci_high"], strict=True)
    ]
    keys = ["model_group", "contrast", "outcome"]
    rows = []
    for key, part in table[table["template"] != ALL].groupby(keys):
        clear = part[part["excludes_zero"]]
        signs = set((clear["estimate"] > 0).tolist())
        flag = ""
        if len(signs) == 2:
            flag = "templates disagree in sign"
        elif len(clear) == 1:
            flag = f"holds only in {clear['template'].iloc[0]}"
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "templates_excluding_zero": len(clear),
                "flag": flag,
            }
        )
    pooled = table[table["template"] == ALL].set_index(keys)["excludes_zero"]
    summary = pd.DataFrame(rows).assign(
        pooled_excludes_zero=lambda t: [bool(pooled[tuple(k)]) for k in t[keys].to_numpy()]
    )
    return table.merge(summary, on=keys, how="left")


def countries(
    frame: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    country_rows: Sequence[Mapping[str, str]],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``correct_local`` and ``correct_crisis`` of S0 with a cue, per country (resampled by
    prompt) and per WHO region (by country), per model group and over all models."""
    info = pd.DataFrame(country_rows)[["iso2", "name_en", "who_region"]]
    data = s0_cued(frame).merge(info, on="iso2")
    by_country, by_region = [], []
    for group, _, part in intervals.scopes(data, groups, per_model=False):
        for (iso2, name, region), cell in part.groupby(["iso2", "name_en", "who_region"]):
            keys = {"model_group": group, "iso2": iso2, "name_en": name, "who_region": region}
            for outcome in COUNTRY_OUTCOMES:
                result = intervals.share(cell, outcome, replicates, seed, (group, iso2))
                by_country += _row(keys, outcome, result)
        for region, cell in part.groupby("who_region"):
            keys = {"model_group": group, "who_region": region, "countries": cell["iso2"].nunique()}
            for outcome in COUNTRY_OUTCOMES:
                result = intervals.share(cell, outcome, replicates, seed, (group, region))
                by_region += _row(keys, outcome, result)
    return pd.DataFrame(by_country), pd.DataFrame(by_region)


def crisis_local_language(
    frame: pd.DataFrame, groups: Mapping[str, Sequence[str]], replicates: int, seed: int
) -> pd.DataFrame:
    """``correct_crisis`` of local-language S0 prompts with a cue, per model group and country
    (resampled by prompt) and over all countries (``iso2`` is ``all``, resampled by country)."""
    data = s0_cued(frame)
    data = data[data["lang_mode"] == "local"]
    rows = []
    for group, _, part in intervals.scopes(data, groups, per_model=False):
        cells = [*part.groupby("iso2", sort=True), (ALL, part)]
        for iso2, cell in cells:
            keys = {"model_group": group, "iso2": iso2}
            result = intervals.share(
                cell, "correct_crisis", replicates, seed, (group, iso2, "local")
            )
            rows += _row(keys, "correct_crisis", result)
    return pd.DataFrame(rows)


def worst_countries(by_country: pd.DataFrame, count: int) -> pd.DataFrame:
    """The ``count`` countries with the lowest ``correct_local`` over all models; ties by iso2.
    One column per outcome, model group and statistic, all models first."""
    index = ["iso2", "name_en", "who_region"]
    wide = by_country.pivot_table(
        index=index, columns=["model_group", "outcome"], values=["estimate", "ci_low", "ci_high"]
    )
    groups = [ALL, *sorted(set(by_country["model_group"]) - {ALL})]
    order = [
        (stat, group, outcome)
        for group in groups
        for outcome in COUNTRY_OUTCOMES
        for stat in ("estimate", "ci_low", "ci_high")
    ]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(order))
    wide.columns = [f"{outcome} {stat} ({group})" for stat, group, outcome in order]
    wide = wide.reset_index()
    return wide.sort_values([f"correct_local estimate ({ALL})", "iso2"]).head(count)


def directories(
    frame: pd.DataFrame, groups: Mapping[str, Sequence[str]], replicates: int, seed: int
) -> pd.DataFrame:
    """Responses that cite a directory, and that cite one with no number, per model and setting."""
    data = frame.assign(setting=setting(frame))
    rows = []
    for group, model, part in intervals.scopes(data, groups):
        for name, cell in part.groupby("setting", sort=True):
            keys = {"model_group": group, "model_id": model, "setting": name}
            for outcome in DIRECTORY_OUTCOMES:
                result = intervals.share(cell, outcome, replicates, seed, tuple(keys.values()))
                rows += _row(keys, outcome, result)
    return pd.DataFrame(rows)
