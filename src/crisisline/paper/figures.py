"""The paper's figures. For each, one function selects exactly the data it plots from the analysis
tables (written next to the figure as CSV, in plotted order) and one draws that data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from crisisline.analysis.descriptive import REFERRAL_CLASSES
from crisisline.paper import style, subsets
from crisisline.paper.style import ModelStyle
from crisisline.paper.subsets import REGIONS, Scope

CONDITIONS = {
    "S0-country": "Country named\nby the user",
    "M1": "Location in\nsystem prompt",
    "M2": "Location and\nnumbers in\nsystem prompt",
}
ALL = "all"  # the pooled rows of the exploratory tables
SHARE_AXIS = "Responses (%)"
CORRECT_AXIS = "Correct number (%)"


def _ordered_groups(styles: Mapping[str, ModelStyle]) -> list[str]:
    return list(dict.fromkeys(s.group for s in styles.values()))


def _offsets(count: int, step: float) -> np.ndarray:
    return step * (np.arange(count) - (count - 1) / 2)


def _group_handles(groups: Mapping[str, tuple[str, str]]) -> list[Line2D]:
    return [
        Line2D(
            [], [], color=c, marker=m, linestyle="none", label=style.group_label(g), **style.MARKER
        )
        for g, (c, m) in groups.items()
    ]


def _bold_labels(ax: Axes, axis: str, bold: Sequence[bool]) -> None:
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for label, is_bold in zip(labels, bold, strict=True):
        if is_bold:
            label.set_fontweight("bold")


# Figure 1: referral outcomes per model.


def referral_data(
    referral: pd.DataFrame, styles: Mapping[str, ModelStyle], scope: Scope
) -> pd.DataFrame:
    """Each model's referral classes, models sorted within their group by the share with a correct
    number; ``left`` is where a class's bar starts. The rows ``correct_local`` hold that share
    (the first two classes) and its interval, the second whisker. Countries with a crisis line."""
    shares = referral.set_index(["model_id", "referral"])
    subset = subsets.referral(scope)
    rows = []
    for group in _ordered_groups(styles):
        models = [m for m, s in styles.items() if s.group == group]
        for model in sorted(models, key=lambda m: -shares.loc[(m, "correct_local"), "share"]):
            left = 0.0
            for name in (*REFERRAL_CLASSES, "correct_local"):
                cell = shares.loc[(model, name)]
                rows.append(
                    {
                        "group": group,
                        "model_id": model,
                        "referral": name,
                        "left": 0.0 if name == "correct_local" else left,
                        "share": cell["share"],
                        "ci_low": cell["ci_low"],
                        "ci_high": cell["ci_high"],
                        "subset": subset,
                    }
                )
                left += cell["share"]
    return pd.DataFrame(rows)


def draw_referral(data: pd.DataFrame, path: Path) -> None:
    """Stacked bars per model under group headings; whiskers: correct crisis line, any correct."""
    with style.paper_style():
        fig = style.new_figure(style.COLUMN, 4.3)
        ax = fig.add_subplot()
        ticks, labels, bold, position = [], [], [], {}
        for group, part in data.groupby("group", sort=False):
            ticks.append(len(ticks))
            labels.append(style.group_label(group))
            bold.append(True)
            for model in part["model_id"].unique():
                position[model] = len(ticks)
                ticks.append(len(ticks))
                labels.append(model)
                bold.append(False)
        for name in REFERRAL_CLASSES:
            rows = data[data["referral"] == name]
            label, colour = style.REFERRAL[name]
            ax.barh(
                rows["model_id"].map(position),
                100 * rows["share"],
                left=100 * rows["left"],
                height=0.72,
                color=colour,
                edgecolor="white",
                linewidth=0.4,
                label=label,
            )
        for name in ("correct_crisis_line", "correct_local"):
            rows = data[data["referral"] == name]
            ax.errorbar(
                100 * rows["share"],
                rows["model_id"].map(position),
                xerr=style.percent_errors(rows["share"], rows["ci_low"], rows["ci_high"]),
                fmt="none",
                ecolor=style.DARK_GREY,
                elinewidth=0.8,
                capsize=1.5,
            )
        ax.set_yticks(ticks, labels)
        _bold_labels(ax, "y", bold)
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel(SHARE_AXIS)
        style.format_axis(ax, categorical="y")
        fig.legend(loc="outside lower center", ncols=1)
        style.save(fig, path)


# Figure 2: knowledge against location given in the system prompt.

GROUNDING_OUTCOMES = {"correct_crisis": "Correct crisis line", "fabricated": "Fabricated number"}


def grounding_data(grounding: pd.DataFrame, groups: Sequence[str], scope: Scope) -> pd.DataFrame:
    """Each group's pooled share per condition, local-language prompts."""
    rows = grounding[
        (grounding["model_id"] == ALL)
        & grounding["model_group"].isin(groups)
        & grounding["measure"].isin(CONDITIONS)
        & grounding["outcome"].isin(GROUNDING_OUTCOMES)
    ]
    order = {
        "outcome": list(GROUNDING_OUTCOMES),
        "model_group": list(groups),
        "measure": list(CONDITIONS),
    }
    rows = rows.sort_values(list(order), key=lambda c: c.map(order[c.name].index))
    rows = rows.assign(
        subset=[
            subsets.condition(scope, m, o)
            for m, o in zip(rows["measure"], rows["outcome"], strict=True)
        ]
    )
    columns = ["outcome", "model_group", "measure", "estimate", "ci_low", "ci_high", "subset"]
    return rows[columns].rename(columns={"measure": "condition"})


def draw_grounding(data: pd.DataFrame, groups: Mapping[str, tuple[str, str]], path: Path) -> None:
    with style.paper_style():
        fig = style.new_figure(style.FULL, 2.45)
        axes = fig.subplots(1, len(GROUNDING_OUTCOMES))
        offsets = _offsets(len(groups), 0.1)
        for ax, (outcome, title) in zip(axes, GROUNDING_OUTCOMES.items(), strict=True):
            for offset, (group, (colour, marker)) in zip(offsets, groups.items(), strict=True):
                rows = data[(data["outcome"] == outcome) & (data["model_group"] == group)]
                ax.errorbar(
                    rows["condition"].map(list(CONDITIONS).index) + offset,
                    100 * rows["estimate"],
                    yerr=style.percent_errors(rows["estimate"], rows["ci_low"], rows["ci_high"]),
                    fmt=f"-{marker}",
                    color=colour,
                    linewidth=1.0,
                    **style.ERRORBAR,
                )
            ax.set_xticks(range(len(CONDITIONS)), list(CONDITIONS.values()))
            ax.set_xlim(-0.4, len(CONDITIONS) - 0.6)
            ax.set_title(title)
            ax.set_ylim(bottom=0, top=100 if outcome == "correct_crisis" else None)
            style.format_axis(ax, categorical="x")
        axes[0].set_ylabel(SHARE_AXIS)
        fig.legend(handles=_group_handles(groups), loc="outside lower center", ncols=len(groups))
        style.save(fig, path)


# Figure 3: prompt language.

LANGUAGES = {"en": "English", "local": "Local language"}


def language_data(
    by_model: pd.DataFrame,
    by_language: pd.DataFrame,
    styles: Mapping[str, ModelStyle],
    groups: Sequence[str],
    scope: Scope,
) -> pd.DataFrame:
    """(a) each model's correct number for English and local-language prompts that name the
    country; (b) each group's correct number per prompt language, languages from the lowest share
    over all models."""
    model_rows = [
        {
            "panel": "by model",
            "model_group": styles[model].group,
            "model_id": model,
            "lang": "",
            "prompt_language": language,
            "estimate": row[f"share_{language}"],
            "ci_low": row[f"share_{language}_ci_low"],
            "ci_high": row[f"share_{language}_ci_high"],
            "subset": subsets.language_by_model(scope, language),
        }
        for model in styles
        for row in [by_model.set_index("model_id").loc[model]]
        for language in LANGUAGES
    ]
    pooled = by_language[by_language["model_group"] == ALL].set_index("lang")["estimate"]
    order = sorted(pooled.index, key=lambda lang: (pooled[lang], lang))
    lang_rows = (
        by_language[by_language["model_group"].isin(groups)]
        .assign(
            panel="by language",
            model_id="",
            prompt_language="",
            subset=lambda t: [
                subsets.language(lang, n) for lang, n in zip(t["lang"], t["countries"], strict=True)
            ],
            lang_rank=lambda t: t["lang"].map(order.index),
            group_rank=lambda t: t["model_group"].map(list(groups).index),
        )
        .sort_values(["lang_rank", "group_rank"])
    )
    columns = list(model_rows[0])
    return pd.concat([pd.DataFrame(model_rows), lang_rows[columns]], ignore_index=True)


def draw_language(
    data: pd.DataFrame,
    styles: Mapping[str, ModelStyle],
    groups: Mapping[str, tuple[str, str]],
    path: Path,
) -> None:
    """(a) a dumbbell per model: English filled, local language hollow; (b) a dot per group and
    language."""
    with style.paper_style():
        fig = style.new_figure(style.FULL, 4.4)
        left, right = fig.subplots(1, 2)
        models = data[data["panel"] == "by model"]
        order = list(models["model_id"].unique())
        group_marker = {m: groups[s.group][1] for m, s in styles.items()}
        for y, model in enumerate(order):
            rows = models[models["model_id"] == model].set_index("prompt_language")
            colour = styles[model].colour
            shifts = (-0.15, 0.15)
            left.plot(
                100 * rows.loc[list(LANGUAGES), "estimate"],
                [y + shift for shift in shifts],
                color=colour,
                linewidth=0.8,
                alpha=0.6,
                zorder=1,
            )
            for shift, language in zip(shifts, LANGUAGES, strict=True):
                row = rows.loc[language]
                left.errorbar(
                    [100 * row["estimate"]],
                    [y + shift],
                    xerr=style.percent_errors([row["estimate"]], [row["ci_low"]], [row["ci_high"]]),
                    fmt=group_marker[model],
                    color=colour,
                    markerfacecolor=colour if language == "en" else "white",
                    zorder=2,
                    **style.ERRORBAR,
                )
        left.set_yticks(range(len(order)), order)
        left.set_title("(a) By model, country named")
        langs = data[data["panel"] == "by language"]
        lang_order = list(langs["lang"].unique())
        offsets = _offsets(len(groups), 0.24)
        for offset, (group, (colour, marker)) in zip(offsets, groups.items(), strict=True):
            rows = langs[langs["model_group"] == group]
            right.errorbar(
                100 * rows["estimate"],
                rows["lang"].map(lang_order.index) + offset,
                xerr=style.percent_errors(rows["estimate"], rows["ci_low"], rows["ci_high"]),
                fmt=marker,
                color=colour,
                **style.ERRORBAR,
            )
        right.set_yticks(range(len(lang_order)), lang_order)
        right.set_title("(b) By prompt language")
        for ax in (left, right):
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.set_xlabel(CORRECT_AXIS)
            ax.tick_params(axis="y", length=0)
            style.format_axis(ax, categorical="y")
        fill = [
            Line2D(
                [],
                [],
                color=style.GREY,
                marker="o",
                linestyle="none",
                markerfacecolor=style.GREY if language == "en" else "white",
                label=f"{label} prompt",
                **style.MARKER,
            )
            for language, label in LANGUAGES.items()
        ]
        fig.legend(
            handles=_group_handles(groups) + fill,
            loc="outside lower center",
            ncols=len(groups) + len(fill),
        )
        style.save(fig, path)


# Figure 4: geography.

GEOGRAPHY_OUTCOMES = {
    "correct_local": "Any correct number",
    "correct_crisis": "Correct crisis line",
}
WORST_MARKERS = {"correct_local": (style.DARK_GREY, "o"), "correct_crisis": (style.PALETTE[0], "s")}


def geography_data(
    who_regions: pd.DataFrame, worst: pd.DataFrame, groups: Sequence[str], scope: Scope
) -> pd.DataFrame:
    """Each group's shares per WHO region, regions from the lowest share of a correct number over
    all models; then the countries with the lowest share, all models."""
    pooled = who_regions[
        (who_regions["model_group"] == ALL) & (who_regions["outcome"] == "correct_local")
    ].set_index("who_region")["estimate"]
    regions = sorted(pooled.index, key=lambda r: (pooled[r], r))
    region_rows = (
        who_regions[
            who_regions["model_group"].isin(groups)
            & who_regions["outcome"].isin(GEOGRAPHY_OUTCOMES)
        ]
        .assign(
            panel="region",
            iso2="",
            name_en="",
            outcome_rank=lambda t: t["outcome"].map(list(GEOGRAPHY_OUTCOMES).index),
            region_rank=lambda t: t["who_region"].map(regions.index),
            group_rank=lambda t: t["model_group"].map(list(groups).index),
            subset=lambda t: [
                subsets.region(scope, r, o)
                for r, o in zip(t["who_region"], t["outcome"], strict=True)
            ],
        )
        .sort_values(["outcome_rank", "region_rank", "group_rank"])
    )
    country_rows = [
        {
            "panel": "country",
            "outcome": outcome,
            "who_region": row["who_region"],
            "iso2": row["iso2"],
            "name_en": row["name_en"],
            "model_group": ALL,
            "estimate": row[f"{outcome} estimate ({ALL})"],
            "ci_low": row[f"{outcome} ci_low ({ALL})"],
            "ci_high": row[f"{outcome} ci_high ({ALL})"],
            "subset": subsets.country(row["name_en"]),
        }
        for outcome in GEOGRAPHY_OUTCOMES
        for _, row in worst.iterrows()
        if pd.notna(row[f"{outcome} estimate ({ALL})"])
    ]
    columns = list(country_rows[0])
    return pd.concat([region_rows[columns], pd.DataFrame(country_rows)], ignore_index=True)


def draw_geography(data: pd.DataFrame, groups: Mapping[str, tuple[str, str]], path: Path) -> None:
    with style.paper_style():
        fig = style.new_figure(style.FULL, 2.8)
        axes = fig.subplots(1, 3, width_ratios=(1, 1, 1.15))
        regions = data[data["panel"] == "region"]
        region_order = list(regions["who_region"].unique())
        offsets = _offsets(len(groups), 0.24)
        for ax, (outcome, title), letter in zip(
            axes[:2], GEOGRAPHY_OUTCOMES.items(), "ab", strict=True
        ):
            for offset, (group, (colour, marker)) in zip(offsets, groups.items(), strict=True):
                rows = regions[(regions["outcome"] == outcome) & (regions["model_group"] == group)]
                ax.errorbar(
                    100 * rows["estimate"],
                    rows["who_region"].map(region_order.index) + offset,
                    xerr=style.percent_errors(rows["estimate"], rows["ci_low"], rows["ci_high"]),
                    fmt=marker,
                    color=colour,
                    **style.ERRORBAR,
                )
            ax.set_yticks(range(len(region_order)), [REGIONS[r] for r in region_order])
            ax.set_title(f"({letter}) {title}")
        axes[1].tick_params(axis="y", labelleft=False)
        countries = data[data["panel"] == "country"]
        country_order = list(dict.fromkeys(countries["name_en"]))
        worst_handles = []
        for offset, (outcome, (colour, marker)) in zip(
            _offsets(2, 0.3), WORST_MARKERS.items(), strict=True
        ):
            rows = countries[countries["outcome"] == outcome]
            ax = axes[2]
            ax.errorbar(
                100 * rows["estimate"],
                rows["name_en"].map(country_order.index) + offset,
                xerr=style.percent_errors(rows["estimate"], rows["ci_low"], rows["ci_high"]),
                fmt=marker,
                color=colour,
                **style.ERRORBAR,
            )
            worst_handles.append(
                Line2D(
                    [],
                    [],
                    color=colour,
                    marker=marker,
                    linestyle="none",
                    label=f"All models, {GEOGRAPHY_OUTCOMES[outcome].lower()}",
                    **style.MARKER,
                )
            )
        axes[2].set_yticks(range(len(country_order)), country_order)
        axes[2].set_title("(c) Lowest-scoring countries")
        for ax in axes:
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.set_xlabel(SHARE_AXIS)
            ax.tick_params(axis="y", length=0)
            style.format_axis(ax, categorical="y")
        fig.legend(
            handles=_group_handles(groups) + worst_handles,
            loc="outside lower center",
            ncols=len(groups) + len(worst_handles),
            columnspacing=0.9,
        )
        style.save(fig, path)


# Appendix: the Qwen size ladder.

CUES = {"country": ("Country named", "-", "o"), "city": ("City named", "--", "s")}


def ladder_data(ladder: pd.DataFrame, sizes: Mapping[str, float], scope: Scope) -> pd.DataFrame:
    """Each model's share of a correct number against its size, prompts that name a city or the
    country."""
    rows = ladder[ladder["outcome"] == "correct_local"].assign(
        size=lambda t: t["model_id"].map(sizes)
    )
    rows = rows.assign(cue_rank=rows["cue"].map(list(CUES).index)).sort_values(["cue_rank", "size"])
    rows = rows.assign(subset=[subsets.cue(scope, c) for c in rows["cue"]])
    return rows[["cue", "model_id", "size", "share", "ci_low", "ci_high", "subset"]]


def draw_ladder(data: pd.DataFrame, colour: str, path: Path) -> None:
    with style.paper_style():
        fig = style.new_figure(style.COLUMN, 2.4)
        ax = fig.add_subplot()
        for cue, (label, line, marker) in CUES.items():
            rows = data[data["cue"] == cue]
            ax.errorbar(
                rows["size"],
                100 * rows["share"],
                yerr=style.percent_errors(rows["share"], rows["ci_low"], rows["ci_high"]),
                fmt=marker,
                linestyle=line,
                color=colour,
                linewidth=1.0,
                label=label,
                **style.ERRORBAR,
            )
        sizes = sorted(data["size"].unique())
        ax.set_xscale("log")
        style.format_axis(ax)
        ax.set_xticks(sizes, [f"{s:g}B" for s in sizes])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0, 100)
        ax.set_xlabel("Parameters")
        ax.set_ylabel(CORRECT_AXIS)
        fig.legend(loc="outside lower center", ncols=len(CUES))
        style.save(fig, path)


# Appendix: every country, by WHO region.


def countries_data(countries: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    """Each group's share of a correct number per country; countries by WHO region, then from the
    highest share over all models."""
    data = countries[countries["outcome"] == "correct_local"]
    pooled = data[data["model_group"] == ALL].set_index("iso2")["estimate"]
    rows = data[data["model_group"].isin(groups)].assign(
        pooled=lambda t: t["iso2"].map(pooled),
        group_rank=lambda t: t["model_group"].map(list(groups).index),
    )
    rows = rows.sort_values(
        ["who_region", "pooled", "iso2", "group_rank"], ascending=[True, False, True, True]
    )
    rows = rows.assign(subset=rows["name_en"].map(subsets.country))
    columns = ["who_region", "iso2", "name_en", "model_group", "estimate", "ci_low", "ci_high"]
    return rows[[*columns, "subset"]]


def draw_countries(data: pd.DataFrame, groups: Mapping[str, tuple[str, str]], path: Path) -> None:
    with style.paper_style():
        fig = style.new_figure(style.FULL, 3.0)
        ax = fig.add_subplot()
        order = list(dict.fromkeys(data["iso2"]))
        for offset, (group, (colour, marker)) in zip(
            _offsets(len(groups), 0.22), groups.items(), strict=True
        ):
            rows = data[data["model_group"] == group]
            ax.errorbar(
                rows["iso2"].map(order.index) + offset,
                100 * rows["estimate"],
                yerr=style.percent_errors(rows["estimate"], rows["ci_low"], rows["ci_high"]),
                fmt=marker,
                color=colour,
                **{**style.ERRORBAR, "markersize": 3.2, "elinewidth": 0.7},
            )
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=8)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.7, len(order) - 0.3)
        ax.set_ylim(0, 100)
        ax.set_ylabel(CORRECT_AXIS)
        style.format_axis(ax, categorical="x")
        regions = data.drop_duplicates("iso2")
        for region, part in regions.groupby("who_region", sort=False):
            first, last = order.index(part["iso2"].iloc[0]), order.index(part["iso2"].iloc[-1])
            ax.text(
                (first + last) / 2,
                1.01,
                REGIONS[region].replace(" ", "\n", 1) if last - first < 6 else REGIONS[region],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
            )
            if last < len(order) - 1:
                ax.axvline(last + 0.5, color=style.GREY, linewidth=0.6)
        fig.legend(handles=_group_handles(groups), loc="outside lower center", ncols=len(groups))
        style.save(fig, path)
