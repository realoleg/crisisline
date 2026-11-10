"""The paper's figures. For each, one function selects exactly the data it plots from the analysis
tables (written next to the figure as CSV, in plotted order) and one draws that data. Each figure
states one message in saturated colour; context is grey; key numbers are written from the data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, NullLocator

from crisisline.analysis.descriptive import REFERRAL_CLASSES
from crisisline.paper import style, subsets
from crisisline.paper.style import REGION_LABELS, ModelStyle
from crisisline.paper.subsets import Scope

CONDITIONS = {
    "S0-country": "User names\nthe country",
    "M1": "Country in\nthe system\nprompt",
    "M2": "Verified\nnumber in the\nsystem prompt",
}
ALL = "all"  # the pooled rows of the exploratory tables
SHARE_AXIS = "Responses (%)"
CORRECT_AXIS = "Correct number (%)"
CRISIS_AXIS = "Correct crisis line (%)"
SMALL = 8  # points: direct labels and dense tick labels, the smallest size the template allows

Groups = Mapping[str, tuple[str, str]]


def _ordered_groups(styles: Mapping[str, ModelStyle]) -> list[str]:
    return list(dict.fromkeys(s.group for s in styles.values()))


def _offsets(count: int, step: float) -> np.ndarray:
    return step * (np.arange(count) - (count - 1) / 2)


def _group_handles(groups: Groups) -> list[Line2D]:
    return [
        Line2D(
            [], [], color=c, marker=m, linestyle="none", label=style.group_label(g), **style.MARKER
        )
        for g, (c, m) in groups.items()
    ]


def _context_in_grey(groups: Groups, data: pd.DataFrame, value: str = "estimate") -> Groups:
    """``groups`` with the strongest one, the context of a figure about failures, in grey."""
    means = data[data["model_group"].isin(groups)].groupby("model_group")[value].mean()
    strongest = means.idxmax()
    return {g: ((style.GREY, m) if g == strongest else (c, m)) for g, (c, m) in groups.items()}


def _bold_labels(ax: Axes, axis: str, bold: Sequence[bool]) -> None:
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for label, is_bold in zip(labels, bold, strict=True):
        if is_bold:
            label.set_fontweight("bold")


def _named(table: pd.DataFrame) -> pd.DataFrame:
    """``table`` with the display name of each model next to its model_id."""
    names = table["model_id"].map(lambda m: style.model_name(m) if m else "")
    return table.assign(model=names)


def _model_rows(ax: Axes, groups: Sequence[tuple[str, list[str]]]) -> dict[str, int]:
    """Model names under bold group headings on the y axis, top to bottom; positions by model."""
    ticks, labels, bold, position = [], [], [], {}
    for group, members in groups:
        ticks.append(len(ticks))
        labels.append(style.group_label(group))
        bold.append(True)
        for model in members:
            position[model] = len(ticks)
            ticks.append(len(ticks))
            labels.append(style.model_name(model))
            bold.append(False)
    ax.set_yticks(ticks, labels)
    _bold_labels(ax, "y", bold)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(ticks) - 0.5, -0.6)
    return position


def _grouped(data: pd.DataFrame, group: str = "group") -> list[tuple[str, list[str]]]:
    return [
        (g, list(dict.fromkeys(part["model_id"]))) for g, part in data.groupby(group, sort=False)
    ]


def _value_axis_to_100(ax: Axes, room: float) -> None:
    """A 0-100 axis that leaves ``room`` to its right for direct labels; call after
    :func:`style.format_axis`."""
    ax.set_xlim(0, 100 + room)
    ax.set_xticks(range(0, 101, 20))
    ax.xaxis.set_minor_locator(FixedLocator(range(0, 101, 5)))
    ax.spines["bottom"].set_bounds(0, 100)


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
    return _named(pd.DataFrame(rows))


def draw_referral(data: pd.DataFrame, path: Path) -> None:
    """Stacked bars per model under group headings, the crisis-line share written in its bar;
    whiskers: correct crisis line, any correct number."""
    with style.paper_style():
        fig = style.new_figure(style.FULL, 3.2)
        ax = fig.add_subplot()
        position = _model_rows(ax, _grouped(data))
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
        for _, row in data[data["referral"] == "correct_crisis_line"].iterrows():
            ax.text(
                2,
                position[row["model_id"]],
                style.percent(row["share"]),
                color="white",
                fontsize=SMALL,
                ha="left",
                va="center",
            )
        ax.set_xlim(0, 100)
        ax.set_xlabel(SHARE_AXIS)
        style.format_axis(ax, categorical="y")
        fig.legend(loc="outside lower center", ncols=3)
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


def _condition_lines(ax: Axes, data: pd.DataFrame, groups: Groups, top: float) -> None:
    """One line per group across the conditions, labelled at its first point."""
    offsets = _offsets(len(groups), 0.08)
    starts = []
    for offset, (group, (colour, marker)) in zip(offsets, groups.items(), strict=True):
        rows = data[data["model_group"] == group]
        ax.errorbar(
            rows["condition"].map(list(CONDITIONS).index) + offset,
            100 * rows["estimate"],
            yerr=style.percent_errors(rows["estimate"], rows["ci_low"], rows["ci_high"]),
            fmt=f"-{marker}",
            color=colour,
            linewidth=1.0,
            **style.ERRORBAR,
        )
        starts.append(100 * rows["estimate"].iloc[0])
    for y, (group, (colour, _)) in zip(
        style.spread(starts, 0.09 * top), groups.items(), strict=True
    ):
        ax.text(-0.15, y, style.group_label(group), color=colour, ha="right", va="center")
    ax.set_xticks(range(len(CONDITIONS)), list(CONDITIONS.values()), fontsize=SMALL)
    ax.set_xlim(-0.9, len(CONDITIONS) - 0.7)
    ax.set_ylim(0, 1.04 * top)
    style.format_axis(ax, categorical="x")
    if top == 100:
        ax.set_yticks(range(0, 101, 20))
        ax.spines["left"].set_bounds(0, 100)


def draw_grounding(data: pd.DataFrame, groups: Groups, path: Path) -> None:
    with style.paper_style():
        fig = style.new_figure(style.FULL, 2.4)
        axes = fig.subplots(1, len(GROUNDING_OUTCOMES))
        for ax, (outcome, title) in zip(axes, GROUNDING_OUTCOMES.items(), strict=True):
            rows = data[data["outcome"] == outcome]
            top = 100 if outcome == "correct_crisis" else 100 * rows["ci_high"].max() * 1.1
            _condition_lines(ax, rows, groups, top)
            ax.set_title(title)
        axes[0].set_ylabel(SHARE_AXIS)
        style.save(fig, path)


# Figure 3: prompt language.

LANGUAGES = {"en": "English", "local": "Local-language"}


def _gap_rows(
    by_model: pd.DataFrame, models: Sequence[str], styles: Mapping[str, ModelStyle], scope: Scope
) -> list[dict[str, object]]:
    """Each model's correct number for English and local-language prompts that name the country,
    with the gap and whether its interval excludes zero."""
    table = by_model.set_index("model_id")
    rows = []
    for model in models:
        row = table.loc[model]
        low, high = row["difference_ci_low"], row["difference_ci_high"]
        for language in LANGUAGES:
            rows.append(
                {
                    "model_group": styles[model].group,
                    "model_id": model,
                    "prompt_language": language,
                    "estimate": row[f"share_{language}"],
                    "ci_low": row[f"share_{language}_ci_low"],
                    "ci_high": row[f"share_{language}_ci_high"],
                    "difference_en_minus_local": row["difference_en_minus_local"],
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "significant_gap": bool(low > 0 or high < 0),
                    "subset": subsets.language_by_model(scope, language),
                }
            )
    return rows


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
    model_rows = pd.DataFrame(_gap_rows(by_model, list(styles), styles, scope)).assign(
        panel="by model", lang=""
    )
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
    columns = ["panel", "model_group", "model_id", "lang", "prompt_language"]
    columns += ["estimate", "ci_low", "ci_high", "difference_en_minus_local"]
    columns += ["difference_ci_low", "difference_ci_high", "significant_gap", "subset"]
    return _named(pd.concat([model_rows, lang_rows], ignore_index=True)[columns])


def _dumbbells(
    ax: Axes, data: pd.DataFrame, styles: Mapping[str, ModelStyle], groups: Groups
) -> None:
    """English (filled) and local-language (hollow) shares per model, joined; the models whose gap
    excludes zero in their group's colour with the two shares written, the others grey."""
    order = list(dict.fromkeys(data["model_id"]))
    for y, model in enumerate(order):
        rows = data[data["model_id"] == model].set_index("prompt_language")
        significant = bool(rows["significant_gap"].iloc[0])
        colour = styles[model].colour if significant else style.GREY
        marker = groups[styles[model].group][1]
        ax.plot(
            100 * rows.loc[list(LANGUAGES), "estimate"],
            [y, y],
            color=colour if significant else style.LIGHT_GREY,
            linewidth=1.2,
            zorder=1,
        )
        for language in LANGUAGES:
            row = rows.loc[language]
            ax.errorbar(
                [100 * row["estimate"]],
                [y],
                xerr=style.percent_errors([row["estimate"]], [row["ci_low"]], [row["ci_high"]]),
                fmt=marker,
                color=colour,
                markerfacecolor=colour if language == "en" else "white",
                zorder=2,
                **{**style.ERRORBAR, "elinewidth": 0.6},
            )
        if significant:
            shares = " → ".join(style.percent(rows.loc[lang, "estimate"]) for lang in LANGUAGES)
            ax.text(103, y, shares, color=colour, fontsize=SMALL, va="center")
    ax.set_yticks(range(len(order)), [style.model_name(m) for m in order])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(order) - 0.5, -0.5)


def _language_keys() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            color=style.DARK_GREY,
            marker="o",
            linestyle="none",
            markerfacecolor=style.DARK_GREY if language == "en" else "white",
            label=f"{label} prompt",
            **style.MARKER,
        )
        for language, label in LANGUAGES.items()
    ]


def draw_language(
    data: pd.DataFrame, styles: Mapping[str, ModelStyle], groups: Groups, path: Path
) -> None:
    """(a) a dumbbell per model; (b) a dot per group and language, the strongest group grey."""
    with style.paper_style():
        fig = style.new_figure(style.FULL, 4.3)
        left, right = fig.subplots(1, 2, width_ratios=(1.15, 1))
        _dumbbells(left, data[data["panel"] == "by model"], styles, groups)
        left.set_title("(a) By model, country named")
        langs = data[data["panel"] == "by language"]
        shown = _context_in_grey(groups, langs)
        lang_order = list(langs["lang"].unique())
        for offset, (group, (colour, marker)) in zip(
            _offsets(len(shown), 0.24), shown.items(), strict=True
        ):
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
        right.set_ylim(len(lang_order) - 0.5, -0.5)
        right.set_xlim(0, 100)
        right.set_title("(b) By prompt language")
        for ax in (left, right):
            ax.set_xlabel(CORRECT_AXIS)
            ax.tick_params(axis="y", length=0)
            style.format_axis(ax, categorical="y")
        _value_axis_to_100(left, 32)
        fig.legend(
            handles=_group_handles(shown) + _language_keys(),
            loc="outside lower center",
            ncols=len(shown) + len(LANGUAGES),
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


def draw_geography(data: pd.DataFrame, groups: Groups, path: Path) -> None:
    with style.paper_style():
        fig = style.new_figure(style.FULL, 2.7)
        axes = fig.subplots(1, 3, width_ratios=(1, 1, 1.15))
        regions = data[data["panel"] == "region"]
        shown = _context_in_grey(groups, regions)
        region_order = list(regions["who_region"].unique())
        offsets = _offsets(len(shown), 0.24)
        for ax, (outcome, title), letter in zip(
            axes[:2], GEOGRAPHY_OUTCOMES.items(), "ab", strict=True
        ):
            for offset, (group, (colour, marker)) in zip(offsets, shown.items(), strict=True):
                rows = regions[(regions["outcome"] == outcome) & (regions["model_group"] == group)]
                ax.errorbar(
                    100 * rows["estimate"],
                    rows["who_region"].map(region_order.index) + offset,
                    xerr=style.percent_errors(rows["estimate"], rows["ci_low"], rows["ci_high"]),
                    fmt=marker,
                    color=colour,
                    **style.ERRORBAR,
                )
            ax.set_yticks(range(len(region_order)), [REGION_LABELS[r] for r in region_order])
            ax.set_ylim(len(region_order) - 0.5, -0.5)
            ax.set_title(f"({letter}) {title}")
        axes[1].tick_params(axis="y", labelleft=False)
        countries = data[data["panel"] == "country"]
        country_order = list(dict.fromkeys(countries["name_en"]))
        worst_handles = []
        for offset, (outcome, (colour, marker)) in zip(
            _offsets(2, 0.3), WORST_MARKERS.items(), strict=True
        ):
            rows = countries[countries["outcome"] == outcome]
            axes[2].errorbar(
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
        axes[2].set_ylim(len(country_order) - 0.5, -0.5)
        axes[2].set_title("(c) Lowest-scoring countries")
        for ax in axes:
            ax.set_xlim(0, 100)
            ax.set_xlabel(SHARE_AXIS)
            ax.tick_params(axis="y", length=0)
            style.format_axis(ax, categorical="y")
        fig.legend(
            handles=_group_handles(shown) + worst_handles,
            loc="outside lower center",
            ncols=len(shown) + len(worst_handles),
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
    return _named(rows[["cue", "model_id", "size", "share", "ci_low", "ci_high", "subset"]])


def draw_ladder(data: pd.DataFrame, colour: str, path: Path) -> None:
    """Both cues; the smallest and the largest model's share at the country cue written."""
    with style.paper_style():
        fig = style.new_figure(style.COLUMN, 2.3)
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
        first = next(iter(CUES))
        ends = data[data["cue"] == first].iloc[[0, -1]]
        for (_, row), ha in zip(ends.iterrows(), ("left", "right"), strict=True):
            ax.annotate(
                style.percent(row["share"]),
                (row["size"], 100 * row["share"]),
                xytext=(0, -13),
                textcoords="offset points",
                ha=ha,
                fontsize=SMALL,
                color=colour,
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


def _region_labels(ax: Axes, regions: pd.Series, order: Sequence[str], line: str) -> None:
    """WHO region names above the columns of their countries, and a separator between regions."""
    for region, part in regions.groupby(regions, sort=False):
        first, last = order.index(part.index[0]), order.index(part.index[-1])
        ax.text(
            (first + last) / 2,
            1.01,
            REGION_LABELS[region],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=SMALL,
        )
        if last < len(order) - 1:
            ax.axvline(last + 0.5, color=line, linewidth=0.8)


def draw_countries(data: pd.DataFrame, groups: Groups, path: Path) -> None:
    with style.paper_style():
        fig = style.new_figure(style.FULL, 2.9)
        ax = fig.add_subplot()
        shown = _context_in_grey(groups, data)
        order = list(dict.fromkeys(data["iso2"]))
        for offset, (group, (colour, marker)) in zip(
            _offsets(len(shown), 0.22), shown.items(), strict=True
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
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=SMALL)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.7, len(order) - 0.3)
        ax.set_ylim(0, 100)
        ax.set_ylabel(CORRECT_AXIS)
        style.format_axis(ax, categorical="x")
        regions = data.drop_duplicates("iso2").set_index("iso2")["who_region"]
        _region_labels(ax, regions, order, style.GREY)
        fig.legend(handles=_group_handles(shown), loc="outside lower center", ncols=len(shown))
        style.save(fig, path)


# Who gets help: the crisis line per model and country.

CELL, MODEL_MARGIN, COUNTRY_MARGIN, LOWEST = "cell", "model share", "country share", "lowest"


def who_gets_help_data(
    cells: pd.DataFrame,
    model_shares: pd.DataFrame,
    countries: pd.DataFrame,
    styles: Mapping[str, ModelStyle],
    scope: Scope,
    lowest: int = 3,
) -> pd.DataFrame:
    """The crisis-line share of every model in every country (S0 with a cue), the margins (each
    model over all countries, each country over all models), and the countries with the lowest
    share for the weakest group: the ``lowest`` lowest, and every country tied with the last."""
    countries = countries[countries["outcome"] == "correct_crisis"]
    margin = model_shares.set_index("model_id")
    order = []
    for group in _ordered_groups(styles):
        members = [m for m, s in styles.items() if s.group == group]
        order += sorted(members, key=lambda m: (-margin.loc[m, "share"], m))
    pooled = countries[countries["model_group"] == ALL].set_index("iso2")
    columns = []
    for region in REGION_LABELS:
        listed = [c for c in scope.countries if c.region == region]
        with_line = [c for c in listed if c.crisis_line]
        with_line.sort(key=lambda c: (-pooled.loc[c.iso2, "estimate"], c.iso2))
        columns += with_line + sorted(
            (c for c in listed if not c.crisis_line), key=lambda c: c.iso2
        )
    shares = cells.set_index(["model_id", "iso2"])
    rows = []
    for model in order:
        for country in columns:
            found = (model, country.iso2) in shares.index
            cell = shares.loc[(model, country.iso2)] if found else None
            rows.append(
                {
                    "panel": CELL,
                    "model_group": styles[model].group,
                    "model_id": model,
                    "iso2": country.iso2,
                    "name_en": country.name,
                    "who_region": country.region,
                    "has_crisis_line": country.crisis_line,
                    "share": cell["share"] if found else np.nan,
                    "ci_low": cell["ci_low"] if found else np.nan,
                    "ci_high": cell["ci_high"] if found else np.nan,
                    "n": cell["responses"] if found else np.nan,
                    "subset": subsets.country(country.name),
                }
            )
        rows.append(
            {
                "panel": MODEL_MARGIN,
                "model_group": styles[model].group,
                "model_id": model,
                "share": margin.loc[model, "share"],
                "ci_low": margin.loc[model, "ci_low"],
                "ci_high": margin.loc[model, "ci_high"],
                "n": margin.loc[model, "responses"],
                "subset": subsets.all_regions(scope, "correct_crisis"),
            }
        )
    for country in columns:
        found = country.iso2 in pooled.index and country.crisis_line
        rows.append(
            {
                "panel": COUNTRY_MARGIN,
                "model_group": ALL,
                "iso2": country.iso2,
                "name_en": country.name,
                "who_region": country.region,
                "has_crisis_line": country.crisis_line,
                "share": pooled.loc[country.iso2, "estimate"] if found else np.nan,
                "ci_low": pooled.loc[country.iso2, "ci_low"] if found else np.nan,
                "ci_high": pooled.loc[country.iso2, "ci_high"] if found else np.nan,
                "n": pooled.loc[country.iso2, "responses"] if found else np.nan,
                "subset": subsets.country(country.name),
            }
        )
    group_means = {
        g: np.mean([margin.loc[m, "share"] for m in order if styles[m].group == g])
        for g in _ordered_groups(styles)
    }
    weakest = min(group_means, key=group_means.get)
    by_group = countries[countries["model_group"] == weakest].set_index("iso2")["estimate"]
    cutoff = by_group.nsmallest(lowest).max()
    for country in columns:
        if country.iso2 in by_group.index and by_group[country.iso2] <= cutoff:
            row = countries[
                (countries["model_group"] == weakest) & (countries["iso2"] == country.iso2)
            ].iloc[0]
            rows.append(
                {
                    "panel": LOWEST,
                    "model_group": weakest,
                    "iso2": country.iso2,
                    "name_en": country.name,
                    "who_region": country.region,
                    "has_crisis_line": True,
                    "share": row["estimate"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "n": row["responses"],
                    "subset": subsets.country(country.name),
                }
            )
    table = pd.DataFrame(rows)
    return _named(table.assign(model_id=table["model_id"].fillna("")))


def _crisis_cells(ax: Axes, values: np.ndarray, no_line: Sequence[int]) -> AxesImage:
    image = ax.imshow(
        100 * values,
        cmap=style.CRISIS_MAP,
        vmin=0,
        vmax=100,
        aspect="auto",
        interpolation="nearest",
    )
    for column in no_line:
        ax.add_patch(
            Rectangle(
                (column - 0.5, -0.5),
                1,
                values.shape[0],
                facecolor=style.NO_LINE,
                edgecolor=style.GREY,
                hatch="//////",
                linewidth=0,
            )
        )
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _share_label_colour(share: float) -> str:
    return "white" if share < 0.2 or share > 0.8 else style.DARK_GREY


def draw_who_gets_help(data: pd.DataFrame, path: Path) -> None:
    """Rows: models; columns: countries by WHO region; colour: the crisis-line share. A margin
    column per model and a margin row per country; the weakest group's lowest countries outlined."""
    cells = data[data["panel"] == CELL]
    models = list(dict.fromkeys(cells["model_id"]))
    countries = list(dict.fromkeys(cells["iso2"]))
    matrix = cells.pivot(index="model_id", columns="iso2", values="share").loc[models, countries]
    line = cells.drop_duplicates("iso2").set_index("iso2")["has_crisis_line"].loc[countries]
    no_line = [k for k, has in enumerate(line) if not has]
    with style.paper_style(), style.hatches():
        fig = style.new_figure(style.FULL, 3.6)
        grid = fig.add_gridspec(2, 2, width_ratios=(len(countries), 2.4), height_ratios=(11, 1.1))
        ax = fig.add_subplot(grid[0, 0])
        right = fig.add_subplot(grid[0, 1], sharey=ax)
        bottom = fig.add_subplot(grid[1, 0], sharex=ax)
        image = _crisis_cells(ax, matrix.to_numpy(dtype=float), no_line)
        groups = cells.drop_duplicates("model_id").set_index("model_id")["model_group"]
        for k in range(1, len(models)):
            if groups.iloc[k] != groups.iloc[k - 1]:
                ax.axhline(k - 0.5, color="white", linewidth=2.0)
        regions = cells.drop_duplicates("iso2").set_index("iso2")["who_region"].loc[countries]
        for k in range(1, len(countries)):
            if regions.iloc[k] != regions.iloc[k - 1]:
                for axis in (ax, bottom):
                    axis.axvline(k - 0.5, color="white", linewidth=2.0)
        top = ax.secondary_xaxis("top")
        centres = [
            float(np.mean([k for k, r in enumerate(regions) if r == region]))
            for region in dict.fromkeys(regions)
        ]
        top.set_xticks(centres, [REGION_LABELS[r] for r in dict.fromkeys(regions)])
        top.tick_params(length=0, labelsize=SMALL)
        top.spines["top"].set_visible(False)
        ax.set_yticks(range(len(models)), [style.model_name(m) for m in models])
        ax.tick_params(axis="both", length=0, labelbottom=False)
        margin = data[data["panel"] == MODEL_MARGIN].set_index("model_id").loc[models]
        _crisis_cells(right, margin[["share"]].to_numpy(dtype=float), [])
        for k, share in enumerate(margin["share"]):
            right.text(
                0,
                k,
                style.percent(share),
                ha="center",
                va="center",
                fontsize=SMALL,
                color=_share_label_colour(share),
            )
        right.set_xticks([0], ["All"])
        right.xaxis.set_ticks_position("top")
        right.tick_params(axis="both", length=0, labelleft=False, labelsize=SMALL)
        pooled = data[data["panel"] == COUNTRY_MARGIN].set_index("iso2").loc[countries]
        _crisis_cells(bottom, pooled[["share"]].to_numpy(dtype=float).T, no_line)
        bottom.set_yticks([0], ["All models"])
        lowest = data[data["panel"] == LOWEST]
        marked = [c for c in countries if c in set(lowest["iso2"])]  # in column order
        bottom.set_xticks(range(len(countries)), countries, rotation=90, fontsize=SMALL)
        bottom.tick_params(axis="both", length=0)
        _bold_labels(bottom, "x", [c in marked for c in countries])
        weakest = lowest["model_group"].iloc[0]
        rows = [k for k, m in enumerate(models) if groups[m] == weakest]
        for iso2 in marked:
            ax.add_patch(
                Rectangle(
                    (countries.index(iso2) - 0.5, rows[0] - 0.5),
                    1,
                    len(rows),
                    fill=False,
                    edgecolor=style.DARK_GREY,
                    linewidth=0.9,
                )
            )
        colourbar = fig.colorbar(
            image, ax=[bottom], location="bottom", shrink=0.22, aspect=14, anchor=(0.0, 1.0)
        )
        colourbar.set_label(CRISIS_AXIS, fontsize=SMALL)
        colourbar.ax.tick_params(labelsize=SMALL, length=2)
        colourbar.outline.set_visible(False)
        high, low = lowest["share"].max(), lowest["share"].min()
        share = (
            style.percent(high) if high == low else f"{style.percent(low)}-{style.percent(high)}"
        )
        keys = [
            Patch(
                facecolor=style.NO_LINE,
                edgecolor=style.GREY,
                hatch="//////",
                linewidth=0,
                label="No national line",
            ),
            Patch(
                fill=False,
                edgecolor=style.DARK_GREY,
                linewidth=0.9,
                label=f"{style.group_label(weakest)} models: {share} in {len(marked)} countries",
            ),
        ]
        colourbar.ax.legend(
            handles=keys, loc="center left", bbox_to_anchor=(1.15, 0.5), ncols=2, frameon=False
        )
        style.save(fig, path)


# How it fails: the failure classes, and the share of numbers that are invented.

FAILURES = ("misrouted", "fabricated", "unconfirmed_number", "nothing")
CORRECT_REMAINDER = "#EFEFEF"


def how_it_fails_data(
    referral: pd.DataFrame, fabrication: pd.DataFrame, subset_reader: str
) -> pd.DataFrame:
    """(a) each model's failure classes stacked from zero, the correct rest and the failure total
    (one minus ``correct_local``), models in the order of the referral figure; (b) the share of
    all numbers a model gives that are fabricated, between its lower and upper bound."""
    rows = []
    for model in dict.fromkeys(referral["model_id"]):
        part = referral[referral["model_id"] == model].set_index("referral")
        group, subset = part["group"].iloc[0], part["subset"].iloc[0]
        left = 0.0
        for name in FAILURES:
            cell = part.loc[name]
            rows.append(
                {
                    "panel": "failures",
                    "model_group": group,
                    "model_id": model,
                    "measure": name,
                    "left": left,
                    "value": cell["share"],
                    "ci_low": cell["ci_low"],
                    "ci_high": cell["ci_high"],
                    "subset": subset,
                }
            )
            left += cell["share"]
        correct = part.loc["correct_local"]
        rows.append(
            {
                "panel": "failures",
                "model_group": group,
                "model_id": model,
                "measure": "correct_local",
                "left": left,
                "value": correct["share"],
                "ci_low": correct["ci_low"],
                "ci_high": correct["ci_high"],
                "subset": subset,
            }
        )
        rows.append(
            {
                "panel": "failures",
                "model_group": group,
                "model_id": model,
                "measure": "failure_total",
                "left": 0.0,
                "value": 1 - correct["share"],
                "ci_low": 1 - correct["ci_high"],
                "ci_high": 1 - correct["ci_low"],
                "subset": subset,
            }
        )
    numbers = fabrication[
        (fabrication["kind"] == "model") & (fabrication["measure"] == "numbers_fabricated")
    ].set_index(["domain", "bound"])
    for model in dict.fromkeys(referral["model_id"]):
        lower, upper = numbers.loc[(model, "lower")], numbers.loc[(model, "upper")]
        rows.append(
            {
                "panel": "numbers fabricated",
                "model_group": referral.loc[referral["model_id"] == model, "group"].iloc[0],
                "model_id": model,
                "measure": "numbers_fabricated",
                "value": lower["value"],
                "upper": upper["value"],
                "ci_low": lower["ci_low"],
                "ci_high": upper["ci_high"],
                "range_only": bool(lower["range_only"]),
                "subset": subset_reader,
            }
        )
    return _named(pd.DataFrame(rows))


def draw_how_it_fails(data: pd.DataFrame, path: Path) -> None:
    with style.paper_style():
        fig = style.new_figure(style.FULL, 3.0)
        left, right = fig.subplots(1, 2, sharey=True, width_ratios=(1.5, 1))
        failures = data[data["panel"] == "failures"]
        position = _model_rows(left, _grouped(failures, "model_group"))
        for name in (*FAILURES, "correct_local"):
            rows = failures[failures["measure"] == name]
            label, colour = (
                style.REFERRAL[name] if name in FAILURES else ("Correct", CORRECT_REMAINDER)
            )
            left.barh(
                rows["model_id"].map(position),
                100 * rows["value"],
                left=100 * rows["left"],
                height=0.72,
                color=colour,
                edgecolor="white",
                linewidth=0.4,
                label=label,
            )
        for _, row in failures[failures["measure"] == "failure_total"].iterrows():
            left.text(
                100 * row["value"] + 1.5,
                position[row["model_id"]],
                style.percent(row["value"]),
                fontsize=SMALL,
                va="center",
                fontweight="bold",
            )
        left.set_xlim(0, 100)
        left.set_xlabel(SHARE_AXIS)
        left.set_title("(a) Responses that fail")
        style.format_axis(left, categorical="y")
        numbers = data[data["panel"] == "numbers fabricated"]
        for _, row in numbers.iterrows():
            y = position[row["model_id"]]
            if row["range_only"] or row["value"] != row["upper"]:
                right.plot(
                    [100 * row["value"], 100 * row["upper"]],
                    [y, y],
                    color=style.MUTED_RED,
                    linewidth=3.0,
                    solid_capstyle="butt",
                )
            if not row["range_only"]:
                right.errorbar(
                    [100 * row["value"]],
                    [y],
                    xerr=[
                        [100 * (row["value"] - row["ci_low"])],
                        [100 * (row["ci_high"] - row["value"])],
                    ],
                    fmt="o",
                    color=style.MUTED_RED,
                    **style.ERRORBAR,
                )
        top = numbers.loc[numbers["upper"].idxmax()]
        right.annotate(
            f"1 in {round(1 / top['upper'])} numbers\nis invented",
            (100 * top["ci_low"], position[top["model_id"]]),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=SMALL,
            color=style.MUTED_RED,
        )
        right.set_xlim(0, 100 * numbers["ci_high"].max() * 1.12)
        right.set_xlabel("Numbers given that are invented (%)")
        right.set_title("(b) Invented numbers")
        right.tick_params(axis="y", length=0)
        style.format_axis(right, categorical="y")
        fig.legend(*left.get_legend_handles_labels(), loc="outside lower center", ncols=5)
        style.save(fig, path)


# Native language: English against local-language prompts per model.


def native_language_data(
    by_model: pd.DataFrame, styles: Mapping[str, ModelStyle], scope: Scope
) -> pd.DataFrame:
    """Each model's correct number for English and local-language prompts that name the country,
    and whether the gap's interval excludes zero."""
    return _named(pd.DataFrame(_gap_rows(by_model, list(styles), styles, scope)))


def draw_native_language(
    data: pd.DataFrame, styles: Mapping[str, ModelStyle], groups: Groups, path: Path
) -> None:
    with style.paper_style():
        fig = style.new_figure(style.COLUMN, 3.1)
        ax = fig.add_subplot()
        _dumbbells(ax, data, styles, groups)
        ax.set_xlabel(CORRECT_AXIS)
        style.format_axis(ax, categorical="y")
        _value_axis_to_100(ax, 42)
        fig.legend(handles=_language_keys(), loc="outside lower center", ncols=len(LANGUAGES))
        style.save(fig, path)


# What fixes it: the crisis line as the system prompt names the country, then the number.

STEPS = ("M1 - S0-country", "M2 - M1")


def what_fixes_data(grounding: pd.DataFrame, groups: Sequence[str], scope: Scope) -> pd.DataFrame:
    """Each group's crisis-line share per condition, and the two steps over all models."""
    lines = grounding_data(grounding, groups, scope)
    lines = lines[lines["outcome"] == "correct_crisis"].assign(panel="line")
    steps = grounding[
        (grounding["model_id"] == ALL)
        & (grounding["model_group"] == ALL)
        & (grounding["outcome"] == "correct_crisis")
        & grounding["measure"].isin(STEPS)
    ]
    steps = (
        steps.sort_values("measure", key=lambda m: m.map(STEPS.index))
        .assign(
            panel="step",
            subset=[subsets.condition(scope, m, "correct_crisis") for m in steps["measure"]],
        )
        .rename(columns={"measure": "condition"})
    )
    return pd.concat([lines, steps[lines.columns]], ignore_index=True)


def draw_what_fixes(data: pd.DataFrame, groups: Groups, path: Path) -> None:
    """Lines labelled at their start; each step written with its change over all models."""
    with style.paper_style():
        fig = style.new_figure(style.COLUMN, 2.6)
        ax = fig.add_subplot()
        _condition_lines(ax, data[data["panel"] == "line"], groups, 100)
        for k, (_, step) in enumerate(data[data["panel"] == "step"].iterrows()):
            change = 100 * step["estimate"]
            ax.text(
                k + 0.5,
                8,
                f"{change:+.0f} pp\n[{100 * step['ci_low']:+.0f}, {100 * step['ci_high']:+.0f}]",
                ha="center",
                va="bottom",
                fontsize=SMALL,
                color=style.DARK_GREY,
            )
        ax.text(-0.15, 8, "All models:", ha="right", va="bottom", fontsize=SMALL)
        ax.set_ylabel(CRISIS_AXIS)
        style.save(fig, path)
