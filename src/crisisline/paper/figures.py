"""The paper's figures. For each, one function selects exactly the data it plots from the analysis
tables (written next to the figure as CSV, in plotted order) and one draws that data. Each figure
states one message in saturated colour; context is grey; numbers belong in numbers.csv."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, PathPatch
from matplotlib.path import Path as Outline
from matplotlib.ticker import FixedLocator, NullLocator
from matplotlib.transforms import ScaledTranslation

from crisisline.analysis.descriptive import REFERRAL_CLASSES
from crisisline.paper import style, subsets
from crisisline.paper.style import LANGUAGE_NAMES, WHO_REGIONS, ModelStyle
from crisisline.paper.subsets import Scope
from crisisline.paper.worldmap import Country, robinson

# The grounding conditions and their labels in figure 4.
CONDITIONS = {
    "S0-country": "Country in\nmessage",
    "M1": "Country in\nsystem prompt",
    "M2": "+ Verified\nnumber",
}
ALL = "all"  # the pooled rows of the exploratory tables
SHARE_AXIS = "Responses (%)"
CORRECT_AXIS = "Correct number (%)"
CRISIS_AXIS = "Correct crisis line (%)"
SMALL = 8  # points: direct labels and dense tick labels, the smallest size the template allows
TINY = 7  # points: the percentages inside bars and the codes of the dense grids
SEPARATOR_REACH = 2  # points: how far the separators of a grid reach beyond it
LABEL_PAD = 2  # points: between a grid and its tick labels


def _ordered_groups(styles: Mapping[str, ModelStyle]) -> list[str]:
    return list(dict.fromkeys(s.group for s in styles.values()))


def _offsets(count: int, step: float) -> np.ndarray:
    return step * (np.arange(count) - (count - 1) / 2)


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


# Figure 2: referral outcomes per model.


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
                fontsize=TINY,
                ha="left",
                va="center",
            )
        ax.set_xlim(0, 100)
        ax.set_xlabel(SHARE_AXIS)
        style.format_axis(ax, categorical="y")
        fig.legend(loc="outside lower center", ncols=3)
        style.save(fig, path)


# The grounding conditions, for figure 4.

FIX_AXES = {"correct_crisis": CRISIS_AXIS, "fabricated": "Fabricated number (%)"}


def grounding_data(grounding: pd.DataFrame, groups: Sequence[str], scope: Scope) -> pd.DataFrame:
    """Each group's pooled share per condition, local-language prompts."""
    rows = grounding[
        (grounding["model_id"] == ALL)
        & grounding["model_group"].isin(groups)
        & grounding["measure"].isin(CONDITIONS)
        & grounding["outcome"].isin(FIX_AXES)
    ]
    order = {
        "outcome": list(FIX_AXES),
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


def _condition_lines(ax: Axes, data: pd.DataFrame, colours: Mapping[str, str], top: float) -> None:
    """One line per group across the conditions, labelled at its first point."""
    offsets = _offsets(len(colours), 0.08)
    starts = []
    for offset, (group, colour) in zip(offsets, colours.items(), strict=True):
        rows = data[data["model_group"] == group]
        ax.errorbar(
            rows["condition"].map(list(CONDITIONS).index) + offset,
            100 * rows["estimate"],
            yerr=style.percent_errors(rows["estimate"], rows["ci_low"], rows["ci_high"]),
            fmt="-o",
            color=colour,
            linewidth=1.0,
            **style.ERRORBAR,
        )
        starts.append(100 * rows["estimate"].iloc[0])
    for y, (group, colour) in zip(style.spread(starts, 0.09 * top), colours.items(), strict=True):
        ax.text(-0.15, y, style.group_label(group), color=colour, ha="right", va="center")
    ax.set_xticks(range(len(CONDITIONS)), list(CONDITIONS.values()), fontsize=SMALL)
    ax.set_xlim(-0.9, len(CONDITIONS) - 0.7)
    ax.set_ylim(0, 1.04 * top)
    style.format_axis(ax, categorical="x")
    if top == 100:
        ax.set_yticks(range(0, 101, 20))
        ax.spines["left"].set_bounds(0, 100)


# English against local-language prompts, for the appendix.

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


def _dumbbells(ax: Axes, data: pd.DataFrame, styles: Mapping[str, ModelStyle]) -> None:
    """English (filled) and local-language (hollow) shares per model, joined; the models whose gap
    excludes zero in their group's colour, the others grey."""
    order = list(dict.fromkeys(data["model_id"]))
    for y, model in enumerate(order):
        rows = data[data["model_id"] == model].set_index("prompt_language")
        significant = bool(rows["significant_gap"].iloc[0])
        colour = styles[model].colour if significant else style.GREY
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
                fmt="o",
                color=colour,
                markerfacecolor=colour if language == "en" else "white",
                zorder=2,
                **{**style.ERRORBAR, "elinewidth": 0.6},
            )
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


# Appendix: the Qwen size ladder.

CUES = ("country", "city")
LADDER_OUTCOMES = ("correct_crisis_line", "misrouted", "fabricated", "nothing")


def ladder_data(ladder: pd.DataFrame, sizes: Mapping[str, float], scope: Scope) -> pd.DataFrame:
    """Each model's share of a correct number against its size, prompts that name a city or the
    country."""
    rows = ladder[ladder["outcome"] == "correct_local"].assign(
        size=lambda t: t["model_id"].map(sizes)
    )
    rows = rows.assign(cue_rank=rows["cue"].map(CUES.index)).sort_values(["cue_rank", "size"])
    rows = rows.assign(subset=[subsets.cue(scope, c) for c in rows["cue"]])
    return _named(rows[["cue", "model_id", "size", "share", "ci_low", "ci_high", "subset"]])


def ladder_outcomes_data(
    referral: pd.DataFrame, sizes: Mapping[str, float], scope: Scope
) -> pd.DataFrame:
    """Each model's share of four referral classes against its size, prompts that name a city or
    the country pooled, countries with a crisis line."""
    rows = referral[referral["referral"].isin(LADDER_OUTCOMES)].assign(
        size=lambda t: t["model_id"].map(sizes),
        outcome_rank=lambda t: t["referral"].map(LADDER_OUTCOMES.index),
        subset=subsets.all_regions(scope, "correct_crisis"),
    )
    rows = rows.sort_values(["outcome_rank", "size"])
    columns = ["referral", "model_id", "size", "share", "ci_low", "ci_high", "responses", "subset"]
    return _named(rows[columns].reset_index(drop=True))


def draw_ladder(data: pd.DataFrame, path: Path) -> None:
    """A line per referral class with its interval as a band, labelled at its right end."""
    with style.paper_style():
        fig = style.new_figure(style.COLUMN, 2.2)
        ax = fig.add_subplot()
        ends = []
        for outcome in LADDER_OUTCOMES:
            rows = data[data["referral"] == outcome]
            colour = style.REFERRAL[outcome][1]
            ax.fill_between(
                rows["size"],
                100 * rows["ci_low"],
                100 * rows["ci_high"],
                color=colour,
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(
                rows["size"],
                100 * rows["share"],
                "-o",
                color=colour,
                linewidth=1.0,
                **{**style.MARKER, "markersize": 3.5},
            )
            ends.append(100 * rows["share"].iloc[-1])
        top = 10 * np.ceil(10 * data["ci_high"].max())
        placed = style.spread(ends, 0.09 * top)
        placed = [y + max(0.0, 0.02 * top - min(placed)) for y in placed]
        sizes = sorted(data["size"].unique())
        for y, outcome in zip(placed, LADDER_OUTCOMES, strict=True):
            label, colour = style.REFERRAL[outcome]
            ax.text(1.15 * sizes[-1], y, label, color=colour, fontsize=SMALL, va="center")
        ax.set_xscale("log")
        style.format_axis(ax)
        ax.set_xticks(sizes, [f"{s:g}B" for s in sizes])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(sizes[0] / 1.15, sizes[-1] * 1.1)
        ax.set_ylim(0, top)
        ax.set_xlabel("Parameters")
        ax.set_ylabel(SHARE_AXIS)
        style.save(fig, path)


# Heatmaps: the crisis line per model and country (appendix), the correct number per model and
# language (figure 3).

CELL, MODEL_MARGIN, COLUMN_MARGIN = "cell", "model share", "column share"
COUNTRY_MARGIN, LOWEST, NO_CRISIS_LINE = "country share", "lowest", "no crisis line"


def _models_by_margin(margin: pd.Series, styles: Mapping[str, ModelStyle]) -> list[str]:
    """Models by group, from the highest margin share within each."""
    order = []
    for group in _ordered_groups(styles):
        members = [m for m, s in styles.items() if s.group == group]
        order += sorted(members, key=lambda m: (-margin[m], m))
    return order


def _weakest_group(margin: pd.Series, styles: Mapping[str, ModelStyle]) -> str:
    """The group with the lowest mean margin share over its models."""
    means = {
        g: np.mean([margin[m] for m, s in styles.items() if s.group == g])
        for g in _ordered_groups(styles)
    }
    return min(means, key=means.get)


def who_gets_help_data(
    cells: pd.DataFrame,
    model_shares: pd.DataFrame,
    countries: pd.DataFrame,
    styles: Mapping[str, ModelStyle],
    scope: Scope,
    lowest: int = 3,
) -> pd.DataFrame:
    """The crisis-line share of every model in every country with a crisis line (S0 with a cue),
    the margins (each model over all countries, each country over all models), the countries with
    the lowest share for the weakest group (the ``lowest`` lowest, and every country tied with the
    last), and the countries without a crisis line, which the figure leaves out."""
    countries = countries[countries["outcome"] == "correct_crisis"]
    margin = model_shares.set_index("model_id")
    order = _models_by_margin(margin["share"], styles)
    pooled = countries[countries["model_group"] == ALL].set_index("iso2")
    columns = []
    for region in WHO_REGIONS:
        with_line = [c for c in scope.countries if c.region == region and c.crisis_line]
        columns += sorted(with_line, key=lambda c: (-pooled.loc[c.iso2, "estimate"], c.iso2))
    shares = cells.set_index(["model_id", "iso2"])
    rows = []
    for model in order:
        for country in columns:
            cell = shares.loc[(model, country.iso2)]
            rows.append(
                {
                    "panel": CELL,
                    "model_group": styles[model].group,
                    "model_id": model,
                    "iso2": country.iso2,
                    "name_en": country.name,
                    "who_region": country.region,
                    "share": cell["share"],
                    "ci_low": cell["ci_low"],
                    "ci_high": cell["ci_high"],
                    "n": cell["responses"],
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
        rows.append(
            {
                "panel": COUNTRY_MARGIN,
                "model_group": ALL,
                "iso2": country.iso2,
                "name_en": country.name,
                "who_region": country.region,
                "share": pooled.loc[country.iso2, "estimate"],
                "ci_low": pooled.loc[country.iso2, "ci_low"],
                "ci_high": pooled.loc[country.iso2, "ci_high"],
                "n": pooled.loc[country.iso2, "responses"],
                "subset": subsets.country(country.name),
            }
        )
    weakest = _weakest_group(margin["share"], styles)
    by_group = countries[countries["model_group"] == weakest].set_index("iso2")
    cutoff = by_group["estimate"].nsmallest(lowest).max()
    for country in columns:
        if country.iso2 in by_group.index and by_group.loc[country.iso2, "estimate"] <= cutoff:
            row = by_group.loc[country.iso2]
            rows.append(
                {
                    "panel": LOWEST,
                    "model_group": weakest,
                    "iso2": country.iso2,
                    "name_en": country.name,
                    "who_region": country.region,
                    "share": row["estimate"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "n": row["responses"],
                    "subset": subsets.country(country.name),
                }
            )
    for region in WHO_REGIONS:
        for country in sorted(
            (c for c in scope.countries if c.region == region and not c.crisis_line),
            key=lambda c: c.iso2,
        ):
            rows.append(
                {
                    "panel": NO_CRISIS_LINE,
                    "model_group": ALL,
                    "iso2": country.iso2,
                    "name_en": country.name,
                    "who_region": country.region,
                    "subset": subsets.NO_LINE,
                }
            )
    table = pd.DataFrame(rows)
    return _named(table.assign(model_id=table["model_id"].fillna("")))


def who_gets_help_language_data(
    cells: pd.DataFrame,
    by_language: pd.DataFrame,
    styles: Mapping[str, ModelStyle],
    scope: Scope,
) -> pd.DataFrame:
    """The correct-number share of every model in every prompt language (local-language S0 with a
    cue), the margins (each model over all languages, each language over all models); languages
    from the highest share of the weakest group."""
    languages = cells[cells["lang"] != ALL].set_index(["model_id", "lang"])
    margin = cells[cells["lang"] == ALL].set_index("model_id")
    order = _models_by_margin(margin["share"], styles)
    weakest = _weakest_group(margin["share"], styles)
    group = by_language[by_language["model_group"] == weakest].set_index("lang")["estimate"]
    columns = sorted(group.index, key=lambda lang: (-group[lang], lang))
    pooled = by_language[by_language["model_group"] == ALL].set_index("lang")
    countries = by_language.drop_duplicates("lang").set_index("lang")["countries"]
    rows = []
    for model in order:
        for lang in columns:
            cell = languages.loc[(model, lang)]
            rows.append(
                {
                    "panel": CELL,
                    "model_group": styles[model].group,
                    "model_id": model,
                    "lang": lang,
                    "language": LANGUAGE_NAMES[lang],
                    "share": cell["share"],
                    "ci_low": cell["ci_low"],
                    "ci_high": cell["ci_high"],
                    "n": cell["responses"],
                    "subset": subsets.language(lang, int(countries[lang])),
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
                "subset": subsets.local_languages(scope),
            }
        )
    for lang in columns:
        rows.append(
            {
                "panel": COLUMN_MARGIN,
                "model_group": ALL,
                "lang": lang,
                "language": LANGUAGE_NAMES[lang],
                "share": pooled.loc[lang, "estimate"],
                "ci_low": pooled.loc[lang, "ci_low"],
                "ci_high": pooled.loc[lang, "ci_high"],
                "n": pooled.loc[lang, "responses"],
                "subset": subsets.language(lang, int(countries[lang])),
            }
        )
    table = pd.DataFrame(rows)
    return _named(table.assign(model_id=table["model_id"].fillna("")))


def _share_cells(ax: Axes, values: np.ndarray) -> AxesImage:
    """The shares as tiles, a very thin white line between neighbours."""
    image = ax.imshow(
        100 * values,
        cmap=style.CRISIS_MAP,
        vmin=0,
        vmax=100,
        aspect="auto",
        interpolation="nearest",
    )
    rows, columns = values.shape
    for y in np.arange(0.5, rows - 1):
        ax.axhline(y, color="white", linewidth=0.3)
    for x in np.arange(0.5, columns - 1):
        ax.axvline(x, color="white", linewidth=0.3)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _breaks(labels: Sequence[str]) -> list[float]:
    """Positions between neighbouring cells whose labels differ."""
    return [k - 0.5 for k in range(1, len(labels)) if labels[k] != labels[k - 1]]


def _share_grid(
    cells: pd.DataFrame,
    model_margin: pd.DataFrame,
    column_margin: pd.DataFrame,
    column: str,
    sections: Sequence[str] | None,
    label: str,
    path: Path,
    names: Mapping[str, str] | None = None,
    height: float = 2.4,
) -> None:
    """Rows: models; columns: ``column``, by their ``names`` if given, under their section if
    ``sections`` (one per column); colour: the share. A margin column per model and a margin row
    per column; lines between sections and before the margin row if ``sections``."""
    models = list(dict.fromkeys(cells["model_id"]))
    columns = list(dict.fromkeys(cells[column]))
    matrix = cells.pivot(index="model_id", columns=column, values="share").loc[models, columns]
    with style.paper_style():
        fig = style.new_figure(style.FULL, height)
        # Room on the right for the separator that reaches past the margin column.
        right_edge = 1 - SEPARATOR_REACH / 72 / style.FULL if sections is not None else 1
        fig.get_layout_engine().set(
            h_pad=0.01, w_pad=0.01, hspace=0.0, wspace=0.0, rect=(0, 0, right_edge, 1)
        )
        grid = fig.add_gridspec(
            2, 2, width_ratios=(len(columns), 1.6), height_ratios=(len(models), 1.0)
        )
        ax = fig.add_subplot(grid[0, 0])
        right = fig.add_subplot(grid[0, 1], sharey=ax)
        bottom = fig.add_subplot(grid[1, 0], sharex=ax)
        image = _share_cells(ax, matrix.to_numpy(dtype=float))
        margin = model_margin.set_index("model_id").loc[models, ["share"]]
        _share_cells(right, margin.to_numpy(dtype=float))
        pooled = column_margin.set_index(column).loc[columns, ["share"]]
        _share_cells(bottom, pooled.to_numpy(dtype=float).T)
        if sections is not None:
            top = ax.secondary_xaxis("top")
            blocks = list(dict.fromkeys(sections))
            centres = [
                float(np.mean([k for k, s in enumerate(sections) if s == name])) for name in blocks
            ]
            top.set_xticks(centres, blocks)
            top.tick_params(length=0, labelsize=SMALL, pad=1)
            top.spines["top"].set_visible(False)
        ax.set_yticks(range(len(models)), [style.model_name(m) for m in models])
        ax.tick_params(axis="both", length=0, labelbottom=False, labelsize=SMALL, pad=LABEL_PAD)
        right.set_xticks([0], ["All"])
        right.xaxis.set_ticks_position("top")
        right.tick_params(axis="both", length=0, labelleft=False, labelsize=SMALL, pad=1)
        bottom.set_yticks([0], ["All models"])
        ticks = [names[c] for c in columns] if names else columns
        bottom.set_xticks(range(len(columns)), ticks, rotation=90, fontsize=TINY)
        bottom.tick_params(axis="both", length=0, labelsize=SMALL, pad=LABEL_PAD)
        bottom.tick_params(axis="x", labelsize=TINY)
        colourbar = fig.colorbar(
            image, ax=[bottom], location="bottom", shrink=0.24, aspect=18, pad=0.0
        )
        colourbar.set_ticks([0, 50, 100])
        colourbar.ax.tick_params(labelsize=TINY, length=2, pad=1)
        colourbar.outline.set_visible(False)
        colourbar.ax.annotate(
            label,
            (0, 0.5),
            xycoords="axes fraction",
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=SMALL,
        )
        if sections is not None:
            _separators(fig, (ax, right, bottom), _breaks(sections))
        style.save(fig, path)


class _LaidOutLine(Line2D):
    """A line in figure coordinates that ``place`` puts from the laid-out axes at each draw."""

    def __init__(
        self, place: Callable[[], tuple[list[float], list[float]]], **kwargs: object
    ) -> None:
        super().__init__([], [], **kwargs)
        self._place = place

    def draw(self, renderer: object) -> None:
        self.set_data(*self._place())
        super().draw(renderer)


def _separators(fig: Figure, axes: tuple[Axes, Axes, Axes], breaks: Sequence[float]) -> None:
    """Black lines with a thin white edge, none in the layout: one between sections and one
    between the grid and the margin column, from a little above the grid to a little below the
    margin row; one between the grid and the margin row, from the model names to a little past
    the margin column. All white edges first, so that crossings stay black."""
    ax, right, bottom = axes
    width, height = fig.get_size_inches()
    reach, pad = SEPARATOR_REACH / 72, LABEL_PAD / 72
    up = ax.transData + ScaledTranslation(0, reach, fig.dpi_scale_trans)
    down = bottom.transData + ScaledTranslation(0, -reach, fig.dpi_scale_trans)

    def before_margin_column() -> tuple[list[float], list[float]]:
        x = (ax.get_position().x1 + right.get_position().x0) / 2
        return [x, x], [
            ax.get_position().y1 + reach / height,
            bottom.get_position().y0 - reach / height,
        ]

    def before_margin_row() -> tuple[list[float], list[float]]:
        y = (ax.get_position().y0 + bottom.get_position().y1) / 2
        return [ax.get_position().x0 - pad / width, right.get_position().x1 + reach / width], [y, y]

    for colour, linewidth, zorder in (("white", 2.0, 3), ("black", 1.0, 4)):
        line = {"color": colour, "linewidth": linewidth, "zorder": zorder}
        lines = [
            ConnectionPatch((x, -0.5), (x, 0.5), coordsA=up, coordsB=down, capstyle="butt", **line)
            for x in breaks
        ]
        lines += [
            _LaidOutLine(place, transform=fig.transFigure, solid_capstyle="butt", **line)
            for place in (before_margin_column, before_margin_row)
        ]
        for artist in lines:
            artist.set_in_layout(False)
            fig.add_artist(artist)


def draw_who_gets_help(data: pd.DataFrame, path: Path) -> None:
    cells = data[data["panel"] == CELL]
    sections = cells.drop_duplicates("iso2")["who_region"].tolist()
    _share_grid(
        cells,
        data[data["panel"] == MODEL_MARGIN],
        data[data["panel"] == COUNTRY_MARGIN],
        "iso2",
        sections,
        CRISIS_AXIS,
        path,
    )


def draw_who_gets_help_language(data: pd.DataFrame, path: Path) -> None:
    _share_grid(
        data[data["panel"] == CELL],
        data[data["panel"] == MODEL_MARGIN],
        data[data["panel"] == COLUMN_MARGIN],
        "lang",
        None,
        CORRECT_AXIS,
        path,
        LANGUAGE_NAMES,
        height=2.7,
    )


# Appendix: English against local-language prompts per model.


def native_language_data(
    by_model: pd.DataFrame, styles: Mapping[str, ModelStyle], scope: Scope
) -> pd.DataFrame:
    """Each model's correct number for English and local-language prompts that name the country,
    and whether the gap's interval excludes zero; models from the largest gap."""
    gap = by_model.set_index("model_id")["difference_en_minus_local"]
    models = sorted(styles, key=lambda m: (-gap[m], m))
    return _named(pd.DataFrame(_gap_rows(by_model, models, styles, scope)))


def draw_native_language(data: pd.DataFrame, styles: Mapping[str, ModelStyle], path: Path) -> None:
    with style.paper_style():
        fig = style.new_figure(style.COLUMN, 2.8)
        ax = fig.add_subplot()
        _dumbbells(ax, data, styles)
        ax.set_xlabel(CORRECT_AXIS)
        style.format_axis(ax, categorical="y")
        _value_axis_to_100(ax, 3)
        fig.legend(handles=_language_keys(), loc="outside lower center", ncols=len(LANGUAGES))
        style.save(fig, path)


# Figure 4: the crisis line and invented numbers as the system prompt names the country, then the
# number.

STEPS = ("M1 - S0-country", "M2 - M1")


def what_fixes_data(grounding: pd.DataFrame, groups: Sequence[str], scope: Scope) -> pd.DataFrame:
    """Each group's crisis-line and fabricated share per condition, and the two steps of the
    crisis line over all models."""
    lines = grounding_data(grounding, groups, scope).assign(panel="line")
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


def draw_what_fixes(data: pd.DataFrame, colours: Mapping[str, str], path: Path) -> None:
    """A panel per outcome, a line per group labelled at its start; the steps are not drawn."""
    lines = data[data["panel"] == "line"]
    with style.paper_style():
        fig = style.new_figure(style.FULL, 2.2)
        axes = fig.subplots(1, len(FIX_AXES))
        for ax, (outcome, label) in zip(axes, FIX_AXES.items(), strict=True):
            rows = lines[lines["outcome"] == outcome]
            top = 100 if outcome == "correct_crisis" else 100 * rows["ci_high"].max() * 1.1
            _condition_lines(ax, rows, colours, top)
            ax.set_ylabel(label)
        style.save(fig, path)


# Figure 1: where small local models do not give the crisis line.

HATCH = "//////"


def percent_value(share: float) -> float:
    """A share in percent, rounded so that a class edge is not missed by a floating-point error."""
    return round(100 * share, 6)


def _class_of(share: float, edges: Sequence[float]) -> int:
    """The class of a share without the crisis line; a class holds its upper edge, the first its
    lower one too: a country in a class above 50% has more than 50%."""
    inner = np.asarray(edges[1:-1], dtype=float)
    return int(np.searchsorted(inner, percent_value(share), side="left"))


def _class_name(k: int, edges: Sequence[float], dash: str = "-") -> str:
    return f"{edges[k]:g}{dash}{edges[k + 1]:g}"


def world_map_data(
    table: pd.DataFrame,
    shapes: Sequence[Country],
    group: str,
    small_km2: float,
    edges: Sequence[float],
    scope: Scope,
) -> pd.DataFrame:
    """The share of responses without the correct crisis line of ``group`` and its class, per
    country of the prompts (local-language S0 with a cue), empty for the countries without a crisis
    line; the land area of each country and whether it is small enough for a marker as well."""
    shares = table[table["model_group"] == group].set_index("iso2")
    area = {c.iso2: c.area for c in shapes if c.iso2}
    rows = []
    for country in sorted(scope.countries, key=lambda c: (c.region, c.iso2)):
        cell = shares.loc[country.iso2] if country.crisis_line else None
        missing = 1 - cell["estimate"] if cell is not None else np.nan
        rows.append(
            {
                "model_group": group,
                "iso2": country.iso2,
                "name_en": country.name,
                "who_region": country.region,
                "has_crisis_line": country.crisis_line,
                "no_crisis_line": missing,
                "ci_low": 1 - cell["ci_high"] if cell is not None else np.nan,
                "ci_high": 1 - cell["ci_low"] if cell is not None else np.nan,
                "n": cell["responses"] if cell is not None else np.nan,
                "class": _class_name(_class_of(missing, edges), edges) if cell is not None else "",
                "area_km2": round(area[country.iso2]),
                "marker": area[country.iso2] < small_km2,
                "subset": subsets.local_language_country(country.name)
                if country.crisis_line
                else subsets.NO_LINE,
            }
        )
    return pd.DataFrame(rows)


def world_map_classes(data: pd.DataFrame, edges: Sequence[float]) -> pd.DataFrame:
    """The countries of the map in each class, from the lowest class; empty classes included."""
    rows = []
    for k in range(len(edges) - 1):
        name = _class_name(k, edges)
        members = data[data["class"] == name].sort_values(["no_crisis_line", "iso2"])
        rows.append(
            {
                "class": name,
                "countries": len(members),
                "names": ", ".join(members["name_en"]),
            }
        )
    return pd.DataFrame(rows)


def world_map_pooled(table: pd.DataFrame, groups: Sequence[str], scope: Scope) -> pd.DataFrame:
    """The share without the correct crisis line of each of ``groups`` over all countries with a
    crisis line, on the subset of the map."""
    pooled = table[table["iso2"] == ALL].set_index("model_group")
    return pd.DataFrame(
        [
            {
                "model_group": group,
                "no_crisis_line": 1 - pooled.loc[group, "estimate"],
                "ci_low": 1 - pooled.loc[group, "ci_high"],
                "ci_high": 1 - pooled.loc[group, "ci_low"],
                "n": pooled.loc[group, "responses"],
                "subset": subsets.local_languages(scope, "correct_crisis"),
            }
            for group in groups
        ]
    )


def _outline(country: Country) -> Outline:
    rings = [ring for polygon in country.polygons for ring in polygon]
    return Outline.make_compound_path(*(Outline(np.vstack([r, r[:1]]), closed=True) for r in rings))


def _graticule(ax: Axes, radius: float, theme: style.MapTheme) -> None:
    """Meridians and parallels every 30 degrees under the countries, and the projection's
    boundary over them."""
    lat = np.linspace(-90, 90, 181)
    lon = np.linspace(-180, 180, 361)
    lines = {"linewidth": 0.3, "color": theme.graticule, "zorder": 0}
    for meridian in range(-150, 151, 30):
        ax.plot(*robinson(np.full_like(lat, meridian), lat, radius), **lines)
    for parallel in range(-60, 61, 30):
        ax.plot(*robinson(lon, np.full_like(lon, parallel), radius), **lines)
    x, y = robinson(
        np.r_[np.full_like(lat, 180), np.full_like(lat, -180), 180],
        np.r_[lat, lat[::-1], -90],
        radius,
    )
    ax.plot(x, y, color=theme.outline, linewidth=0.5)


def draw_world_map(
    data: pd.DataFrame,
    shapes: Sequence[Country],
    edges: Sequence[float],
    radius: float,
    path: Path,
    theme: style.MapTheme = style.PAPER_MAP,
) -> None:
    """One Robinson map: countries of the prompts by their class of the share without the crisis
    line, hatched grey without a national line, other land faint; small countries also as a
    circle. The legend counts the countries in each class."""
    rows = data.set_index("iso2")

    def colour(iso2: str) -> str:
        row = rows.loc[iso2]
        if not row["has_crisis_line"]:
            return theme.no_line
        return style.MISSING_CLASSES[_class_of(row["no_crisis_line"], edges)]

    rc = {
        "hatch.linewidth": 0.5,
        "text.color": theme.text,
        "figure.facecolor": theme.background,
        "savefig.facecolor": theme.background,
    }
    x_edge, _ = robinson(180.0, 0.0, radius)
    _, y_edge = robinson(0.0, 90.0, radius)
    with style.paper_style(), mpl.rc_context(rc):
        fig = style.new_figure(style.FULL, 3.55)
        fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01)
        ax = fig.add_subplot()
        ax.set_facecolor(theme.background)
        _graticule(ax, radius, theme)
        for country in shapes:
            outline = _outline(country)
            if not country.iso2:
                # An edge in the fill colour: no seams between neighbours, no visible border.
                outside = {"facecolor": theme.outside, "edgecolor": theme.outside}
                ax.add_patch(PathPatch(outline, linewidth=0.3, **outside))
                continue
            face = colour(country.iso2)
            ax.add_patch(PathPatch(outline, facecolor=face, edgecolor=theme.border, linewidth=0.3))
            if not rows.loc[country.iso2, "has_crisis_line"]:
                ax.add_patch(
                    PathPatch(outline, fill=False, hatch=HATCH, edgecolor=theme.hatch, linewidth=0)
                )
        for country in shapes:
            if country.iso2 and rows.loc[country.iso2, "marker"]:
                ax.plot(
                    *country.centroid,
                    "o",
                    markersize=4.5,
                    markerfacecolor=colour(country.iso2),
                    markeredgecolor=theme.marker_edge,
                    markeredgewidth=0.5,
                )
        ax.set_xlim(-1.005 * x_edge, 1.005 * x_edge)
        ax.set_ylim(-1.01 * y_edge, 1.01 * y_edge)
        ax.set_aspect("equal")
        ax.set_axis_off()
        counts = world_map_classes(data, edges)["countries"]
        handles, labels = [], []
        for k, count in enumerate(counts):
            handles.append(Patch(facecolor=style.MISSING_CLASSES[k], linewidth=0))
            labels.append(f"{_class_name(k, edges, '–')}% · {count}")
        handles += [
            Patch(facecolor=theme.no_line, hatch=HATCH, edgecolor=theme.hatch, linewidth=0),
            Patch(facecolor=theme.outside, edgecolor=style.LIGHT_GREY, linewidth=0.4),
        ]
        labels += ["No national crisis line", "Not studied"]
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncols=len(handles),
            title="No crisis line given",
            title_fontsize=SMALL,
            handlelength=0.9,
            handletextpad=0.3,
            columnspacing=0.6,
        )
        style.save(fig, path)
