"""Figures as vector PDF sized for the ACL template: one column 7.7 cm, full width 16 cm."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl

mpl.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from crisisline.analysis.descriptive import ERROR_TYPES  # noqa: E402

CM = 1 / 2.54
COLUMN = 7.7 * CM
FULL = 16 * CM
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#e4e4e4"
# Fixed order; colour follows the series, and every series also has its own marker.
SERIES = ("#0072B2", "#D55E00", "#009E73", "#7A5195")
MARKERS = ("o", "s", "^", "D")
ERROR_COLOURS = {
    "correct number": "#0072B2",
    "foreign number only": "#D55E00",
    "fabricated number": "#CC79A7",
    "other number": "#E2B33C",
    "directory only": "#56B4E9",
    "no resource": "#9A9A9A",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", metadata={"CreationDate": None, "Creator": None})
    plt.close(fig)


def _percent(values: pd.Series) -> pd.Series:
    return 100 * values


def correct_by_model_and_cue(table: pd.DataFrame, path: Path) -> None:
    """``correct_local`` per model and prompt setting, with 95% intervals."""
    _style()
    settings = list(table["setting"].unique())
    models = table[table["setting"] == settings[0]].sort_values("share")["model_id"].tolist()
    fig, ax = plt.subplots(figsize=(FULL, 8.5 * CM), layout="constrained")
    for k, setting in enumerate(settings):
        part = table[table["setting"] == setting].set_index("model_id").loc[models]
        y = [i + 0.2 * (k - (len(settings) - 1) / 2) for i in range(len(models))]
        share = _percent(part["share"])
        ax.errorbar(
            share,
            y,
            xerr=[share - _percent(part["ci_low"]), _percent(part["ci_high"]) - share],
            fmt=MARKERS[k],
            color=SERIES[k],
            markersize=4,
            elinewidth=0.8,
            label=setting,
        )
    ax.set_yticks(range(len(models)), models)
    ax.set_xlabel("Responses with a correct number (%)")
    ax.set_xlim(0, 102)
    ax.set_xticks(range(0, 101, 20))
    ax.grid(axis="y", visible=False)
    fig.legend(loc="outside upper center", ncols=len(settings), frameon=False)
    _save(fig, path)


def correct_by_country(table: pd.DataFrame, path: Path) -> None:
    """``correct_local`` per country, grouped by WHO region: stands in for a map."""
    _style()
    table = table.sort_values(["who_region", "share"], ascending=[True, False]).reset_index()
    fig, ax = plt.subplots(figsize=(FULL, 7.5 * CM), layout="constrained")
    ax.plot(table.index, _percent(table["share"]), "o", color=SERIES[0], markersize=3.5)
    ax.set_xticks(table.index, table["iso2"], rotation=90)
    ax.set_ylabel("Responses with a correct number (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    for region, part in table.groupby("who_region"):
        ax.text(part.index.to_numpy().mean(), 100.5, region, ha="center", va="bottom", color=MUTED)
        ax.axvline(part.index.max() + 0.5, color=GRID, linewidth=0.8)
    ax.set_xlim(-0.8, len(table) - 0.2)
    _save(fig, path)


def error_types_by_model(table: pd.DataFrame, path: Path) -> None:
    """Each model's responses by error class."""
    _style()
    order = table[table["error_type"] == ERROR_TYPES[0]].sort_values("share")["model_id"].tolist()
    fig, ax = plt.subplots(figsize=(FULL, 8 * CM), layout="constrained")
    wide = table.pivot(index="model_id", columns="error_type", values="share").fillna(0).loc[order]
    left = 0.0
    for error in ERROR_TYPES:
        share = 100 * wide[error] if error in wide else 0.0 * wide[ERROR_TYPES[0]]
        ax.barh(
            order,
            share,
            left=left,
            height=0.7,
            color=ERROR_COLOURS[error],
            edgecolor="white",
            linewidth=0.8,
            label=error,
        )
        left = left + share
    ax.set_xlabel("Responses to prompts about a country (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    fig.legend(loc="outside upper center", ncols=3, frameon=False)
    _save(fig, path)


def qwen_ladder(table: pd.DataFrame, sizes: dict[str, float], path: Path) -> None:
    """``correct_local`` against nominal size, per cue, with 95% intervals."""
    _style()
    fig, ax = plt.subplots(figsize=(COLUMN, 6 * CM), layout="constrained")
    for k, (cue, part) in enumerate(table.groupby("cue")):
        part = part.assign(size=part["model_id"].map(sizes)).sort_values("size")
        share = _percent(part["share"])
        ax.errorbar(
            part["size"],
            share,
            yerr=[share - _percent(part["ci_low"]), _percent(part["ci_high"]) - share],
            fmt=f"-{MARKERS[k]}",
            color=SERIES[k],
            markersize=4,
            linewidth=1.2,
            elinewidth=0.8,
            label=f"{cue} cue",
        )
    ax.set_xscale("log")
    ax.set_xticks(sorted(sizes.values()), [f"{v:g}" for v in sorted(sizes.values())])
    ax.minorticks_off()
    ax.set_xlabel("Nominal size (billions of parameters)")
    ax.set_ylabel("Correct number (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    _save(fig, path)


def default_country_heatmap(
    table: pd.DataFrame, top: int, path: Path, count: str = "responses"
) -> None:
    """The share of responses counted in ``count`` for the commonest countries, per language."""
    _style()
    keep = table.groupby("country")[count].sum().nlargest(top).index
    table = table.assign(share=table[count] / table["language_responses"])
    wide = (
        table[table["country"].isin(keep)]
        .pivot(index="lang", columns="country", values="share")
        .reindex(columns=keep)
        .reindex(sorted(table["lang"].unique()))
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(FULL, 13.5 * CM), layout="constrained")
    image = ax.imshow(100 * wide.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(wide.columns)), wide.columns, rotation=90)
    ax.set_yticks(range(len(wide.index)), wide.index)
    ax.set_xlabel("Country whose number is given")
    ax.set_ylabel("Prompt language (no country named)")
    ax.grid(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, label="Responses (%)", shrink=0.6)
    _save(fig, path)


def setting_rows(table: pd.DataFrame, settings: Sequence[tuple[str, str, str]]) -> pd.DataFrame:
    """Rows of a share table for the settings ``(label, condition, cue)``, labelled."""
    parts = [
        table[(table["condition"] == condition) & (table["cue"] == cue)].assign(setting=label)
        for label, condition, cue in settings
    ]
    return pd.concat(parts)
