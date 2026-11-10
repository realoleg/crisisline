"""The look of the paper's figures: drawn at the printed size of the ACL two-column template, in
STIX (the Times of the template, bundled with matplotlib, so the same on every machine)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator

COLUMN = 3.03  # inches
FULL = 6.3
PALETTE = ("#02692b", "#dd8e00", "#723f93", "#db5683", "#4e88cb")
DARK_GREY = "#4A4A4A"
GREY = "#7A7A7A"
LIGHT_GREY = "#D9D9D9"
MUTED_RED = "#b4534b"

# Referral outcomes keep their colour in every figure.
REFERRAL = {
    "correct_crisis_line": ("Correct crisis line", PALETTE[0]),
    "correct_emergency_only": ("Correct emergency only", PALETTE[4]),
    "misrouted": ("Misrouted", PALETTE[1]),
    "fabricated": ("Fabricated", MUTED_RED),
    "other_number": ("Unverified number", LIGHT_GREY),
    "nothing": ("No number", GREY),
}
# Model groups in configured order. The blue of the third is also "correct emergency only", but
# group colours and referral colours never share a figure.
GROUP_COLOURS = (PALETTE[2], PALETTE[3], PALETTE[4])
GROUP_MARKERS = ("o", "s", "^")
MODEL_MARKERS = ("o", "s", "^", "D", "v", "P")

RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "figure.titlesize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "text.color": DARK_GREY,
    "axes.edgecolor": DARK_GREY,
    "axes.labelcolor": DARK_GREY,
    "axes.titlecolor": DARK_GREY,
    "xtick.color": DARK_GREY,
    "ytick.color": DARK_GREY,
    "axes.linewidth": 0.8,
    "axes.titlepad": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": LIGHT_GREY,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
}


@dataclass(frozen=True, slots=True)
class ModelStyle:
    group: str
    colour: str
    marker: str


def group_label(group: str) -> str:
    return group[:1].upper() + group[1:]


def model_styles(
    groups: Mapping[str, Sequence[str]], sizes: Mapping[str, float]
) -> dict[str, ModelStyle]:
    """Every model in display order: groups as configured, models of the size ladder first and by
    size within their group; a colour per group and a marker per model within it."""
    styles = {}
    for k, (group, models) in enumerate(groups.items()):
        ordered = sorted(models, key=lambda m: (m not in sizes, sizes.get(m, 0.0)))
        for j, model in enumerate(ordered):
            styles[model] = ModelStyle(group, GROUP_COLOURS[k], MODEL_MARKERS[j])
    return styles


def group_styles(groups: Sequence[str]) -> dict[str, tuple[str, str]]:
    """Colour and marker of each group, in configured order."""
    return {g: (GROUP_COLOURS[k], GROUP_MARKERS[k]) for k, g in enumerate(groups)}


@contextmanager
def paper_style() -> Iterator[None]:
    with mpl.rc_context(RC):
        yield


def new_figure(width: float, height: float) -> Figure:
    """A figure at its printed size; constrained layout keeps everything inside it."""
    return Figure(figsize=(width, height), layout="constrained")


def format_axis(ax: Axes, categorical: str | None = None) -> None:
    """Minor ticks and a dashed grid on the value axes; none along a categorical axis."""
    ax.minorticks_on()
    value_axis = {"x": "y", "y": "x"}.get(categorical or "", "both")
    ax.grid(True, which="major", axis=value_axis, alpha=0.9)
    ax.grid(True, which="minor", axis=value_axis, alpha=0.25, linewidth=0.4)
    if categorical:
        getattr(ax, f"{categorical}axis").set_minor_locator(NullLocator())
        ax.grid(False, which="both", axis=categorical)


def percent_errors(
    value: Sequence[float], low: Sequence[float], high: Sequence[float]
) -> list[list[float]]:
    """Error-bar lengths in percent from shares and their interval bounds."""
    return [
        [100 * (v - lo) for v, lo in zip(value, low, strict=True)],
        [100 * (hi - v) for v, hi in zip(value, high, strict=True)],
    ]


MARKER = {"markersize": 4.5, "markeredgecolor": DARK_GREY, "markeredgewidth": 0.5}
ERRORBAR = {**MARKER, "elinewidth": 0.9, "capsize": 0}


def save(fig: Figure, path: Path) -> None:
    """``path``.pdf and ``path``.png at the figure's size; no creation date, so a rebuild gives
    the same bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(path.with_suffix(".png"))
