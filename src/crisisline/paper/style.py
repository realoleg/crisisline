"""The look of the paper's figures: drawn at the printed size of the ACL two-column template, in
STIX (the Times of the template, bundled with matplotlib, so the same on every machine)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.text import Text
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
    "correct_local_not_crisis_line": ("Correct local number, not the crisis line", PALETTE[4]),
    "misrouted": ("Misrouted", PALETTE[1]),
    "fabricated": ("Fabricated", MUTED_RED),
    "unconfirmed_number": ("Unconfirmed number", LIGHT_GREY),
    "nothing": ("No number", GREY),
}
# Model groups in configured order. The blue of the third is also a referral colour, but
# group colours and referral colours never share a figure.
GROUP_COLOURS = (PALETTE[2], PALETTE[3], PALETTE[4])
# The share of the crisis line: red at none, near-white at half, green at all.
CRISIS_MAP = LinearSegmentedColormap.from_list("crisis", (MUTED_RED, "#F7F7F7", PALETTE[0]))
# Classes of the share without the crisis line, light to near-black red.
MISSING_CLASSES = ("#F2D7D4", "#CF7F76", "#8E3129", "#3F1210")


@dataclass(frozen=True, slots=True)
class MapTheme:
    background: str
    outside: str  # land outside the study
    no_line: str  # countries without a national crisis line, hatched
    hatch: str
    border: str  # between countries of the study
    graticule: str
    outline: str  # the projection's boundary
    marker_edge: str
    text: str


PAPER_MAP = MapTheme(
    "white", "#F4F4F4", "#9A9A9A", DARK_GREY, "white", "#E6E6E6", "#BBBBBB", DARK_GREY, DARK_GREY
)
# For talks only.
DARK_MAP = MapTheme(
    "#111111", "#2A2A2A", "#9A9A9A", DARK_GREY, "white", "#222222", "#555555", "#DDDDDD", "white"
)

# Display names of the models in every figure and label column; model_id stays the key.
MODEL_NAMES = {
    "gpt-6-luna": "GPT-6 Luna",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "claude-haiku-5.5": "Claude Haiku 5.5",
    "gemma4-e4b": "Gemma 4 E4B",
    "gemma4-e2b": "Gemma 4 E2B",
    "qwen3.5-9b": "Qwen3.5 9B",
    "qwen3.5-4b": "Qwen3.5 4B",
    "qwen3.5-2b": "Qwen3.5 2B",
    "qwen3.5-0.8b": "Qwen3.5 0.8B",
    "llama3.1-8b": "Llama 3.1 8B",
    "local-8": "Ministral 3 8B",
}
# Prompt languages by code, as in the translation config, with English.
LANGUAGE_NAMES = {
    "am": "Amharic",
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sv": "Swedish",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}
# WHO regions: figures and CSVs show the code, the caption the name.
WHO_REGIONS = {
    "AFR": "Africa",
    "AMR": "Americas",
    "EMR": "Eastern Mediterranean",
    "EUR": "Europe",
    "SEAR": "South-East Asia",
    "WPR": "Western Pacific",
}

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
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "legend.labelspacing": 0.3,
    "legend.borderpad": 0.2,
    "legend.borderaxespad": 0.2,
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


def group_label(group: str) -> str:
    return group[:1].upper() + group[1:]


def model_name(model: str) -> str:
    return MODEL_NAMES.get(model, model)


def percent(share: float) -> str:
    return f"{100 * share:.0f}%"


def spread(values: Sequence[float], gap: float) -> list[float]:
    """``values`` moved apart, in their order, until neighbours are at least ``gap`` apart: the
    positions of direct labels that would otherwise overlap."""
    order = sorted(range(len(values)), key=lambda k: values[k])
    placed = [float(values[k]) for k in order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + gap)
    shift = (sum(placed) - sum(values[k] for k in order)) / len(placed)
    result = [0.0] * len(values)
    for k, value in zip(order, placed, strict=True):
        result[k] = value - shift
    return result


def model_styles(
    groups: Mapping[str, Sequence[str]], sizes: Mapping[str, float]
) -> dict[str, ModelStyle]:
    """Every model in display order: groups as configured, models of the size ladder first and by
    size within their group; a colour per group."""
    styles = {}
    for k, (group, models) in enumerate(groups.items()):
        for model in sorted(models, key=lambda m: (m not in sizes, sizes.get(m, 0.0))):
            styles[model] = ModelStyle(group, GROUP_COLOURS[k])
    return styles


def group_colours(groups: Sequence[str]) -> dict[str, str]:
    """The colour of each group, in configured order."""
    return {g: GROUP_COLOURS[k] for k, g in enumerate(groups)}


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
    """``path``.pdf and ``path``.png at the figure's size, every multi-line text at single line
    spacing; no creation date, so a rebuild gives the same bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.draw_without_rendering()  # creates every tick label
    for text in fig.findobj(Text):
        text.set_linespacing(1.0)
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(path.with_suffix(".png"))
