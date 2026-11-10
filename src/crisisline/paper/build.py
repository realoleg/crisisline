"""``crisisline paper``: figures, tables and numbers of the paper from ``artifacts/analysis``.

Reads only the analysis tables and the configs; every estimate and interval comes from there.
Ground truth v2 throughout, v1 only in the sensitivity table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from crisisline.cli import load_config
from crisisline.paper import figures, style, tables

logger = logging.getLogger(__name__)

PRIMARY = "v2"


def _write(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n", float_format="%.6g")


def build(config: dict[str, Any]) -> int:
    """Every output of the paper stage in ``paths.paper``."""
    # Font subsetting in the PDF backend logs every table it prunes.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    paths, settings = config["paths"], config["analysis"]
    analysis, out = Path(paths["analysis"]), Path(paths["paper"])
    model_groups = settings["exploratory"]["model_groups"]
    groups = list(model_groups)
    sizes = settings["model_sizes"]
    styles = style.model_styles(model_groups, sizes)
    group_styles = style.group_styles(groups)
    models_config = load_config(Path(paths["models_config"]))

    def read(name: str, version: str = "version") -> pd.DataFrame:
        """An analysis table, ground truth v2 only where it holds both; "NA" is Namibia."""
        table = pd.read_csv(analysis / name, keep_default_na=False, na_values=[""])
        if version not in table:
            return table
        return table[table[version] == PRIMARY].drop(columns=version).reset_index(drop=True)

    hypotheses = pd.read_csv(analysis / "hypotheses.csv", keep_default_na=False, na_values=[""])
    primary = hypotheses[hypotheses["analysis"] == PRIMARY]
    by_model = read("outcomes_by_model.csv")
    grounding = read("exploratory/grounding.csv")
    who_regions = read("exploratory/who_regions.csv")
    worst = read("exploratory/worst_countries.csv")
    near_miss = read("exploratory/near_miss_by_model.csv")
    temperature = read("robustness_temperature.csv")
    template_levels = read("exploratory/templates_levels.csv")

    figure_dir = out / "figures"
    referral = figures.referral_data(
        read("referral_outcomes.csv"),
        by_model[by_model["outcome"] == "correct_local"],
        styles,
    )
    ladder = figures.ladder_data(read("qwen_ladder.csv"), sizes)
    drawn = {
        "fig1_referral_outcomes": (referral, figures.draw_referral),
        "fig2_knowledge_vs_location": (
            figures.grounding_data(grounding, groups),
            lambda d, p: figures.draw_grounding(d, group_styles, p),
        ),
        "fig3_language": (
            figures.language_data(
                read("exploratory/language_gap_by_model.csv"),
                read("exploratory/correct_local_by_language.csv"),
                styles,
                groups,
            ),
            lambda d, p: figures.draw_language(d, styles, group_styles, p),
        ),
        "fig4_geography": (
            figures.geography_data(who_regions, worst, groups),
            lambda d, p: figures.draw_geography(d, group_styles, p),
        ),
        "appendix_qwen_ladder": (
            ladder,
            lambda d, p: figures.draw_ladder(d, styles[next(iter(sizes))].colour, p),
        ),
        "appendix_countries": (
            figures.countries_data(read("exploratory/countries.csv"), groups),
            lambda d, p: figures.draw_countries(d, group_styles, p),
        ),
    }
    for name, (data, draw) in drawn.items():
        _write(data, figure_dir / f"{name}.csv")
        draw(data, figure_dir / name)
    logger.info("Figures done")

    fabrication = tables.fabrication_table(
        read("exploratory/fabrication_reader.csv"), read("fabrication_by_model.csv"), near_miss
    )
    no_location = tables.no_location_table(
        primary[primary["hypothesis"] == "us_default"].iloc[0],
        read("us_default_breakdown.csv", version="analysis"),
        read("asks_location.csv"),
        read("outcomes_cells.csv"),
        read("default_country.csv"),
        list(styles),
    )
    built = {
        "models": tables.models_table(read("models.csv"), models_config, styles),
        "fabrication": fabrication,
        "no_location": no_location,
        "languages": tables.languages_table(
            read("exploratory/correct_local_by_language.csv"),
            read("exploratory/language_gap_by_language.csv"),
        ),
        "templates": tables.templates_table(
            template_levels, read("exploratory/templates_contrasts.csv")
        ),
        "sensitivity": tables.sensitivity_table(hypotheses),
        "robustness_temperature": tables.robustness_table(temperature),
        "near_miss": tables.near_miss_table(near_miss),
    }
    for name, table in built.items():
        _write(table, out / "tables" / f"{name}.csv")
    numbers = tables.numbers_table(
        hypotheses=primary,
        no_location=no_location,
        who_regions=who_regions,
        worst=worst,
        georgia=read("georgia_case.csv"),
        template_levels=template_levels,
        by_model=by_model,
        blocked_model=settings["blocked_model"],
        ladder=ladder,
        temperature=temperature,
        grounding=grounding,
        groups=groups,
        fabrication_rows=fabrication,
        referral=referral,
    )
    _write(numbers, out / "numbers.csv")
    logger.info("Paper -> %s", out)
    return 0
