"""``crisisline paper``: figures, tables and numbers of the paper from ``artifacts/analysis``.

Reads only the analysis tables and the configs; every estimate and interval comes from there.
Ground truth v2 throughout, v1 only in the sensitivity table.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd

from crisisline.analysis.frames import write_table
from crisisline.cli import load_config, read_csv
from crisisline.paper import figures, style, tables, worldmap
from crisisline.paper.subsets import Scope

logger = logging.getLogger(__name__)

PRIMARY = "v2"
# Columns of the analysis tables that can name a model, or the group of its own the analysis
# gives a model that is in no group; a contrast between groups names both ("A - B, S0").
MODEL_COLUMNS = ("model_id", "model_group", "level", "domain")
CONTRAST_PARTS = re.compile(r" - |, ")


def without_models(table: pd.DataFrame, models: Collection[str]) -> pd.DataFrame:
    """``table`` without the rows that name one of ``models``."""
    drop = pd.Series(False, index=table.index)
    for column in MODEL_COLUMNS:
        if column in table:
            drop |= table[column].isin(models)
    if "contrast" in table:
        drop |= table["contrast"].map(
            lambda c: isinstance(c, str) and any(p in models for p in CONTRAST_PARTS.split(c))
        )
    return table[~drop].reset_index(drop=True)


def build(config: dict[str, Any]) -> int:
    """Every output of the paper stage in ``paths.paper``."""
    paths, settings = config["paths"], config["analysis"]
    analysis, out = Path(paths["analysis"]), Path(paths["paper"])
    model_groups = settings["exploratory"]["model_groups"]
    groups = list(model_groups)
    sizes = settings["model_sizes"]
    styles = style.model_styles(model_groups, sizes)
    group_colours = style.group_colours(groups)
    models_config = load_config(Path(paths["models_config"]))
    if unlisted := sorted(set(pd.read_csv(analysis / "models.csv")["model_id"]) - set(styles)):
        logger.warning(
            "Not in analysis.exploratory.model_groups, so left out of every paper figure and "
            "table: %s",
            ", ".join(unlisted),
        )
    temperature_block = {
        iso2
        for block in config["robustness"]["blocks"]
        for iso2 in block["prompts"].get("iso2", [])
        if iso2 != config["no_country_iso2"]
    }
    scope = Scope.from_rows(
        read_csv(Path(paths["countries"])),
        temperature_block,
        settings["grounding_excluded_models"],
    )
    min_digits = settings["exploratory"]["near_miss"]["min_digits"]

    def read(name: str, version: str = "version") -> pd.DataFrame:
        """An analysis table, ground truth v2 only where it holds both, without the models in no
        group; "NA" is Namibia."""
        table = pd.read_csv(analysis / name, keep_default_na=False, na_values=[""])
        if unlisted:
            table = without_models(table, unlisted)
        if version not in table:
            return table
        return table[table[version] == PRIMARY].drop(columns=version).reset_index(drop=True)

    results = pd.read_csv(analysis / "comparisons.csv", keep_default_na=False, na_values=[""])
    primary = results[results["analysis"] == PRIMARY]
    by_model = read("outcomes_by_model.csv")
    grounding = read("exploratory/grounding.csv")
    who_regions = read("exploratory/who_regions.csv")
    worst = read("exploratory/worst_countries.csv")
    near_miss = read("exploratory/near_miss_by_model.csv")
    temperature = read("robustness_temperature.csv")
    template_levels = read("exploratory/templates_levels.csv")

    estimate = json.loads(Path(paths["fabrication_estimate"]).read_text(encoding="utf-8"))
    fabrication = tables.fabrication_table(
        read("exploratory/fabrication_reader.csv"),
        estimate,
        read("fabrication_by_model.csv"),
        near_miss,
        min_digits,
    )
    gaps = read("exploratory/language_gap_by_model.csv")
    countries = read("exploratory/countries.csv")
    by_condition = read("outcomes_by_model_condition.csv")
    figure_dir = out / "figures"
    referral = figures.referral_data(read("referral_outcomes.csv"), styles, scope)
    ladder = figures.ladder_data(read("qwen_ladder.csv"), sizes, scope)
    ladder_outcomes = figures.ladder_outcomes_data(read("qwen_ladder_referral.csv"), sizes, scope)
    who = figures.who_gets_help_data(
        read("crisis_by_model_country.csv"),
        by_condition[
            (by_condition["condition"] == "S0") & (by_condition["outcome"] == "correct_crisis")
        ],
        countries,
        styles,
        scope,
    )
    who_language = figures.who_gets_help_language_data(
        read("correct_local_by_model_language.csv"),
        read("exploratory/correct_local_by_language.csv"),
        styles,
        scope,
    )
    world = config["paper"]["world_map"]
    features = json.loads(Path(paths["world_map"]).read_text(encoding="utf-8"))["features"]
    for cession in world["un_boundaries"]:
        features = worldmap.ceded(features, **cession)
    shapes = worldmap.countries(
        features,
        {c.iso2 for c in scope.countries},
        world["iso2_to_adm0"],
        world["drop_continents"],
        world["earth_radius_km"],
        world["central_longitude"],
    )
    local_crisis = read("exploratory/crisis_local_language.csv")
    world_data = figures.world_map_data(
        local_crisis, shapes, world["group"], world["small_country_km2"], world["classes"], scope
    )
    world_pooled = figures.world_map_pooled(
        local_crisis, [world["group"], world["comparison_group"]], scope
    )
    native = figures.native_language_data(gaps, styles, scope)
    fixes = figures.what_fixes_data(grounding, groups, scope)
    # Named by their place in the paper.
    drawn = {
        "fig1_world_map": (
            world_data,
            lambda d, p: figures.draw_world_map(
                d, shapes, world["classes"], world["earth_radius_km"], world["central_longitude"], p
            ),
        ),
        "fig2_referral_outcomes": (referral, figures.draw_referral),
        "fig3_language_heatmap": (who_language, figures.draw_who_gets_help_language),
        "fig4_what_fixes": (fixes, lambda d, p: figures.draw_what_fixes(d, group_colours, p)),
        "appendix_country_heatmap": (who, figures.draw_who_gets_help),
        "appendix_native_language": (
            native,
            lambda d, p: figures.draw_native_language(d, styles, p),
        ),
        "appendix_qwen_ladder": (ladder_outcomes, figures.draw_ladder),
    }
    for name, (data, draw) in drawn.items():
        write_table(data, figure_dir / f"{name}.csv")
        draw(data, figure_dir / name)
    # A slide version of the world map, on a dark background; the paper does not use it.
    figures.draw_world_map(
        world_data,
        shapes,
        world["classes"],
        world["earth_radius_km"],
        world["central_longitude"],
        out / "talk" / "talk_world_map_dark",
        style.DARK_MAP,
    )
    logger.info("Figures done")

    no_location = tables.no_location_table(
        primary[primary["comparison"] == "us_default"].iloc[0],
        read("us_default_breakdown.csv", version="analysis"),
        read("asks_location.csv"),
        read("outcomes_cells.csv"),
        read("default_country.csv"),
        list(styles),
        scope,
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
            template_levels, read("exploratory/templates_contrasts.csv"), grounding, scope
        ),
        "sensitivity": tables.sensitivity_table(results, scope),
        "robustness_temperature": tables.robustness_table(temperature, scope),
        "near_miss": tables.near_miss_table(near_miss, min_digits),
    }
    for name, table in built.items():
        write_table(table, out / "tables" / f"{name}.csv")
    numbers = tables.numbers_table(
        comparisons=primary,
        translation_correlation=read("exploratory/language_gap_translation_correlation.csv"),
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
        scope=scope,
    )
    numbers = pd.concat(
        [
            numbers,
            tables.figure_numbers(
                who,
                native,
                fixes,
                ladder_outcomes,
                world_data,
                world_pooled,
                figures.world_map_classes(world_data, world["classes"]),
                world["small_country_km2"],
            ),
        ],
        ignore_index=True,
    )
    write_table(numbers, out / "numbers.csv")
    regions = pd.DataFrame(style.WHO_REGIONS.items(), columns=["who_region", "name"])
    write_table(regions, out / "who_regions.csv")
    logger.info("Paper -> %s", out)
    return 0
