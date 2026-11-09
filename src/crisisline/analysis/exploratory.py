"""Exploratory analyses: their tables for ``analyze``.

Grounding by condition, the language gap, fabrication in reader terms, near-misses, template
consistency, countries and WHO regions, directories. Shares carry a cluster bootstrap
interval (:mod:`crisisline.analysis.intervals`) with the seed of ``analysis``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crisisline.analysis import breakdowns, fabrication, intervals, language_gap, near_miss
from crisisline.analysis.descriptive import _rng
from crisisline.analysis.intervals import ALL
from crisisline.cli import read_csv, read_yaml_list
from crisisline.extract import normalize
from crisisline.score import FABRICATED, NOT_A_NUMBER, UNKNOWN

logger = logging.getLogger(__name__)

Key = tuple[str, str, int]
OUTSIDE_GROUND_TRUTH = ("unknown", "fabricated", "other_service", "unverified")
VERSIONS = ("v2", "v1")


def labelling_files(config: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    """The rows of each labelling file of unknown numbers, by its name in ``paths``."""
    paths = config["paths"]
    files = dict.fromkeys(str(s["path"]) for s in config["unknowns"]["estimate"]["strata"])
    return {name: read_csv(Path(paths[name])) for name in files}


def ground_truth_targets(config: Mapping[str, Any]) -> dict[str, list[near_miss.Target]]:
    """Ground truth v2 per country for the near-miss test; country names do not identify a
    service either."""
    settings = config["analysis"]["exploratory"]["near_miss"]
    countries = read_csv(Path(config["paths"]["countries"]))
    country_words = {w for c in countries for w in f"{c['name_en']} {c['country_local']}".split()}
    return near_miss.targets(
        read_csv(Path(config["paths"]["resources"])),
        [*settings["generic_name_words"], *country_words],
        settings["name_min_chars"],
    )


def _translations(directory: Path) -> list[dict[str, Any]]:
    return [
        {**entry, "lang": path.stem}
        for path in sorted(directory.glob("*.yaml"))
        for entry in read_yaml_list(path)
    ]


def _by_version(
    build: Callable[[pd.DataFrame], pd.DataFrame], variants: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    return pd.concat([build(variants[v]).assign(version=v) for v in VERSIONS], ignore_index=True)


def language_tables(
    frame: pd.DataFrame,
    entries: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Sequence[str]],
    settings: Mapping[str, Any],
    replicates: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Per-model gap, the strength interaction with and without chrF, gap by language."""
    step, chrf_step = settings["strength_step"], settings["chrf_step"]
    strength = language_gap.strength(frame)
    held_out = language_gap.strength(frame, leave_out_gap_data=True)
    chrf = language_gap.prompt_chrf(frame, entries)
    interaction = pd.concat(
        [
            language_gap.strength_interaction(frame, strength, step, "all English prompts"),
            language_gap.strength_interaction(
                frame, held_out, step, "English prompts outside the gap data"
            ),
            language_gap.strength_interaction(
                frame, strength, step, "all English prompts, + chrF", chrf, chrf_step
            ),
            language_gap.leave_one_model_out(frame, strength, step),
        ],
        ignore_index=True,
    )
    quality = language_gap.translation_quality(entries)
    penalty = language_gap.penalty_by_language(frame, quality, replicates, seed)
    return {
        "language_gap_by_model": language_gap.per_model(frame, groups, replicates, seed),
        "model_strength": pd.DataFrame(
            {"strength_all_english": strength, "strength_held_out": held_out}
        )
        .rename_axis("model_id")
        .reset_index(),
        "language_gap_strength": interaction,
        "language_gap_by_language": penalty,
        "language_gap_translation_correlation": language_gap.correlations(
            penalty, replicates, seed
        ),
        "correct_local_by_language": breakdowns.by_language(frame, groups, replicates, seed),
    }


def _domains(
    row: Mapping[str, Any], group_of: Mapping[str, str], lang_group: Mapping[str, str]
) -> tuple[tuple[str, str], ...]:
    return (
        ("model", row["model_id"]),
        ("model group", group_of[row["model_id"]]),
        ("language group", lang_group[row["lang"]]),
        ("all", ALL),
    )


def reader_fabrication(
    config: Mapping[str, Any],
    prelabels: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    language_groups: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Per model, model group and language group, fabricated numbers over all numbers given and
    responses with at least one fabricated number (the labels on the scores before labels, where
    the strata are defined)."""
    paths, settings = config["paths"], config["unknowns"]
    specs = settings["estimate"]["strata"]
    flags = read_csv(Path(paths["unknowns_flags"]))
    labels = labelling_files(config)
    label_of = {r["pair_id"]: r["label"] for rows in labels.values() for r in rows}
    stratum_of = fabrication.stratum_index(specs, flags)
    group_of = intervals.group_of(groups)
    lang_group = {lang: g for g, langs in language_groups.items() for lang in langs}

    mentions: dict[str, Counter] = defaultdict(Counter)
    numbers: Counter = Counter()
    patterns: dict[tuple[str, str], list[tuple[bool, tuple[int, ...]]]] = defaultdict(list)
    data = prelabels[prelabels["no_resource"].notna()]
    for row in data[["model_id", "lang", "iso2", "numbers"]].to_dict("records"):
        domains = _domains(row, group_of, lang_group)
        known, pattern = False, [0] * len(specs)
        for number in row["numbers"]:
            numbers.update(domains)
            if number["number_class"] != UNKNOWN:
                continue
            pair = f"{row['iso2']}:{number['number']}"
            mentions[pair].update(domains)
            if pair not in label_of:
                pattern[stratum_of[pair]] += 1
            elif label_of[pair] == FABRICATED:
                known = True
        for domain in domains:
            patterns[domain].append((known, tuple(pattern)))
    domains = sorted(patterns)
    responses = []
    for domain in domains:
        found = patterns[domain]
        unknown = Counter(p for known, p in found if not known and any(p))
        responses.append(
            fabrication.Responses(
                total=len(found),
                known=sum(known for known, _ in found),
                patterns=np.array(list(unknown), dtype=float).reshape(-1, len(specs)),
                counts=np.array(list(unknown.values()), dtype=float),
            )
        )
    strata = {
        label: fabrication.build_strata(specs, flags, labels, mentions, domains, label)
        for label in (FABRICATED, NOT_A_NUMBER)
    }
    table = fabrication.reader_terms(
        strata[FABRICATED],
        strata[NOT_A_NUMBER],
        np.array([numbers[d] for d in domains], dtype=float),
        responses,
        domains,
        settings["estimate"]["replicates"],
        config["unknowns"]["seed"],
    )
    kind, name = zip(*table.pop("domain"), strict=True)
    return table.assign(kind=kind, domain=name)[["kind", "domain", *table.columns]]


def _context(
    texts: Mapping[Key, str], cache: dict[Key, str], key: Key, start: int, raw: str
) -> str:
    if key not in cache:
        cache[key] = normalize(texts.get(key, ""))
    return near_miss.number_context(cache[key], start, start + len(raw))


def near_miss_tables(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    texts: Mapping[Key, str],
    groups: Mapping[str, Sequence[str]],
    replicates: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Every long number outside ground truth v2 against the numbers of its country."""
    settings = config["analysis"]["exploratory"]["near_miss"]
    no_country = config["no_country_iso2"]
    min_digits, max_distance = settings["min_digits"], settings["max_distance"]
    by_country = ground_truth_targets(config)
    labelled = [row for rows in labelling_files(config).values() for row in rows]
    belongs = {r["pair_id"]: r["belongs_to_iso2"] for r in labelled if r["belongs_to_iso2"]}
    group_of = intervals.group_of(groups)
    chance: dict[tuple[str, int, str], float] = {}
    normalised: dict[Key, str] = {}
    rows, unresolved = [], 0
    columns = ["model_id", "prompt_id", "sample_idx", "iso2", "numbers"]
    for row in frame[columns].to_dict("records"):
        key = (row["model_id"], row["prompt_id"], row["sample_idx"])
        for number in row["numbers"]:
            value = number["number"]
            if number["number_class"] not in OUTSIDE_GROUND_TRUTH:
                continue
            if len(near_miss.digits(value)) < min_digits:
                continue
            country = row["iso2"]
            if country == no_country:
                country = belongs.get(f"{no_country}:{value}", "")
                if not country:
                    unresolved += 1
                    continue
            candidates = by_country.get(country, [])
            found = near_miss.match(value, candidates, max_distance)
            cell = (country, len(near_miss.digits(value)), near_miss.random_prefix(value))
            if cell not in chance:
                chance[cell] = near_miss.chance(value, candidates, max_distance)
            named = found is not None and near_miss.names_service(
                _context(texts, normalised, key, number["start"], number["raw"]), found.target
            )
            rows.append(
                {
                    "model_group": group_of[row["model_id"]],
                    "model_id": row["model_id"],
                    "prompt_id": row["prompt_id"],
                    "sample_idx": row["sample_idx"],
                    "iso2": row["iso2"],
                    "country": country,
                    "number": value,
                    "number_class": number["number_class"],
                    "near_miss": float(found is not None),
                    "kind": found.kind if found else "",
                    "target": found.target.number if found else "",
                    "target_type": found.target.type if found else "",
                    "target_name": found.target.name if found else "",
                    "names_service": float(named),
                    "chance": chance[cell],
                }
            )
    mentions = pd.DataFrame(rows)
    logger.info("Near-misses: %d mentions tested, %d with no country", len(mentions), unresolved)
    return {
        "near_miss_by_model": _near_miss_summary(mentions, groups, replicates, seed, unresolved),
        "near_miss_pairs": _near_miss_pairs(mentions),
        "near_miss_validation": _near_miss_validation(
            labelled, by_country, settings, no_country, min_digits, max_distance
        ),
    }


def _near_miss_summary(
    mentions: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    replicates: int,
    seed: int,
    unresolved: int,
) -> pd.DataFrame:
    """Per model, group and all: near-miss share of tested mentions and its chance baseline, both
    resampled by the user's country, and how many near-misses name the matched service."""
    rows = []
    for group, model, part in intervals.scopes(mentions, groups):
        row: dict[str, Any] = {"model_group": group, "model_id": model, "mentions": len(part)}
        for column in ("near_miss", "chance"):
            rng = _rng(seed, group, model, column)
            result = intervals.ratio_interval(part[column], part["iso2"], replicates, rng)
            row |= {
                column: result["estimate"],
                f"{column}_ci_low": result["ci_low"],
                f"{column}_ci_high": result["ci_high"],
            }
        hits = part[part["near_miss"] == 1]
        row |= {
            "near_miss_mentions": len(hits),
            **{
                f"near_miss_{label}": int((hits["number_class"] == label).sum())
                for label in OUTSIDE_GROUND_TRUTH
            },
            "names_service": int(hits["names_service"].sum()),
            "names_service_share": hits["names_service"].mean() if len(hits) else float("nan"),
            "countries": part["iso2"].nunique(),
        }
        rows.append(row)
    return pd.DataFrame(rows).assign(mentions_without_country=unresolved)


def _near_miss_pairs(mentions: pd.DataFrame) -> pd.DataFrame:
    hits = mentions[mentions["near_miss"] == 1]
    return (
        hits.groupby(["country", "number", "kind", "target", "target_type", "target_name"])
        .agg(
            mentions=("model_id", "size"),
            models=("model_id", lambda m: ";".join(sorted(set(m)))),
            names_service=("names_service", "sum"),
            classes=("number_class", lambda c: ";".join(sorted(set(c)))),
        )
        .reset_index()
        .sort_values(["mentions", "country", "number"], ascending=[False, True, True])
    )


def _near_miss_validation(
    labelled: Sequence[Mapping[str, str]],
    by_country: Mapping[str, Sequence[near_miss.Target]],
    settings: Mapping[str, Any],
    no_country: str,
    min_digits: int,
    max_distance: int,
) -> pd.DataFrame:
    """Every labelled pair in the method's scope whose note gives the real number: whether the
    method finds it, and whether the noted number is itself within the method's distance."""
    pattern = re.compile(settings["note_pattern"], re.IGNORECASE)
    rows = []
    for r in labelled:
        if not pattern.search(r["note"]) or len(near_miss.digits(r["number_norm"])) < min_digits:
            continue
        country = r["belongs_to_iso2"] if r["iso2"] == no_country else r["iso2"]
        found = near_miss.match(r["number_norm"], by_country.get(country, []), max_distance)
        noted = near_miss.noted_numbers(r["note"], country)
        mine = near_miss.digits(r["number_norm"])
        distances = [
            near_miss.levenshtein(mine, form, 99)
            for n in noted
            for form in (near_miss.forms(n, country) if n.startswith("+") else (n,))
        ]
        rows.append(
            {
                "pair_id": r["pair_id"],
                "label": r["label"],
                "responses": int(r["responses"]),
                "note": r["note"],
                "noted_numbers": ";".join(noted),
                "noted_distance": min(distances) if distances else float("nan"),
                "noted_within_method": bool(distances) and min(distances) <= max_distance,
                "found": found is not None,
                "found_target": found.target.number if found else "",
                "found_is_noted": bool(found) and found.target.number in noted,
            }
        )
    return pd.DataFrame(rows)


def model_groups(
    listed: Mapping[str, Sequence[str]], models: Iterable[str]
) -> Mapping[str, Sequence[str]]:
    """``listed`` with every model it leaves out as a group of its own, named by its model_id."""
    known = {model for group in listed.values() for model in group}
    unlisted = sorted(set(models) - known)
    if not unlisted:
        return listed
    logger.warning(
        "Not in analysis.exploratory.model_groups, each in a group of its own: %s",
        ", ".join(unlisted),
    )
    return {**listed, **{model: [model] for model in unlisted}}


def tables(
    config: Mapping[str, Any],
    variants: Mapping[str, pd.DataFrame],
    texts: Mapping[Key, str],
) -> dict[str, pd.DataFrame]:
    """Every exploratory table, by file name."""
    settings = config["analysis"]
    exploratory = settings["exploratory"]
    groups = model_groups(
        exploratory["model_groups"],
        (model for frame in variants.values() for model in frame["model_id"].unique()),
    )
    seed, replicates = settings["seed"], settings["interval_replicates"]
    countries = read_csv(Path(config["paths"]["countries"]))
    entries = _translations(Path(config["paths"]["translations"]))

    out: dict[str, pd.DataFrame] = {
        "grounding": _by_version(
            lambda f: breakdowns.grounding(f, groups, replicates, seed), variants
        ),
    }
    logger.info("Grounding done")
    language = {
        v: language_tables(variants[v], entries, groups, exploratory, replicates, seed)
        for v in VERSIONS
    }
    for name in language["v2"]:
        out[name] = pd.concat(
            [language[v][name].assign(version=v) for v in VERSIONS], ignore_index=True
        )
    logger.info("Language gap done")
    out["fabrication_reader"] = reader_fabrication(
        config, variants["prelabels"], groups, exploratory["language_groups"]
    )
    logger.info("Fabrication in reader terms done")
    out |= near_miss_tables(config, variants["v2"], texts, groups, replicates, seed)
    logger.info("Near-misses done")
    levels, contrasts = {}, {}
    for v in VERSIONS:
        levels[v], contrasts[v] = breakdowns.templates(variants[v], groups, replicates, seed)
    out["templates_levels"] = pd.concat([levels[v].assign(version=v) for v in VERSIONS])
    out["templates_contrasts"] = pd.concat([contrasts[v].assign(version=v) for v in VERSIONS])
    logger.info("Templates done")
    by_country, by_region = {}, {}
    for v in VERSIONS:
        by_country[v], by_region[v] = breakdowns.countries(
            variants[v], groups, countries, replicates, seed
        )
    out["countries"] = pd.concat([by_country[v].assign(version=v) for v in VERSIONS])
    out["who_regions"] = pd.concat([by_region[v].assign(version=v) for v in VERSIONS])
    out["worst_countries"] = breakdowns.worst_countries(
        by_country["v2"], exploratory["worst_countries"]
    ).assign(version="v2")
    logger.info("Countries done")
    out["directories"] = _by_version(
        lambda f: breakdowns.directories(f, groups, replicates, seed), variants
    )
    logger.info("Directories done")
    return out
