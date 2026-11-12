"""``crisisline analyze``: from scores to everything in ``artifacts/analysis``."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from crisisline import unknowns as unknowns_module
from crisisline.analysis import (
    comparisons,
    descriptive,
    examples,
    exploratory,
    fabrication,
    frames,
)
from crisisline.analysis.frames import write_table
from crisisline.cli import (
    UNKNOWN_COMMIT,
    file_sha256,
    git_commit,
    load_config,
    read_csv,
    read_jsonl,
    repo_root,
)

logger = logging.getLogger(__name__)

Key = tuple[str, str, int]
MAIN_OUTCOMES = ("correct_local", "correct_crisis", "wrong_country", "na_default", "no_resource")
TABLE_OUTCOMES = (
    "correct_local",
    "correct_crisis",
    "wrong_country",
    "na_default",
    "fabricated",
    "no_usable_resource",
)
ROBUSTNESS_OUTCOMES = (
    "correct_local",
    "correct_crisis",
    "wrong_country",
    "na_default",
    "fabricated",
    "no_resource",
    "loop",
)
# In the order of comparisons.csv; Holm runs over all five.
MAIN_COMPARISONS = (
    "cue_country_vs_city",
    "language_en_vs_local",
    "model_size",
    comparisons.US_DEFAULT,
    "grounding_verified_number",
)
GEE_COMPARISONS = tuple(c for c in MAIN_COMPARISONS if c != comparisons.US_DEFAULT)


def read_generations(
    directory: Path, ngram: int, threshold: float
) -> tuple[dict[Key, frames.Record], dict[Key, str], dict[Key, str], dict[str, set[str]]]:
    """Loop flags, texts, refusal details and model versions of a directory of generations."""
    records: dict[Key, frames.Record] = {}
    texts: dict[Key, str] = {}
    details: dict[Key, str] = {}
    model_versions: dict[str, set[str]] = defaultdict(set)
    for path in sorted(directory.glob("*.jsonl")):
        for raw in read_jsonl(path):
            key = (raw["model_id"], raw["prompt_id"], raw["sample_idx"])
            records[key] = frames.read_record(raw, ngram, threshold)
            texts[key] = raw.get("response") or ""
            details[key] = raw.get("detail") or ""
            if raw.get("model_version"):
                model_versions[raw["model_id"]].add(raw["model_version"])
    return records, texts, details, model_versions


def git_state(root: Path) -> tuple[str, bool]:
    """HEAD and whether tracked files have changes; ``unknown`` outside a git checkout or
    without git."""
    commit = git_commit(root)
    if commit == UNKNOWN_COMMIT:
        return commit, False
    return commit.removesuffix("-dirty"), commit.endswith("-dirty")


def input_checksums(inputs: Sequence[Path], root: Path) -> dict[str, str]:
    """sha256 of every input, keyed by its path relative to ``root``."""
    return {os.path.relpath(path, root): file_sha256(path) for path in inputs}


# Fields of a score that come from the response alone, the same against every ground truth.
RESPONSE_FIELDS = ("model_id", "prompt_id", "sample_idx", "error", "blocked", "empty", "truncated")


def out_of_date(
    scores: Mapping[str, Sequence[Mapping[str, Any]]], current: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Names of the ``scores`` from other generations than ``current``: other records, or another
    error, blocked, empty or truncated flag on one."""

    def responses(rows: Sequence[Mapping[str, Any]]) -> set[tuple[Any, ...]]:
        return {tuple(row[f] for f in RESPONSE_FIELDS) for row in rows}

    reference = responses(current)
    return [name for name, rows in scores.items() if responses(rows) != reference]


def _read_scores(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _main_comparisons(
    specs: Mapping[str, comparisons.GeeSpec],
    frame: pd.DataFrame,
    us_default_frame: pd.DataFrame,
    analysis: str,
    sizes: Mapping[str, float],
    lang: str,
) -> list[comparisons.TestResult]:
    """The five main comparisons in :data:`MAIN_COMPARISONS` order. ``us_default`` also counts
    the temperature block, whose prompts name no country, hence its own frame."""
    return [
        comparisons.run_us_default(us_default_frame, analysis, lang)
        if c == comparisons.US_DEFAULT
        else comparisons.run_gee(specs[c], frame, analysis, sizes)
        for c in MAIN_COMPARISONS
    ]


def exchangeable_results(
    specs: Mapping[str, comparisons.GeeSpec],
    variants: Mapping[str, pd.DataFrame],
    sizes: Mapping[str, float],
) -> list[comparisons.TestResult]:
    """The exchangeable working correlation next to the main analyses: a model that does not
    converge reports no estimate. ``grounding_verified_number`` also with every model."""
    results = []
    for version in ("v2", "v1"):
        frame = variants[version]
        for c in ("cue_country_vs_city", "language_en_vs_local", "grounding_verified_number"):
            spec = replace(specs[c], exchangeable=True)
            results.append(comparisons.run_gee(spec, frame, f"{version}, Exchangeable", sizes))
        all_models = replace(
            specs["grounding_verified_number"], exchangeable=True, excluded_models=()
        )
        results.append(
            comparisons.run_gee(all_models, frame, f"{version}, Exchangeable, all models", sizes)
        )
    return results


def sensitivity_results(
    specs: Mapping[str, comparisons.GeeSpec],
    variants: Mapping[str, pd.DataFrame],
    us_default_frames: Mapping[str, pd.DataFrame],
    sizes: Mapping[str, float],
    blocked_model: str,
    georgia: str,
    lang: str,
) -> list[comparisons.TestResult]:
    """Sensitivity analyses of the main comparisons, without Holm adjustment. ``georgia`` is the
    country left out once: its English name is also a US state."""
    v2, us_default_frame = variants["v2"], us_default_frames["v2"]

    def gee(data: pd.DataFrame, ids: Sequence[str], analysis: str) -> list:
        return [comparisons.run_gee(specs[c], data, analysis, sizes) for c in ids]

    results = gee(v2[v2["truncated"] != 1], GEE_COMPARISONS, "without truncated")
    results.append(
        comparisons.run_us_default(
            us_default_frame[us_default_frame["truncated"] != 1], "without truncated", lang
        )
    )
    results += gee(v2[v2["has_crisis_line"]], GEE_COMPARISONS, "only countries with a crisis line")
    results += gee(variants["official"], GEE_COMPARISONS, "official sources only")
    results += gee(
        v2[v2["iso2"] != georgia],
        ("cue_country_vs_city", "language_en_vs_local"),
        f"without {georgia}",
    )
    blocked = (v2["model_id"] == blocked_model) & (v2["blocked"] == 1)
    results += gee(
        v2[~blocked],
        ("cue_country_vs_city", "language_en_vs_local", "grounding_verified_number"),
        f"{blocked_model} without blocked",
    )
    results.append(comparisons.run_us_default(us_default_frame, "v2", lang, "default_countries"))
    results.append(
        comparisons.run_us_default(us_default_frames["v1"], "v1", lang, "default_countries")
    )
    results += comparisons.simple_effects_cue(specs["cue_country_vs_city"], v2, "v2", sizes)
    results += comparisons.simple_effects_cue(
        specs["cue_country_vs_city"], variants["v1"], "v1", sizes
    )
    return results


def extraction_comparison(
    variants: Mapping[str, pd.DataFrame], replicates: int, seed: int, lang: str
) -> tuple[pd.DataFrame, list[comparisons.TestResult]]:
    """Main outcomes and ``us_default``, main block only, by extraction: pre-fix and final without
    labels on ground truth v1 isolate the fixes; ``v2`` is the final scores."""
    tables_by_version = []
    us_default_results = []
    for name in ("prefix", "prelabels", "v2"):
        frame = variants[name]
        tables_by_version.append(
            descriptive.share_table(frame, ["model_id"], replicates, seed, MAIN_OUTCOMES).assign(
                version=name
            )
        )
        us_default_results.append(comparisons.run_us_default(frame, f"main block, {name}", lang))
    return pd.concat(tables_by_version), us_default_results


def icc_table(frame: pd.DataFrame, results: Sequence[comparisons.TestResult]) -> pd.DataFrame:
    """The intraclass correlations."""
    rows = [
        {
            "measure": f"residual correlation by country, {r.comparison}",
            "value": r.residual_correlation,
        }
        for r in results
        if r.comparison != comparisons.US_DEFAULT
    ]
    cued = frames.s0_cued(frame)
    rows.append(
        {
            "measure": "ANOVA ICC of correct_local by country, S0 with cue",
            "value": comparisons.anova_icc(cued["correct_local"], cued["iso2"]),
        }
    )
    api = frame[frame["is_api"] & frame["correct_local"].notna()]
    rows.extend(
        {
            "measure": f"ANOVA ICC of correct_local within a prompt, {model}",
            "value": comparisons.anova_icc(part["correct_local"], part["prompt_id"]),
        }
        for model, part in api.groupby("model_id")
    )
    rows.append(
        {
            "measure": "ANOVA ICC of correct_local within a prompt, all API models",
            "value": comparisons.anova_icc(
                api["correct_local"], api["model_id"] + "/" + api["prompt_id"]
            ),
        }
    )
    return pd.DataFrame(rows)


def fabrication_tables(
    config: Mapping[str, Any], prelabels: Sequence[Mapping[str, Any]], replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The estimator per model, and over all models as a check on ``unknowns --estimate``."""
    mentions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pairs = unknowns_module.collect_pairs(prelabels)
    for pair in pairs:
        for mention in pair.mentions:
            mentions[pair.pair_id][mention.response_id.split("/", 1)[0]] += 1
    flags = exploratory.unknown_flags(config, pairs)
    specs = config["unknowns"]["estimate"]["strata"]
    labels = exploratory.labelling_files(config)
    models = sorted({m for by_model in mentions.values() for m in by_model})
    strata = fabrication.build_strata(specs, flags, labels, mentions, models)
    table = fabrication.per_model(strata, models, replicates, seed)
    pooled = {pair: {"all models": sum(m.values())} for pair, m in mentions.items()}
    all_strata = fabrication.build_strata(specs, flags, labels, pooled, ["all models"])
    table = pd.concat([table, fabrication.per_model(all_strata, ["all models"], replicates, seed)])
    return table, fabrication.cells(strata, models)


def _grounding_excluded_table(
    variants: Mapping[str, pd.DataFrame], excluded: Sequence[str]
) -> pd.DataFrame:
    """The excluded models' outcomes in each condition of ``grounding_verified_number``, per
    version."""
    rows = []
    for version in ("v2", "v1"):
        frame = variants[version]
        data = frame[frame["model_id"].isin(excluded) & frame["lang_mode"].eq("local")]
        data = data[(data["condition"] == "M2") | (data["cue"] == "country")]
        for (model, condition), part in data.groupby(["model_id", "condition"]):
            rows.append(
                {
                    "version": version,
                    "model_id": model,
                    "condition": condition,
                    "responses": len(part),
                    "correct_local": int(part["correct_local"].sum()),
                }
            )
    return pd.DataFrame(rows)


def _model_table(
    config: Mapping[str, Any], frame: pd.DataFrame, model_versions: Mapping[str, set[str]]
) -> pd.DataFrame:
    settings = load_config(Path(config["paths"]["models_config"]))["models"]
    rows = []
    for model, part in frame.groupby("model_id"):
        spec = settings[model]
        rows.append(
            {
                "model_id": model,
                "backend": spec["backend"],
                "identifier": spec.get("tag") or spec.get("model"),
                "responses_per_prompt": part["sample_idx"].nunique(),
                "records": len(part),
                "versions_seen": len(model_versions[model]),
            }
        )
    return pd.DataFrame(rows)


def results_notes(bootstrap: pd.DataFrame | None, fabrication_table: pd.DataFrame) -> str:
    """What a reader of the tables has to know about the bootstrap and the fabrication intervals."""
    lines = ["# Notes on the results", ""]
    if bootstrap is not None:
        for row in bootstrap[bootstrap["failed"] > 0].itertuples():
            lines.append(
                f"- Cluster bootstrap of {row.comparison}: {row.failed} of {row.replicates} "
                "replicates did not converge and were excluded."
            )
    for row in fabrication.collapsed_intervals(fabrication_table).itertuples():
        lines.append(
            f"- Fabrication share of {row.model_id}: report only the range "
            f"{100 * row.share_lower:.1f}-{100 * row.share_upper:.1f}%. An interval in "
            "`fabrication_by_model.csv` collapses to a point; that is not real precision."
        )
    return "\n".join(lines) + "\n"


def _us_default_frame(main: pd.DataFrame, robustness: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([main.assign(block="main"), robustness.assign(block="robustness")])


def _with_block(frame: pd.DataFrame, blocks: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Each robustness row with the name of the block whose prompt fields it matches
    (``robustness.blocks``); a row in no block, or in two, is an error."""
    names = pd.Series(None, index=frame.index, dtype=object)
    for block in blocks:
        fields = block["prompts"].items()
        match = pd.concat([frame[f].isin(values) for f, values in fields], axis=1).all(axis=1)
        if names[match].notna().any():
            raise ValueError(f"robustness rows in block {block['name']} and another")
        names[match] = block["name"]
    if names.isna().any():
        raise ValueError(f"{int(names.isna().sum())} robustness rows are in no block")
    return frame.assign(block=names)


def _ladder_table(
    frame: pd.DataFrame, sizes: Mapping[str, float], replicates: int, seed: int
) -> pd.DataFrame:
    cued = frames.s0_cued(frame)[lambda d: d["model_id"].isin(sizes)]
    return descriptive.share_table(cued, ["model_id", "cue"], replicates, seed, ["correct_local"])


def _country_table(frame: pd.DataFrame, countries: Sequence[Mapping[str, str]]) -> pd.DataFrame:
    """Each country's share: the mean over models of each model's mean, S0 with a cue."""
    cued = frames.s0_cued(frame)
    per_model = cued.groupby(["iso2", "model_id"])["correct_local"].mean().reset_index()
    table = per_model.groupby("iso2")["correct_local"].mean().rename("share").reset_index()
    return table.merge(pd.DataFrame(countries)[["iso2", "who_region", "name_en"]], on="iso2")


def _figure_tables(
    frame: pd.DataFrame,
    version: str,
    out: Path,
    countries: Sequence[Mapping[str, str]],
    sizes: Mapping[str, float],
    seed: int,
    replicates: int,
) -> None:
    """Each country's share and the Qwen size ladder; the v1 tables have a suffix."""
    suffix = "" if version == "v2" else f"_{version}"
    write_table(_country_table(frame, countries), out / f"correct_by_country{suffix}.csv")
    write_table(_ladder_table(frame, sizes, replicates, seed), out / f"qwen_ladder{suffix}.csv")


def analyze(config: dict[str, Any], variant_scores: Mapping[str, Path], run_bootstrap: bool) -> int:
    """Every output of the analysis for the scores named by ``config`` and ``variant_scores`` (the
    robustness block and the official-sources-only run)."""
    paths, settings = config["paths"], config["analysis"]
    out = Path(paths["analysis"])
    seed, replicates = settings["seed"], settings["interval_replicates"]
    no_country = config["no_country_iso2"]
    sizes = settings["model_sizes"]

    # official is scored by run_analyze from the generations on disk; `score` alone rewrites only
    # the v2 scores.
    def responses(path: Path) -> list[dict[str, Any]]:
        return pq.read_table(path, columns=list(RESPONSE_FIELDS)).to_pylist()

    current = responses(Path(variant_scores["official"]))
    main_scores = [Path(paths[name]) for name in ("scores", "scores_v1", "scores_prelabels")]
    stale = out_of_date({str(path): responses(path) for path in main_scores}, current)
    if stale:
        logger.error("scores are out of date; run `crisisline score` (%s)", ", ".join(stale))
        return 1
    # The downloaded pre-fix scores must come from the same generations as well, for the models
    # they hold: the code no longer produces them, so a model run since is not among them.
    prefix_path = Path(paths["scores_prefix"])
    prefix = responses(prefix_path)
    prefix_models = {row["model_id"] for row in prefix}
    if out_of_date({"prefix": prefix}, [r for r in current if r["model_id"] in prefix_models]):
        logger.error(
            "%s is not from these generations of its models; download both again", prefix_path
        )
        return 1
    if missing := sorted({row["model_id"] for row in current} - prefix_models):
        logger.warning(
            "Not in %s, so left out of extraction_comparison.csv: %s",
            prefix_path,
            ", ".join(missing),
        )

    prompts = {p["prompt_id"]: p for p in read_jsonl(Path(paths["prompts"]))}
    templates = load_config(Path(paths["prompts_config"]))["templates"]
    template_types = {name: template["type"] for name, template in templates.items()}
    lang = settings["us_default_lang"]
    georgia = settings["georgia"]
    country_rows = read_csv(Path(paths["countries"]))
    countries = {c["iso2"]: c for c in country_rows}
    models_config = load_config(Path(paths["models_config"]))["models"]
    api_models = [m for m, spec in models_config.items() if spec["backend"] == "openrouter"]
    ngram, threshold = settings["loop_ngram"], settings["loop_threshold"]
    records, texts, details, model_versions = read_generations(
        Path(paths["generations"]), ngram, threshold
    )
    rb_records, _, _, _ = read_generations(Path(paths["robustness_generations"]), ngram, threshold)

    def main_frame(path: Path | str) -> pd.DataFrame:
        scores = _read_scores(Path(path))
        return frames.build_frame(scores, prompts, countries, records, api_models, template_types)

    def robustness_frame(path: Path) -> pd.DataFrame:
        scores = _read_scores(path)
        frame = frames.build_frame(
            scores, prompts, countries, rb_records, api_models, template_types
        )
        return _with_block(frame, config["robustness"]["blocks"])

    variants = {
        "v2": main_frame(paths["scores"]),
        "v1": main_frame(paths["scores_v1"]),
        "prefix": main_frame(paths["scores_prefix"]),
        "prelabels": main_frame(paths["scores_prelabels"]),
        "official": main_frame(variant_scores["official"]),
    }
    robustness = {
        "v2": robustness_frame(variant_scores["robustness_v2"]),
        "v1": robustness_frame(variant_scores["robustness_v1"]),
    }
    us_default_frames = {v: _us_default_frame(variants[v], robustness[v]) for v in ("v2", "v1")}
    logger.info("Frames built: %d main, %d robustness", len(variants["v2"]), len(robustness["v2"]))

    # Holm over the five main comparisons, within each version.
    specs = comparisons.gee_specs(settings["grounding_excluded_models"])
    main = comparisons.holm(
        _main_comparisons(specs, variants["v2"], us_default_frames["v2"], "v2", sizes, lang)
    )
    versus = comparisons.holm(
        _main_comparisons(specs, variants["v1"], us_default_frames["v1"], "v1", sizes, lang)
    )
    logger.info("Main comparisons done")
    extraction, extraction_us_default = extraction_comparison(
        {name: frame[frame["model_id"].isin(prefix_models)] for name, frame in variants.items()},
        replicates,
        seed,
        lang,
    )
    sensitivity = sensitivity_results(
        specs, variants, us_default_frames, sizes, settings["blocked_model"], georgia["iso2"], lang
    )
    exchangeable = exchangeable_results(specs, variants, sizes)
    other = pd.DataFrame([asdict(r) for r in [*exchangeable, *sensitivity, *extraction_us_default]])
    table = pd.concat([main, versus, other], ignore_index=True)
    write_table(table, out / "comparisons.csv")
    write_table(
        pd.concat(
            [comparisons.us_default_breakdown(us_default_frames[v], v, lang) for v in ("v2", "v1")],
            ignore_index=True,
        ),
        out / "us_default_breakdown.csv",
    )
    write_table(extraction, out / "extraction_comparison.csv")
    write_table(
        _grounding_excluded_table(variants, settings["grounding_excluded_models"]),
        out / "grounding_verified_number_excluded.csv",
    )
    logger.info("Sensitivity done")
    boot = None
    if run_bootstrap:
        boot = comparisons.cluster_bootstrap(
            variants["v2"],
            sizes,
            specs,
            GEE_COMPARISONS,
            template_types,
            settings["bootstrap_replicates"],
            seed,
            settings["workers"],
        )
        write_table(boot, out / "bootstrap.csv")
        logger.info("Bootstrap done")
    write_table(icc_table(variants["v2"], list(main.itertuples())), out / "icc.csv")

    # Descriptive, in both versions.
    for name, by in (
        ("outcomes_by_model", ["model_id"]),
        ("outcomes_by_model_condition", ["model_id", "condition"]),
        ("outcomes_cells", ["model_id", "condition", "lang_mode", "cue", "type"]),
    ):
        write_table(
            pd.concat(
                [
                    descriptive.share_table(variants[v], by, replicates, seed).assign(version=v)
                    for v in ("v2", "v1")
                ]
            ),
            out / f"{name}.csv",
        )
    logger.info("Outcome tables done")
    outside = descriptive.outside_scope_map(
        read_csv(Path(paths["valid_outside_scope"])), no_country
    )
    write_table(
        pd.concat(
            [
                descriptive.foreign_numbers(variants[v], settings["top_numbers"]).assign(version=v)
                for v in ("v2", "v1")
            ]
        ),
        out / "foreign_numbers.csv",
    )
    write_table(
        pd.concat(
            [
                descriptive.default_country(variants[v], outside, no_country).assign(version=v)
                for v in ("v2", "v1")
            ]
        ),
        out / "default_country.csv",
    )
    words = settings["asks_location_words"]
    write_table(
        pd.concat(
            [
                descriptive.location_split(variants[v], texts, words, no_country).assign(version=v)
                for v in ("v2", "v1")
            ]
        ),
        out / "asks_location.csv",
    )
    pattern = re.compile(georgia["pattern"], re.IGNORECASE)
    ge_rows, ge_numbers = [], []
    for v in ("v2", "v1"):
        ge_table, numbers = descriptive.georgia_case(
            variants[v], texts, pattern, settings["top_numbers"], georgia["iso2"]
        )
        ge_rows.append(ge_table.assign(version=v))
        ge_numbers.append(numbers.assign(version=v))
    write_table(pd.concat(ge_rows), out / "georgia_case.csv")
    write_table(pd.concat(ge_numbers), out / "georgia_numbers.csv")
    blocked_model = settings["blocked_model"]
    write_table(
        descriptive.blocked_share(variants["v2"], blocked_model), out / "claude_blocked_share.csv"
    )
    claude = variants["v2"][variants["v2"]["model_id"] == blocked_model]
    claude_kept = claude[claude["blocked"] != 1]
    write_table(
        pd.concat(
            [
                descriptive.share_table(
                    part, ["model_id"], replicates, seed, TABLE_OUTCOMES
                ).assign(denominator=label)
                for label, part in (("with blocked", claude), ("without blocked", claude_kept))
            ]
        ),
        out / "claude_with_without_blocked.csv",
    )
    write_table(
        pd.concat(
            [
                descriptive.temperature_comparison(
                    variants[v], robustness[v], ROBUSTNESS_OUTCOMES, replicates, seed
                ).assign(version=v)
                for v in ("v2", "v1")
            ]
        ),
        out / "robustness_temperature.csv",
    )
    aborted = variants["v2"][variants["v2"]["aborted_loop"]]
    write_table(aborted[["model_id", "prompt_id", "sample_idx"]], out / "aborted_loops.csv")
    labelled_prelabels = _read_scores(Path(paths["scores_prelabels"]))
    by_model, cells = fabrication_tables(
        config,
        labelled_prelabels,
        config["unknowns"]["estimate"]["replicates"],
        config["unknowns"]["seed"],
    )
    write_table(by_model, out / "fabrication_by_model.csv")
    write_table(cells, out / "fabrication_cells.csv")
    (out / "notes.md").write_text(results_notes(boot, by_model), encoding="utf-8")
    logger.info("Descriptive tables done")

    # Tables and examples.
    for v, frame in (("v2", variants["v2"]), ("v1", variants["v1"])):
        _figure_tables(frame, v, out, country_rows, sizes, seed, replicates)
        write_table(descriptive.error_types(frame), out / f"error_types_{v}.csv")
    write_table(
        pd.concat(
            [
                # By model and condition, so that the crisis-line class is S0's correct_crisis in
                # outcomes_by_model_condition, interval included.
                descriptive.referral_outcomes(
                    frames.s0_cued(variants[v]), replicates, seed, ["model_id", "condition"]
                )
                .drop(columns="condition")
                .assign(version=v)
                for v in ("v2", "v1")
            ]
        ),
        out / "referral_outcomes.csv",
    )
    write_table(
        pd.concat(
            [
                descriptive.share_table(
                    frames.s0_cued(variants[v]),
                    ["model_id", "iso2"],
                    replicates,
                    seed,
                    ["correct_crisis"],
                ).assign(version=v)
                for v in ("v2", "v1")
            ]
        ),
        out / "crisis_by_model_country.csv",
    )
    write_table(
        pd.concat(
            [
                descriptive.correct_local_by_model_language(variants[v], replicates, seed).assign(
                    version=v
                )
                for v in ("v2", "v1")
            ]
        ),
        out / "correct_local_by_model_language.csv",
    )
    write_table(
        pd.concat(
            [
                descriptive.referral_outcomes(
                    frames.s0_cued(variants[v])[lambda d: d["model_id"].isin(sizes)],
                    replicates,
                    seed,
                ).assign(version=v)
                for v in ("v2", "v1")
            ]
        ),
        out / "qwen_ladder_referral.csv",
    )
    write_table(_model_table(config, variants["v2"], model_versions), out / "models.csv")
    examples_tests = examples.classes(
        {c["iso2"]: c["lang_group"] for c in country_rows},
        config["score"]["na_default_numbers"],
        lambda row: descriptive.asks_location(
            texts.get((row["model_id"], row["prompt_id"], row["sample_idx"]), ""),
            words[row["lang"]],
        ),
    )
    found = examples.candidates(
        variants["v2"],
        texts,
        details,
        examples_tests,
        settings["examples_per_class"],
        settings["snippet_chars"],
        seed,
    )
    with (out / "examples.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in found)
    for name, table in exploratory.tables(config, variants, texts, template_types).items():
        write_table(table, out / "exploratory" / f"{name}.csv")
    logger.info("Exploratory tables done")

    inputs = [
        paths["scores"],
        paths["scores_v1"],
        paths["scores_prefix"],
        paths["scores_prelabels"],
        *variant_scores.values(),
        paths["prompts"],
        paths["resources"],
        paths["resources_v1"],
        paths["valid_outside_scope"],
        *sorted(Path(paths["unknown_labels"]).parent.glob("*.csv")),
        *sorted(Path(paths["prompts_config"]).parent.glob("*.yaml")),
        *(
            path
            for directory in (paths["generations"], paths["robustness_generations"])
            for path in sorted(Path(directory).glob("*"))
            if path.suffix in (".jsonl", ".json")
        ),
    ]
    commit, dirty = git_state(Path.cwd())
    manifest = {
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": seed,
        "bootstrap_replicates": settings["bootstrap_replicates"] if run_bootstrap else 0,
        "packages": {
            p: package_version(p)
            for p in ("statsmodels", "scipy", "pandas", "numpy", "matplotlib", "phonenumbers")
        },
        "inputs_sha256": input_checksums([Path(p) for p in inputs], repo_root(Path.cwd())),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Analysis -> %s", out)
    return 0
