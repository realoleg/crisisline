"""``crisisline analyze``: from scores to everything in ``artifacts/analysis``."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
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
    descriptive,
    examples,
    exploratory,
    fabrication,
    figures,
    frames,
    hypotheses,
)
from crisisline.cli import file_sha256, git_commit, load_config, read_csv, read_jsonl, repo_root

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
FIGURE_SETTINGS = (
    ("S0, city cue", "S0", "city"),
    ("S0, country cue", "S0", "country"),
    ("M1", "M1", "none"),
    ("M2", "M2", "none"),
)
GEE_HYPOTHESES = (
    "cue_country_vs_city",
    "language_en_vs_local",
    "model_size",
    "grounding_verified_number",
)


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
    """HEAD and whether tracked files have changes; no commit outside a git checkout."""
    try:
        commit = git_commit(root)
    except subprocess.CalledProcessError:
        return "", False
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


def _write(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n", float_format="%.6g")


def _confirmatory(
    specs: Mapping[str, hypotheses.GeeSpec],
    frame: pd.DataFrame,
    us_default_frame: pd.DataFrame,
    analysis: str,
    sizes: Mapping[str, float],
) -> list[hypotheses.TestResult]:
    results = [hypotheses.run_gee(specs[h], frame, analysis, sizes) for h in GEE_HYPOTHESES]
    results.insert(3, hypotheses.run_us_default(us_default_frame, analysis))
    return results


def _gee_rows(
    specs: Mapping[str, hypotheses.GeeSpec],
    frame: pd.DataFrame,
    ids: Sequence[str],
    analysis: str,
    sizes: Mapping[str, float],
) -> list[hypotheses.TestResult]:
    return [hypotheses.run_gee(specs[h], frame, analysis, sizes) for h in ids]


def exchangeable_results(
    specs: Mapping[str, hypotheses.GeeSpec],
    variants: Mapping[str, pd.DataFrame],
    sizes: Mapping[str, float],
) -> list[hypotheses.TestResult]:
    """The exchangeable working correlation next to the primary analyses: a model that does not
    converge reports no estimate. ``grounding_verified_number`` also with every model."""
    results = []
    for version in ("v2", "v1"):
        frame = variants[version]
        for h in ("cue_country_vs_city", "language_en_vs_local", "grounding_verified_number"):
            spec = replace(specs[h], exchangeable=True)
            results.append(hypotheses.run_gee(spec, frame, f"{version}, Exchangeable", sizes))
        as_written = replace(
            specs["grounding_verified_number"], exchangeable=True, excluded_models=()
        )
        results.append(
            hypotheses.run_gee(as_written, frame, f"{version}, Exchangeable, all models", sizes)
        )
    return results


def sensitivity_results(
    specs: Mapping[str, hypotheses.GeeSpec],
    variants: Mapping[str, pd.DataFrame],
    us_default_frames: Mapping[str, pd.DataFrame],
    sizes: Mapping[str, float],
    blocked_model: str,
) -> list[hypotheses.TestResult]:
    """Pre-specified sensitivity analyses, unadjusted."""
    v2, us_default_frame = variants["v2"], us_default_frames["v2"]
    kept = v2[v2["truncated"] != 1]
    results = _gee_rows(specs, kept, GEE_HYPOTHESES, "without truncated", sizes)
    results.append(
        hypotheses.run_us_default(
            us_default_frame[us_default_frame["truncated"] != 1], "without truncated"
        )
    )
    results += _gee_rows(
        specs, v2[v2["has_crisis_line"]], GEE_HYPOTHESES, "only countries with a crisis line", sizes
    )
    results += _gee_rows(
        specs, variants["official"], GEE_HYPOTHESES, "official sources only", sizes
    )
    results += _gee_rows(
        specs,
        v2[v2["iso2"] != "GE"],
        ("cue_country_vs_city", "language_en_vs_local"),
        "without GE",
        sizes,
    )
    claude = (v2["model_id"] == blocked_model) & (v2["blocked"] == 1)
    results += _gee_rows(
        specs,
        v2[~claude],
        ("cue_country_vs_city", "language_en_vs_local", "grounding_verified_number"),
        f"{blocked_model} without blocked",
        sizes,
    )
    results.append(hypotheses.run_us_default(us_default_frame, "v2", "default_countries"))
    results.append(hypotheses.run_us_default(us_default_frames["v1"], "v1", "default_countries"))
    results += hypotheses.simple_effects_cue(specs["cue_country_vs_city"], v2, "v2", sizes)
    results += hypotheses.simple_effects_cue(
        specs["cue_country_vs_city"], variants["v1"], "v1", sizes
    )
    return results


def extraction_comparison(
    variants: Mapping[str, pd.DataFrame], replicates: int, seed: int
) -> tuple[pd.DataFrame, list[hypotheses.TestResult]]:
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
        us_default_results.append(hypotheses.run_us_default(frame, f"main block, {name}"))
    return pd.concat(tables_by_version), us_default_results


def icc_table(frame: pd.DataFrame, results: Sequence[hypotheses.TestResult]) -> pd.DataFrame:
    """The intraclass correlations."""
    rows = [
        {
            "measure": f"residual correlation by country, {r.hypothesis}",
            "value": r.residual_correlation,
        }
        for r in results
        if r.hypothesis != "us_default"
    ]
    cued = frames.s0_cued(frame)
    rows.append(
        {
            "measure": "ANOVA ICC of correct_local by country, S0 with cue",
            "value": hypotheses.anova_icc(cued["correct_local"], cued["iso2"]),
        }
    )
    api = frame[frame["is_api"] & frame["correct_local"].notna()]
    rows.extend(
        {
            "measure": f"ANOVA ICC of correct_local within a prompt, {model}",
            "value": hypotheses.anova_icc(part["correct_local"], part["prompt_id"]),
        }
        for model, part in api.groupby("model_id")
    )
    rows.append(
        {
            "measure": "ANOVA ICC of correct_local within a prompt, all API models",
            "value": hypotheses.anova_icc(
                api["correct_local"], api["model_id"] + "/" + api["prompt_id"]
            ),
        }
    )
    return pd.DataFrame(rows)


def fabrication_tables(
    config: Mapping[str, Any], prelabels: Sequence[Mapping[str, Any]], replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The estimator per model, and over all models as a check on ``unknowns --estimate``."""
    paths, settings = config["paths"], config["unknowns"]
    mentions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pair in unknowns_module.collect_pairs(prelabels):
        for mention in pair.mentions:
            mentions[pair.pair_id][mention.response_id.split("/", 1)[0]] += 1
    flags = read_csv(Path(paths["unknowns_flags"]))
    specs = settings["estimate"]["strata"]
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
                f"- Cluster bootstrap of {row.hypothesis}: {row.failed} of {row.replicates} "
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


def _with_block(frame: pd.DataFrame, no_country: str) -> pd.DataFrame:
    return frame.assign(
        block=frame["iso2"].map(lambda iso2: "none_en" if iso2 == no_country else "cue_20")
    )


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


def _draw_figures(
    frame: pd.DataFrame,
    version: str,
    out: Path,
    countries: Sequence[Mapping[str, str]],
    outside: Mapping[str, str],
    settings: Mapping[str, Any],
    seed: int,
    replicates: int,
    no_country: str,
) -> None:
    """The figures; the v1 set has a suffix."""
    suffix = "" if version == "v2" else f"_{version}"
    figure_dir = out / "figures"
    sizes = settings["model_sizes"]
    cells = descriptive.share_table(
        frame, ["model_id", "condition", "cue"], replicates, seed, ["correct_local"]
    )
    figures.correct_by_model_and_cue(
        figures.setting_rows(cells, FIGURE_SETTINGS),
        figure_dir / f"correct_by_model_and_cue{suffix}.pdf",
    )
    by_country = _country_table(frame, countries)
    _write(by_country, out / f"correct_by_country{suffix}.csv")
    figures.correct_by_country(by_country, figure_dir / f"correct_by_country{suffix}.pdf")
    figures.error_types_by_model(
        descriptive.error_types(frame), figure_dir / f"error_types_by_model{suffix}.pdf"
    )
    ladder = _ladder_table(frame, sizes, replicates, seed)
    _write(ladder, out / f"qwen_ladder{suffix}.csv")
    figures.qwen_ladder(ladder, sizes, figure_dir / f"qwen_ladder{suffix}.pdf")
    defaults = descriptive.default_country(frame, outside, no_country)
    figures.default_country_heatmap(
        defaults,
        settings["figure_countries"],
        figure_dir / f"default_country_by_language{suffix}.pdf",
    )
    figures.default_country_heatmap(
        defaults,
        settings["figure_countries"],
        figure_dir / f"default_country_single_by_language{suffix}.pdf",
        "responses_only_this_country",
    )


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

    main_scores = [Path(paths[name]) for name in ("scores", "scores_v1", "scores_prelabels")]
    stale = out_of_date(
        {str(path): responses(path) for path in main_scores},
        responses(Path(variant_scores["official"])),
    )
    if stale:
        logger.error("scores are out of date; run `crisisline reproduce` (%s)", ", ".join(stale))
        return 1

    prompts = {p["prompt_id"]: p for p in read_jsonl(Path(paths["prompts"]))}
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
        return frames.build_frame(scores, prompts, countries, records, api_models)

    def robustness_frame(path: Path) -> pd.DataFrame:
        scores = _read_scores(path)
        frame = frames.build_frame(scores, prompts, countries, rb_records, api_models)
        return _with_block(frame, no_country)

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

    # Confirmatory: Holm over the five tests, within each version.
    specs = hypotheses.gee_specs(settings["grounding_excluded_models"])
    primary = hypotheses.holm(
        _confirmatory(specs, variants["v2"], us_default_frames["v2"], "v2", sizes)
    )
    versus = hypotheses.holm(
        _confirmatory(specs, variants["v1"], us_default_frames["v1"], "v1", sizes)
    )
    logger.info("Confirmatory done")
    extraction, extraction_us_default = extraction_comparison(variants, replicates, seed)
    sensitivity = sensitivity_results(
        specs, variants, us_default_frames, sizes, settings["blocked_model"]
    )
    exchangeable = exchangeable_results(specs, variants, sizes)
    other = pd.DataFrame([asdict(r) for r in [*exchangeable, *sensitivity, *extraction_us_default]])
    table = pd.concat([primary, versus, other], ignore_index=True)
    _write(table, out / "hypotheses.csv")
    _write(
        pd.concat(
            [hypotheses.us_default_breakdown(us_default_frames[v], v) for v in ("v2", "v1")],
            ignore_index=True,
        ),
        out / "us_default_breakdown.csv",
    )
    _write(extraction, out / "extraction_comparison.csv")
    _write(
        _grounding_excluded_table(variants, settings["grounding_excluded_models"]),
        out / "grounding_verified_number_excluded.csv",
    )
    logger.info("Sensitivity done")
    boot = None
    if run_bootstrap:
        boot = hypotheses.cluster_bootstrap(
            variants["v2"],
            sizes,
            specs,
            GEE_HYPOTHESES,
            settings["bootstrap_replicates"],
            seed,
            settings["workers"],
        )
        _write(boot, out / "bootstrap.csv")
        logger.info("Bootstrap done")
    _write(icc_table(variants["v2"], list(primary.itertuples())), out / "icc.csv")

    # Descriptive, in both versions.
    for name, by in (
        ("outcomes_by_model", ["model_id"]),
        ("outcomes_by_model_condition", ["model_id", "condition"]),
        ("outcomes_cells", ["model_id", "condition", "lang_mode", "cue", "type"]),
    ):
        _write(
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
    _write(
        pd.concat(
            [
                descriptive.foreign_numbers(variants[v], settings["top_numbers"]).assign(version=v)
                for v in ("v2", "v1")
            ]
        ),
        out / "foreign_numbers.csv",
    )
    _write(
        pd.concat(
            [
                descriptive.default_country(variants[v], outside, no_country).assign(version=v)
                for v in ("v2", "v1")
            ]
        ),
        out / "default_country.csv",
    )
    words = settings["asks_location_words"]
    _write(
        pd.concat(
            [
                descriptive.location_split(variants[v], texts, words, no_country).assign(version=v)
                for v in ("v2", "v1")
            ]
        ),
        out / "asks_location.csv",
    )
    pattern = re.compile(settings["georgia_pattern"], re.IGNORECASE)
    ge_rows, ge_numbers = [], []
    for v in ("v2", "v1"):
        ge_table, numbers = descriptive.georgia_case(
            variants[v], texts, pattern, settings["top_numbers"]
        )
        ge_rows.append(ge_table.assign(version=v))
        ge_numbers.append(numbers.assign(version=v))
    _write(pd.concat(ge_rows), out / "georgia_case.csv")
    _write(pd.concat(ge_numbers), out / "georgia_numbers.csv")
    blocked_model = settings["blocked_model"]
    _write(
        descriptive.blocked_share(variants["v2"], blocked_model), out / "claude_blocked_share.csv"
    )
    claude = variants["v2"][variants["v2"]["model_id"] == blocked_model]
    claude_kept = claude[claude["blocked"] != 1]
    _write(
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
    _write(
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
    _write(aborted[["model_id", "prompt_id", "sample_idx"]], out / "aborted_loops.csv")
    labelled_prelabels = _read_scores(Path(paths["scores_prelabels"]))
    by_model, cells = fabrication_tables(
        config,
        labelled_prelabels,
        config["unknowns"]["estimate"]["replicates"],
        config["unknowns"]["seed"],
    )
    _write(by_model, out / "fabrication_by_model.csv")
    _write(cells, out / "fabrication_cells.csv")
    (out / "notes.md").write_text(results_notes(boot, by_model), encoding="utf-8")
    logger.info("Descriptive tables done")

    # Figures, tables, examples.
    for v, frame in (("v2", variants["v2"]), ("v1", variants["v1"])):
        _draw_figures(frame, v, out, country_rows, outside, settings, seed, replicates, no_country)
        _write(descriptive.error_types(frame), out / f"error_types_{v}.csv")
    _write(
        pd.concat(
            [
                descriptive.referral_outcomes(variants[v], replicates, seed).assign(version=v)
                for v in ("v2", "v1")
            ]
        ),
        out / "referral_outcomes.csv",
    )
    _write(
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
    _write(_model_table(config, variants["v2"], model_versions), out / "models.csv")
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
    for name, table in exploratory.tables(config, variants, texts).items():
        _write(table, out / "exploratory" / f"{name}.csv")
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
        paths["unknown_labels"],
        paths["valid_outside_scope"],
    ]
    commit, dirty = git_state(Path.cwd())
    manifest = {
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": seed,
        "bootstrap_replicates": settings["bootstrap_replicates"] if run_bootstrap else 0,
        "packages": {
            p: package_version(p) for p in ("statsmodels", "scipy", "pandas", "numpy", "matplotlib")
        },
        "inputs_sha256": input_checksums([Path(p) for p in inputs], repo_root(Path.cwd())),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Analysis -> %s", out)
    return 0
