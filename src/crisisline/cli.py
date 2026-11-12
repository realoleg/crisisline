"""Command-line entry point: ``crisisline <command>``, one subcommand per pipeline stage."""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime
import fcntl
import hashlib
import http.client
import json
import logging
import os
import re
import subprocess
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from dotenv import dotenv_values

from crisisline import audit, colab, generate, golden, groundtruth, llm, prompts, unknowns
from crisisline.extract import ExtractConfig, build_index, fold, normalize
from crisisline.groundtruth import Issue, Severity
from crisisline.score import ScoreConfig, UnknownNumber, collect_unknowns, score_response

logger = logging.getLogger(__name__)

# Relative to the repository root, like every path inside the config.
DEFAULT_CONFIG = Path("configs/experiment.yaml")
COLAB_CONFIG = Path("configs/colab.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all pipeline subcommands."""
    parser = argparse.ArgumentParser(
        prog="crisisline",
        description="Crisis line benchmark pipeline. One command per stage.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('crisisline')}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--config",
        type=Path,
        help=f"Experiment config (default: {DEFAULT_CONFIG} in the repository).",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    validate = sub.add_parser("validate", help="Validate ground-truth CSVs in data/.")
    validate.add_argument(
        "--check-urls",
        action="store_true",
        help="Also fetch every source_url (network access; not for CI).",
    )

    translate = sub.add_parser(
        "translate", help="Translate and back-translate prompts (OpenRouter API)."
    )
    target = translate.add_mutually_exclusive_group(required=True)
    target.add_argument("--lang", metavar="L", help="Target language code.")
    target.add_argument("--all", action="store_true", help="Every non-English language.")
    target.add_argument(
        "--recheck",
        action="store_true",
        help="Recompute chrF and QA flags of every language file; no API calls.",
    )
    translate.add_argument(
        "--force",
        action="store_true",
        help="Also re-translate qa = auto entries (manual and native are never touched).",
    )

    sub.add_parser("build", help="Build the full prompt matrix.")

    generate = sub.add_parser("generate", help="Generate responses for one model (resumable).")
    generate.add_argument("--model", required=True, metavar="M", help="model_id from models.yaml.")
    generate.add_argument(
        "--robustness", action="store_true", help="Run the model's robustness blocks."
    )
    generate.add_argument(
        "--limit", type=int, metavar="N", help="Only the first N prompts (smoke test)."
    )
    generate.add_argument(
        "--retry-errors",
        action="store_true",
        help="Regenerate the records that are technical errors and replace them in place.",
    )

    colab_cmd = sub.add_parser(
        "colab",
        help="Run generate for a local model on a Colab runtime (colab CLI); resumable.",
    )
    colab_cmd.add_argument(
        "--model",
        required=True,
        nargs="+",
        metavar="M",
        help="Local model_ids, run one after another.",
    )
    colab_cmd.add_argument(
        "--robustness", action="store_true", help="Run the model's robustness blocks."
    )
    colab_cmd.add_argument(
        "--limit", type=int, metavar="N", help="Only the first N prompts (smoke test)."
    )
    colab_cmd.add_argument("--retry-errors", action="store_true", help="Passed on to generate.")

    sub.add_parser("score", help="Extract numbers and score responses.")

    audit = sub.add_parser("audit", help="Extraction audit.")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True, metavar="<step>")
    audit_sample = audit_sub.add_parser(
        "sample",
        help="Draw a sample over all languages and the targeted responses into CSVs for labelling.",
    )
    audit_score = audit_sub.add_parser("score", help="Precision and recall from the labelled CSVs.")
    for audit_cmd in (audit_sample, audit_score):
        audit_cmd.add_argument(
            "--changed",
            action="store_true",
            help="The re-audit: responses whose numbers differ between scores_prefix and scores.",
        )
    unknowns_cmd = sub.add_parser(
        "unknowns", help="Write the blind labelling files of unknown numbers."
    )
    unknowns_step = unknowns_cmd.add_mutually_exclusive_group()
    unknowns_step.add_argument(
        "--flagged",
        action="store_true",
        help="Only the sample of pairs that libphonenumber rules out.",
    )
    unknowns_step.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate the share of fabricated mentions from the labelled files.",
    )
    analyze = sub.add_parser(
        "analyze",
        help="Score the main run against ground truth v1, the robustness block and the "
        "official-sources variant, then write every analysis table and the example candidates.",
    )
    analyze.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip the cluster bootstrap of the four GEE comparisons.",
    )
    sub.add_parser("paper", help="Figures, tables and numbers of the paper from the analysis.")
    sub.add_parser(
        "reproduce",
        help="Every artifact and number from the generations: validate, build, score, audit "
        "score, unknowns --estimate, analyze, paper; then check the outputs against checks/golden.",
    )
    return parser


class RelativePaths(logging.Filter):
    """Writes every path under ``root`` in a log message relative to it."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.prefix = f"{root}{os.sep}"

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = record.getMessage().replace(self.prefix, ""), None
        return True


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_root(start: Path) -> Path:
    """The nearest directory at or above ``start`` that holds ``pyproject.toml``."""
    for directory in (start, *start.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    raise FileNotFoundError(f"not inside the repository: no pyproject.toml above {start}")


def resolve_paths(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """``config`` with every entry of ``paths`` resolved against the repository root."""
    return {**config, "paths": {name: root / path for name, path in config["paths"].items()}}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, restval=""))


def ascii_url(url: str) -> str:
    """IDNA-encode the host and percent-encode the path and query; ``urllib`` sends ASCII only."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc.encode("idna").decode("ascii"),
            urllib.parse.quote(parts.path, safe="/%"),
            urllib.parse.quote(parts.query, safe="=&%"),
            parts.fragment,
        )
    )


def url_problem(url: str, timeout_s: float) -> str | None:
    """Fetch ``url``; return why it failed, or ``None`` if it answered with a non-error status."""
    try:
        request = urllib.request.Request(
            ascii_url(url), headers={"User-Agent": f"crisisline/{version('crisisline')}"}
        )
        with urllib.request.urlopen(request, timeout=timeout_s):
            return None
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}"
    # URLError and TimeoutError are OSErrors; reading the answer can also raise a bare
    # ConnectionResetError or an HTTPException (a server that closes without answering).
    except (OSError, http.client.HTTPException, ValueError) as exc:
        return str(getattr(exc, "reason", exc))


def check_urls(resources: Sequence[dict[str, str]], timeout_s: float, workers: int) -> list[Issue]:
    """Warn on every distinct ``source_url`` that does not respond."""
    urls = sorted({r["source_url"] for r in resources if r["source_url"]})
    logger.info("Checking %d URLs", len(urls))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        problems = pool.map(lambda url: url_problem(url, timeout_s), urls)
    return [
        Issue(Severity.WARNING, "url", groundtruth.RESOURCES_FILE, None, f"{url}: {problem}")
        for url, problem in zip(urls, problems, strict=True)
        if problem
    ]


def run_validate(config: dict[str, Any], check_urls_flag: bool) -> int:
    """Validate the ground truth and log every issue. Exit code 1 if there are errors."""
    paths = {name: Path(p) for name, p in config["paths"].items()}
    resources = read_csv(paths["resources"])
    issues = groundtruth.validate(
        read_csv(paths["countries"]),
        resources,
        read_csv(paths["shared_numbers"]),
        read_csv(paths["unknown_labels"]),
    )
    if check_urls_flag:
        settings = config["validate"]
        issues += check_urls(resources, settings["url_timeout_s"], settings["url_workers"])

    errors = [i for i in issues if i.severity is Severity.ERROR]
    warnings = [i for i in issues if i.severity is Severity.WARNING]
    for issue in errors:
        logger.error("%s", issue)
    for issue in warnings:
        logger.warning("%s", issue)
    logger.info("%d errors, %d warnings", len(errors), len(warnings))
    return 1 if errors else 0


_NUMBER = pa.struct(
    [
        ("number", pa.string()),
        ("raw", pa.string()),
        ("start", pa.int32()),
        ("number_class", pa.string()),
        ("type", pa.string()),
        ("other_iso2", pa.list_(pa.string())),
        ("other_status", pa.list_(pa.string())),
        ("also_valid_in", pa.list_(pa.string())),
    ]
)
SCORES_SCHEMA = pa.schema(
    [
        ("prompt_id", pa.string()),
        ("model_id", pa.string()),
        ("sample_idx", pa.int32()),
        ("iso2", pa.string()),
        ("error", pa.bool_()),
        ("numbers", pa.list_(_NUMBER)),
        ("correct_local", pa.bool_()),
        ("correct_crisis", pa.bool_()),
        ("wrong_country", pa.bool_()),
        ("na_default", pa.bool_()),
        ("fabricated", pa.bool_()),
        ("mentions_directory", pa.bool_()),
        ("no_resource", pa.bool_()),
        ("blocked", pa.bool_()),
        ("empty", pa.bool_()),
        ("truncated", pa.bool_()),
        ("default_countries", pa.list_(pa.string())),
    ]
)


def extract_config(config: dict[str, Any]) -> ExtractConfig:
    settings = config["extract"]

    def words(*groups: str) -> tuple[str, ...]:
        """Every word of the per-language lists in ``groups``, NFKC-normalised and folded."""
        return tuple(
            sorted(
                {
                    fold(unicodedata.normalize("NFKC", word))
                    for group in groups
                    for language_words in settings[group].values()
                    for word in language_words
                }
            )
        )

    return ExtractConfig(
        min_digits=settings["min_digits"],
        max_digits=settings["max_digits"],
        short_max_digits=settings["short_max_digits"],
        context_chars=settings["context_chars"],
        context_words=words("context_words", "service_words"),
        service_words=words("service_words"),
        service_after_chars=settings["service_after_chars"],
        item_articles=tuple(
            fold(unicodedata.normalize("NFKC", a)) for a in settings["item_articles"]
        ),
        verb_final_languages=frozenset(settings["verb_final_languages"]),
        unit_words=words("unit_words"),
        abbreviations=frozenset(words("abbreviations")),
        unit_numbers=frozenset(settings["unit_numbers"]),
        whole_min_digits=settings["whole_min_digits"],
        whole_min_trailing=settings["whole_min_trailing"],
    )


def audit_targets(config: dict[str, Any]) -> audit.Targets:
    settings = config["audit"]["main"]["targeted"]
    short_number = settings["short_number"]
    return audit.Targets(
        false_number=settings["false_number"],
        false_number_languages=frozenset(settings["false_number_languages"]),
        short_number_language=settings["short_number_language"],
        short_number=re.compile(short_number),
        call_verb=re.compile(short_number + settings["call_verb"]),
    )


def score_config(config: dict[str, Any]) -> ScoreConfig:
    settings = config["score"]
    return ScoreConfig(
        no_country_iso2=config["no_country_iso2"],
        na_default_numbers=frozenset(settings["na_default_numbers"]),
        truncated_finish_reasons=frozenset(settings["truncated_finish_reasons"]),
        directories=tuple(settings["directories"]),
        snippet_chars=settings["snippet_chars"],
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_records(directory: Path) -> list[dict[str, Any]]:
    """Every record of the generation files in ``directory``, file by file in name order."""
    return [record for path in sorted(directory.glob("*.jsonl")) for record in read_jsonl(path)]


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_score(config: dict[str, Any]) -> int:
    """Score every generation record; write ``scores.parquet`` and ``unknowns.csv``.

    Exit code 1, with nothing written, if a ``(prompt_id, model_id, sample_idx)`` repeats.
    """
    paths = config["paths"]
    records: list[dict[str, Any]] = []
    first_seen: dict[tuple[str, str, int], str] = {}
    duplicates = 0
    for path in sorted(Path(paths["generations"]).glob("*.jsonl")):
        for n, record in enumerate(read_jsonl(path), start=1):
            key = (record["prompt_id"], record["model_id"], record["sample_idx"])
            where = f"{path.name} record {n}"
            if key in first_seen:
                duplicates += 1
                logger.error(
                    "Duplicate (prompt_id, model_id, sample_idx) %s at %s, first at %s",
                    key,
                    where,
                    first_seen[key],
                )
            else:
                first_seen[key] = where
            records.append(record)
    if duplicates:
        logger.error("%d duplicate generation records; nothing written", duplicates)
        return 1

    index = build_index(read_csv(Path(paths["resources"])))
    labels = {
        (r["iso2"], r["number_norm"]): r["label"] for r in read_csv(Path(paths["unknown_labels"]))
    }
    extract_cfg, score_cfg = extract_config(config), score_config(config)
    scored = [
        (
            record.get("response") or "",
            score_response(record, index, extract_cfg, score_cfg, labels),
        )
        for record in records
    ]
    errors = sum(score.error for _, score in scored)

    scores_path = Path(paths["scores"])
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dataclasses.asdict(score) for _, score in scored]
    pq.write_table(pa.Table.from_pylist(rows, schema=SCORES_SCHEMA), scores_path)
    unknowns = collect_unknowns(scored, score_cfg.snippet_chars)
    unknowns_path = Path(paths["unknowns"])
    write_csv(
        unknowns_path,
        [dataclasses.asdict(u) for u in unknowns],
        [f.name for f in dataclasses.fields(UnknownNumber)],
    )
    logger.info("Scored %d records (%d errors) -> %s", len(rows), errors, scores_path)
    logger.info("%d unlabelled unknown numbers -> %s", len(unknowns), unknowns_path)
    return 0


def read_yaml_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def write_yaml_list(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=1_000_000)


def run_translate(
    config: dict[str, Any], lang: str | None, force: bool, client: llm.ChatClient
) -> int:
    """Translate the missing units of ``lang`` (every non-English language if ``None``).

    Writes ``data/translations/{lang}.yaml`` after each language, with flags recomputed over the
    whole file, and logs every flagged ``qa = auto`` entry. Exit code 1 if a language is unknown,
    or if an API call failed or the translator's answer was malformed (the other entries are
    still written; a rerun fills the gaps).
    """
    paths = config["paths"]
    prompts_cfg = load_config(Path(paths["prompts_config"]))
    models_cfg = load_config(Path(paths["models_config"]))
    settings = models_cfg["translation"]
    translator = llm.ModelSpec.from_config(settings["translator"])
    back_translator = llm.ModelSpec.from_config(settings["back_translator"])
    qa_cfg = config["translation"]
    units = prompts.translation_units(read_csv(Path(paths["countries"])), prompts_cfg)
    langs = list(dict.fromkeys(u.lang for u in units))
    if lang is not None and lang not in langs:
        logger.error("No non-English country speaks '%s'; choose from %s", lang, ", ".join(langs))
        return 1
    today = datetime.datetime.now(datetime.UTC).date().isoformat()

    failed = 0
    for current in [lang] if lang else langs:
        path = Path(paths["translations"]) / f"{current}.yaml"
        lang_units = [u for u in units if u.lang == current]
        existing = read_yaml_list(path)
        todo = prompts.pending(lang_units, {prompts.entry_key(e): e for e in existing}, force)
        logger.info("%s: translating %d of %d strings", current, len(todo), len(lang_units))

        def work(unit: prompts.Unit) -> prompts.Translation | None:
            try:
                return prompts.translate(
                    unit,
                    client,
                    translator,
                    back_translator,
                    settings["params"],
                    prompts_cfg,
                    today,
                )
            except (llm.TransientError, prompts.MalformedTranslation) as exc:
                logger.error("%s %s: %s", unit.lang, unit.key, exc)
                return None

        try:
            with ThreadPoolExecutor(max_workers=qa_cfg["workers"]) as pool:
                results = list(pool.map(work, todo))
        except llm.RequestError as exc:
            logger.error("%s; stopping, nothing written for %s", exc, current)
            return 1
        new = [t for t in results if t is not None]
        failed += len(results) - len(new)
        if new:
            write_yaml_list(path, prompts.merge(lang_units, existing, new, qa_cfg))
        report_flags(current, read_yaml_list(path), path)
    if failed:
        logger.error("%d strings failed technically; rerun to fill them", failed)
    return 1 if failed else 0


def run_recheck(config: dict[str, Any]) -> int:
    """Recompute chrF and QA flags of every ``data/translations/{lang}.yaml`` (no API calls)."""
    paths = config["paths"]
    units = prompts.translation_units(
        read_csv(Path(paths["countries"])), load_config(Path(paths["prompts_config"]))
    )
    for path in sorted(Path(paths["translations"]).glob("*.yaml")):
        lang = path.stem
        entries = prompts.recheck(
            [u for u in units if u.lang == lang], read_yaml_list(path), config["translation"]
        )
        write_yaml_list(path, entries)
        report_flags(lang, entries, path)
    return 0


def report_flags(lang: str, entries: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Log every flagged ``qa = auto`` entry of a language file."""
    flagged = [e for e in entries if e["qa"] == "auto" and e["flags"]]
    for entry in flagged:
        logger.warning(
            "%s %s flagged %s: %s",
            lang,
            prompts.entry_key(entry),
            ",".join(entry["flags"]),
            entry["translation"],
        )
    logger.info("%s: %d entries flagged for review -> %s", lang, len(flagged), path)
    declined = sum(prompts.failed(e) for e in flagged)
    if declined:
        logger.warning(
            "%s: %d translations declined by a content filter or empty; build refuses them "
            "until they are fixed by hand (qa: manual)",
            lang,
            declined,
        )
    added = sum(prompts.ADDED_CONTENT in e["flags"] for e in entries)
    if added:
        logger.warning(
            "%s: %d translations with added content; build refuses them until the addition is "
            "removed by hand (qa: manual) and the file is rechecked",
            lang,
            added,
        )


def run_build(config: dict[str, Any]) -> int:
    """Write ``prompts.jsonl`` and ``m2_resources.csv``.

    Exit code 1, with nothing written, if translations are missing or stale.
    """
    paths = config["paths"]
    translations = {
        prompts.entry_key(entry): entry
        for path in sorted(Path(paths["translations"]).glob("*.yaml"))
        for entry in read_yaml_list(path)
    }
    countries = read_csv(Path(paths["countries"]))
    prompts_cfg = load_config(Path(paths["prompts_config"]))
    # Ground truth v1: the M2 prompts are the ones the models saw, whatever v2 adds.
    m2 = prompts.m2_table(countries, read_csv(Path(paths["resources_v1"])), prompts_cfg)
    try:
        matrix = prompts.build_prompts(
            countries, m2, prompts_cfg, translations, config["no_country_iso2"]
        )
    except prompts.BuildError as exc:
        for problem in exc.problems:
            logger.error("%s", problem)
        logger.error("%s; nothing written. Run crisisline translate first.", exc)
        return 1
    out = Path(paths["prompts"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in matrix)
    logger.info("Built %d prompts -> %s", len(matrix), out)
    m2_path = Path(paths["m2_resources"])
    write_csv(m2_path, prompts.m2_rows(m2.values(), prompts_cfg), prompts.M2_COLUMNS)
    logger.info("M2 resources of %d countries -> %s", len(m2), m2_path)
    return 0


def openrouter_client(config: dict[str, Any], root: Path) -> llm.OpenRouterClient:
    """Client keyed from the environment, or from ``.env`` at the repository root."""
    settings = load_config(Path(config["paths"]["models_config"]))["openrouter"]
    env = {**dotenv_values(root / ".env"), **os.environ}
    key = llm.api_key({k: v for k, v in env.items() if v is not None}, settings["api_key_env"])
    return llm.OpenRouterClient(key, settings)


GENERATION_DIRS = {"main": "generations", "robustness": "robustness_generations"}


UNKNOWN_COMMIT = "unknown"


def git_commit(root: Path) -> str:
    """HEAD of the repository, with ``-dirty`` if tracked files have changes;
    :data:`UNKNOWN_COMMIT` outside a git checkout or without git."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN_COMMIT
    return commit + ("-dirty" if dirty else "")


def gpu_info() -> dict[str, str | None]:
    """GPU name, driver and CUDA version from ``nvidia-smi``; all ``None`` without a GPU."""
    query = ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    try:
        name, driver = (
            subprocess.run(query, capture_output=True, text=True, check=True)
            .stdout.splitlines()[0]
            .split(", ")
        )
        header = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return {"name": None, "driver_version": None, "cuda_version": None}
    cuda = header.split("CUDA Version:")[1].split()[0] if "CUDA Version:" in header else None
    return {"name": name, "driver_version": driver, "cuda_version": cuda}


@dataclasses.dataclass(frozen=True)
class GenerationSetup:
    """A ready backend and what defines its environment."""

    backend: generate.Backend
    samples: int
    workers: int
    environment: dict[str, Any]  # must not change within one model's output
    session: dict[str, Any]  # recorded per session; a change is only logged
    max_errors: int  # technical errors in a row that stop the run


def setup_generation(
    config: dict[str, Any], root: Path, model_id: str, mode: str
) -> GenerationSetup:
    """Connect the model's backend and probe its environment.

    Raises ``ValueError`` for an unknown model and
    :class:`~crisisline.llm.CompletionError` if Ollama cannot serve it.
    """
    models_cfg = load_config(Path(config["paths"]["models_config"]))
    if model_id not in models_cfg["models"]:
        raise ValueError(f"unknown model '{model_id}'; see configs/models.yaml")
    model = models_cfg["models"][model_id]
    settings = models_cfg["generation"][model["backend"]]
    robustness = config["robustness"]
    samples = robustness["samples"] if mode == "robustness" else settings["samples"]
    max_errors = models_cfg["generation"]["max_consecutive_errors"]
    environment: dict[str, Any] = {
        "model_id": model_id,
        "backend": model["backend"],
        "git_commit": git_commit(root),
    }

    if model["backend"] == generate.BACKEND_OPENROUTER:
        spec = llm.ModelSpec.from_config({**model, **settings})
        client = openrouter_client(config, root)
        backend: generate.Backend = generate.OpenRouterBackend(client, spec, settings["max_tokens"])
        environment["model"] = spec.model
        return GenerationSetup(backend, samples, settings["workers"], environment, {}, max_errors)

    tag = model["tag"]
    ollama = generate.OllamaClient(models_cfg["ollama"])
    ollama_version = ollama.version()
    try:
        show = ollama.show(tag)
    except llm.RequestError as exc:
        raise llm.RequestError(f"{exc}; is the model pulled? ollama pull {tag}") from exc
    capabilities = sorted(show.get("capabilities", []))
    options = {**settings["params"], "num_predict": settings["max_tokens"]}
    if mode == "robustness":
        options["temperature"] = robustness["temperature"]
    digest = ollama.digest(tag)
    backend = generate.OllamaBackend(
        client=ollama,
        tag=tag,
        digest=digest or "",
        options=options,
        think=False if generate.OLLAMA_THINKING in capabilities else None,
    )
    gpu = gpu_info()
    environment |= {
        "ollama_version": ollama_version,
        "tag": tag,
        "digest": digest,
        "quantization": show.get("details", {}).get("quantization_level"),
        "capabilities": capabilities,
        "gpu_name": gpu["name"],
    }
    session = {"driver_version": gpu["driver_version"], "cuda_version": gpu["cuda_version"]}
    return GenerationSetup(backend, samples, settings["workers"], environment, session, max_errors)


def repair_tail(path: Path) -> None:
    """Drop an incomplete last line, left when a run was killed mid-write. It is not a record."""
    data = path.read_bytes() if path.exists() else b""
    if not data or data.endswith(b"\n"):
        return
    keep = data.rfind(b"\n") + 1
    logger.warning(
        "%s: dropping an incomplete last line (%d bytes) left by an interrupted run",
        path,
        len(data) - keep,
    )
    with path.open("r+b") as f:
        f.truncate(keep)


def done_keys(path: Path, model_id: str) -> set[tuple[str, int]]:
    """``(prompt_id, sample_idx)`` already in a generation file. Raises ``ValueError`` on a record
    of another model or a repeated pair."""
    done: set[tuple[str, int]] = set()
    for n, record in enumerate(read_jsonl(path) if path.exists() else [], start=1):
        key = (record["prompt_id"], record["sample_idx"])
        if record["model_id"] != model_id:
            raise ValueError(f"{path} record {n} is for model {record['model_id']}")
        if key in done:
            raise ValueError(f"{path} record {n} repeats {key}")
        done.add(key)
    return done


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_manifest(path: Path, environment: dict[str, Any], session: dict[str, Any]) -> list[str]:
    """Check the run's environment against the manifest and record the session.

    Returns the environment fields that differ from the manifest (nothing is written then).
    """
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if manifest is None:
        manifest = {"environment": environment, "sessions": []}
    elif manifest["environment"] != environment:
        fields = sorted(set(environment) | set(manifest["environment"]))
        return [
            f"{f}: {manifest['environment'].get(f)!r} in the manifest, now {environment.get(f)!r}"
            for f in fields
            if manifest["environment"].get(f) != environment.get(f)
        ]
    for previous in manifest["sessions"][:1]:
        for field, value in session.items():
            if field in previous and previous[field] != value:
                logger.warning("%s changed: %r -> %r", field, previous[field], value)
    manifest["sessions"].append(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return []


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds")


@contextlib.contextmanager
def output_lock(out: Path) -> Iterator[bool]:
    """Hold an exclusive lock on ``{out}.lock`` for the block; ``False`` if another process
    holds it. Two processes on one output would append the same records twice, and a retry
    renaming its file over the output would leave the other appending to a deleted file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_name(out.name + ".lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def run_generate(
    config: dict[str, Any],
    model_id: str,
    mode: str,
    limit: int | None,
    setup: GenerationSetup,
    now: Callable[[], str] = utc_now,
    retry_errors: bool = False,
) -> int:
    """Generate the model's missing records and append each one as soon as it exists.

    Resumable: ``(prompt_id, sample_idx)`` pairs already in the output are skipped. With
    ``retry_errors``, regenerate only the records that are technical errors instead, replacing
    them in place (:func:`retry_error_records`). Exit code 1 if another process is writing the
    same output, the output has records but no manifest, the environment differs from the
    model's manifest, a request fails every time, or ``max_errors`` records in a row are
    technical errors.
    """
    out = Path(config["paths"][GENERATION_DIRS[mode]]) / f"{model_id}.jsonl"
    with output_lock(out) as locked:
        if not locked:
            logger.error("%s is being written by another process; nothing generated", out)
            return 1
        return _generate(config, model_id, mode, limit, setup, now, retry_errors)


def _generate(
    config: dict[str, Any],
    model_id: str,
    mode: str,
    limit: int | None,
    setup: GenerationSetup,
    now: Callable[[], str],
    retry_errors: bool,
) -> int:
    paths = config["paths"]
    if retry_errors and limit is not None:
        logger.error("--retry-errors retries every error record; it takes no --limit")
        return 1
    prompts_path = Path(paths["prompts"])
    if not prompts_path.exists():
        logger.error("%s does not exist; run crisisline build first", prompts_path)
        return 1
    backend = setup.backend
    out_dir = Path(paths[GENERATION_DIRS[mode]])
    out = out_dir / f"{model_id}.jsonl"
    environment = setup.environment | {
        "mode": mode,
        "prompts_sha256": file_sha256(prompts_path),
        "params": backend.params(0),
        "samples": setup.samples,
    }
    try:
        selected = generate.select_prompts(
            read_jsonl(prompts_path), model_id, backend.name, mode, config
        )[:limit]
        repair_tail(out)
        done = done_keys(out, model_id)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    manifest = out_dir / f"{model_id}.manifest.json"
    if done and not manifest.exists():
        logger.error(
            "%s has %d records but no manifest %s: their environment cannot be checked; "
            "nothing generated",
            out,
            len(done),
            manifest,
        )
        return 1
    session = {"started_utc": now(), "records_before": len(done), **setup.session}
    if retry_errors:
        session["retry_errors"] = True
    if mismatches := update_manifest(manifest, environment, session):
        for mismatch in mismatches:
            logger.error("Environment differs from the manifest: %s", mismatch)
        logger.error("One model runs in one environment; nothing generated")
        return 1
    log_every = load_config(Path(paths["models_config"]))["generation"]["log_every"]
    if retry_errors:
        by_id = {p["prompt_id"]: p for p in selected}
        return retry_error_records(out, by_id, model_id, setup, now, log_every)

    tasks = generate.pending_tasks(selected, setup.samples, done)
    logger.info(
        "%s (%s): %d prompts x %d samples, %d done, %d to generate -> %s",
        model_id,
        mode,
        len(selected),
        setup.samples,
        len(done),
        len(tasks),
        out,
    )
    written = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:

        def write(record: dict[str, Any]) -> None:
            nonlocal written
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            written += 1
            if written % log_every == 0:
                logger.info("%d / %d records", written, len(tasks))

        try:
            counts = generate.run(
                tasks, model_id, backend, setup.workers, write, now, setup.max_errors
            )
        except llm.RequestError as exc:
            logger.error("%s; stopped after %d records (rerun resumes)", exc, written)
            return 1
    logger.info("%d records written: %s", written, dict(sorted(counts.items())))
    if counts[llm.Result.TRANSIENT.value]:
        logger.warning(
            "%d records are technical errors (result transient)", counts[llm.Result.TRANSIENT.value]
        )
    return 0


def run_colab_models(
    config: dict[str, Any],
    root: Path,
    model_ids: Sequence[str],
    mode: str,
    limit: int | None,
    retry_errors: bool,
) -> int:
    """Generate local models on Colab one after another, unattended.

    Every model is checked before the first starts. A model whose outputs are already complete
    is skipped, so rerunning the same command resumes the list; the first model that fails
    stops it. The runtime gets HEAD as a git bundle, so the working tree must be clean.
    """
    paths = config["paths"]
    models_cfg = load_config(Path(paths["models_config"]))
    for model_id in model_ids:
        model = models_cfg["models"].get(model_id)
        if model is None:
            logger.error("unknown model '%s'; see configs/models.yaml", model_id)
            return 1
        if model["backend"] != generate.BACKEND_OLLAMA:
            logger.error("%s is an API model: run crisisline generate locally", model_id)
            return 1
    commit = git_commit(root)
    if commit == UNKNOWN_COMMIT:
        logger.error("The runtime gets the current commit as a git bundle: run from a git checkout")
        return 1
    if commit.endswith("-dirty"):
        logger.error("Tracked files have changes; the runtime gets the commit: commit them first")
        return 1
    prompts_path = Path(paths["prompts"])
    if not prompts_path.exists():
        logger.error("%s does not exist; run crisisline build first", prompts_path)
        return 1
    all_prompts = read_jsonl(prompts_path)
    samples = (
        config["robustness"]["samples"]
        if mode == "robustness"
        else models_cfg["generation"][generate.BACKEND_OLLAMA]["samples"]
    )
    for n, model_id in enumerate(model_ids, start=1):
        out = Path(paths[GENERATION_DIRS[mode]]) / f"{model_id}.jsonl"
        selected = generate.select_prompts(
            all_prompts, model_id, generate.BACKEND_OLLAMA, mode, config
        )[:limit]
        pending = generate.pending_tasks(selected, samples, done_keys(out, model_id))
        if not pending and not retry_errors:
            logger.info("[%d/%d] %s (%s) is complete; skipped", n, len(model_ids), model_id, mode)
            continue
        logger.info(
            "[%d/%d] %s (%s): %d to generate", n, len(model_ids), model_id, mode, len(pending)
        )
        generate_args = [
            "--model",
            model_id,
            *([f"--{mode}"] if mode != "main" else []),
            *(["--limit", str(limit)] if limit is not None else []),
            *(["--retry-errors"] if retry_errors else []),
        ]
        with output_lock(out) as locked:
            if not locked:
                logger.error("%s is being driven by another process; nothing started", out)
                return 1
            code = run_colab(config, root, model_id, mode, generate_args, commit)
        if code != 0:
            logger.error("Stopped the list at %s; rerun the same command to resume", model_id)
            return code
    return 0


def run_colab(
    config: dict[str, Any],
    root: Path,
    model_id: str,
    mode: str,
    generate_args: Sequence[str],
    commit: str,
) -> int:
    """Generate one local model's outputs on a Colab runtime and keep them verified locally.

    Resuming installs the Ollama version of the model's manifest.
    """
    paths = config["paths"]
    model = load_config(Path(paths["models_config"]))["models"][model_id]
    cfg = load_config(root / COLAB_CONFIG)
    out_dir = Path(paths[GENERATION_DIRS[mode]])
    manifest = out_dir / f"{model_id}.manifest.json"
    ollama_version = (
        json.loads(manifest.read_text(encoding="utf-8"))["environment"]["ollama_version"]
        if manifest.exists()
        else ""
    )
    job = colab.Job(
        session=f"{cfg['session_prefix']}-{model_id}-{mode}",
        commit=commit,
        tag=model["tag"],
        ollama_version=ollama_version,
        generate_args=tuple(generate_args),
        outputs=tuple(
            str((out_dir / name).relative_to(root)) for name in (f"{model_id}.jsonl", manifest.name)
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "crisisline.bundle"
        subprocess.run(
            ["git", "bundle", "create", str(bundle), "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
        )
        try:
            return colab.run(
                colab.Colab(job.session, cfg["exec_timeout_s"]), job, cfg, root, bundle
            )
        except colab.ColabError as exc:
            logger.error("%s", exc)
            logger.error("Runtime %s left as is; rerun the same command to resume", job.session)
            return 1


def rewrite_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Replace ``path`` atomically: write a temporary file next to it, then rename it over."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def retry_error_records(
    out: Path,
    prompts_by_id: dict[str, dict[str, Any]],
    model_id: str,
    setup: GenerationSetup,
    now: Callable[[], str],
    checkpoint_every: int,
) -> int:
    """Regenerate the records with ``error`` set and put each new record in the old one's place.

    Only technical errors are retried, never ``content_filter`` or ``empty`` answers. The file
    is rewritten atomically every ``checkpoint_every`` new records and at the end, so it never
    holds a duplicate ``(prompt_id, sample_idx)`` and an interrupted retry leaves a valid file
    (the state of the last checkpoint).

    The least recently tried go first (``ts_utc``): a retry stopped by errors in a row puts the
    records that failed again at the back, so records that fail every time cannot keep every
    later retry from the others.
    """
    records = read_jsonl(out) if out.exists() else []
    errors = sorted(
        (i for i, r in enumerate(records) if r.get("error")), key=lambda i: records[i]["ts_utc"]
    )
    unknown = sorted({records[i]["prompt_id"] for i in errors} - prompts_by_id.keys())
    if unknown:
        logger.error("Error records for prompts not in this run: %s", ", ".join(unknown))
        return 1
    logger.info(
        "%s: retrying %d error records of %d in %s", model_id, len(errors), len(records), out
    )
    tasks = [(prompts_by_id[records[i]["prompt_id"]], records[i]["sample_idx"]) for i in errors]
    replaced = 0

    def write(record: dict[str, Any]) -> None:
        nonlocal replaced
        records[errors[replaced]] = record  # run() writes in task order
        replaced += 1
        if replaced % checkpoint_every == 0:
            rewrite_jsonl(out, records)
            logger.info("%d / %d records retried", replaced, len(tasks))

    try:
        counts = generate.run(
            tasks, model_id, setup.backend, setup.workers, write, now, setup.max_errors
        )
    except llm.RequestError as exc:
        rewrite_jsonl(out, records)
        logger.error("%s; stopped after %d retried records", exc, replaced)
        return 1
    if replaced:
        rewrite_jsonl(out, records)
    logger.info("%d records retried: %s", replaced, dict(sorted(counts.items())))
    return 0


def run_audit_sample(config: dict[str, Any]) -> int:
    """Write the two labelling CSVs of the main-run extraction audit: a balanced random sample
    and the targeted responses outside it. Never overwrites labels."""
    paths, settings = config["paths"], config["audit"]["main"]
    outs = Path(paths["main_audit"]), Path(paths["main_audit_targeted"])
    if existing := [str(out) for out in outs if out.exists()]:
        logger.error("%s exist and may hold labels; move them away to draw a new sample", existing)
        return 1
    records = read_records(Path(paths["generations"]))
    index = build_index(read_csv(Path(paths["resources"])))
    extract_cfg = extract_config(config)
    chosen = audit.balanced_sample(records, settings["per_language"], settings["seed"])
    targeted = audit.targeted(
        records, {audit.response_id(r) for r in chosen}, audit_targets(config), index, extract_cfg
    )
    write_csv(outs[0], audit.audit_rows(chosen, index, extract_cfg), audit.COLUMNS)
    write_csv(
        outs[1], audit.audit_rows([r for _, r in targeted], index, extract_cfg), audit.COLUMNS
    )
    logger.info("%d responses of %d records -> %s", len(chosen), len(records), outs[0])
    logger.info(
        "%d targeted responses %s -> %s",
        len(targeted),
        dict(Counter(stratum for stratum, _ in targeted)),
        outs[1],
    )
    return 0


def scored_numbers(path: Path) -> dict[str, frozenset[str]]:
    """Response id to the numbers of a scores file."""
    columns = ["prompt_id", "model_id", "sample_idx", "numbers"]
    return {
        audit.response_id(row): frozenset(n["number"] for n in row["numbers"] or ())
        for row in pq.read_table(path, columns=columns).to_pylist()
    }


def run_audit_sample_changed(config: dict[str, Any]) -> int:
    """Write the labelling CSV of responses whose numbers changed between the pre-fix scores and
    the current ones, outside the rows already labelled. Never overwrites labels."""
    paths, settings = config["paths"], config["audit"]["changed"]
    out = Path(paths["main_audit_changed"])
    if out.exists():
        logger.error("%s exists and may hold labels; move it away to draw a new sample", out)
        return 1
    records = read_records(Path(paths["generations"]))
    labelled = {
        row["response_id"]
        for name in ("main_audit", "main_audit_targeted")
        for row in read_csv(Path(paths[name]))
    }
    population = audit.changed_responses(
        records,
        scored_numbers(Path(paths["scores_prefix"])),
        scored_numbers(Path(paths["scores"])),
        labelled,
    )
    chosen = audit.changed_sample(population, settings["size"], settings["seed"])
    if len(chosen) < settings["size"]:
        logger.warning("Only %d responses for %d", len(chosen), settings["size"])
    index = build_index(read_csv(Path(paths["resources"])))
    write_csv(out, audit.audit_rows(chosen, index, extract_config(config)), audit.COLUMNS)
    logger.info("%d of %d changed responses -> %s", len(chosen), len(population), out)
    return 0


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _log_labels(report: audit.AuditReport, findings: bool = True) -> None:
    for problem in report.problems:
        logger.error("%s", problem)
    if report.unlabelled:
        logger.warning(
            "%d rows not labelled yet: %s", len(report.unlabelled), ", ".join(report.unlabelled)
        )
    for finding in report.findings if findings else ():
        logger.info(
            "%s %s [%s] %s%s",
            finding.kind,
            finding.number,
            finding.language,
            finding.response_id,
            f" ({finding.note})" if finding.note else "",
        )


def _log_rates(rows: Iterable[tuple[str, audit.Counts]]) -> None:
    for name, counts in rows:
        logger.info(
            "%-8s precision %s, recall %s (%d extracted, %d false, %d missed)",
            name,
            _rate(counts.precision),
            _rate(counts.recall),
            counts.extracted,
            counts.false,
            counts.missed,
        )


def _criterion(report: audit.AuditReport, threshold: float) -> int:
    """Check that overall precision and recall both reach ``threshold``. Exit code 1 if either
    does not, if a label cannot be matched, or if nothing is labelled."""
    if report.problems or not report.by_language:
        return 1
    rates = (report.overall.precision, report.overall.recall)
    if not all(rate is not None and rate >= threshold for rate in rates):
        logger.error(
            "Extraction criterion not met: precision %s and recall %s must both reach %.2f",
            _rate(rates[0]),
            _rate(rates[1]),
            threshold,
        )
        return 1
    logger.info("Criterion precision and recall >= %.2f: met", threshold)
    return 0


def _score_sample(path: Path, settings: Mapping[str, Any]) -> int:
    """Log precision and recall of a labelled sample overall, per script group and per language,
    with every miss and false extraction. Exit code 1 as :func:`_criterion`."""
    report = audit.score(read_csv(path))
    _log_labels(report)
    _log_rates(report.by_language.items())
    groups = audit.by_group(report.by_language, settings["groups"])
    _log_rates([*groups.items(), ("all", report.overall)])
    return _criterion(report, settings["threshold"])


def run_audit_score(config: dict[str, Any], changed: bool) -> int:
    """Score the labelled random sample, or with ``changed`` the re-audit (:func:`_score_sample`):
    exit code 1 if precision or recall misses the threshold. For the random sample, then log how
    many targeted responses show each labelled pattern per language; exit code 1 also if a
    targeted label cannot be matched."""
    paths, settings = config["paths"], config["audit"]["main"]
    if changed:
        return _score_sample(Path(paths["main_audit_changed"]), settings)
    code = _score_sample(Path(paths["main_audit"]), settings)

    targeted = audit.score(read_csv(Path(paths["main_audit_targeted"])))
    _log_labels(targeted, findings=False)
    for language, patterns in audit.pattern_counts(targeted).items():
        logger.info(
            "Targeted %-4s %d responses: %s",
            language,
            targeted.by_language[language].responses,
            ", ".join(f"{p} {n}" for p, n in sorted(patterns.items())) or "none",
        )
    return code or (1 if targeted.problems else 0)


@dataclasses.dataclass(frozen=True, slots=True)
class UnknownsInput:
    """Unknown number pairs with what the labelling files need of them."""

    pairs: Sequence[unknowns.Pair]
    texts: Mapping[str, str]
    groups: Mapping[str, str]
    checks: Mapping[str, str]
    flagged: frozenset[str]
    seed: int
    model_names: re.Pattern[str]


def read_unknowns_input(config: dict[str, Any]) -> UnknownsInput:
    paths, settings = config["paths"], config["unknowns"]
    columns = ["prompt_id", "model_id", "sample_idx", "iso2", "numbers"]
    pairs = unknowns.collect_pairs(pq.read_table(paths["scores"], columns=columns).to_pylist())
    no_country = config["no_country_iso2"]
    checks = {p.pair_id: unknowns.check_number(p.number, p.iso2, no_country) for p in pairs}
    return UnknownsInput(
        pairs=pairs,
        texts={
            audit.response_id(record): normalize(record["response"])
            for record in read_records(Path(paths["generations"]))
            if record.get("response")
        },
        groups=unknowns.group_pairs(pairs, no_country),
        checks=checks,
        flagged=frozenset(p for p, check in checks.items() if check in unknowns.IMPOSSIBLE),
        seed=settings["seed"],
        model_names=re.compile(settings["model_names"]),
    )


def write_labelling(
    out: Path,
    chosen: Sequence[unknowns.Pair],
    size: int,
    fields: Sequence[str],
    data: UnknownsInput,
) -> None:
    """A labelling file of ``chosen`` with ``size`` contexts per pair. A group with one member in
    the file is not shown."""
    members = Counter(data.groups.get(p.pair_id) for p in chosen)
    rows = []
    for pair in chosen:
        group = data.groups.get(pair.pair_id, "")
        contexts = unknowns.blind_contexts(pair, data.texts, size, data.seed, data.model_names)
        rows.append(
            unknowns.labelling_row(pair, group if members[group] > 1 else "", contexts, size)
        )
    write_csv(out, rows, fields)


def run_unknowns_flagged(config: dict[str, Any]) -> int:
    """Write the sample of flagged pairs, allocated over its bands by mentions, and nothing else.
    No model appears in it. Never overwrites labels."""
    settings = config["unknowns"]["flagged_sample"]
    out = Path(config["paths"][settings["path"]])
    if out.exists():
        logger.error("%s exists and may hold labels; move it away to draw a new sample", out)
        return 1
    data = read_unknowns_input(config)
    unflagged = {p.pair_id for p in data.pairs} - data.flagged
    strata = [
        [
            p
            for p in data.pairs
            if b["low"] <= p.responses <= b["high"] and p.pair_id in data.flagged
        ]
        for b in settings["bands"]
    ]
    mentions = [sum(p.responses for p in stratum) for stratum in strata]
    sizes = unknowns.allocate(settings["size"], mentions)
    chosen = [
        pair
        for b, size in zip(settings["bands"], sizes, strict=True)
        for pair in unknowns.draw_sample(
            data.pairs, unflagged, b["low"], b["high"], size, data.seed
        )
    ]
    write_labelling(out, sorted(chosen, key=unknowns.pair_order), 1, unknowns.SAMPLE_COLUMNS, data)
    for b, stratum, weight, size in zip(settings["bands"], strata, mentions, sizes, strict=True):
        logger.info(
            "Responses %d-%d: %d of %d flagged pairs, %d mentions",
            b["low"],
            b["high"],
            size,
            len(stratum),
            weight,
        )
    logger.info("%d pairs -> %s", len(chosen), out)
    return 0


def run_analyze(config: dict[str, Any], run_bootstrap: bool) -> int:
    """Score the main run against ground truth v1 (:func:`score_variants`), the robustness block
    and the official-sources-only variant, then write every output of the analysis to
    ``paths.analysis``."""
    # Imported here: pandas and statsmodels are slow to load, and only this command needs them.
    from crisisline.analysis import pipeline

    for variant in score_variants(config).values():
        if code := run_score(variant):
            return code
    paths = config["paths"]
    out = Path(paths["analysis"]) / "scores"
    out.mkdir(parents=True, exist_ok=True)
    marker = config["analysis"]["secondary_source_marker"]
    resources = read_csv(Path(paths["resources"]))
    official = out / "resources_official.csv"
    write_csv(official, [r for r in resources if marker not in r["note"]], list(resources[0]))
    variants = {
        "robustness_v2": (paths["robustness_generations"], paths["resources"]),
        "robustness_v1": (paths["robustness_generations"], paths["resources_v1"]),
        "official": (paths["generations"], official),
    }
    scores = {}
    for name, (generations, ground_truth) in variants.items():
        scores[name] = out / f"{name}.parquet"
        overrides = {
            "generations": generations,
            "resources": ground_truth,
            "scores": scores[name],
            "unknowns": out / f"unknowns_{name}.csv",
        }
        code = run_score({**config, "paths": {**paths, **overrides}})
        if code:
            return code
    return pipeline.analyze(config, scores, run_bootstrap)


def run_paper(config: dict[str, Any]) -> int:
    """Figures, tables and numbers of the paper from ``paths.analysis`` to ``paths.paper``."""
    # Imported here, like the analysis: pandas and matplotlib are slow to load.
    from crisisline.paper import build

    return build.build(config)


def score_variants(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``config`` for the other scorings of the main run: ground truth v1, and v1 without labels."""
    paths = config["paths"]
    overrides = {
        "v1": {
            "resources": paths["resources_v1"],
            "scores": paths["scores_v1"],
            "unknowns": paths["unknowns_v1"],
        },
        "prelabels": {
            "resources": paths["resources_v1"],
            "unknown_labels": Path(os.devnull),  # no labels
            "scores": paths["scores_prelabels"],
            "unknowns": paths["unknowns_prelabels"],
        },
    }
    return {name: {**config, "paths": {**paths, **o}} for name, o in overrides.items()}


def check_golden(manifest: Path, root: Path) -> int:
    """Check every output listed in ``manifest`` (paths relative to ``root``) as its ``check``
    says: ``hash``, its sha256; ``numeric``, :func:`golden.differences` from the reference copy at
    the same path under the manifest's directory; ``exists``, only that it is there. Exit code 1,
    with every missing or different file logged, if one fails."""
    entries = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    problems = []
    for entry in entries:
        name = entry["path"]
        file = root / name
        if not file.exists():
            problems.append(f"{name}: missing")
            continue
        match entry["check"]:
            case "exists":
                found = []
            case "hash":
                found = [] if file_sha256(file) == entry["sha256"] else ["sha256 differs"]
            case "numeric":
                found = golden.differences(
                    file.suffix,
                    (manifest.parent / name).read_text(encoding="utf-8"),
                    file.read_text(encoding="utf-8"),
                    entry.get("unchecked_lines"),
                )
            case other:
                raise ValueError(f"{name}: unknown check {other!r} in {manifest}")
        if found:
            more = f" (and {len(found) - 1} more)" if len(found) > 1 else ""
            problems.append(f"{name}: {found[0]}{more}")
    for problem in problems:
        logger.error("%s", problem)
    checks = Counter(entry["check"] for entry in entries)
    logger.info(
        "%d of %d outputs pass %s (%s)",
        len(entries) - len(problems),
        len(entries),
        manifest,
        ", ".join(f"{n} {check}" for check, n in sorted(checks.items())),
    )
    return 1 if problems else 0


def run_reproduce(config: dict[str, Any], root: Path) -> int:
    """Every offline stage in order, then :func:`check_golden`; stops at the first that fails.
    The extraction audit fails if precision or recall misses its threshold (:func:`_criterion`)."""
    steps: list[Callable[[], int]] = [
        lambda: run_validate(config, False),
        lambda: run_build(config),
        lambda: run_score(config),
        lambda: run_audit_score(config, False),
        lambda: run_audit_score(config, True),
        lambda: run_unknowns_estimate(config),
        lambda: run_analyze(config, True),
        lambda: run_paper(config),
        lambda: check_golden(Path(config["paths"]["golden"]), root),
    ]
    for step in steps:
        if code := step():
            return code
    return 0


def run_unknowns_estimate(config: dict[str, Any]) -> int:
    """Write the stratified estimate of fabricated mentions."""
    paths, settings = config["paths"], config["unknowns"]
    flags = read_csv(Path(paths["unknowns_flags"]))
    strata = [
        unknowns.stratum(
            s["name"], flags, read_csv(Path(paths[s["path"]])), s["bands"], s.get("impossible")
        )
        for s in settings["estimate"]["strata"]
    ]
    estimate = unknowns.fabrication_estimate(
        strata,
        settings["estimate"]["replicates"],
        settings["seed"],
        settings["estimate"]["confidence"],
    )
    out = Path(paths["fabrication_estimate"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(estimate, indent=2) + "\n", encoding="utf-8")
    low, high = estimate["interval"]
    logger.info(
        "Fabricated: %.4f of mentions, CI [%.4f, %.4f] -> %s", estimate["share"], low, high, out
    )
    return 0


def run_unknowns(config: dict[str, Any]) -> int:
    """Write the blind labelling files of unknown numbers and the libphonenumber check of every
    pair. No model appears in them. Never overwrites labels."""
    paths, settings = config["paths"], config["unknowns"]
    samples = settings["samples"]
    outs = [Path(paths["unknowns_head"]), Path(paths["unknowns_flags"])]
    outs += [Path(paths[sample["path"]]) for sample in samples]
    if existing := [str(out) for out in outs if out.exists()]:
        logger.error("%s exist and may hold labels; move them away to prepare new files", existing)
        return 1

    data = read_unknowns_input(config)
    pairs, checks, flagged = data.pairs, data.checks, data.flagged
    seed = data.seed
    bands = settings["bands"]

    head = settings["head"]
    chosen_head = [p for p in pairs if p.responses >= head["min_responses"]]
    write_labelling(outs[0], chosen_head, head["contexts"], unknowns.HEAD_COLUMNS, data)
    logger.info(
        "%d pairs, %d mentions -> %s",
        len(chosen_head),
        sum(p.responses for p in chosen_head),
        outs[0],
    )
    write_csv(
        outs[1],
        [unknowns.flag_row(p, bands, checks[p.pair_id]) for p in pairs],
        unknowns.FLAG_COLUMNS,
    )
    for sample, out in zip(samples, outs[2:], strict=True):
        chosen = unknowns.draw_sample(
            pairs, flagged, sample["low"], sample["high"], sample["size"], seed
        )
        if len(chosen) < sample["size"]:
            logger.warning("Only %d pairs for %d -> %s", len(chosen), sample["size"], out)
        write_labelling(out, chosen, 1, unknowns.SAMPLE_COLUMNS, data)
        logger.info("%d pairs -> %s", len(chosen), out)
    for b in bands:
        members = [p for p in pairs if unknowns.band_of(p.responses, bands) == b["name"]]
        hit = [p for p in members if p.pair_id in flagged]
        logger.info(
            "Band %-5s %5d pairs %6d mentions; flagged %5d pairs %6d mentions",
            b["name"],
            len(members),
            sum(p.responses for p in members),
            len(hit),
            sum(p.responses for p in hit),
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, configure logging and dispatch to the requested command."""
    args = build_parser().parse_args(argv)
    root = repo_root(Path.cwd())
    handler = logging.StreamHandler()
    handler.addFilter(RelativePaths(root))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[handler],
    )
    # Font subsetting while matplotlib writes a PDF logs every table it prunes.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    config = resolve_paths(load_config(args.config or root / DEFAULT_CONFIG), root)
    mode = "robustness" if getattr(args, "robustness", False) else "main"
    commands: dict[str, Callable[[], int]] = {
        "validate": lambda: run_validate(config, args.check_urls),
        "translate": lambda: (
            run_recheck(config) if args.recheck else _translate(config, root, args)
        ),
        "build": lambda: run_build(config),
        "generate": lambda: _generate_command(config, root, args, mode),
        "colab": lambda: run_colab_models(
            config, root, args.model, mode, args.limit, args.retry_errors
        ),
        "score": lambda: run_score(config),
        "audit": lambda: _audit(config, args),
        "unknowns": lambda: _unknowns(config, args),
        "analyze": lambda: run_analyze(config, not args.no_bootstrap),
        "paper": lambda: run_paper(config),
        "reproduce": lambda: run_reproduce(config, root),
    }
    return commands[args.command]()


def _translate(config: dict[str, Any], root: Path, args: argparse.Namespace) -> int:
    try:
        client = openrouter_client(config, root)
    except llm.MissingKeyError as exc:
        logger.error("%s", exc)
        return 1
    return run_translate(config, args.lang, args.force, client)


def _generate_command(
    config: dict[str, Any], root: Path, args: argparse.Namespace, mode: str
) -> int:
    try:
        setup = setup_generation(config, root, args.model, mode)
    except (llm.MissingKeyError, llm.CompletionError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return run_generate(config, args.model, mode, args.limit, setup, retry_errors=args.retry_errors)


def _audit(config: dict[str, Any], args: argparse.Namespace) -> int:
    if args.audit_command == "score":
        return run_audit_score(config, args.changed)
    return run_audit_sample_changed(config) if args.changed else run_audit_sample(config)


def _unknowns(config: dict[str, Any], args: argparse.Namespace) -> int:
    if args.estimate:
        return run_unknowns_estimate(config)
    return run_unknowns_flagged(config) if args.flagged else run_unknowns(config)
