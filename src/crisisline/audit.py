"""Extraction audit: labelling samples and their score.

``balanced_sample`` draws responses evenly over all prompt languages, ``targeted`` picks the
responses that show the two known extraction patterns, and ``changed_sample`` draws from the
responses whose numbers differ between two scorings. ``audit_rows`` turns responses into
rows of the labelling CSV with the numbers the scorer extracts; ``score`` reads the labels back.
A labeller marks a row ``labelled = yes`` and lists, separated by ``;``, the numbers the scorer
missed (``missed``, as written in the response) and the extracted numbers that are not phone
numbers (``false``, copied from ``extracted``).
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from crisisline.extract import YEAR, ExtractConfig, NumberIndex, extract
from crisisline.score import RESULT_CONTENT_FILTER, RESULT_EMPTY, prompt_iso2, prompt_lang

COLUMNS = (
    "response_id",
    "language",
    "model_id",
    "prompt_id",
    "sample_idx",
    "response",
    "extracted",
    "extracted_raw",
    "labelled",
    "missed",
    "false",
    "note",
)
SEPARATOR = ";"
LABELLED = "yes"
# Targeted strata, in the order of the targeted CSV.
STRATA = (
    "false_number",
    "false_number_extracted",
    "short_number_call_verb",
    "short_number_unextracted",
)


def response_id(record: Mapping[str, Any]) -> str:
    return f"{record['model_id']}/{record['prompt_id']}/{record['sample_idx']}"


def auditable(record: Mapping[str, Any]) -> bool:
    """A response with text: no technical error, not blocked, not empty."""
    return (
        not record.get("error")
        and record.get("result") not in (RESULT_CONTENT_FILTER, RESULT_EMPTY)
        and bool((record.get("response") or "").strip())
    )


def balanced_sample(
    records: Iterable[Mapping[str, Any]], per_language: int, seed: int
) -> list[Mapping[str, Any]]:
    """``per_language`` auditable responses for every prompt language, one per (language, model).

    Models follow a seeded cyclic rotation over the languages in a seeded order, so every model
    gets the same number of responses, give or take one; each cell gives one seeded response.
    """
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if auditable(record):
            cells[(prompt_lang(record["prompt_id"]), record["model_id"])].append(record)
    languages = sorted({language for language, _ in cells})
    models = sorted({model for _, model in cells})
    rng = random.Random(seed)
    rng.shuffle(languages)
    rng.shuffle(models)
    chosen = [
        rng.choice(
            sorted(cells[(language, models[(per_language * i + k) % len(models)])], key=response_id)
        )
        for i, language in enumerate(languages)
        for k in range(per_language)
    ]
    return sorted(chosen, key=row_order)


@dataclass(frozen=True, slots=True)
class Targets:
    """The two known extraction patterns."""

    false_number: str  # "365" in "24時間365日"
    false_number_languages: frozenset[str]  # every response containing it; elsewhere if extracted
    short_number_language: str
    short_number: re.Pattern[str]  # a standalone short number
    call_verb: re.Pattern[str]  # a short number followed by a call verb


def stratum(
    record: Mapping[str, Any], targets: Targets, index: NumberIndex, config: ExtractConfig
) -> str | None:
    """The targeted stratum of a response, or None.

    ``false_number``: the number appears in a response in one of its languages;
    ``false_number_extracted``: it is extracted from a response in another language;
    ``short_number_call_verb``: a short number followed by a call verb, in the short-number
    language; ``short_number_unextracted``: there, a short number that is not a year and was
    not extracted.
    """
    prompt_id, text = record["prompt_id"], record["response"]
    language = prompt_lang(prompt_id)

    def extracted() -> set[str]:
        return {n.number for n in extract(text, prompt_iso2(prompt_id), language, index, config)}

    if targets.false_number in unicodedata.normalize("NFKC", text):
        if language in targets.false_number_languages:
            return "false_number"
        if targets.false_number in extracted():
            return "false_number_extracted"
    if language != targets.short_number_language:
        return None
    if targets.call_verb.search(text):
        return "short_number_call_verb"
    found = extracted()
    for match in targets.short_number.finditer(text):
        digits = match.group(1).replace(" ", "")
        if digits not in found and not YEAR.fullmatch(digits):
            return "short_number_unextracted"
    return None


def targeted(
    records: Iterable[Mapping[str, Any]],
    exclude: Collection[str],
    targets: Targets,
    index: NumberIndex,
    config: ExtractConfig,
) -> list[tuple[str, Mapping[str, Any]]]:
    """Auditable responses in a targeted stratum, except ``exclude`` (response ids), with their
    stratum; ordered by stratum, language and response id."""
    rows = [
        (found, record)
        for record in records
        if auditable(record)
        and response_id(record) not in exclude
        and (found := stratum(record, targets, index, config))
    ]
    return sorted(
        rows,
        key=lambda row: (
            STRATA.index(row[0]),
            prompt_lang(row[1]["prompt_id"]),
            response_id(row[1]),
        ),
    )


def row_order(record: Mapping[str, Any]) -> tuple[str, str, str, int]:
    """Order of a labelling CSV: language, model, prompt, sample."""
    prompt_id = record["prompt_id"]
    return (prompt_lang(prompt_id), record["model_id"], prompt_id, record["sample_idx"])


def changed_responses(
    records: Iterable[Mapping[str, Any]],
    before: Mapping[str, Collection[str]],
    after: Mapping[str, Collection[str]],
    exclude: Collection[str],
) -> list[Mapping[str, Any]]:
    """Auditable responses, except ``exclude``, whose numbers differ between two scorings
    (response id to numbers), ordered by response id."""
    return sorted(
        (
            record
            for record in records
            if auditable(record)
            and (rid := response_id(record)) not in exclude
            and set(before[rid]) != set(after[rid])
        ),
        key=response_id,
    )


def changed_sample(
    population: Sequence[Mapping[str, Any]], size: int, seed: int
) -> list[Mapping[str, Any]]:
    """Up to ``size`` random responses of ``population``, in the order of a labelling CSV."""
    chosen = random.Random(seed).sample(list(population), min(size, len(population)))
    return sorted(chosen, key=row_order)


def audit_rows(
    records: Iterable[Mapping[str, Any]], index: NumberIndex, config: ExtractConfig
) -> list[dict[str, str]]:
    """Labelling rows: each response with the numbers the scorer extracts, label columns empty."""
    rows = []
    for record in records:
        prompt_id = record["prompt_id"]
        numbers = extract(
            record["response"], prompt_iso2(prompt_id), prompt_lang(prompt_id), index, config
        )
        rows.append(
            {
                "response_id": response_id(record),
                "language": prompt_lang(prompt_id),
                "model_id": record["model_id"],
                "prompt_id": prompt_id,
                "sample_idx": str(record["sample_idx"]),
                "response": record["response"],
                "extracted": f"{SEPARATOR} ".join(n.number for n in numbers),
                "extracted_raw": f"{SEPARATOR} ".join(n.raw for n in numbers),
                "labelled": "",
                "missed": "",
                "false": "",
                "note": "",
            }
        )
    return rows


def split(cell: str) -> list[str]:
    return [item.strip() for item in cell.split(SEPARATOR) if item.strip()]


@dataclass
class Counts:
    responses: int = 0
    extracted: int = 0
    false: int = 0
    missed: int = 0

    @property
    def correct(self) -> int:
        return self.extracted - self.false

    @property
    def precision(self) -> float | None:
        return self.correct / self.extracted if self.extracted else None

    @property
    def recall(self) -> float | None:
        found = self.correct + self.missed
        return self.correct / found if found else None


@dataclass(frozen=True, slots=True)
class Finding:
    response_id: str
    language: str
    kind: str  # "missed" or "false"
    number: str
    note: str


@dataclass
class AuditReport:
    overall: Counts = field(default_factory=Counts)
    by_language: dict[str, Counts] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    unlabelled: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def score(rows: Iterable[Mapping[str, str]]) -> AuditReport:
    """Precision and recall of extraction over the labelled rows, overall and per language.

    Rows not marked ``labelled = yes`` are skipped and listed. A ``false`` entry that is not
    among the row's ``extracted`` numbers is a problem: the label cannot be matched.
    """
    report = AuditReport()
    for row in rows:
        rid, language = row["response_id"], row["language"]
        if row["labelled"].strip().lower() != LABELLED:
            report.unlabelled.append(rid)
            continue
        extracted, false, missed = (
            split(row["extracted"]),
            split(row["false"]),
            split(row["missed"]),
        )
        if unknown := [n for n in false if n not in extracted]:
            report.problems.append(f"{rid}: false {unknown} not among extracted {extracted}")
            continue
        counts = report.by_language.setdefault(language, Counts())
        for target in (counts, report.overall):
            target.responses += 1
            target.extracted += len(extracted)
            target.false += len(false)
            target.missed += len(missed)
        note = row["note"].strip()
        report.findings += [Finding(rid, language, "missed", n, note) for n in missed]
        report.findings += [Finding(rid, language, "false", n, note) for n in false]
    report.by_language = dict(sorted(report.by_language.items()))
    return report


def by_group(
    by_language: Mapping[str, Counts], groups: Mapping[str, Sequence[str]]
) -> dict[str, Counts]:
    """Counts summed over the languages of each group, in the order of ``groups``."""
    totals = {}
    for group, languages in groups.items():
        total = Counts()
        for counts in (by_language[lang] for lang in languages if lang in by_language):
            total.responses += counts.responses
            total.extracted += counts.extracted
            total.false += counts.false
            total.missed += counts.missed
        totals[group] = total
    return totals


def pattern_counts(report: AuditReport) -> dict[str, Counter[str]]:
    """Per language, the responses with each finding: ``false 365``, ``missed 116 117``."""
    counts: dict[str, Counter[str]] = {language: Counter() for language in report.by_language}
    for finding in report.findings:
        counts[finding.language][f"{finding.kind} {finding.number}"] += 1
    return counts
