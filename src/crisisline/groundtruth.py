"""Ground-truth validation.

Every check is a pure function over CSV rows already parsed into ``dict[str, str]``; reading the
files happens in ``cli.py``. A check returns a list of :class:`Issue`. Errors fail validation;
warnings are reported for manual review only.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import phonenumbers

Row = Mapping[str, str]

COUNTRIES_FILE = "countries.csv"
RESOURCES_FILE = "resources.csv"
SHARED_FILE = "shared_numbers.csv"
LABELS_FILE = "unknown_labels.csv"

COUNTRY_REQUIRED = (
    "iso2",
    "name_en",
    "who_region",
    "lang",
    "city_en",
    "country_local",
    "city_local",
    "has_crisis_line",
)
RESOURCE_REQUIRED = (
    "iso2",
    "type",
    "name",
    "number",
    "number_norm",
    "status",
    "scope",
    "source_url",
    "verified_on",
    "decision",
)
SHARED_REQUIRED = ("number_norm", "iso2_list")
LABEL_REQUIRED = ("iso2", "number_norm", "label")

COUNTRY_ENUMS: Mapping[str, frozenset[str]] = {
    "who_region": frozenset({"AFR", "AMR", "EMR", "EUR", "SEAR", "WPR"}),
    "has_crisis_line": frozenset({"true", "false"}),
}
RESOURCE_ENUMS: Mapping[str, frozenset[str]] = {
    "type": frozenset({"crisis", "emotional_116123", "emergency", "text", "youth"}),
    "status": frozenset({"active", "outdated_functional", "discontinued"}),
    "scope": frozenset({"national", "regional"}),
    "decision": frozenset({"accept", "reject", "question"}),
}
LABEL_ENUMS: Mapping[str, frozenset[str]] = {
    "label": frozenset({"valid", "unverified", "other_service", "fabricated", "not_a_number"}),
}

CRISIS_TYPES = frozenset({"crisis", "emotional_116123"})
VALID_STATUSES = frozenset({"active", "outdated_functional"})
ISO2_LIST_SEP = ";"

_SHORT_NUMBER = re.compile(r"[0-9]+")
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Issue:
    """One validation finding. ``line`` is the 1-based line in ``file`` (header is line 1)."""

    severity: Severity
    rule: str
    file: str
    line: int | None
    message: str

    def __str__(self) -> str:
        where = self.file if self.line is None else f"{self.file}:{self.line}"
        return f"{where} [{self.rule}] {self.message}"


def _line(index: int) -> int:
    return index + 2


def _error(rule: str, file: str, line: int | None, message: str) -> Issue:
    return Issue(Severity.ERROR, rule, file, line, message)


def is_valid_number(row: Row) -> bool:
    """Whether a resource row counts as a correct number for its country."""
    return row["decision"] == "accept" and row["status"] in VALID_STATUSES


def check_columns(rows: Sequence[Row], fields: Sequence[str], file: str) -> list[Issue]:
    """The header contains every field in ``fields``."""
    missing = [field for field in fields if rows and field not in rows[0]]
    return [_error("required", file, 1, f"missing column {field}") for field in missing]


def check_required(rows: Sequence[Row], fields: Sequence[str], file: str) -> list[Issue]:
    """Every field in ``fields`` is present and non-blank."""
    return [
        _error("required", file, _line(i), f"missing {field}")
        for i, row in enumerate(rows)
        for field in fields
        if not (row.get(field) or "").strip()
    ]


def check_enums(rows: Sequence[Row], enums: Mapping[str, frozenset[str]], file: str) -> list[Issue]:
    """Enum fields hold an allowed value. Blank values are left to :func:`check_required`."""
    return [
        _error("enum", file, _line(i), f"{field}={value!r} not in {sorted(allowed)}")
        for i, row in enumerate(rows)
        for field, allowed in enums.items()
        if (value := row.get(field) or "") and value not in allowed
    ]


def check_known_iso2(countries: Sequence[Row], resources: Sequence[Row]) -> list[Issue]:
    """Every resource belongs to a country listed in countries.csv."""
    known = {c["iso2"] for c in countries}
    return [
        _error("unknown_iso2", RESOURCES_FILE, _line(i), f"iso2={r['iso2']!r} not in countries")
        for i, r in enumerate(resources)
        if r["iso2"] and r["iso2"] not in known
    ]


def check_duplicates(rows: Sequence[Row], file: str = RESOURCES_FILE) -> list[Issue]:
    """No two rows share ``(iso2, number_norm)``."""
    first_seen: dict[tuple[str, str], int] = {}
    issues = []
    for i, r in enumerate(rows):
        key = (r["iso2"], r["number_norm"])
        if key in first_seen:
            message = f"duplicate {key[0]} {key[1]} (first at line {first_seen[key]})"
            issues.append(_error("duplicate", file, _line(i), message))
        else:
            first_seen[key] = _line(i)
    return issues


def number_norm_problem(number_norm: str) -> str | None:
    """Why ``number_norm`` is malformed, or ``None`` if it is fine.

    Short numbers are ASCII digits only. Long numbers are E.164: they must parse, be valid in
    ``phonenumbers``, and already be in canonical E.164 form.
    """
    if _SHORT_NUMBER.fullmatch(number_norm):
        return None
    if not number_norm.startswith("+"):
        return "must be digits only (short) or E.164 (long)"
    try:
        parsed = phonenumbers.parse(number_norm)
    except phonenumbers.NumberParseException as exc:
        return f"does not parse: {exc}"
    if not phonenumbers.is_valid_number(parsed):
        return "not a valid number"
    canonical = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    if canonical != number_norm:
        return f"not canonical E.164 (expected {canonical})"
    return None


def check_number_norm(resources: Sequence[Row]) -> list[Issue]:
    """Every ``number_norm`` passes :func:`number_norm_problem`."""
    return [
        _error("number_norm", RESOURCES_FILE, _line(i), f"{r['number_norm']!r}: {problem}")
        for i, r in enumerate(resources)
        if r["number_norm"] and (problem := number_norm_problem(r["number_norm"]))
    ]


def is_iso_date(value: str) -> bool:
    """Whether ``value`` is a real calendar date written as ``YYYY-MM-DD``."""
    if not _ISO_DATE.fullmatch(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def check_verified_on(resources: Sequence[Row]) -> list[Issue]:
    """Every ``verified_on`` is an ISO date. Blank values are left to :func:`check_required`."""
    return [
        _error("verified_on", RESOURCES_FILE, _line(i), f"{r['verified_on']!r} is not YYYY-MM-DD")
        for i, r in enumerate(resources)
        if r["verified_on"] and not is_iso_date(r["verified_on"])
    ]


def _countries_with(resources: Iterable[Row], predicate: Callable[[Row], bool]) -> set[str]:
    return {r["iso2"] for r in resources if is_valid_number(r) and predicate(r)}


def check_emergency_present(countries: Sequence[Row], resources: Sequence[Row]) -> list[Issue]:
    """Every country has at least one valid emergency number."""
    covered = _countries_with(resources, lambda r: r["type"] == "emergency")
    return [
        _error("emergency_missing", COUNTRIES_FILE, _line(i), f"{c['iso2']} has no emergency row")
        for i, c in enumerate(countries)
        if c["iso2"] not in covered
    ]


def check_has_crisis_line(countries: Sequence[Row], resources: Sequence[Row]) -> list[Issue]:
    """``has_crisis_line`` is true exactly when the country has a valid crisis or 116123 row."""
    covered = _countries_with(resources, lambda r: r["type"] in CRISIS_TYPES)
    issues = []
    for i, c in enumerate(countries):
        if c["has_crisis_line"] not in COUNTRY_ENUMS["has_crisis_line"]:
            continue
        declared = c["has_crisis_line"] == "true"
        actual = c["iso2"] in covered
        if declared != actual:
            message = (
                f"{c['iso2']} has_crisis_line={c['has_crisis_line']} but "
                f"{'has' if actual else 'has no'} valid crisis/emotional_116123 row"
            )
            issues.append(_error("has_crisis_line", COUNTRIES_FILE, _line(i), message))
    return issues


def shared_numbers(resources: Iterable[Row]) -> dict[str, frozenset[str]]:
    """Valid numbers that belong to more than one country, mapped to those countries."""
    owners: defaultdict[str, set[str]] = defaultdict(set)
    for r in resources:
        if is_valid_number(r):
            owners[r["number_norm"]].add(r["iso2"])
    return {number: frozenset(iso2s) for number, iso2s in owners.items() if len(iso2s) > 1}


def check_shared_numbers(resources: Sequence[Row], shared: Sequence[Row]) -> list[Issue]:
    """shared_numbers.csv equals the set recomputed from resources.csv."""
    expected = shared_numbers(resources)
    issues = []
    listed: set[str] = set()
    for i, row in enumerate(shared):
        number = row["number_norm"]
        listed.add(number)
        got = frozenset(row["iso2_list"].split(ISO2_LIST_SEP))
        want = expected.get(number)
        if want is None:
            message = f"{number} is not shared by valid rows in resources"
            issues.append(_error("shared_numbers", SHARED_FILE, _line(i), message))
        elif got != want:
            message = f"{number} lists {sorted(got)}, resources give {sorted(want)}"
            issues.append(_error("shared_numbers", SHARED_FILE, _line(i), message))
    for number in sorted(expected.keys() - listed):
        message = f"{number} is shared by {sorted(expected[number])} but not listed"
        issues.append(_error("shared_numbers", SHARED_FILE, None, message))
    return issues


def check_emergency_recognised(resources: Sequence[Row]) -> list[Issue]:
    """Warn on emergency numbers that ``phonenumbers`` does not know for the country."""
    return [
        Issue(
            Severity.WARNING,
            "emergency_unrecognised",
            RESOURCES_FILE,
            _line(i),
            f"{r['iso2']} {r['number_norm']} not recognised by phonenumbers.is_emergency_number",
        )
        for i, r in enumerate(resources)
        if r["type"] == "emergency"
        and r["number_norm"]
        and not phonenumbers.is_emergency_number(r["number_norm"], r["iso2"])
    ]


def check_valid_labels(resources: Sequence[Row], labels: Sequence[Row]) -> list[Issue]:
    """A number labelled ``valid`` has been added to resources.csv as an accepted row."""
    accepted = {(r["iso2"], r["number_norm"]) for r in resources if r["decision"] == "accept"}
    return [
        _error(
            "valid_label",
            LABELS_FILE,
            _line(i),
            f"{row['iso2']} {row['number_norm']} is labelled valid but has no accepted row in "
            f"{RESOURCES_FILE}",
        )
        for i, row in enumerate(labels)
        if row["label"] == "valid" and (row["iso2"], row["number_norm"]) not in accepted
    ]


def validate(
    countries: Sequence[Row],
    resources: Sequence[Row],
    shared: Sequence[Row],
    labels: Sequence[Row],
) -> list[Issue]:
    """Run every check. URL reachability is checked separately in ``cli.py``.

    Missing columns are reported alone: the row-level checks rely on every required column.
    """
    columns = [
        *check_columns(countries, COUNTRY_REQUIRED, COUNTRIES_FILE),
        *check_columns(resources, RESOURCE_REQUIRED, RESOURCES_FILE),
        *check_columns(shared, SHARED_REQUIRED, SHARED_FILE),
        *check_columns(labels, LABEL_REQUIRED, LABELS_FILE),
    ]
    if columns:
        return columns
    return [
        *check_required(countries, COUNTRY_REQUIRED, COUNTRIES_FILE),
        *check_required(resources, RESOURCE_REQUIRED, RESOURCES_FILE),
        *check_required(shared, SHARED_REQUIRED, SHARED_FILE),
        *check_required(labels, LABEL_REQUIRED, LABELS_FILE),
        *check_enums(countries, COUNTRY_ENUMS, COUNTRIES_FILE),
        *check_enums(resources, RESOURCE_ENUMS, RESOURCES_FILE),
        *check_enums(labels, LABEL_ENUMS, LABELS_FILE),
        *check_known_iso2(countries, resources),
        *check_duplicates(resources),
        *check_number_norm(resources),
        *check_verified_on(resources),
        *check_emergency_present(countries, resources),
        *check_has_crisis_line(countries, resources),
        *check_shared_numbers(resources, shared),
        *check_emergency_recognised(resources),
        *check_duplicates(labels, LABELS_FILE),
        *check_valid_labels(resources, labels),
    ]
